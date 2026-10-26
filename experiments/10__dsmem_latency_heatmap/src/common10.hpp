/*
 * common10.hpp — experiment 10: SM map with TPC indices, placeholder-CTA steering of a single measuring cluster,
 * stream-ordered device buffers and small statistics helpers.
 *
 * SmMap10 (host)       latest run of the 03 map: gpc[smid], max_cs_observed[smid]; per GPC the list of its TPCs in
 *                      ascending smid order (TPC t = SMs 2t, 2t+1); tpc_index[smid] = rank of the SM's TPC in that list
 *                      (the "TPC index within the GPC"); capable(t) = both SMs joined clusters of ≥ 4 CTAs in 03
 *                      (false only for TPCs 62–65, which pair only with their mate).
 * Mask132              set of SMs (bits) passed by value to the blocker kernel.
 * k_block (device)     the placeholder ("blocker") kernel: 66 clusters of 2 CTAs, 1024 threads, 200 KB dynamic smem
 *                      (→ exactly 1 CTA per SM, all 132 SMs; in a first-wave CS-2 launch the two ranks are TPC mates).
 *                      Each CTA writes its %smid to smem; after a cluster.sync() rank 0 reads its partner's smid through
 *                      DSMEM. The cluster exits if either SM is in the target set T, otherwise it stays resident until
 *                      the host-mapped word *stop becomes non-zero (set by the host: releasing the placeholders needs
 *                      no GPU work, so it cannot queue behind a measuring grid that is waiting for SMs).
 *                      ready[cluster] = 1 + kept is written to host-mapped memory (system-scope fence) once the decision
 *                      is made; info[block] = smid | kept << 16.
 *                      While resident it runs one of three loads (load condition of the session):
 *                        mode 0 idle  : thread 0 polls *stop (volatile load over PCIe) with nanosleep(20 µs) between polls;
 *                                       threads 1–1023 wait in __syncthreads (no issue).
 *                        mode 1 light : all 1024 threads stream the CTA's own 192 KiB with ld.shared (8 independent
 *                                       4-B loads per thread per group, 6 groups per pass); *stop polled every 16 passes.
 *                        mode 2 heavy : the same loads with ld.shared::cluster from the partner CTA (TPC mate) —
 *                                       SM-to-SM traffic at the source-injection limit; *stop polled every 4 passes.
 *                      A final cluster.sync() keeps a CTA resident while its partner may still read its smem.
 * Steer (host)         one session: zero the flags, launch k_block on stream B, wait until all 66 clusters reported,
 *                      run the measuring kernels on stream A (only the SMs of T are free, so a cluster of |T| CTAs
 *                      must land on T), then set *stop (host write) and synchronise B. A measuring kernel
 *                      writes started[0] = 1 (mapped) when it begins; if it has not begun within kStartTimeoutMs the
 *                      session is aborted (stop set, kernel completes wherever it lands, outcome "unplaceable").
 *                      No call in a session may synchronise the device (no cudaMalloc/cudaFree/legacy-stream copies):
 *                      all buffers are allocated up front, all copies are async on stream A. Lazy module loading would
 *                      make the first launch of any kernel wait for the resident placeholders (deadlock): main() sets
 *                      CUDA_MODULE_LOADING=EAGER and preload() runs every memset/copy path once before the first
 *                      session. Stream waits poll with a timeout (sync_or_abort): on a hang the placeholders are released
 *                      and the run dies instead of blocking forever.
 * SBuf<T>              device buffer with async zero/get on a given stream.
 * DBG(...)             stage-by-stage progress of the steering (placeholders, launch, start, release) when the
 *                      environment variable H10_DEBUG is set.
 * Stats helpers        median, MAD (median absolute deviation from the median), linear-interpolated percentiles.
 */
#pragma once
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "harness.cuh"
#include "host.hpp"

constexpr size_t kSmem = 200u * 1024u;  // → 1 CTA per SM for every kernel of this experiment
constexpr int kBlockClusters = 66, kBlockThreads = 1024;
constexpr int kStartTimeoutMs = 1000, kReadyTimeoutMs = 5000;
inline bool dbg() { static const bool d = std::getenv("H10_DEBUG") != nullptr; return d; }
#define DBG(...) do { if (dbg()) { std::printf(__VA_ARGS__); std::fflush(stdout); } } while (0)
constexpr uint32_t kBlkWords = 1024u * 8u * 6u;  // 192 KiB streamed per pass by a light/heavy blocker CTA

enum LoadCond { L_IDLE = 0, L_LIGHT = 1, L_HEAVY = 2, L_NONE = 3 };
inline const char* cond_name(int c) { return c == L_IDLE ? "idle" : c == L_LIGHT ? "light" : c == L_HEAVY ? "heavy" : "none"; }

struct Mask132 {
    uint32_t w[5] = {0, 0, 0, 0, 0};
    void set(uint32_t s) { w[s >> 5] |= 1u << (s & 31); }
    __host__ __device__ bool has(uint32_t s) const { return (w[s >> 5] >> (s & 31)) & 1u; }
};

// ---------------------------------------------------------------- device: blocker kernel
__device__ __forceinline__ uint32_t lds_vol_u32(uint32_t a) {
    uint32_t v; asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory"); return v; }

__device__ __forceinline__ uint32_t ld_mapped_u32(const uint32_t* a) { return *(const volatile uint32_t*)a; }

__global__ void __launch_bounds__(1024) k_block(Mask132 target, uint32_t mode, const uint32_t* stop, uint32_t* ready, uint32_t* info) {
    extern __shared__ __align__(16) uint32_t bw[];
    __shared__ uint32_t s_smid, s_keep, s_stop;
    const uint32_t tid = threadIdx.x;
    const ClusterRegs c = read_cluster_regs();
    const uint32_t me = read_smid();
    if (tid == 0) s_smid = me;
    if (mode != L_IDLE)
        for (uint32_t i = tid; i < kBlkWords; i += kBlockThreads) bw[i] = i;
    __syncthreads();
    cluster_sync();
    if (tid == 0) {
        const uint32_t other = ld_dsmem_u32(mapa_u32(smem_u32(&s_smid), c.ctarank ^ 1u));
        s_keep = !(target.has(me) || target.has(other));
        s_stop = 0;
        info[blockIdx.x] = me | (s_keep << 16);
    }
    __syncthreads();
    cluster_sync();  // partner smid read before anybody exits
    const uint32_t keep = s_keep;
    if (c.ctarank == 0 && tid == 0) {
        *(volatile uint32_t*)(ready + c.clusterid_x) = 1u + keep;
        __threadfence_system();
    }
    if (!keep) return;  // both CTAs of the cluster take the same decision
    uint32_t sink = 0;
    if (mode == L_IDLE) {
        if (tid == 0)
            while (ld_mapped_u32(stop) == 0u) nanosleep_ns(20000);
        __syncthreads();
    } else {
        const uint32_t base = mode == L_HEAVY ? mapa_u32(smem_u32(bw), c.ctarank ^ 1u) : smem_u32(bw);
        const uint32_t every = mode == L_HEAVY ? 4u : 16u;  // ≈ 15–20 µs of streaming between polls
        for (uint32_t pass = 0;; ++pass) {
            if (pass % every == 0) {
                if (tid == 0) s_stop = ld_mapped_u32(stop);
                __syncthreads();
                if (s_stop) break;
                __syncthreads();
            }
#pragma unroll 1
            for (uint32_t g = 0; g < 6; ++g) {
                const uint32_t a = base + (g * 8 * kBlockThreads + tid) * 4;
                uint32_t x[8];
                if (mode == L_HEAVY) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) x[j] = ld_dsmem_u32(a + j * kBlockThreads * 4);
                } else {
#pragma unroll
                    for (int j = 0; j < 8; ++j) x[j] = lds_vol_u32(a + j * kBlockThreads * 4);
                }
                sink += x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7];
            }
        }
    }
    if (sink == 0xffffffffu) info[blockIdx.x] |= 1u << 31;  // keeps the loads (never true for these data)
    __syncthreads();
    cluster_sync();  // no CTA exits while its partner may still read its smem
}

// measuring kernels call this first: rank 0, thread 0 tells the host that the cluster has been placed
__device__ __forceinline__ void signal_started(uint32_t* started) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(volatile uint32_t*)started = 1u;
        __threadfence_system();
    }
}

// ---------------------------------------------------------------- host: map
struct SmMap10 {
    int n = 0, ngpc = 0;
    std::vector<int> gpc, maxcs, tpc_index;
    std::vector<std::vector<int>> tpcs;  // tpcs[g] = TPC ids of GPC g, ascending
    bool capable(int t) const { return maxcs[2 * t] >= 4 && maxcs[2 * t + 1] >= 4; }
    std::vector<int> capable_tpcs(int g) const {
        std::vector<int> v;
        for (int t : tpcs[g]) if (capable(t)) v.push_back(t);
        return v;
    }
};

inline SmMap10 load_map10(int sm_count, const char* path = "../03__sm_gpc_topology/results/03__sm_gpc_topology__map.csv") {
    std::ifstream f(path);
    if (!f) hb::die(std::string("cannot open the 03 map ") + path);
    auto split = [](const std::string& l) {
        std::vector<std::string> v;
        std::stringstream ss(l);
        for (std::string c; std::getline(ss, c, ',');) v.push_back(c);
        return v;
    };
    std::string line;
    std::getline(f, line);
    const auto h = split(line);
    int c_run = -1, c_smid = -1, c_gpc = -1, c_max = -1;
    for (int i = 0; i < (int)h.size(); ++i) {
        if (h[i] == "run_id") c_run = i;
        if (h[i] == "smid") c_smid = i;
        if (h[i] == "gpc") c_gpc = i;
        if (h[i] == "max_cs_observed") c_max = i;
    }
    if (c_run < 0 || c_smid < 0 || c_gpc < 0 || c_max < 0) hb::die("03 map lacks run_id/smid/gpc/max_cs_observed columns");
    std::vector<std::vector<std::string>> rows;
    std::string latest;
    while (std::getline(f, line)) {
        auto r = split(line);
        if ((int)r.size() <= std::max(c_gpc, c_max)) continue;
        if (r[c_run] > latest) latest = r[c_run];
        rows.push_back(std::move(r));
    }
    SmMap10 m;
    m.n = sm_count;
    m.gpc.assign(sm_count, -1);
    m.maxcs.assign(sm_count, 0);
    m.tpc_index.assign(sm_count, -1);
    for (const auto& r : rows)
        if (r[c_run] == latest) {
            const int s = std::stoi(r[c_smid]);
            if (s >= 0 && s < sm_count) { m.gpc[s] = std::stoi(r[c_gpc]); m.maxcs[s] = std::stoi(r[c_max]); }
        }
    for (int s = 0; s < sm_count; ++s) {
        if (m.gpc[s] < 0) hb::die("03 map has no GPC for smid " + std::to_string(s));
        m.ngpc = std::max(m.ngpc, m.gpc[s] + 1);
    }
    m.tpcs.assign(m.ngpc, {});
    for (int t = 0; t < sm_count / 2; ++t) {
        if (m.gpc[2 * t] != m.gpc[2 * t + 1]) hb::die("TPC " + std::to_string(t) + " spans two GPCs in the 03 map");
        m.tpcs[m.gpc[2 * t]].push_back(t);
    }
    for (int g = 0; g < m.ngpc; ++g)
        for (int i = 0; i < (int)m.tpcs[g].size(); ++i) m.tpc_index[2 * m.tpcs[g][i]] = m.tpc_index[2 * m.tpcs[g][i] + 1] = i;
    std::printf("03 map run %s: %d GPCs\n", latest.c_str(), m.ngpc);
    return m;
}

// ---------------------------------------------------------------- host: buffers on a stream
template <class T>
struct SBuf {
    T* d = nullptr;
    size_t n = 0;
    explicit SBuf(size_t n_) : n(std::max<size_t>(1, n_)) { CUDA_CHECK(cudaMalloc(&d, n * sizeof(T))); CUDA_CHECK(cudaMemset(d, 0, n * sizeof(T))); }
    ~SBuf() { cudaFree(d); }
    SBuf(const SBuf&) = delete;
    SBuf& operator=(const SBuf&) = delete;
    void zero(cudaStream_t s, size_t m = 0) { CUDA_CHECK(cudaMemsetAsync(d, 0, (m ? m : n) * sizeof(T), s)); }
    // copy the first m elements back (stream-ordered, then synchronised)
    void get(std::vector<T>& h, cudaStream_t s, size_t m = 0) {
        h.resize(m ? m : n);
        CUDA_CHECK(cudaMemcpyAsync(h.data(), d, h.size() * sizeof(T), cudaMemcpyDeviceToHost, s));
        const auto t0 = std::chrono::steady_clock::now();
        while (cudaStreamQuery(s) == cudaErrorNotReady) {
            if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(60)) hb::die("device-to-host copy timed out");
            usleep(5);
        }
        CUDA_CHECK(cudaGetLastError());
    }
};

// ---------------------------------------------------------------- host: steering
struct Steer {
    cudaStream_t A{}, B{}, C{};
    uint32_t* h_flags = nullptr;  // mapped: [0, 66) ready, [100] started, [101] stop
    uint32_t* d_flags = nullptr;
    SBuf<uint32_t> info{132};
    std::vector<uint32_t> h_info;
    hb::ClusterLaunch blk;
    bool active = false;

    Steer() {
        CUDA_CHECK(cudaStreamCreateWithFlags(&A, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&B, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&C, cudaStreamNonBlocking));
        CUDA_CHECK(cudaHostAlloc(&h_flags, 128 * sizeof(uint32_t), cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(&d_flags, h_flags, 0));
        std::memset(h_flags, 0, 128 * sizeof(uint32_t));
        blk.grid = dim3(kBlockClusters * 2);
        blk.block = dim3(kBlockThreads);
        blk.cluster = dim3(2, 1, 1);
        blk.smem = kSmem;
        blk.stream = B;
        blk.policy = cudaClusterSchedulingPolicyDefault;
        if (blk.max_active_clusters(k_block) < kBlockClusters)
            hb::die("blocker: fewer than 66 CS-2 clusters can be co-resident at 200 KB");
    }
    uint32_t* started() const { return d_flags + 100; }
    volatile uint32_t* hv() const { return h_flags; }
    void set_stop() { hv()[101] = 1; std::atomic_thread_fence(std::memory_order_seq_cst); }
    // cudaStreamSynchronize with a timeout: on a hang release the placeholders and die
    void sync_or_abort(cudaStream_t s, const char* what, int timeout_s = 60) {
        const auto t0 = std::chrono::steady_clock::now();
        for (;;) {
            const cudaError_t e = cudaStreamQuery(s);
            if (e == cudaSuccess) return;
            if (e != cudaErrorNotReady) CUDA_CHECK(e);
            if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(timeout_s)) {
                set_stop();
                hb::die(std::string("stream wait timed out: ") + what);
            }
            usleep(20);
        }
    }
    // run the memset and copy paths once on every stream before any placeholder is resident
    void preload() {
        std::vector<uint32_t> h;
        for (cudaStream_t s : {A, B, C}) {
            info.zero(s);
            info.get(h, s);
        }
    }

    // Start the placeholder CTAs; returns the number of kept clusters.
    int begin(const Mask132& T, int mode) {
        for (int i = 0; i < 128; ++i) hv()[i] = 0;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        DBG("[steer] launching placeholders mode %d\n", mode);
        CUDA_CHECK(blk.launch(k_block, T, (uint32_t)mode, (const uint32_t*)(d_flags + 101), d_flags, info.d));
        CUDA_CHECK(cudaGetLastError());
        active = true;
        const auto t0 = std::chrono::steady_clock::now();
        for (;;) {
            int n = 0, kept = 0;
            for (int i = 0; i < kBlockClusters; ++i) { const uint32_t r = hv()[i]; n += r != 0; kept += r == 2; }
            if (n == kBlockClusters) { DBG("[steer] all 66 reported, %d kept\n", kept); return kept; }
            if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(kReadyTimeoutMs)) {
                end();
                hb::die("blocker clusters did not all start (" + std::to_string(n) + " of 66)");
            }
            usleep(10);
        }
    }
    void reset_started() { hv()[100] = 0; std::atomic_thread_fence(std::memory_order_seq_cst); }
    // Wait until the measuring kernel just launched on A has begun; false on timeout.
    bool wait_started() {
        const auto t0 = std::chrono::steady_clock::now();
        while (hv()[100] == 0) {
            if (cudaStreamQuery(A) == cudaSuccess) return hv()[100] != 0;
            if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(kStartTimeoutMs)) return false;
            usleep(20);
        }
        return true;
    }
    // Release the placeholders and wait for them; afterwards h_info holds smid | kept << 16 per blocker CTA.
    void end() {
        if (!active) return;
        DBG("[steer] end: setting stop\n");
        set_stop();
        sync_or_abort(B, "placeholders");
        DBG("[steer] placeholders done\n");
        CUDA_CHECK(cudaGetLastError());
        info.get(h_info, C);
        active = false;
    }
};

// ---------------------------------------------------------------- statistics
template <class T>
inline double pct_sorted(const std::vector<T>& s, double q) {
    if (s.empty()) return NAN;
    const double x = q * (s.size() - 1);
    const size_t i = (size_t)x;
    const double f = x - i;
    return i + 1 < s.size() ? s[i] * (1 - f) + s[i + 1] * f : (double)s[i];
}
template <class T>
inline double median_sorted(const std::vector<T>& s) { return pct_sorted(s, 0.5); }
template <class T>
inline double mad_sorted(const std::vector<T>& s) {  // s sorted
    if (s.empty()) return NAN;
    const double m = median_sorted(s);
    std::vector<double> d(s.size());
    for (size_t i = 0; i < s.size(); ++i) d[i] = std::fabs((double)s[i] - m);
    std::sort(d.begin(), d.end());
    return median_sorted(d);
}
