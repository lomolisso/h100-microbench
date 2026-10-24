/*
 * sync_common.hpp — experiment 07: pieces shared by the sync_* binaries.
 *
 * Device
 *   Rec                      one sample: c0 = cycles of warp 0 (lane 0), c1 = a second warp's value where a kernel
 *                            needs one (bar.arrive producer, mbarrier wake-up), cmax = slowest warp of the block
 *                            (each warp's own t1 − t0, same SM clock), smid.
 *   spin_cycles(n)           busy wait of ≥ n SM cycles on %clock64 (delays; not a measured quantity).
 *   alu_chain(n, x, zero)    n dependent mad.lo.u32 (the "independent work" of split-phase tests; its own time is
 *                            measured as K1 primitive alu_chain).
 *   bar_sync0_then_lds(a)    bar.sync 0 followed by a volatile shared load; the caller reads t0 with
 *                            read_clock64_after(value, zero). bar.sync compiles to BAR.SYNC.DEFER_BLOCKING: a warp that
 *                            reaches it early keeps issuing until its first memory instruction, so a %clock64 read
 *                            right after bar.sync can precede the barrier's completion by the arrival skew (measured:
 *                            150 cycles with no skew source, the full skew otherwise). The load blocks until the barrier
 *                            has completed, and t0 waits for the load.
 *   block_record(...)        after a timed region: lane 0 of every warp publishes its cycles, thread 0 writes the Rec
 *                            (one bar.sync 0, skipped when the block has a single active thread).
 *   k_timer_floor            32 × (clock64 → clock64) per rep, one thread: the timer-pair cost subtracted from every
 *                            single-op sample of the same run (rows kernel k0, primitive timer_floor).
 * Host
 *   Out07                    the two results files of a binary: primitives (07__sync__primitives__<kN>.csv) and
 *                            07__sync__propagation.csv, with the mandatory columns first; per-launch flush, per-config
 *                            fsync (CsvWriter: append, flock). open_alt(k) adds a second primitives file
 *                            (07__sync__primitives__<k>.csv) for binaries whose rows would exceed ~600 MB in one file.
 *   ClusterOnce              one cluster per launch, 1 CTA/SM forced with 200 KB dynamic smem, explicit cluster
 *                            attribute (also for CS 1), Default scheduling policy; checks that the ranks sit on
 *                            distinct SMs (a launch that fails is discarded and repeated, counted in `discarded`).
 */
#pragma once
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "harness.cuh"
#include "host.hpp"
#include "../../04__memory_hierarchy_latency/src/gpc_map.hpp"

constexpr uint32_t kReps = 1024, kQuickReps = 64;
constexpr int kLaunches = 5, kQuickLaunches = 2, kWarmups = 2;
constexpr size_t kOneCtaPerSmSmem = 200u * 1024u;

struct Rec { uint32_t c0, c1, cmax, smid; };

// ---------------------------------------------------------------- device helpers
__device__ __forceinline__ void spin_cycles(uint64_t n) {
    const uint64_t t = read_clock64();
    while (read_clock64() - t < n) { }
}
__device__ __forceinline__ uint32_t mad_lo_u32(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r; asm volatile("mad.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
__device__ __forceinline__ uint32_t alu_chain(uint32_t n, uint32_t x, uint32_t zero) {
#pragma unroll 1
    for (uint32_t i = 0; i < n / 8; ++i) {
#pragma unroll
        for (int u = 0; u < 8; ++u) x = mad_lo_u32(x, 3u, zero + 1u);
    }
    return x;
}

__device__ __forceinline__ uint32_t lds_volatile_u32(uint32_t a) {
    uint32_t v; asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory"); return v; }
__device__ __forceinline__ uint32_t bar_sync0_then_lds(uint32_t a) { bar_sync0(); return lds_volatile_u32(a); }

// Called by every active thread of the block after its timed region. `cyc` = this warp's measured cycles (lane 0's
// value is used), `c1` = a second value carried by warp 1 (0 if unused). Thread 0 writes out[idx].
__device__ __forceinline__ void block_record(uint32_t* s_w, uint32_t cyc, uint32_t c1, uint32_t smid, Rec* out,
                                             uint32_t idx, bool single_thread) {
    const uint32_t warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    if (single_thread) {
        out[idx] = Rec{cyc, c1, cyc, smid};
        return;
    }
    if ((threadIdx.x & 31) == 0) { s_w[warp] = cyc; s_w[32 + warp] = c1; }
    bar_sync0();
    if (threadIdx.x == 0) {
        uint32_t m = 0;
        for (uint32_t w = 0; w < nw; ++w) m = s_w[w] > m ? s_w[w] : m;
        out[idx] = Rec{s_w[0], nw > 1 ? s_w[33] : 0u, m, smid};
    }
    bar_sync0();
}

__global__ void __launch_bounds__(32) k_timer_floor(uint32_t reps, Rec* out) {
    if (threadIdx.x != 0) return;
    const uint32_t smid = read_smid();
#pragma unroll 1
    for (uint32_t r = 0; r < reps; ++r) {
        uint64_t s = 0;
#pragma unroll
        for (int u = 0; u < 32; ++u) {
            const uint64_t t0 = read_clock64();
            const uint64_t t1 = read_clock64();
            s += t1 - t0;
        }
        out[r] = Rec{(uint32_t)s, 0u, (uint32_t)s, smid};
    }
}

// ---------------------------------------------------------------- host helpers
inline std::vector<std::string> prim_cols() {
    return hb::with_mandatory({"kernel", "primitive", "cs", "block_threads", "variant", "rep", "cycles", "rank", "smid",
                               "n_ops", "cycles_warp_max"});
}
inline std::vector<std::string> prop_cols() {
    return hb::with_mandatory({"primitive", "cs", "smid_src", "smid_dst", "one_way_ns", "round_trip_cycles",
                               "block_threads", "variant", "rep"});
}

struct Out07 {
    hb::Run& run;
    hb::CsvWriter prim, prop;
    std::unique_ptr<hb::CsvWriter> alt;  // optional second primitives file (open_alt)
    long long li = 0;  // launch_idx, counts every recorded launch of the binary invocation
    int discarded = 0;
    Out07(hb::Run& r, const std::string& kfile)
        : run(r), prim(hb::Run::results("07__sync__primitives__" + kfile + ".csv"), prim_cols()),
          prop(hb::Run::results("07__sync__propagation.csv"), prop_cols()) {}

    void open_alt(const std::string& kfile) {
        alt = std::make_unique<hb::CsvWriter>(hb::Run::results("07__sync__primitives__" + kfile + ".csv"), prim_cols());
    }
    // One row per sample. recs[rank * reps + rep]; `use_c1` selects Rec::c1 as the cycles value; `to_alt` writes into
    // the open_alt file.
    void prim_rows(const hb::ClockSample& s, const char* kernel, const std::string& primitive, int cs, int bt,
                   const std::string& variant, const std::vector<Rec>& recs, uint32_t reps, int nranks, int n_ops,
                   bool use_c1 = false, bool to_alt = false) {
        hb::CsvWriter& w = to_alt ? *alt : prim;
        const std::string pre = run.prefix(li, s);
        char b[256];
        for (int k = 0; k < nranks; ++k)
            for (uint32_t r = 0; r < reps; ++r) {
                const Rec& x = recs[(size_t)k * reps + r];
                std::snprintf(b, sizeof b, ",%s,%s,%d,%d,%s,%u,%u,%d,%u,%d,%u", kernel, primitive.c_str(), cs, bt,
                              variant.c_str(), r, use_c1 ? x.c1 : x.c0, k, x.smid, n_ops, x.cmax);
                w.add(pre + b);
            }
    }
    // round_trip < 0 → empty cell; one_way_ns may be negative (timer offsets ≤ 1 tick between SMs).
    void prop_row(const hb::ClockSample& s, const std::string& primitive, int cs, uint32_t smid_src, uint32_t smid_dst,
                  long long one_way_ns, long long round_trip, int bt, const std::string& variant, uint32_t rep,
                  const std::string& pre_cached = "") {
        char b[256];
        char rt[32] = "";
        if (round_trip >= 0) std::snprintf(rt, sizeof rt, "%lld", round_trip);
        std::snprintf(b, sizeof b, ",%s,%d,%u,%u,%lld,%s,%d,%s,%u", primitive.c_str(), cs, smid_src, smid_dst, one_way_ns,
                      rt, bt, variant.c_str(), rep);
        prop.add((pre_cached.empty() ? run.prefix(li, s) : pre_cached) + b);
    }
    void flush() { prim.flush(); prop.flush(); if (alt) alt->flush(); }
    void sync() { prim.sync(); prop.sync(); if (alt) alt->sync(); }
};

// One cluster of `cs` CTAs per launch, 1 CTA/SM.
struct ClusterOnce {
    hb::ClusterLaunch L;
    ClusterOnce(int cs, int block, size_t smem = kOneCtaPerSmSmem) {
        L.grid = dim3(cs);
        L.block = dim3(block);
        L.cluster = dim3(cs, 1, 1);
        L.smem = smem;
        L.explicit_cluster = true;
        L.policy = cudaClusterSchedulingPolicyDefault;
    }
};

inline bool ranks_distinct(const std::vector<Rec>& recs, uint32_t reps, int nranks) {
    std::vector<uint32_t> sm(nranks);
    for (int k = 0; k < nranks; ++k) sm[k] = recs[(size_t)k * reps].smid;
    return hb::all_distinct(sm);
}

template <class T>
struct DevBuf {
    T* d = nullptr;
    size_t n = 0;
    explicit DevBuf(size_t n_) : n(n_) { CUDA_CHECK(cudaMalloc(&d, n * sizeof(T))); CUDA_CHECK(cudaMemset(d, 0, n * sizeof(T))); }
    ~DevBuf() { cudaFree(d); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    std::vector<T> get(size_t m = 0) const {
        std::vector<T> h(m ? m : n);
        CUDA_CHECK(cudaMemcpy(h.data(), d, h.size() * sizeof(T), cudaMemcpyDeviceToHost));
        return h;
    }
    void zero() { CUDA_CHECK(cudaMemset(d, 0, n * sizeof(T))); }
};

// The timer-pair floor of this run: kLaunches recorded launches (after warm-ups) of k_timer_floor → kernel k0 rows
// (cycles = sum of 32 pairs; n_ops = 32).
inline void timer_floor(Out07& o, uint32_t reps, int launches) {
    DevBuf<Rec> d(reps);
    for (int l = 0; l < kWarmups + launches; ++l) {
        k_timer_floor<<<1, 32>>>(reps, d.d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        if (l < kWarmups) continue;
        const hb::ClockSample s = o.run.clock_note("k0_timer_floor", "launch", o.li);
        o.prim_rows(s, "k0", "timer_floor", 0, 1, "pair", d.get(), reps, 1, 32);
        o.flush();
        ++o.li;
    }
    o.sync();
    double sum = 0;
    for (const Rec& r : d.get()) sum += r.c0;
    std::printf("timer floor: %.2f cycles per clock64 pair\n", sum / (32.0 * reps));
}
