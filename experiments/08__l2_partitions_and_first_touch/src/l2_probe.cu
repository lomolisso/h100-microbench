/*
 * l2_probe.cu — experiment 08: latency of individual L2 lines from chosen SMs under controlled first touch.
 * Which partition holds a line, whether reads leave copies, how a line's home depends on its address bits, and the
 * near/far signature of every SM.
 *
 * Usage: bin/l2_probe {calib|k1|k2|k3|k5|ncu_lines|ncu_fabric|clockcheck} [--quick] [--label L]. One invocation = one run_id.
 *
 * Building blocks (every mode):
 *   Lines: 128 B lines of a cudaMalloc buffer (≥ 8 MiB, 2 MiB pages, base 2 MiB-aligned: checked). Line offsets in
 *     the CSVs are byte offsets from the buffer base. The last line of every 2 MiB page is reserved as that page's
 *     TLB-warm line and never probed.
 *   Cache state: CleanL2 (05's bw_common.hpp: 128 MB read-modify-write flush, then a 128 MB clean read) before every
 *     configuration, so that no line of the buffer is in L2 and no dirty write-back overlaps the probe.
 *   k_touch: a one-warp block claimed on SM T (claim_target_sm) writes every listed line COMPLETELY (16 lanes per
 *     line: qword 0 = the line's own address, qword 1 = a token, qwords 2-15 = 0). After CleanL2 this is the line's
 *     first touch; whole lines leave no partially valid sectors that would need a fill from HBM.
 *   k_probe<V>: 132 blocks × 32 threads with 200 KB dynamic smem (exactly 1 CTA/SM; the block on SM P wins the claim,
 *     every other block returns). Lane 0 walks the line list; per line:
 *       a = list[i] (+ (p & zero): ordered after the previous line); one untimed ld.global.cg of the page's TLB-warm
 *       line (so the first timed read pays no TLB miss); `warm` untimed loads of the line (the first t0 waits for
 *       them); then S timed loads:
 *         t0 = %clock64; p = ld.global.{cg|ca}.u64 [p]; t1 = read_clock64_after(p)  →  cycles = t1 − t0 (u32, smem).
 *       qword 0 of the line points to the line itself, so every load depends on the previous one and hits the same
 *       line. Sample 0 is the line's first read by this SM. After the S samples one untimed .cg load of qword 1
 *       returns the token (coherence check in K3). Samples are copied to global in batches of 51 200 (smem full),
 *       outside every timed pair. cycles include the timer pair (calib measures it).
 *   label_panel: CleanL2 → SM 0 (GPC 0) touches N candidate lines → SM 2 (GPC 1) reads each once (S = 1). 2-means
 *     on these first reads: far ⇒ the line's home is SM 0's partition (label 0), near ⇒ SM 2's (label 1). The
 *     partitions are named after these two reference SMs. Label samples → probe.csv, kernel "label".
 * Modes:
 *   calib — 04's per-access ld.shared chain (1024 words, stride 1) with the same timer pair, 4096 samples, 3
 *     launches on SM 0 and SM 2 → timer-pair cost = median − 23 (ld.shared latency). probe.csv kernel "calib".
 *   k1 — spec K1. 32 lines (16 per home, 8 MiB buffer). For P ∈ {0, 2}, toucher T ∈ {P, the other one},
 *     warm ∈ {0, 1}, variant ∈ {cg, ca}: CleanL2 → touch on T → probe on P (S = 64). 5 launches.
 *   k2 — spec K2 (partition hash bits). 2 GiB buffer (so that bit 30 can be flipped inside it). Lines: 32 random
 *     base lines L; L ^ 2^b for b = 7..30 (single); L ^ 2^b1 ^ 2^b2 for 7 ≤ b1 < b2 ≤ 20 (pair); 32 random masks of
 *     bits 7..20 per base (mask); 4 fixed in-page offsets in every one of the 1024 pages (page). Duplicates are probed once. Directions (P, T) = (2, 0) and
 *     (0, 2), 3 reps (fresh random probe order each). CleanL2 → T touches all lines → P probes all (S = 64). Class
 *     from sample 0 (2-means): near ⇒ home = P's partition, far ⇒ T's. → probe_k2.csv (every sample, line_home
 *     filled), hashbits.csv (one row per (base, variant) pair of the single/pair/mask sets).
 *   k3 — spec K3 (replication). SM triples (A, B, C): B in the other partition than A, C in B's partition but another
 *     GPC (prior labels from 04, re-measured here); plus one control triple inside one partition. 32 lines
 *     (16 per home). Phases: CleanL2; A touches (token t0); B reads (S = 64); C reads; A reads; A rewrites the lines
 *     (token t1); B reads, then checks qword 1 == t1. seq_mode "launch": one kernel per phase; "flag": one 132-block
 *     launch, roles by %smid, phases ordered by st.release / ld.acquire.gpu flags (2 s timeout), %globaltimer at
 *     every phase start/end. 5 reps. → probe.csv kernel "K3" (every sample) and replication.csv (per line and phase:
 *     sample 0 = *_first, median of samples 1..63 = *_repeat).
 *   k5 — spec K5 (per-SM signature). 256 lines of an 8 MiB buffer: 128 homed in each partition, each touched by its
 *     home reference SM (0 or 2), so every line exists only at home. For every SM P of the 132 and 5 reps:
 *     CleanL2 → both touches → probe on P (S = 64, fixed random line order). → sm_signature_samples.csv, one row per
 *     (rep, P, line) with sample 0 (first_cycles) and all 64 samples in order (cycles_seq, ';'-joined).
 *   ncu_lines — for ncu (application replay; profile regex k_ncu): 32 fixed lines (first line of 4 KiB pages 0..31),
 *     each: CleanL2 → SM 0 touches → k_ncu_line on P (S = 64), P ∈ {2, 0}; then k_ncu_l1_cg and k_ncu_l1_ca on SM 0
 *     over the same 32 lines (S = 64, L1 hit counters). Kernel order + sample 0 → ncu_manifest.csv. ncu 2025.2.1
 *     exports no per-slice instance values for lts__ counters, so the .max rollup over the 96 slices shows whether
 *     one slice served all reads of the line.
 *   ncu_fabric — label a 16 MiB buffer (SM 0 touches, SM 2 reads every line once); CleanL2 → SM 0 touches all →
 *     k_ncu_empty, then k_ncu_fab on SM 2 (S = 1): far lines first read, near lines first read, far lines second read.
 *   clockcheck — clock/throttle note only.
 * Every configuration samples the clocks (clock_log); claim retries per mode → launch_stats.csv.
 */
#include <algorithm>
#include <cstring>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "../../04__memory_hierarchy_latency/src/gpc_map.hpp"
#include "../../05__memory_hierarchy_bandwidth/src/bw_common.hpp"
#include "harness.cuh"
#include "host.hpp"

static const char* EXP = "08__l2_partitions_and_first_touch";
static constexpr uint64_t kLine = 128, kPage = 2ull << 20;
static constexpr uint32_t kS = 64, kBatch = 51200, kCalibHops = 4096, kCalibWords = 1024;
static constexpr size_t kProbeSmem = kBatch * sizeof(uint32_t);  // 200 KB → 1 CTA/SM
static constexpr uint32_t kRef0 = 0, kRef1 = 2;                  // reference SMs: partition labels 0 and 1
static constexpr int kMaxRetries = 20;

// ---------------------------------------------------------------- device
__device__ __forceinline__ uint64_t ld_cg(uint64_t p) {
    uint64_t v;
    asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint64_t ld_ca(uint64_t p) {
    uint64_t v;
    asm volatile("ld.global.ca.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
template <int V> __device__ __forceinline__ uint64_t ld_line(uint64_t p) { return V == 1 ? ld_ca(p) : ld_cg(p); }

// Whole-line writes by one warp: lanes 0-15 write line i, lanes 16-31 line i+1 (qword 0 = own address, 1 = token).
__device__ __forceinline__ void touch_list(const uint64_t* lines, uint32_t n, uint64_t token) {
    const uint32_t lane = threadIdx.x & 31, q = lane & 15;
    for (uint32_t i = lane >> 4; i < n; i += 2) {
        const uint64_t a = lines[i];
        reinterpret_cast<uint64_t*>(a)[q] = q == 0 ? a : (q == 1 ? token : 0ull);
    }
    __syncwarp();
}

// The whole warp calls this; lane 0 measures, all lanes copy the samples out. out[i*S + k], q1[i] (if non-null),
// tns[0..1] = %globaltimer at start / end.
template <int V>
__device__ __forceinline__ void probe_list(const uint64_t* lines, const uint64_t* tlb, uint32_t n, uint32_t S, uint32_t warm,
                                           uint64_t zero, uint32_t* s, uint32_t* out, uint64_t* q1, uint64_t* tns, uint64_t* sink) {
    const uint32_t lane = threadIdx.x & 31, per_batch = kBatch / S;
    uint64_t p = 0;
    if (lane == 0 && tns) tns[0] = read_globaltimer();
    for (uint32_t i0 = 0; i0 < n; i0 += per_batch) {
        const uint32_t m = min(per_batch, n - i0);
        if (lane == 0) {
#pragma unroll 1
            for (uint32_t i = 0; i < m; ++i) {
                uint64_t a = lines[i0 + i] + (p & zero);
                if (tlb) a += ld_cg(tlb[i0 + i] + (a & zero)) & zero;
                p = a;
#pragma unroll 1
                for (uint32_t w = 0; w < warm; ++w) p = ld_line<V>(p);
                p += read_clock64_after(p, zero) & zero;  // the first t0 must not issue before the warm load returned
#pragma unroll 1
                for (uint32_t k = 0; k < S; ++k) {
                    const uint64_t t0 = read_clock64();
                    p = ld_line<V>(p);
                    const uint64_t t1 = read_clock64_after(p, zero);
                    s[i * S + k] = (uint32_t)(t1 - t0);
                }
                if (q1) q1[i0 + i] = ld_cg(p + 8);
            }
        }
        __syncwarp();
        for (uint32_t k = lane; k < m * S; k += 32) out[(size_t)i0 * S + k] = s[k];
        __syncwarp();
    }
    if (lane == 0) {
        if (tns) tns[1] = read_globaltimer();
        sink[0] = p;
    }
}

__global__ void __launch_bounds__(32) k_touch(const uint64_t* lines, uint32_t n, uint64_t token, uint32_t target, uint32_t* claim) {
    if (!claim_target_sm(target, claim)) return;
    touch_list(lines, n, token);
}

template <int V>
__global__ void __launch_bounds__(32) k_probe(const uint64_t* lines, const uint64_t* tlb, uint32_t n, uint32_t S, uint32_t warm,
                                               uint32_t target, uint32_t* claim, uint64_t zero, uint32_t* out, uint64_t* q1,
                                               uint64_t* tns, uint64_t* sink) {
    extern __shared__ uint32_t s[];
    if (!claim_target_sm(target, claim)) return;
    probe_list<V>(lines, tlb, n, S, warm, zero, s, out, q1, tns, sink);
}

// Same code under names that ncu selects with --kernel-name regex:k_ncu.
#define NCU_PROBE(name, V)                                                                                                     \
    __global__ void __launch_bounds__(32) name(const uint64_t* lines, const uint64_t* tlb, uint32_t n, uint32_t S, uint32_t warm, \
                                                uint32_t target, uint32_t* claim, uint64_t zero, uint32_t* out, uint64_t* q1,   \
                                                uint64_t* tns, uint64_t* sink) {                                                \
        extern __shared__ uint32_t s[];                                                                                         \
        if (!claim_target_sm(target, claim)) return;                                                                            \
        probe_list<V>(lines, tlb, n, S, warm, zero, s, out, q1, tns, sink);                                                     \
    }
NCU_PROBE(k_ncu_line, 0)
NCU_PROBE(k_ncu_l1_cg, 0)
NCU_PROBE(k_ncu_l1_ca, 1)
NCU_PROBE(k_ncu_fab, 0)
__global__ void k_ncu_empty() {}

// K3 in one launch: roles by %smid (1 CTA/SM, all 132 resident), phases ordered by a flag in global memory.
// Phase p (0 b, 1 c, 2 a_after, 3 b_after_a_write) writes out[p*n*S ...], q1[p*n ...], tns[2p..2p+1].
__device__ __forceinline__ bool flag_wait(const uint32_t* flag, uint32_t v, uint32_t* status) {
    uint32_t ok = 1;
    if ((threadIdx.x & 31) == 0) {
        const uint64_t t_end = read_globaltimer() + 2000000000ull;
        while (ld_acquire_gpu_u32(flag) < v)
            if (read_globaltimer() > t_end) { atomicExch(status, v); ok = 0; break; }
    }
    return __shfl_sync(0xffffffffu, ok, 0) != 0;
}
__device__ __forceinline__ void flag_set(uint32_t* flag, uint32_t v) {
    __syncwarp();
    if ((threadIdx.x & 31) == 0) st_release_gpu_u32(flag, v);
}
__global__ void __launch_bounds__(32) k3_flag(const uint64_t* lines, const uint64_t* tlb, uint32_t n, uint32_t A, uint32_t B, uint32_t C,
                                               uint64_t tok0, uint64_t tok1, uint32_t* flag, uint64_t zero, uint32_t* out,
                                               uint64_t* q1, uint64_t* tns, uint64_t* sink, uint32_t* status) {
    extern __shared__ uint32_t s[];
    const uint32_t sm = read_smid();
    const size_t ph = (size_t)n * kS;
    if (sm == A) {
        touch_list(lines, n, tok0);
        flag_set(flag, 1);
        if (!flag_wait(flag, 3, status)) return;
        probe_list<0>(lines, tlb, n, kS, 0, zero, s, out + 2 * ph, q1 + 2 * n, tns + 4, sink);
        touch_list(lines, n, tok1);
        flag_set(flag, 4);
    } else if (sm == B) {
        if (!flag_wait(flag, 1, status)) return;
        probe_list<0>(lines, tlb, n, kS, 0, zero, s, out, q1, tns, sink);
        flag_set(flag, 2);
        if (!flag_wait(flag, 4, status)) return;
        probe_list<0>(lines, tlb, n, kS, 0, zero, s, out + 3 * ph, q1 + 3 * n, tns + 6, sink);
    } else if (sm == C) {
        if (!flag_wait(flag, 2, status)) return;
        probe_list<0>(lines, tlb, n, kS, 0, zero, s, out + ph, q1 + n, tns + 2, sink);
        flag_set(flag, 3);
    }
}

// 04's timer calibration: per-access ld.shared chain with the same timer pair.
__global__ void __launch_bounds__(32) k_calib(uint32_t hops, uint32_t target, uint32_t* claim, uint64_t zero, uint32_t* out, uint64_t* sink) {
    extern __shared__ uint32_t s[];  // [0, kCalibWords): chain; then hops samples
    if (!claim_target_sm(target, claim)) return;
    uint32_t* samples = s + kCalibWords;
    for (uint32_t i = threadIdx.x; i < kCalibWords; i += 32) s[i] = smem_u32(&s[(i + 1) % kCalibWords]);
    __syncwarp();
    if (threadIdx.x == 0) {
        uint32_t p = smem_u32(&s[0]);
#pragma unroll 1
        for (uint32_t k = 0; k < hops; ++k) {
            const uint64_t t0 = read_clock64();
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(p) : "r"(p) : "memory");
            const uint64_t t1 = read_clock64_after(p, zero);
            samples[k] = (uint32_t)(t1 - t0);
        }
        sink[0] = p;
    }
    __syncwarp();
    for (uint32_t k = threadIdx.x; k < hops; k += 32) out[k] = samples[k];
}

// ---------------------------------------------------------------- host
static uint64_t tlb_line(uint64_t off) { return (off & ~(kPage - 1)) + kPage - kLine; }
static bool probe_able(uint64_t off) { return (off & (kPage - 1)) != kPage - kLine; }

// Line list on the device: addresses of the lines and of their pages' TLB-warm lines.
struct DevList {
    std::vector<uint64_t> off;
    uint64_t* addr = nullptr;
    uint64_t* tlb = nullptr;
    DevList(uint64_t base, const std::vector<uint64_t>& o) : off(o) {
        std::vector<uint64_t> a(o.size()), t(o.size());
        for (size_t i = 0; i < o.size(); ++i) {
            if (!probe_able(o[i])) hb::die("a TLB-warm line was listed for probing");
            a[i] = base + o[i];
            t[i] = base + tlb_line(o[i]);
        }
        CUDA_CHECK(cudaMalloc(&addr, std::max<size_t>(1, a.size()) * 8));
        CUDA_CHECK(cudaMalloc(&tlb, std::max<size_t>(1, t.size()) * 8));
        CUDA_CHECK(cudaMemcpy(addr, a.data(), a.size() * 8, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(tlb, t.data(), t.size() * 8, cudaMemcpyHostToDevice));
    }
    ~DevList() { cudaFree(addr); cudaFree(tlb); }
    DevList(const DevList&) = delete;
    DevList& operator=(const DevList&) = delete;
    uint32_t n() const { return (uint32_t)off.size(); }
};

// 2-means on 1-D values: returns the threshold (largest value of the low class); dies if the classes are < 80 apart.
static double split2(const std::vector<double>& v, const char* what) {
    double lo = *std::min_element(v.begin(), v.end()), hi = *std::max_element(v.begin(), v.end());
    double thr = 0.5 * (lo + hi);
    for (int it = 0; it < 100; ++it) {
        double s0 = 0, s1 = 0;
        size_t n0 = 0, n1 = 0;
        for (double x : v) (x <= thr ? (s0 += x, ++n0) : (s1 += x, ++n1));
        if (!n0 || !n1) break;
        lo = s0 / n0;
        hi = s1 / n1;
        const double t = 0.5 * (lo + hi);
        if (t == thr) break;
        thr = t;
    }
    if (hi - lo < 80) hb::die(std::string("no separate far class in ") + what + " (centres " + std::to_string(lo) + ", " + std::to_string(hi) + ")");
    return thr;
}

static uint32_t median_u32(std::vector<uint32_t> v) {
    if (v.empty()) return 0;
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

class Lab {
  public:
    Lab(hb::Run& r, uint64_t bytes) : run_(r), sm_count_(r.dev().sm_count), bytes_(bytes) {
        CUDA_CHECK(cudaMalloc(&buf_, bytes));
        CUDA_CHECK(cudaMemset(buf_, 0, bytes));
        if ((uint64_t)buf_ % kPage) hb::die("buffer base is not 2 MiB-aligned");
        CUDA_CHECK(cudaMalloc(&d_sink_, 8));
        CUDA_CHECK(cudaMalloc(&d_tns_, 16 * 8));
        CUDA_CHECK(cudaMalloc(&d_flag_, 2 * 4));
        for (auto f : {(const void*)k_probe<0>, (const void*)k_probe<1>, (const void*)k_ncu_line, (const void*)k_ncu_l1_cg,
                       (const void*)k_ncu_l1_ca, (const void*)k_ncu_fab, (const void*)k3_flag, (const void*)k_calib})
            CUDA_CHECK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kProbeSmem));
        std::printf("buffer %llu MiB at 0x%llx\n", (unsigned long long)(bytes >> 20), (unsigned long long)base());
    }
    ~Lab() { cudaFree(buf_); cudaFree(d_sink_); cudaFree(d_tns_); cudaFree(d_flag_); cudaFree(d_out_); cudaFree(d_q1_); }
    uint64_t base() const { return (uint64_t)buf_; }
    uint64_t bytes() const { return bytes_; }
    void clean() { clean_(); }

    void touch(const DevList& l, uint64_t token, uint32_t T) {
        claimed([&] { k_touch<<<hb::SmClaim::blocks(sm_count_), 32>>>(l.addr, l.n(), token, T, claim_.ptr()); }, T);
    }
    // kernel: 0 k_probe<cg>, 1 k_probe<ca>, 2 k_ncu_line, 3 k_ncu_l1_cg, 4 k_ncu_l1_ca, 5 k_ncu_fab
    void probe(int kernel, const DevList& l, uint32_t S, uint32_t warm, uint32_t P, bool with_q1 = false) {
        reserve(l.n(), S);
        using K = void (*)(const uint64_t*, const uint64_t*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t*, uint64_t, uint32_t*,
                           uint64_t*, uint64_t*, uint64_t*);
        static const K ks[] = {k_probe<0>, k_probe<1>, k_ncu_line, k_ncu_l1_cg, k_ncu_l1_ca, k_ncu_fab};
        claimed([&] { ks[kernel]<<<sm_count_, 32, kProbeSmem>>>(l.addr, l.tlb, l.n(), S, warm, P, claim_.ptr(), 0ull, d_out_,
                                                                with_q1 ? d_q1_ : nullptr, d_tns_, d_sink_); }, P);
        fetch(l.n(), S, with_q1, 2);
    }
    // K3 in one launch; returns false on a flag timeout (the configuration is then repeated by the caller).
    bool k3_flag_run(const DevList& l, uint32_t A, uint32_t B, uint32_t C, uint64_t tok0, uint64_t tok1) {
        reserve(4 * l.n(), kS);
        CUDA_CHECK(cudaMemset(d_flag_, 0, 8));
        CUDA_CHECK(cudaMemset(d_tns_, 0, 16 * 8));
        k3_flag<<<sm_count_, 32, kProbeSmem>>>(l.addr, l.tlb, l.n(), A, B, C, tok0, tok1, d_flag_, 0ull, d_out_, d_q1_, d_tns_, d_sink_,
                                              d_flag_ + 1);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        uint32_t st = 0;
        CUDA_CHECK(cudaMemcpy(&st, d_flag_ + 1, 4, cudaMemcpyDeviceToHost));
        fetch(4 * l.n(), kS, true, 8);
        return st == 0;
    }
    void calib(uint32_t P) {
        reserve(kCalibHops, 1);
        claimed([&] { k_calib<<<sm_count_, 32, kProbeSmem>>>(kCalibHops, P, claim_.ptr(), 0ull, d_out_, d_sink_); }, P);
        samples_.resize(kCalibHops);
        CUDA_CHECK(cudaMemcpy(samples_.data(), d_out_, kCalibHops * 4, cudaMemcpyDeviceToHost));
    }
    void empty() {
        k_ncu_empty<<<1, 32>>>();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    const std::vector<uint32_t>& samples() const { return samples_; }
    const std::vector<uint64_t>& q1() const { return q1_; }
    const std::vector<uint64_t>& tns() const { return tns_; }
    int retries() const { return retries_; }
    int sm_count() const { return sm_count_; }

  private:
    void reserve(uint32_t n, uint32_t S) {
        const size_t need = (size_t)n * S;
        if (need > cap_out_) { cudaFree(d_out_); CUDA_CHECK(cudaMalloc(&d_out_, need * 4)); cap_out_ = need; }
        if (n > cap_q1_) { cudaFree(d_q1_); CUDA_CHECK(cudaMalloc(&d_q1_, (size_t)n * 8)); cap_q1_ = n; }
    }
    void fetch(uint32_t n, uint32_t S, bool with_q1, int ntns) {
        samples_.resize((size_t)n * S);
        CUDA_CHECK(cudaMemcpy(samples_.data(), d_out_, samples_.size() * 4, cudaMemcpyDeviceToHost));
        q1_.assign(with_q1 ? n : 0, 0);
        if (with_q1) CUDA_CHECK(cudaMemcpy(q1_.data(), d_q1_, (size_t)n * 8, cudaMemcpyDeviceToHost));
        tns_.resize(ntns);
        CUDA_CHECK(cudaMemcpy(tns_.data(), d_tns_, ntns * 8, cudaMemcpyDeviceToHost));
    }
    template <class L>
    void claimed(L launch, uint32_t target) {
        for (int r = 0; r <= kMaxRetries; ++r) {
            claim_.reset();
            launch();
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            if (claim_.ok(target)) return;
            ++retries_;
        }
        hb::die("no block ran on SM " + std::to_string(target));
    }
    hb::Run& run_;
    int sm_count_;
    uint64_t bytes_;
    char* buf_ = nullptr;
    uint64_t* d_sink_ = nullptr;
    uint64_t* d_tns_ = nullptr;
    uint32_t* d_flag_ = nullptr;
    uint32_t* d_out_ = nullptr;
    uint64_t* d_q1_ = nullptr;
    size_t cap_out_ = 0, cap_q1_ = 0;
    hb::SmClaim claim_;
    CleanL2 clean_;
    std::vector<uint32_t> samples_;
    std::vector<uint64_t> q1_, tns_;
    int retries_ = 0;
};

// ---------------------------------------------------------------- CSV helpers
static std::vector<std::string> probe_cols() {
    return hb::with_mandatory({"kernel", "probe_smid", "first_touch_smid", "line_offset", "sample_idx", "cycles", "load_variant",
                               "phase", "rep", "warm", "line_home", "seq_mode"});
}
struct Ctx {
    hb::Run& run;
    bool quick;
    long long li = 0;
    int bad = 0;
    hb::ClockSample note(const std::string& label) {
        hb::ClockSample s = run.clock_note(label, "launch", li);
        if (s.bad()) ++bad;
        return s;
    }
};
// One row per sample: samples[i*S + k] of line i.
static void add_samples(hb::CsvWriter& csv, const std::string& pre, const char* kernel, uint32_t P, uint32_t T, const std::vector<uint64_t>& off,
                        const std::vector<int>& home, const uint32_t* smp, uint32_t S, const char* variant, const char* phase, int rep,
                        uint32_t warm, const char* seq) {
    char b[512];
    for (size_t i = 0; i < off.size(); ++i)
        for (uint32_t k = 0; k < S; ++k) {
            std::snprintf(b, sizeof b, "%s,%s,%u,%u,%llu,%u,%u,%s,%s,%d,%u,%d,%s", pre.c_str(), kernel, P, T, (unsigned long long)off[i], k,
                          smp[i * S + k], variant, phase, rep, warm, home.empty() ? -1 : home[i], seq);
            csv.add(b);
        }
}
static void write_stats(hb::Run& run, const char* mode, long long configs, int retries, int bad) {
    hb::CsvWriter st(hb::Run::results("08__l2__launch_stats.csv"), hb::with_mandatory({"mode", "configs", "claim_retries", "bad_clock_configs"}));
    char b[512];
    std::snprintf(b, sizeof b, "%s,%s,%lld,%d,%d", run.prefix(-1, run.dev().sample()).c_str(), mode, configs, retries, bad);
    st.add(b);
    st.sync();
    std::printf("%s: %lld configurations, %d claim retries, %d with a clock warning\n", mode, configs, retries, bad);
}
static void note_buffer(hb::Run& run, const char* mode, const Lab& lab) {
    hb::CsvWriter b(hb::Run::results("08__l2__buffers.csv"), hb::with_mandatory({"mode", "base_hex", "bytes"}));
    char r[512];
    std::snprintf(r, sizeof r, "%s,%s,0x%llx,%llu", run.prefix(-1, run.dev().sample()).c_str(), mode, (unsigned long long)lab.base(),
                  (unsigned long long)lab.bytes());
    b.add(r);
    b.sync();
}

struct Panel {
    std::vector<uint64_t> off;
    std::vector<int> home;  // 0: SM 0's partition, 1: SM 2's
};
// Label n_cand random lines of [0, region) and return per_home lines of each home (home-0 lines first).
static Panel label_panel(Ctx& c, Lab& lab, hb::CsvWriter& csv, uint64_t region, uint32_t n_cand, uint32_t per_home, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::set<uint64_t> seen;
    std::vector<uint64_t> cand;
    while (cand.size() < n_cand) {
        const uint64_t o = (rng() % (region / kLine)) * kLine;
        if (probe_able(o) && seen.insert(o).second) cand.push_back(o);
    }
    DevList l(lab.base(), cand);
    lab.clean();
    lab.touch(l, 0, kRef0);
    lab.probe(0, l, 1, 0, kRef1);
    const hb::ClockSample s = c.note("label");
    std::vector<double> v(lab.samples().begin(), lab.samples().end());
    const double thr = split2(v, "label_panel");
    Panel all;
    for (size_t i = 0; i < cand.size(); ++i) { all.off.push_back(cand[i]); all.home.push_back(v[i] > thr ? 0 : 1); }
    add_samples(csv, c.run.prefix(c.li, s), "label", kRef1, kRef0, all.off, all.home, lab.samples().data(), 1, "cg", "label", 0, 0, "launch");
    csv.flush();
    ++c.li;
    Panel p;
    for (int h : {0, 1}) {
        uint32_t k = 0;
        for (size_t i = 0; i < cand.size() && k < per_home; ++i)
            if (all.home[i] == h) { p.off.push_back(cand[i]); p.home.push_back(h); ++k; }
        if (k < per_home) hb::die("label_panel: too few lines of home " + std::to_string(h));
    }
    return p;
}

// ---------------------------------------------------------------- modes
static void run_calib(Ctx& c) {
    Lab lab(c.run, 8ull << 20);
    hb::CsvWriter csv(hb::Run::results("08__l2__probe.csv"), probe_cols());
    for (int rep = 0; rep < 3; ++rep)
        for (uint32_t P : {kRef0, kRef1}) {
            lab.calib(P);
            const hb::ClockSample s = c.note("calib");
            add_samples(csv, c.run.prefix(c.li, s), "calib", P, P, {0}, {}, lab.samples().data(), kCalibHops, "shared", "calib", rep, 0, "launch");
            csv.flush();
            std::printf("calib SM %u rep %d: median %u cycles\n", P, rep, median_u32(lab.samples()));
            ++c.li;
        }
    csv.sync();
    write_stats(c.run, "calib", c.li, lab.retries(), c.bad);
}

static void run_k1(Ctx& c) {
    Lab lab(c.run, 8ull << 20);
    note_buffer(c.run, "k1", lab);
    hb::CsvWriter csv(hb::Run::results("08__l2__probe.csv"), probe_cols());
    const Panel pn = label_panel(c, lab, csv, 8ull << 20, 512, 16, 0x08000001ull);
    DevList l(lab.base(), pn.off);
    const int reps = c.quick ? 1 : 5;
    c.run.lock_clocks_note("k1", "before");
    for (int rep = 0; rep < reps; ++rep)
        for (uint32_t P : {kRef0, kRef1})
            for (uint32_t T : {P, P == kRef0 ? kRef1 : kRef0})
                for (uint32_t warm : {0u, 1u})
                    for (int V : {0, 1}) {
                        lab.clean();
                        lab.touch(l, 0, T);
                        lab.probe(V, l, kS, warm, P);
                        const hb::ClockSample s = c.note("k1");
                        add_samples(csv, c.run.prefix(c.li, s), "K1", P, T, pn.off, pn.home, lab.samples().data(), kS, V ? "ca" : "cg", "probe",
                                    rep, warm, "launch");
                        csv.flush();
                        std::vector<uint32_t> s0, rest;
                        for (uint32_t i = 0; i < l.n(); ++i) {
                            s0.push_back(lab.samples()[i * kS]);
                            for (uint32_t k = 1; k < kS; ++k) rest.push_back(lab.samples()[i * kS + k]);
                        }
                        std::printf("K1 rep %d P %3u T %3u warm %u %s: sample0 median %u, samples 1-63 median %u\n", rep, P, T, warm, V ? "ca" : "cg",
                                    median_u32(s0), median_u32(rest));
                        ++c.li;
                    }
    csv.sync();
    c.run.lock_clocks_note("k1", "after");
    write_stats(c.run, "k1", c.li, lab.retries(), c.bad);
}

struct HashPair { std::string set; int bit, bit2; uint64_t mask; uint32_t pair_idx; uint32_t base, flipped; };

static void run_k2(Ctx& c) {
    const uint64_t kBuf = 2ull << 30;  // bits up to 30 can be flipped inside the buffer
    Lab lab(c.run, kBuf);
    note_buffer(c.run, "k2", lab);
    std::mt19937_64 rng(0x08000002ull);
    std::vector<uint64_t> lines;
    std::map<uint64_t, uint32_t> idx;
    auto add = [&](uint64_t o) -> int64_t {
        if (!probe_able(o)) return -1;
        auto it = idx.find(o);
        if (it != idx.end()) return it->second;
        idx[o] = (uint32_t)lines.size();
        lines.push_back(o);
        return lines.size() - 1;
    };
    const uint32_t nb = c.quick ? 8 : 32;
    std::vector<uint64_t> bases;
    while (bases.size() < nb) {  // ≥ 3 zero bits among 7..20, so that no single or pair flip yields the TLB-warm line
        const uint64_t o = (rng() % (kBuf / kLine)) * kLine;
        if (__builtin_popcountll(~(o >> 7) & 0x3fffull) >= 3) bases.push_back(o);
    }
    std::vector<HashPair> pairs;
    auto pair = [&](const char* set, int b1, int b2, uint64_t mask, uint32_t j) {
        const int64_t a = add(bases[j]), f = add(bases[j] ^ mask);
        if (a < 0 || f < 0) return;
        pairs.push_back({set, b1, b2, mask, j, (uint32_t)a, (uint32_t)f});
    };
    for (uint32_t j = 0; j < nb; ++j) add(bases[j]);
    for (int b = 7; b <= 30; ++b)
        for (uint32_t j = 0; j < nb; ++j) pair("single", b, -1, 1ull << b, j);
    for (int b1 = 7; b1 <= 20; ++b1)
        for (int b2 = b1 + 1; b2 <= 20; ++b2)
            for (uint32_t j = 0; j < nb; ++j) pair("pair", b1, b2, (1ull << b1) | (1ull << b2), j);
    for (uint32_t j = 0; j < nb; ++j)
        for (uint32_t m = 0; m < nb; ++m) {
            uint64_t mask = 0;
            while (!mask) mask = (rng() & 0x3fffull) << 7;
            pair("mask", -1, -1, mask, j);
        }
    std::vector<uint64_t> in_page;
    while (in_page.size() < 4) {
        const uint64_t o = (rng() % (kPage / kLine)) * kLine;
        if (probe_able(o)) in_page.push_back(o);
    }
    for (uint64_t pg = 0; pg < kBuf / kPage; ++pg)
        for (uint64_t o : in_page) add(pg * kPage + o);
    std::printf("K2: %zu unique lines, %zu pairs\n", lines.size(), pairs.size());

    hb::CsvWriter csv(hb::Run::results("08__l2__probe_k2.csv"), probe_cols());
    hb::CsvWriter hcsv(hb::Run::results("08__l2__hashbits.csv"),
                       hb::with_mandatory({"probe_smid", "bit", "pair_idx", "base_offset", "flipped_offset", "base_median", "flipped_median",
                                           "flipped_class_changed", "first_touch_smid", "rep", "set", "bit2", "mask_hex", "base_first_cycles",
                                           "flipped_first_cycles"}));
    const int reps = c.quick ? 1 : 3;
    c.run.lock_clocks_note("k2", "before");
    for (int rep = 0; rep < reps; ++rep)
        for (auto [P, T] : {std::pair<uint32_t, uint32_t>{kRef1, kRef0}, {kRef0, kRef1}}) {
            std::vector<uint32_t> order(lines.size());
            std::iota(order.begin(), order.end(), 0u);
            std::shuffle(order.begin(), order.end(), rng);
            std::vector<uint64_t> off(order.size());
            for (size_t i = 0; i < order.size(); ++i) off[i] = lines[order[i]];
            DevList l(lab.base(), off);
            lab.clean();
            lab.touch(l, 0, T);
            lab.probe(0, l, kS, 0, P);
            const hb::ClockSample s = c.note("k2");
            const uint32_t* smp = lab.samples().data();
            std::vector<double> first(order.size());
            for (size_t i = 0; i < order.size(); ++i) first[i] = smp[i * kS];
            const double thr = split2(first, "k2");
            const int partP = P == kRef0 ? 0 : 1, partT = 1 - partP;
            std::vector<int> home(order.size());
            for (size_t i = 0; i < order.size(); ++i) home[i] = first[i] > thr ? partT : partP;
            const std::string pre = c.run.prefix(c.li, s);
            add_samples(csv, pre, "K2", P, T, off, home, smp, kS, "cg", "probe", rep, 0, "launch");
            csv.flush();
            // per original line: position in this probe order
            std::vector<uint32_t> pos(lines.size());
            for (size_t i = 0; i < order.size(); ++i) pos[order[i]] = (uint32_t)i;
            auto med = [&](uint32_t li) {
                std::vector<uint32_t> v(smp + (size_t)pos[li] * kS, smp + (size_t)pos[li] * kS + kS);
                return median_u32(v);
            };
            char b[512];
            for (const auto& hp : pairs) {
                const uint32_t pb = pos[hp.base], pf = pos[hp.flipped];
                std::snprintf(b, sizeof b, "%s,%u,%d,%u,%llu,%llu,%u,%u,%d,%u,%d,%s,%d,0x%llx,%u,%u", pre.c_str(), P, hp.bit, hp.pair_idx,
                              (unsigned long long)lines[hp.base], (unsigned long long)lines[hp.flipped], med(hp.base), med(hp.flipped),
                              (int)(home[pb] != home[pf]), T, rep, hp.set.c_str(), hp.bit2, (unsigned long long)hp.mask, smp[(size_t)pb * kS],
                              smp[(size_t)pf * kS]);
                hcsv.add(b);
            }
            hcsv.flush();
            size_t far = 0;
            for (size_t i = 0; i < order.size(); ++i) far += first[i] > thr;
            std::printf("K2 rep %d P %u T %u: threshold %.1f, %zu of %zu first reads far\n", rep, P, T, thr, far, order.size());
            ++c.li;
        }
    csv.sync();
    hcsv.sync();
    c.run.lock_clocks_note("k2", "after");
    write_stats(c.run, "k2", c.li, lab.retries(), c.bad);
}

static void run_k3(Ctx& c) {
    const std::vector<int> gpc = load_gpc_map(c.run.dev().sm_count);
    struct Triple { uint32_t A, B, C; };
    const std::vector<Triple> triples = c.quick ? std::vector<Triple>{{0, 2, 6}} : std::vector<Triple>{{0, 2, 6}, {2, 8, 0}, {14, 126, 12}, {0, 8, 10}};
    for (const auto& t : triples)
        std::printf("K3 triple A %u (G%d, prior partition %d)  B %u (G%d, %d)  C %u (G%d, %d)\n", t.A, gpc[t.A], kPartitionOfGpc[gpc[t.A]], t.B, gpc[t.B],
                    kPartitionOfGpc[gpc[t.B]], t.C, gpc[t.C], kPartitionOfGpc[gpc[t.C]]);
    Lab lab(c.run, 8ull << 20);
    note_buffer(c.run, "k3", lab);
    hb::CsvWriter csv(hb::Run::results("08__l2__probe.csv"), probe_cols());
    hb::CsvWriter rcsv(hb::Run::results("08__l2__replication.csv"),
                       hb::with_mandatory({"a_smid", "b_smid", "phase", "median_cycles", "c_smid", "seq_mode", "rep", "line_offset", "line_home",
                                           "qword1_ok", "phase_t0_ns", "phase_t1_ns"}));
    const Panel pn = label_panel(c, lab, csv, 8ull << 20, 256, 16, 0x08000003ull);
    DevList l(lab.base(), pn.off);
    const uint32_t n = l.n();
    const int reps = c.quick ? 1 : 5;
    static const char* phase_name[4] = {"b_read", "c_read", "a_read", "b_read_after_a_write"};
    static const char* rep_name[4][2] = {{"b_first", "b_repeat"}, {"c_first", "c_repeat"}, {"a_after_first", "a_after_repeat"},
                                         {"b_after_a_write_first", "b_after_a_write_repeat"}};
    int timeouts = 0;
    c.run.lock_clocks_note("k3", "before");
    for (int rep = 0; rep < reps; ++rep)
        for (size_t ti = 0; ti < triples.size(); ++ti)
            for (const char* seq : {"launch", "flag"}) {
                const Triple t = triples[ti];
                const uint64_t tok0 = 0x5a00000000000000ull | (uint64_t)rep << 16 | ti << 8, tok1 = tok0 | 0xff;
                std::vector<uint32_t> smp(4 * (size_t)n * kS);
                std::vector<uint64_t> q1(4 * (size_t)n), tns(8);
                lab.clean();
                if (!std::strcmp(seq, "launch")) {
                    lab.touch(l, tok0, t.A);
                    const uint32_t who[4] = {t.B, t.C, t.A, t.B};
                    for (int ph = 0; ph < 4; ++ph) {
                        if (ph == 3) lab.touch(l, tok1, t.A);
                        lab.probe(0, l, kS, 0, who[ph], true);
                        std::copy(lab.samples().begin(), lab.samples().end(), smp.begin() + (size_t)ph * n * kS);
                        std::copy(lab.q1().begin(), lab.q1().end(), q1.begin() + (size_t)ph * n);
                        tns[2 * ph] = lab.tns()[0];
                        tns[2 * ph + 1] = lab.tns()[1];
                    }
                } else {
                    if (!lab.k3_flag_run(l, t.A, t.B, t.C, tok0, tok1)) {
                        ++timeouts;
                        std::printf("K3 flag timeout (triple %zu rep %d): configuration discarded\n", ti, rep);
                        continue;
                    }
                    smp = lab.samples();
                    q1 = lab.q1();
                    tns = lab.tns();
                }
                const hb::ClockSample s = c.note("k3");
                const std::string pre = c.run.prefix(c.li, s);
                const uint32_t who[4] = {t.B, t.C, t.A, t.B};
                for (int ph = 0; ph < 4; ++ph)
                    add_samples(csv, pre, "K3", who[ph], t.A, pn.off, pn.home, smp.data() + (size_t)ph * n * kS, kS, "cg", phase_name[ph], rep, 0,
                                seq);
                csv.flush();
                char b[512];
                for (int ph = 0; ph < 4; ++ph)
                    for (uint32_t i = 0; i < n; ++i) {
                        const uint32_t* x = smp.data() + ((size_t)ph * n + i) * kS;
                        const uint64_t want = ph == 3 ? tok1 : tok0;
                        const int ok = q1[(size_t)ph * n + i] == want;
                        for (int r = 0; r < 2; ++r) {
                            const uint32_t v = r == 0 ? x[0] : median_u32(std::vector<uint32_t>(x + 1, x + kS));
                            std::snprintf(b, sizeof b, "%s,%u,%u,%s,%u,%u,%s,%d,%llu,%d,%d,%lld,%lld", pre.c_str(), t.A, t.B, rep_name[ph][r], v, t.C, seq, rep,
                                          (unsigned long long)pn.off[i], pn.home[i], ok, (long long)(tns[2 * ph] - tns[0]),
                                          (long long)(tns[2 * ph + 1] - tns[0]));
                            rcsv.add(b);
                        }
                    }
                rcsv.flush();
                std::printf("K3 rep %d A %3u B %3u C %3u %-6s:", rep, t.A, t.B, t.C, seq);
                for (int ph = 0; ph < 4; ++ph)
                    for (int h : {0, 1}) {
                        std::vector<uint32_t> f;
                        for (uint32_t i = 0; i < n; ++i)
                            if (pn.home[i] == h) f.push_back(smp[((size_t)ph * n + i) * kS]);
                        std::printf(" %s/h%d %u", phase_name[ph], h, median_u32(f));
                    }
                std::printf("\n");
                ++c.li;
            }
    csv.sync();
    rcsv.sync();
    c.run.lock_clocks_note("k3", "after");
    write_stats(c.run, "k3", c.li, lab.retries() + timeouts, c.bad);
}

static void run_k5(Ctx& c) {
    Lab lab(c.run, 8ull << 20);
    note_buffer(c.run, "k5", lab);
    hb::CsvWriter csv(hb::Run::results("08__l2__probe.csv"), probe_cols());
    hb::CsvWriter scsv(hb::Run::results("08__l2__sm_signature_samples.csv"),
                       hb::with_mandatory({"rep", "probe_smid", "line_offset", "line_home", "first_touch_smid", "load_variant", "first_cycles",
                                           "cycles_seq"}));
    const Panel pn = label_panel(c, lab, csv, 8ull << 20, 1024, 128, 0x08000005ull);
    csv.sync();
    std::vector<uint64_t> o0, o1;
    for (size_t i = 0; i < pn.off.size(); ++i) (pn.home[i] ? o1 : o0).push_back(pn.off[i]);
    std::vector<uint32_t> perm(pn.off.size());
    std::iota(perm.begin(), perm.end(), 0u);
    std::mt19937_64 rng(0x08000055ull);
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<uint64_t> off;
    std::vector<int> home;
    for (uint32_t i : perm) { off.push_back(pn.off[i]); home.push_back(pn.home[i]); }
    DevList l0(lab.base(), o0), l1(lab.base(), o1), la(lab.base(), off);
    const int reps = c.quick ? 1 : 5;
    std::vector<uint32_t> probes;
    for (int p = 0; p < lab.sm_count(); ++p)
        if (!c.quick || p % 11 == 0) probes.push_back((uint32_t)p);
    c.run.lock_clocks_note("k5", "before");
    for (int rep = 0; rep < reps; ++rep)
        for (uint32_t P : probes) {
            lab.clean();
            lab.touch(l0, 0, kRef0);
            lab.touch(l1, 0, kRef1);
            lab.probe(0, la, kS, 0, P);
            const hb::ClockSample s = c.note("k5");
            const std::string pre = c.run.prefix(c.li, s);
            const uint32_t* smp = lab.samples().data();
            std::string row;
            std::vector<uint32_t> f[2];
            for (size_t i = 0; i < off.size(); ++i) {
                row = pre + "," + std::to_string(rep) + "," + std::to_string(P) + "," + std::to_string(off[i]) + "," + std::to_string(home[i]) + "," +
                      std::to_string(home[i] ? kRef1 : kRef0) + ",cg," + std::to_string(smp[i * kS]) + ",";
                for (uint32_t k = 0; k < kS; ++k) {
                    if (k) row += ';';
                    row += std::to_string(smp[i * kS + k]);
                }
                scsv.add(row);
                f[home[i]].push_back(smp[i * kS]);
            }
            scsv.flush();
            if (P % 11 == 0) std::printf("K5 rep %d SM %3u: first-read median home0 %u, home1 %u\n", rep, P, median_u32(f[0]), median_u32(f[1]));
            ++c.li;
        }
    scsv.sync();
    c.run.lock_clocks_note("k5", "after");
    write_stats(c.run, "k5", c.li, lab.retries(), c.bad);
}

// Kernel order of the ncu modes → ncu_manifest.csv (one row per profiled kernel, in launch order).
static hb::CsvWriter manifest_writer() {
    return hb::CsvWriter(hb::Run::results("08__l2__ncu_manifest.csv"),
                         hb::with_mandatory({"mode", "ncu_order", "kernel", "probe_smid", "first_touch_smid", "lines", "line_offset", "what",
                                             "first_median", "repeat_median"}));
}

static void run_ncu_lines(Ctx& c) {
    Lab lab(c.run, 8ull << 20);
    hb::CsvWriter man = manifest_writer();
    std::vector<uint64_t> off;
    for (uint64_t k = 0; k < (c.quick ? 4u : 32u); ++k) off.push_back(k * 4096);
    int order = 0;
    char b[512];
    auto row = [&](const char* kernel, uint32_t P, size_t nlines, long long line, const char* what) {
        std::vector<uint32_t> f, r;
        for (size_t i = 0; i < nlines; ++i) {
            f.push_back(lab.samples()[i * kS]);
            for (uint32_t k = 1; k < kS; ++k) r.push_back(lab.samples()[i * kS + k]);
        }
        std::snprintf(b, sizeof b, "%s,ncu_lines,%d,%s,%u,%u,%zu,%lld,%s,%u,%u", c.run.prefix(c.li, c.note("ncu_lines")).c_str(), order++, kernel, P,
                      kRef0, nlines, line, what, median_u32(f), median_u32(r));
        man.add(b);
        man.flush();
        ++c.li;
    };
    for (uint32_t P : {kRef1, kRef0})
        for (uint64_t o : off) {
            DevList l(lab.base(), {o});
            lab.clean();
            lab.touch(l, 0, kRef0);
            lab.probe(2, l, kS, 0, P);
            row("k_ncu_line", P, 1, (long long)o, "line");
        }
    DevList la(lab.base(), off);
    for (int k : {3, 4}) {
        lab.clean();
        lab.touch(la, 0, kRef0);
        lab.probe(k, la, kS, 0, kRef0);
        row(k == 3 ? "k_ncu_l1_cg" : "k_ncu_l1_ca", kRef0, off.size(), -1, k == 3 ? "l1_cg" : "l1_ca");
    }
    man.sync();
    write_stats(c.run, "ncu_lines", c.li, lab.retries(), c.bad);
}

static void run_ncu_fabric(Ctx& c) {
    const uint64_t kBuf = 16ull << 20;
    Lab lab(c.run, kBuf);
    hb::CsvWriter man = manifest_writer();
    std::vector<uint64_t> all;
    for (uint64_t o = 0; o < kBuf; o += kLine)
        if (probe_able(o)) all.push_back(o);
    std::vector<uint64_t> far, near;
    {
        DevList l(lab.base(), all);
        lab.clean();
        lab.touch(l, 0, kRef0);
        lab.probe(0, l, 1, 0, kRef1);
        std::vector<double> v(lab.samples().begin(), lab.samples().end());
        const double thr = split2(v, "ncu_fabric");
        for (size_t i = 0; i < all.size(); ++i) (v[i] > thr ? far : near).push_back(all[i]);
    }
    std::printf("ncu_fabric: %zu far, %zu near lines for SM %u\n", far.size(), near.size(), kRef1);
    DevList la(lab.base(), all), lf(lab.base(), far), ln(lab.base(), near);
    lab.clean();
    lab.touch(la, 0, kRef0);
    int order = 0;
    char b[512];
    auto row = [&](const char* kernel, size_t nlines, const char* what) {
        std::vector<uint32_t> f(lab.samples().begin(), lab.samples().end());
        std::snprintf(b, sizeof b, "%s,ncu_fabric,%d,%s,%u,%u,%zu,-1,%s,%u,0", c.run.prefix(c.li, c.note("ncu_fabric")).c_str(), order++, kernel, kRef1,
                      kRef0, nlines, what, median_u32(f));
        man.add(b);
        man.flush();
        ++c.li;
    };
    lab.empty();
    std::snprintf(b, sizeof b, "%s,ncu_fabric,%d,k_ncu_empty,-1,-1,0,-1,empty,0,0", c.run.prefix(c.li, c.note("ncu_fabric")).c_str(), order++);
    man.add(b);
    ++c.li;
    lab.probe(5, lf, 1, 0, kRef1);
    row("k_ncu_fab", far.size(), "far_first");
    lab.probe(5, ln, 1, 0, kRef1);
    row("k_ncu_fab", near.size(), "near_first");
    lab.probe(5, lf, 1, 0, kRef1);
    row("k_ncu_fab", far.size(), "far_second");
    man.sync();
    write_stats(c.run, "ncu_fabric", c.li, lab.retries(), c.bad);
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "k1";
    hb::Run run(EXP, argc, argv);
    Ctx c{run, hb::has_flag(argc, argv, "--quick")};
    if (mode == "clockcheck") run.lock_clocks_note(hb::arg_str(argc, argv, "--label", "check"), "check");
    else if (mode == "calib") run_calib(c);
    else if (mode == "k1") run_k1(c);
    else if (mode == "k2") run_k2(c);
    else if (mode == "k3") run_k3(c);
    else if (mode == "k5") run_k5(c);
    else if (mode == "ncu_lines") run_ncu_lines(c);
    else if (mode == "ncu_fabric") run_ncu_fabric(c);
    else hb::die("unknown mode " + mode);
    run.complete();
    return 0;
}
