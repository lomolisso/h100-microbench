/*
 * tma_chase.cu — experiment 06, K1 + K5: latency of one dependent 16-B load through the async proxy (TMA bulk copy,
 * tensor TMA) versus working-set size, next to generic-proxy twins on exactly the same pointer chain.
 *
 * Usage: bin/tma_chase {chase|ncu|clockcheck} [--quick] [--label L]. One invocation = one run_id.
 *
 * Chain (04's pchase_chain.hpp, kChainSeed): one 2 GiB cudaMalloc buffer `base`; footprint F = n lines of 128 B;
 *   line i holds at qword 0 the address of line next[i] and at qword 1 the address of line next[next[i]] (two hops
 *   ahead; only variant bulk16_prefetch uses it). next = one random cycle over the n lines (Sattolo). The host writes
 *   the chain into a zeroed pinned buffer and copies F bytes with one cudaMemcpy; the chase starts at line 0.
 * Placement: claim_target_sm — 2 × 132 one-warp blocks; the first block on the probe SM runs the chase in thread 0,
 *   every other block returns at once. Probe SMs: the first SM of GPC 1 and of GPC 4 of the 03 map (one per L2
 *   partition group). Static smem only (128 B destination + one mbarrier).
 * One hop of each variant (one thread; parity of hop k = k mod 2; the barrier is initialised with count 1):
 *   bulk16            mbarrier.arrive.expect_tx(bar, 16); cp.async.bulk.shared::cluster.global [dst], [p], 16, [bar]
 *                     (UBLKCP.S.G); try_wait.parity loop (no hint); p = ld.shared.u64 [dst]
 *   tensor1d          expect_tx 16; cp.async.bulk.tensor.1d tile, uint64 tensor over the whole buffer, box = 2
 *                     elements, coordinate (p − base)/8 (UTMALDG.1D); try_wait loop; ld.shared
 *   tensor2d          expect_tx 64; cp.async.bulk.tensor.2d, float tensor {4, 2^27} with a 16-B row stride (the buffer
 *                     seen as rows of 16 B), box 4×4 = 64 B at (0, (p − base)/16): bytes 0–63 of the pointer's line,
 *                     so every hop still touches exactly one 128-B line; try_wait loop; ld.shared
 *   bulk16_evict_last bulk16 with .L2::cache_hint and a createpolicy.fractional.L2::evict_last (1.0) policy
 *   bulk16_prefetch   bulk16, then ld.shared.v2 {p, q} and cp.async.bulk.prefetch.L2 [q], 16: the line two hops ahead is
 *                     prefetched one hop before it is loaded
 *   bulk16_hint{0,100,1000}  bulk16 with try_wait.parity + suspendTimeHint 0 / 100 / 1000 ns
 *   bulk16_testwait   bulk16 with a test_wait.parity loop
 *   ld_cg / ld_ca     generic p = ld.global.cg/.ca.u64 [p] (04's K2 chase)
 *   cp_async16        cp.async.cg.shared.global [dst], [p], 16; commit_group; wait_group 0; p = ld.shared [dst]
 * Timed region (thread 0): warm_hops untimed hops, t0 = read_clock64_after(p), `hops` hops (#pragma unroll 1 around 16
 *   inlined hops), t1 = read_clock64_after(p). cycles_per_hop = (t1 − t0)/hops; the fixed timer region (≤ 192 cycles,
 *   experiment 01) is < 0.003 cycles/hop at ≥ 2^16 hops and is not subtracted.
 * Sweep (04's footprints): F = 1 KiB · 1.25^k rounded to 128 B up to 2 GiB, plus 128 KiB, 25/30/50/60 MiB, 2 GiB.
 *   n ≤ 2^20 lines: each launch warms in-kernel (warm_hops = n, the whole cycle), then times max(n, 2^16) hops
 *               (rounded up to 16): the timed hops cover the whole cycle at least once.
 *   n > 2^20:   one untimed launch of 2^19 hops (TLB, instruction cache), then launches of 2^19 timed hops
 *               (warm_hops = 0). Every launch starts at line 0, so all of them visit the same 2^19 lines (64 MiB):
 *               the L2 is cleaned (05's CleanL2: flush + clean read of other buffers) before every one of these
 *               launches, otherwise lines left by the previous launch would be L2 hits.
 *   5 recorded launches per (F, variant, SM). --quick: F ∈ {16 KiB, 1 MiB, 8 MiB, 256 MiB}, one SM, 2 launches.
 * Polling floor: at start, the predicate latency of every wait form on an already completed barrier (tma_common.hpp
 *   k_pred, glue subtracted, median of 5 launches) → poll_floor_cycles of the variants that use that form (empty for
 *   the generic variants). The notebook reports cycles_per_hop raw and minus this floor.
 *   → results/06__tma__chase.csv: variant, footprint_bytes, hops, warm_hops, rep, cycles_per_hop, poll_floor_cycles,
 *     total_cycles, smid, gpc, retries (one row per recorded launch).
 * ncu mode: k1_ncu<V> (same body, the only kernel src/run.sh profiles, application replay) for bulk16 / tensor2d /
 *   ld_cg / cp_async16 at F = 8 MiB (after one warm launch) and bulk16 at F = 256 MiB (after CleanL2), 2^16 hops each.
 *   Writes bin/ncu_manifest_chase.csv (one row per k1_ncu launch).
 */
#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>
#include <string>
#include <vector>

#include "../../04__memory_hierarchy_latency/src/pchase_chain.hpp"
#include "../../05__memory_hierarchy_bandwidth/src/bw_common.hpp"  // CleanL2 and 04's gpc_map.hpp
#include "tma_common.hpp"

static const char* EXP = "06__tma_latency_and_issue";
static constexpr int kUnroll = 16, kMaxRetries = 20;
static constexpr uint64_t kInKernelWarmMax = 1ull << 20, kMinHops = 1ull << 16, kColdHops = 1ull << 19;
static constexpr uint64_t kBufBytes = 2ull << 30;

enum Variant {
    V_BULK16, V_T1D, V_T2D, V_EVL, V_PF, V_H0, V_H100, V_H1000, V_TEST, V_LDCG, V_LDCA, V_CPASYNC, V_COUNT
};
static const char* kName[V_COUNT] = {"bulk16", "tensor1d", "tensor2d", "bulk16_evict_last", "bulk16_prefetch",
                                     "bulk16_hint0", "bulk16_hint100", "bulk16_hint1000", "bulk16_testwait",
                                     "ld_cg", "ld_ca", "cp_async16"};
__host__ __device__ constexpr int wait_form(int v) { return v == V_TEST ? W_TEST : (v == V_H0 || v == V_H100 || v == V_H1000) ? W_HINT : W_TRY; }
__host__ __device__ constexpr uint32_t hint_of(int v) { return v == V_H100 ? 100u : v == V_H1000 ? 1000u : 0u; }
__host__ __device__ constexpr bool is_generic(int v) { return v == V_LDCG || v == V_LDCA || v == V_CPASYNC; }

struct ChaseOut { uint64_t cycles, sink; uint32_t smid; };

struct HopCtx {
    uint32_t d, b;  // smem destination, mbarrier
    uint64_t base, policy;
    const CUtensorMap *tm1, *tm2;
};

template <int V>
__device__ __forceinline__ uint64_t hop(uint64_t p, uint32_t parity, const HopCtx& c) {
    if constexpr (V == V_LDCG) {
        uint64_t v; asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory"); return v;
    } else if constexpr (V == V_LDCA) {
        uint64_t v; asm volatile("ld.global.ca.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory"); return v;
    } else if constexpr (V == V_CPASYNC) {
        cp_async_cg16(c.d, (const void*)p);
        cp_async_commit();
        cp_async_wait_all();
        return lds_u64(c.d);
    } else {
        if constexpr (V == V_T2D) mbar_arrive_expect_tx(c.b, 64);
        else mbar_arrive_expect_tx(c.b, 16);
        if constexpr (V == V_T1D) tma_load_1d(c.d, c.tm1, (int32_t)((p - c.base) >> 3), c.b);
        else if constexpr (V == V_T2D) tma_load_2d(c.d, c.tm2, 0, (int32_t)((p - c.base) >> 4), c.b);
        else if constexpr (V == V_EVL) bulk_g2s_hint(c.d, (const void*)p, 16, c.b, c.policy);
        else bulk_g2s(c.d, (const void*)p, 16, c.b);
        wait_loop<wait_form(V)>(c.b, parity, hint_of(V));
        if constexpr (V == V_PF) {
            uint64_t q;
            asm volatile("ld.shared.v2.u64 {%0, %1}, [%2];" : "=l"(p), "=l"(q) : "r"(c.d) : "memory");
            bulk_prefetch_l2((const void*)q, 16);
            return p;
        } else {
            return lds_u64(c.d);
        }
    }
}

template <int V>
__device__ __forceinline__ void chase_body(const uint64_t* start, uint64_t warm_hops, uint64_t hops, uint32_t target,
                                           uint32_t* claim, uint64_t zero, ChaseOut* out, const CUtensorMap* tm1,
                                           const CUtensorMap* tm2) {
    __shared__ alignas(128) uint64_t dst[16];
    __shared__ alignas(8) uint64_t bar;
    if (!claim_target_sm(target, claim)) return;
    if (threadIdx.x != 0) return;
    HopCtx c;
    c.d = smem_u32(dst);
    c.b = smem_u32(&bar);
    c.base = (uint64_t)start;  // the chain starts at line 0 = the buffer base
    c.policy = 0;
    if constexpr (V == V_EVL) c.policy = createpolicy_evict_last();
    c.tm1 = tm1;
    c.tm2 = tm2;
    if constexpr (!is_generic(V)) {
        mbar_init(c.b, 1);
        fence_mbarrier_init();
    }
    uint64_t p = (uint64_t)start;
#pragma unroll 1
    for (uint64_t i = 0; i < warm_hops / kUnroll; ++i) {
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) p = hop<V>(p, u & 1, c);
    }
    const uint64_t t0 = read_clock64_after(p, zero);
#pragma unroll 1
    for (uint64_t i = 0; i < hops / kUnroll; ++i) {
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) p = hop<V>(p, u & 1, c);
    }
    const uint64_t t1 = read_clock64_after(p, zero);
    out->cycles = t1 - t0;
    out->sink = p;
    out->smid = read_smid();
}

template <int V>
__global__ void __launch_bounds__(32) k1_chase(const uint64_t* start, uint64_t warm_hops, uint64_t hops, uint32_t target,
                                                uint32_t* claim, uint64_t zero, ChaseOut* out,
                                                const __grid_constant__ CUtensorMap tm1, const __grid_constant__ CUtensorMap tm2) {
    chase_body<V>(start, warm_hops, hops, target, claim, zero, out, &tm1, &tm2);
}
// Identical body under another name: the only kernel that src/run.sh profiles.
template <int V>
__global__ void __launch_bounds__(32) k1_ncu(const uint64_t* start, uint64_t warm_hops, uint64_t hops, uint32_t target,
                                              uint32_t* claim, uint64_t zero, ChaseOut* out,
                                              const __grid_constant__ CUtensorMap tm1, const __grid_constant__ CUtensorMap tm2) {
    chase_body<V>(start, warm_hops, hops, target, claim, zero, out, &tm1, &tm2);
}

static uint64_t round16(uint64_t h) { return (h + kUnroll - 1) / kUnroll * kUnroll; }

static std::vector<uint64_t> footprints(bool quick) {
    if (quick) return {16ull << 10, 1ull << 20, 8ull << 20, 256ull << 20};
    std::vector<uint64_t> f;
    for (double x = 1024; x < (double)kBufBytes; x *= 1.25) f.push_back((uint64_t)std::llround(x / 128.0) * 128);
    for (uint64_t e : {128ull << 10, 25ull << 20, 30ull << 20, 50ull << 20, 60ull << 20, (unsigned long long)kBufBytes}) f.push_back(e);
    std::sort(f.begin(), f.end());
    f.erase(std::unique(f.begin(), f.end()), f.end());
    return f;
}

struct Ctx {
    hb::Run& run;
    hb::SmClaim claim;
    CleanL2 clean;
    uint64_t* d_buf = nullptr;
    uint64_t* h_buf = nullptr;
    ChaseOut* d_out = nullptr;
    CUtensorMap tm1{}, tm2{};
    int blocks = 0;
    explicit Ctx(hb::Run& r) : run(r), blocks(hb::SmClaim::blocks(r.dev().sm_count)) {
        CUDA_CHECK(cudaMalloc(&d_buf, kBufBytes));
        CUDA_CHECK(cudaMallocHost(&h_buf, kBufBytes));
        std::memset(h_buf, 0, kBufBytes);
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(ChaseOut)));
        // 1-D uint64 tensor over the whole buffer, box of 2 elements (16 B).
        // (the driver rejects globalStrides = nullptr even for rank 1, where no stride is read: pass a dummy)
        const cuuint64_t g1[1] = {kBufBytes / 8}, s1[1] = {16};
        const cuuint32_t b1[1] = {2};
        CUresult e1 = encode_tiled(&tm1, CU_TENSOR_MAP_DATA_TYPE_UINT64, 1, d_buf, g1, s1, b1, CU_TENSOR_MAP_SWIZZLE_NONE);
        // 2-D float tensor: rows of 4 floats (16 B), row stride 16 B, 2^27 rows; box 4×4 floats = 64 B.
        const cuuint64_t g2[2] = {4, kBufBytes / 16};
        const cuuint64_t s2[1] = {16};
        const cuuint32_t b2[2] = {4, 4};
        CUresult e2 = encode_tiled(&tm2, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, d_buf, g2, s2, b2, CU_TENSOR_MAP_SWIZZLE_NONE);
        if (e1 != CUDA_SUCCESS || e2 != CUDA_SUCCESS)
            hb::die("cuTensorMapEncodeTiled failed: 1d " + std::to_string(e1) + ", 2d " + std::to_string(e2));
        auto carve = [](const void* k) {
            CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, (int)cudaSharedmemCarveoutMaxL1));
        };
        carve((const void*)k1_chase<V_LDCA>);  // L1 as large as possible for the L1-cached twin (as in 04)
    }
    ~Ctx() { cudaFree(d_buf); cudaFreeHost(h_buf); cudaFree(d_out); }

    uint64_t fill(uint64_t F) {
        const uint32_t n = (uint32_t)(F / chain::kLineBytes);
        const std::vector<uint32_t> next = chain::sattolo_cycle(n);
        const uint64_t base = (uint64_t)d_buf;
        for (uint32_t i = 0; i < n; ++i) {
            h_buf[(uint64_t)i * chain::kQwordsPerLine] = base + (uint64_t)next[i] * chain::kLineBytes;
            h_buf[(uint64_t)i * chain::kQwordsPerLine + 1] = base + (uint64_t)next[next[i]] * chain::kLineBytes;
        }
        CUDA_CHECK(cudaMemcpy(d_buf, h_buf, F, cudaMemcpyHostToDevice));
        return n;
    }

    template <class K>
    int launch(K kernel, uint64_t warm_hops, uint64_t hops, uint32_t target) {
        for (int r = 0; r <= kMaxRetries; ++r) {
            claim.reset();
            kernel<<<blocks, 32>>>(d_buf, warm_hops, hops, target, claim.ptr(), 0ull, d_out, tm1, tm2);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            if (claim.ok(target)) return r;
        }
        hb::die("no block ran on SM " + std::to_string(target) + " after retries");
    }
    ChaseOut out() const {
        ChaseOut o;
        CUDA_CHECK(cudaMemcpy(&o, d_out, sizeof o, cudaMemcpyDeviceToHost));
        return o;
    }
};

struct Floors { double tr = 0, test = 0, h0 = 0, h100 = 0, h1000 = 0; };
static double floor_of(const Floors& f, int v) {
    switch (v) {
        case V_TEST: return f.test;
        case V_H0: return f.h0;
        case V_H100: return f.h100;
        case V_H1000: return f.h1000;
        default: return f.tr;
    }
}

template <int V>
static void run_point(Ctx& c, hb::CsvWriter& csv, const Floors& fl, uint64_t F, uint64_t n, uint32_t sm, int gpc, int reps,
                      long long& launch_idx) {
    const std::string label = std::string("k1_") + kName[V] + "_F" + std::to_string(F) + "_sm" + std::to_string(sm);
    const bool inkernel_warm = n <= kInKernelWarmMax;
    const uint64_t hops = inkernel_warm ? round16(std::max<uint64_t>(n, kMinHops)) : kColdHops;
    const uint64_t warm = inkernel_warm ? round16(n) : 0;
    int retries = 0;
    if (!inkernel_warm) {  // untimed warm launch
        c.clean();
        retries += c.launch(k1_chase<V>, 0, kColdHops, sm);
    }
    char floor_s[32] = "";
    if (!is_generic(V)) std::snprintf(floor_s, sizeof floor_s, "%.3f", floor_of(fl, V));
    double sum = 0;
    for (int r = 0; r < reps; ++r) {
        if (!inkernel_warm) c.clean();
        retries += c.launch(k1_chase<V>, warm, hops, sm);
        const hb::ClockSample s = c.run.clock_note(label, "launch", launch_idx);
        const ChaseOut o = c.out();
        if (o.smid != sm) hb::die("claimed block reported smid " + std::to_string(o.smid));
        const double cph = double(o.cycles) / hops;
        sum += cph;
        char b[700];
        std::snprintf(b, sizeof b, "%s,%s,%llu,%llu,%llu,%d,%.4f,%s,%llu,%u,%d,%d", c.run.prefix(launch_idx, s).c_str(), kName[V],
                      (unsigned long long)F, (unsigned long long)hops, (unsigned long long)warm, r, cph, floor_s,
                      (unsigned long long)o.cycles, o.smid, gpc, retries);
        csv.add(b);
        csv.flush();
        ++launch_idx;
    }
    csv.sync();
    std::printf("k1 %-18s F %11llu sm %3u: %8.2f cycles/hop (mean of %d), retries %d\n", kName[V], (unsigned long long)F, sm,
                sum / reps, reps, retries);
    std::fflush(stdout);
}

template <int... Vs>
static void run_all(std::integer_sequence<int, Vs...>, Ctx& c, hb::CsvWriter& csv, const Floors& fl, uint64_t F, uint64_t n,
                    uint32_t sm, int gpc, int reps, long long& li) {
    (run_point<Vs>(c, csv, fl, F, n, sm, gpc, reps, li), ...);
}

static Floors measure_floors() {
    Floors f;
    f.tr = measure_pred_floor<W_TRY>(0);
    f.test = measure_pred_floor<W_TEST>(0);
    f.h0 = measure_pred_floor<W_HINT>(0);
    f.h100 = measure_pred_floor<W_HINT>(100);
    f.h1000 = measure_pred_floor<W_HINT>(1000);
    std::printf("poll floors (cycles/call): try %.2f test %.2f hint0 %.2f hint100 %.2f hint1000 %.2f\n", f.tr, f.test, f.h0,
                f.h100, f.h1000);
    return f;
}

static void run_chase(hb::Run& run, bool quick) {
    const std::vector<int> gpc = load_gpc_map(run.dev().sm_count);
    std::vector<uint32_t> probes = {(uint32_t)first_sm_of_gpc(gpc, 1), (uint32_t)first_sm_of_gpc(gpc, 4)};
    if (quick) probes.resize(1);
    hb::CsvWriter csv(hb::Run::results("06__tma__chase.csv"),
                      hb::with_mandatory({"variant", "footprint_bytes", "hops", "warm_hops", "rep", "cycles_per_hop",
                                          "poll_floor_cycles", "total_cycles", "smid", "gpc", "retries"}));
    Ctx c(run);
    run.lock_clocks_note("k1", "before");
    const Floors fl = measure_floors();
    long long launch_idx = 0;
    for (uint64_t F : footprints(quick)) {
        const uint64_t n = c.fill(F);
        const int reps = quick ? 2 : 5;
        for (uint32_t sm : probes)
            run_all(std::make_integer_sequence<int, V_COUNT>{}, c, csv, fl, F, n, sm, gpc[sm], reps, launch_idx);
    }
    run.lock_clocks_note("k1", "after");
}

template <int V>
static void ncu_point(Ctx& c, FILE* man, uint64_t F, uint64_t n, uint32_t sm, int gpc, long long& li) {
    const bool warm = n <= kInKernelWarmMax;
    if (!warm) c.clean();
    int retries = warm ? c.launch(k1_chase<V>, round16(n), 0, sm) : 0;  // warm (not profiled)
    retries += c.launch(k1_ncu<V>, 0, kMinHops, sm);
    const hb::ClockSample s = c.run.clock_note(std::string("ncu_") + kName[V] + "_F" + std::to_string(F), "launch", li);
    const ChaseOut o = c.out();
    std::fprintf(man, "%s,%s,%llu,%llu,%.4f,%u,%d,%d\n", c.run.prefix(li, s).c_str(), kName[V], (unsigned long long)F,
                 (unsigned long long)kMinHops, double(o.cycles) / kMinHops, o.smid, gpc, retries);
    std::fflush(man);
    ++li;
}

static void run_ncu(hb::Run& run) {
    const std::vector<int> gpc = load_gpc_map(run.dev().sm_count);
    const uint32_t sm = (uint32_t)first_sm_of_gpc(gpc, 1);
    FILE* man = std::fopen("bin/ncu_manifest_chase.csv", "w");
    if (!man) hb::die("cannot write bin/ncu_manifest_chase.csv");
    std::fprintf(man, "%s,variant,footprint_bytes,hops,cycles_per_hop_under_ncu,smid,gpc,retries\n", hb::join(hb::mandatory_cols()).c_str());
    Ctx c(run);
    long long li = 0;
    run.lock_clocks_note("ncu_chase", "before");
    uint64_t n = c.fill(8ull << 20);
    ncu_point<V_BULK16>(c, man, 8ull << 20, n, sm, gpc[sm], li);
    ncu_point<V_T2D>(c, man, 8ull << 20, n, sm, gpc[sm], li);
    ncu_point<V_LDCG>(c, man, 8ull << 20, n, sm, gpc[sm], li);
    ncu_point<V_CPASYNC>(c, man, 8ull << 20, n, sm, gpc[sm], li);
    n = c.fill(256ull << 20);
    ncu_point<V_BULK16>(c, man, 256ull << 20, n, sm, gpc[sm], li);
    run.lock_clocks_note("ncu_chase", "after");
    std::fclose(man);
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "chase";
    hb::Run run(EXP, argc, argv);
    if (mode == "clockcheck") run.lock_clocks_note(hb::arg_str(argc, argv, "--label", "check"), "check");
    else if (mode == "chase") run_chase(run, hb::has_flag(argc, argv, "--quick"));
    else if (mode == "ncu") run_ncu(run);
    else hb::die("unknown mode " + mode);
    run.complete();
    return 0;
}
