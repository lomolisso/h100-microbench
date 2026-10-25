/*
 * tma_completion.cu — experiment 06, K2: completion time of one bulk copy (L2 → smem) versus payload, the polling
 * floor of each mbarrier wait form, the cost of mbarrier.arrive.expect_tx, and the timer floor.
 *
 * Usage: bin/tma_completion {k2|ncu|clockcheck} [--quick] [--label L]. One invocation = one run_id.
 *
 * k2_copy<W, HINT>(src, payload, reps, hbm, base, target, claim, out): pinned with claim_target_sm to the chase's probe
 *   SM (first SM of GPC 1 of the 03 map; 2 × 132 one-warp blocks, the others return at once), thread 0 works, 64 KiB
 *   dynamic smem destination, one mbarrier (count 1). L2 source: a 4 MiB buffer, warmed at kernel start by the same
 *   thread with 64 untimed 64-KiB bulk copies. Sample r copies `payload` bytes from src + sample_offset(r): 64-KiB slot
 *   (17 r mod 61) plus (r mod 16) · 4 KiB, so the samples start in different 4-KiB pages with varying L2 homes:
 *     arrive.expect_tx(bar, payload) (untimed); t0 = clock64; cp.async.bulk.shared::cluster.global [dst], [src+off],
 *     payload, [bar]; t_issue = clock64; wait loop of form W until the phase of sample r completes (counts calls);
 *     t1 = clock64 (issued after the loop's final branch, i.e. after the predicate was true).
 *   W: try = try_wait.parity (no hint), test = test_wait.parity, hint = try_wait.parity with suspendTimeHint HINT ns
 *   (compile-time, so ptxas encodes it as in K1). Each sample → cycles_issue = t_issue − t0, cycles_issue_to_complete
 *   = t1 − t0, polls = wait calls made.
 *   Payloads: try: fine sweep 16..2048 B step 16, 2112..16384 step 64, 17408..65536 step 1024 (399 sizes, staircase);
 *   test, hint 0/100/1000: 19 coarse sizes 16 B … 64 KiB. Cold source (HBM): every form × the 19 coarse sizes, src =
 *   a 1 GiB buffer, L2 cleaned before each launch (05's CleanL2), no warm pass, sample r reads the never touched slot
 *   pair base + 2r (plus the same 4-KiB phase rotation). Per (W, source, payload): 2 unrecorded warm-up launches, then 5 recorded launches × 32 samples.
 *   parity = the mbarrier phase parity the sample waits on (r mod 2), recorded because a long wait's observed
 *   completion turned out to depend on it.
 * Predicate / issue costs (tma_common.hpp, one thread, 5 recorded launches after 2 warm-ups):
 *   pred            k_pred<W, false>, phase complete, 1024 dependent calls → cycles per call (includes the selp + mad glue)
 *   pred_baseline   k_pred<W, true>, same glue with setp instead of the wait (subtract from pred)
 *   pred_incomplete k_pred<W, false>, phase never completes: time per call that returns false (256 calls; 64 for
 *                   hint ≥ 100)
 *   expect_tx_lat   k_expect<true>: 1024 arrive.expect_tx(16) whose returned state feeds the next address
 *   expect_tx_issue k_expect<false>: 1024 independent back-to-back arrive.expect_tx(16)
 *   timer_floor     k_timer_floor: 32 × (clock64 → clock64)
 *   → results/06__tma__completion.csv: kind, wait_variant, hint_ns, source, payload_bytes, rep, parity, src_offset_bytes, cycles_issue,
 *     cycles_issue_to_complete, polls, cycles_predicate_only (per call), calls, total_cycles, smid
 *     (one row per sample, or per launch for the predicate kinds).
 * ncu mode: k2_ncu (= k2_copy<try>) with 64 KiB and 4 KiB payloads, 32 samples; manifest bin/ncu_manifest_completion.csv.
 */
#include <cstring>
#include <string>
#include <vector>

#include "../../05__memory_hierarchy_bandwidth/src/bw_common.hpp"  // CleanL2, and 04's gpc_map.hpp
#include "tma_common.hpp"

static const char* EXP = "06__tma_latency_and_issue";
static constexpr uint32_t kSrcBytes = 4u << 20, kSlot = 64u << 10, kSlots = kSrcBytes / kSlot, kReps = 32;
static constexpr uint64_t kHbmBytes = 1ull << 30;
static constexpr uint32_t kHbmSlots = (uint32_t)(kHbmBytes / kSlot);

struct CopySample { uint32_t issue, complete, polls, smid, parity, offset; };

// Source offset of sample r: different 64-KiB slots and 16 different 4-KiB phases, so that the first lines of the
// samples' L2 homes vary (home partition per 4-KiB page); a fixed in-slot offset gave every sample the same home.
__host__ __device__ __forceinline__ uint64_t sample_offset(uint32_t r, int hbm, uint32_t base_slot) {
    const uint64_t slot = hbm ? (uint64_t)base_slot + 2ull * r : (uint64_t)((r * 17u) % 61u);  // hbm: slot pairs, never reused
    return slot * kSlot + (uint64_t)(r % 16u) * 4096u;
}

// hbm = 0: src = the 4 MiB L2 buffer (warm pass), hbm = 1: src = the 1 GiB buffer, L2 cleaned by the host before the
// launch, no warm pass (cold: L2 miss → HBM). Sample r reads src + sample_offset(r). The block is pinned to SM `target`.
template <int W, uint32_t HINT>
__device__ __forceinline__ void copy_body(const uint8_t* src, uint32_t payload, uint32_t reps, int hbm, uint32_t base_slot,
                                          uint32_t target, uint32_t* claim, CopySample* out) {
    extern __shared__ __align__(1024) uint8_t dyn[];
    __shared__ alignas(8) uint64_t bar;
    if (!claim_target_sm(target, claim)) return;
    if (threadIdx.x != 0) return;
    const uint32_t d = smem_u32(dyn), b = smem_u32(&bar);
    mbar_init(b, 1);
    fence_mbarrier_init();
    uint32_t parity = 0;
#pragma unroll 1
    for (uint32_t s = 0; s < (hbm ? 0u : kSlots); ++s) {  // untimed warm pass over the whole 4 MiB
        mbar_arrive_expect_tx(b, kSlot);
        bulk_g2s(d, src + (size_t)s * kSlot, kSlot, b);
        wait_loop<W_TRY>(b, parity, 0);
        parity ^= 1;
    }
    const uint32_t smid = read_smid();
#pragma unroll 1
    for (uint32_t r = 0; r < reps; ++r) {
        const uint64_t off = sample_offset(r, hbm, base_slot);
        const uint8_t* p = src + off;
        mbar_arrive_expect_tx(b, payload);
        const uint64_t t0 = read_clock64();
        bulk_g2s(d, p, payload, b);
        const uint64_t ti = read_clock64();
        const uint32_t polls = wait_loop_count<W>(b, parity, HINT);
        const uint64_t t1 = read_clock64();
        out[r] = CopySample{(uint32_t)(ti - t0), (uint32_t)(t1 - t0), polls, smid, parity, (uint32_t)(off & 0xffffffffu)};
        parity ^= 1;
    }
}

template <int W, uint32_t HINT>
__global__ void __launch_bounds__(32) k2_copy(const uint8_t* src, uint32_t payload, uint32_t reps, int hbm, uint32_t base_slot,
                                               uint32_t target, uint32_t* claim, CopySample* out) {
    copy_body<W, HINT>(src, payload, reps, hbm, base_slot, target, claim, out);
}
__global__ void __launch_bounds__(32) k2_ncu(const uint8_t* src, uint32_t payload, uint32_t reps, int hbm, uint32_t base_slot,
                                              uint32_t target, uint32_t* claim, CopySample* out) {
    copy_body<W_TRY, 0>(src, payload, reps, hbm, base_slot, target, claim, out);
}

static std::vector<uint32_t> fine_payloads() {
    std::vector<uint32_t> v;
    for (uint32_t p = 16; p <= 2048; p += 16) v.push_back(p);
    for (uint32_t p = 2112; p <= 16384; p += 64) v.push_back(p);
    for (uint32_t p = 17408; p <= 65536; p += 1024) v.push_back(p);
    return v;
}
static std::vector<uint32_t> coarse_payloads() {
    return {16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152, 65536};
}

struct Ctx {
    hb::Run& run;
    hb::CsvWriter& csv;
    uint8_t* d_src = nullptr;
    uint8_t* d_hbm = nullptr;
    CleanL2 clean;
    hb::SmClaim claim;
    uint32_t target = 0;
    CopySample* d_out = nullptr;
    PredOut* d_pred = nullptr;
    uint64_t* d_tf = nullptr;
    long long li = 0;
    Ctx(hb::Run& r, hb::CsvWriter& c) : run(r), csv(c) {
        CUDA_CHECK(cudaMalloc(&d_src, kSrcBytes));
        CUDA_CHECK(cudaMemset(d_src, 0x5a, kSrcBytes));
        CUDA_CHECK(cudaMalloc(&d_hbm, kHbmBytes));
        CUDA_CHECK(cudaMemset(d_hbm, 0x3c, kHbmBytes));
        CUDA_CHECK(cudaMalloc(&d_out, kReps * sizeof(CopySample)));
        CUDA_CHECK(cudaMalloc(&d_pred, sizeof(PredOut)));
        CUDA_CHECK(cudaMalloc(&d_tf, kReps * sizeof(uint64_t)));
    }
    ~Ctx() { cudaFree(d_src); cudaFree(d_hbm); cudaFree(d_out); cudaFree(d_pred); cudaFree(d_tf); }

    template <class K>
    void launch_copy(K k, uint32_t payload, int hbm = 0) {
        uint32_t base = 0;
        if (hbm) {  // fresh slots each launch, spread over the buffer
            clean();
            base = (uint32_t)(((uint64_t)(li + 1) * 2654435761ull) % (kHbmSlots - 2 * kReps - 1));
        }
        for (int t = 0; t < 20; ++t) {
            claim.reset();
            k<<<hb::SmClaim::blocks(run.dev().sm_count), 32, kSlot>>>(hbm ? d_hbm : d_src, payload, kReps, hbm, base, target,
                                                                     claim.ptr(), d_out);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            if (claim.ok(target)) return;
        }
        hb::die("no block ran on SM " + std::to_string(target));
    }
    // One CSV row; empty strings for the columns a kind does not use.
    void row(const hb::ClockSample& s, const char* kind, const char* wv, uint32_t hint, const char* source, uint32_t payload,
             int rep, const std::string& parity, const std::string& offset, const std::string& issue, const std::string& complete,
             const std::string& polls, const std::string& per_call, const std::string& calls, const std::string& total, uint32_t smid) {
        char b[900];
        std::snprintf(b, sizeof b, "%s,%s,%s,%u,%s,%u,%d,%s,%s,%s,%s,%s,%s,%s,%s,%u", run.prefix(li, s).c_str(), kind, wv, hint, source,
                      payload, rep, parity.c_str(), offset.c_str(), issue.c_str(), complete.c_str(), polls.c_str(), per_call.c_str(),
                      calls.c_str(), total.c_str(), smid);
        csv.add(b);
    }
};

template <int W, uint32_t HINT>
static void copy_point(Ctx& c, const char* wv, uint32_t payload, int launches, int hbm = 0) {
    const char* source = hbm ? "hbm" : "l2";
    const std::string label = std::string("k2_") + wv + "_" + source + "_P" + std::to_string(payload);
    for (int w = 0; w < 2; ++w) c.launch_copy(k2_copy<W, HINT>, payload, hbm);  // warm-up launches
    double sum = 0;
    for (int l = 0; l < launches; ++l) {
        c.launch_copy(k2_copy<W, HINT>, payload, hbm);
        const hb::ClockSample s = c.run.clock_note(label, "launch", c.li);
        std::vector<CopySample> h(kReps);
        CUDA_CHECK(cudaMemcpy(h.data(), c.d_out, kReps * sizeof(CopySample), cudaMemcpyDeviceToHost));
        for (uint32_t r = 0; r < kReps; ++r) {
            c.row(s, "copy", wv, HINT, source, payload, (int)r, std::to_string(h[r].parity), std::to_string(h[r].offset),
                  std::to_string(h[r].issue), std::to_string(h[r].complete), std::to_string(h[r].polls), "", "", "", h[r].smid);
            sum += h[r].complete;
        }
        c.csv.flush();
        ++c.li;
    }
    c.csv.sync();
    std::printf("k2 copy %-8s %-3s payload %6u: %8.1f cycles issue→complete (mean)\n", wv, source, payload, sum / (launches * kReps));
    std::fflush(stdout);
}

template <class K, class... A>
static PredOut pred_launch(Ctx& c, K k, A... a) {
    k<<<1, 32>>>(a..., c.d_pred);
    CUDA_CHECK(cudaGetLastError());
    PredOut o;
    CUDA_CHECK(cudaMemcpy(&o, c.d_pred, sizeof o, cudaMemcpyDeviceToHost));
    return o;
}

template <int W>
static void pred_point(Ctx& c, const char* wv, uint32_t hint, int launches) {
    struct Kind { const char* name; bool baseline; int complete; uint32_t n; };
    const uint32_t n_incomplete = hint >= 100 ? 64 : 256;
    const Kind kinds[] = {{"pred", false, 1, 1024}, {"pred_baseline", true, 1, 1024}, {"pred_incomplete", false, 0, n_incomplete}};
    for (const Kind& k : kinds) {
        double sum = 0;
        for (int l = 0; l < launches + 2; ++l) {
            const PredOut o = k.baseline ? pred_launch(c, k_pred<W, true>, k.n, hint, k.complete, 0u)
                                         : pred_launch(c, k_pred<W, false>, k.n, hint, k.complete, 0u);
            if (l < 2) continue;  // warm-up launches
            const hb::ClockSample s = c.run.clock_note(std::string("k2_") + k.name + "_" + wv, "launch", c.li);
            const double per = double(o.cycles) / k.n;
            sum += per;
            char pc[32];
            std::snprintf(pc, sizeof pc, "%.4f", per);
            c.row(s, k.name, wv, hint, "", 0, l - 2, "", "", "", "", std::to_string(o.ok_count), pc, std::to_string(k.n),
                  std::to_string(o.cycles), o.smid);
            c.csv.flush();
            ++c.li;
        }
        c.csv.sync();
        std::printf("k2 %-16s %-8s: %8.2f cycles/call (mean)\n", k.name, wv, sum / launches);
    }
}

static void expect_and_timer(Ctx& c, int launches) {
    const uint32_t n = 1024;
    for (int dep = 1; dep >= 0; --dep) {
        const char* name = dep ? "expect_tx_lat" : "expect_tx_issue";
        double sum = 0;
        for (int l = 0; l < launches + 2; ++l) {
            const PredOut o = dep ? pred_launch(c, k_expect<true>, n, 0u) : pred_launch(c, k_expect<false>, n, 0u);
            if (l < 2) continue;
            const hb::ClockSample s = c.run.clock_note(std::string("k2_") + name, "launch", c.li);
            const double per = double(o.cycles) / n;
            sum += per;
            char pc[32];
            std::snprintf(pc, sizeof pc, "%.4f", per);
            c.row(s, name, "", 0, "", 16, l - 2, "", "", "", "", "", pc, std::to_string(n), std::to_string(o.cycles), o.smid);
            c.csv.flush();
            ++c.li;
        }
        c.csv.sync();
        std::printf("k2 %-16s: %8.2f cycles/call (mean)\n", name, sum / launches);
    }
    for (int l = 0; l < launches + 2; ++l) {
        k_timer_floor<<<1, 32>>>(kReps, c.d_tf);
        CUDA_CHECK(cudaGetLastError());
        std::vector<uint64_t> h(kReps);
        CUDA_CHECK(cudaMemcpy(h.data(), c.d_tf, kReps * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        if (l < 2) continue;
        const hb::ClockSample s = c.run.clock_note("k2_timer_floor", "launch", c.li);
        for (uint32_t r = 0; r < kReps; ++r) c.row(s, "timer_floor", "", 0, "", 0, (int)r, "", "", "", std::to_string(h[r]), "", "", "", "", 0);
        c.csv.flush();
        ++c.li;
    }
    c.csv.sync();
}

static void run_k2(hb::Run& run, bool quick) {
    hb::CsvWriter csv(hb::Run::results("06__tma__completion.csv"),
                      hb::with_mandatory({"kind", "wait_variant", "hint_ns", "source", "payload_bytes", "rep", "parity", "src_offset_bytes", "cycles_issue",
                                          "cycles_issue_to_complete", "polls", "cycles_predicate_only", "calls",
                                          "total_cycles", "smid"}));
    Ctx c(run, csv);
    c.target = (uint32_t)first_sm_of_gpc(load_gpc_map(run.dev().sm_count), 1);
    for (auto k : {k2_copy<W_TRY, 0>, k2_copy<W_TEST, 0>, k2_copy<W_HINT, 0>, k2_copy<W_HINT, 100>, k2_copy<W_HINT, 1000>, k2_ncu})
        CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kSlot));
    const int launches = quick ? 2 : 5;
    run.lock_clocks_note("k2", "before");
    expect_and_timer(c, launches);
    pred_point<W_TRY>(c, "try", 0, launches);
    pred_point<W_TEST>(c, "test", 0, launches);
    pred_point<W_HINT>(c, "hint0", 0, launches);
    pred_point<W_HINT>(c, "hint100", 100, launches);
    pred_point<W_HINT>(c, "hint1000", 1000, launches);
    for (uint32_t p : quick ? coarse_payloads() : fine_payloads()) copy_point<W_TRY, 0>(c, "try", p, launches);
    for (uint32_t p : coarse_payloads()) {
        copy_point<W_TEST, 0>(c, "test", p, launches);
        copy_point<W_HINT, 0>(c, "hint0", p, launches);
        copy_point<W_HINT, 100>(c, "hint100", p, launches);
        copy_point<W_HINT, 1000>(c, "hint1000", p, launches);
    }
    for (uint32_t p : coarse_payloads()) {  // cold source: completion after a long wait (HBM)
        copy_point<W_TRY, 0>(c, "try", p, launches, 1);
        copy_point<W_TEST, 0>(c, "test", p, launches, 1);
        copy_point<W_HINT, 0>(c, "hint0", p, launches, 1);
        copy_point<W_HINT, 100>(c, "hint100", p, launches, 1);
        copy_point<W_HINT, 1000>(c, "hint1000", p, launches, 1);
    }
    run.lock_clocks_note("k2", "after");
}

static void run_ncu(hb::Run& run) {
    FILE* man = std::fopen("bin/ncu_manifest_completion.csv", "w");
    if (!man) hb::die("cannot write bin/ncu_manifest_completion.csv");
    std::fprintf(man, "%s,wait_variant,payload_bytes,samples,mean_cycles_issue_to_complete_under_ncu,smid\n",
                 hb::join(hb::mandatory_cols()).c_str());
    hb::CsvWriter dummy(hb::Run::results("06__tma__completion.csv"),
                        hb::with_mandatory({"kind", "wait_variant", "hint_ns", "source", "payload_bytes", "rep", "parity", "src_offset_bytes", "cycles_issue",
                                            "cycles_issue_to_complete", "polls", "cycles_predicate_only", "calls",
                                            "total_cycles", "smid"}));  // header check only; ncu mode writes no rows here
    Ctx c(run, dummy);
    c.target = (uint32_t)first_sm_of_gpc(load_gpc_map(run.dev().sm_count), 1);
    CUDA_CHECK(cudaFuncSetAttribute(k2_ncu, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kSlot));
    run.lock_clocks_note("ncu_completion", "before");
    for (uint32_t p : {65536u, 4096u}) {
        c.launch_copy(k2_ncu, p);
        const hb::ClockSample s = run.clock_note("ncu_k2_P" + std::to_string(p), "launch", c.li);
        std::vector<CopySample> h(kReps);
        CUDA_CHECK(cudaMemcpy(h.data(), c.d_out, kReps * sizeof(CopySample), cudaMemcpyDeviceToHost));
        double sum = 0;
        for (auto& x : h) sum += x.complete;
        std::fprintf(man, "%s,try,%u,%u,%.1f,%u\n", run.prefix(c.li, s).c_str(), p, kReps, sum / kReps, h[0].smid);
        std::fflush(man);
        ++c.li;
    }
    run.lock_clocks_note("ncu_completion", "after");
    std::fclose(man);
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "k2";
    hb::Run run(EXP, argc, argv);
    if (mode == "clockcheck") run.lock_clocks_note(hb::arg_str(argc, argv, "--label", "check"), "check");
    else if (mode == "k2") run_k2(run, hb::has_flag(argc, argv, "--quick"));
    else if (mode == "ncu") run_ncu(run);
    else hb::die("unknown mode " + mode);
    run.complete();
    return 0;
}
