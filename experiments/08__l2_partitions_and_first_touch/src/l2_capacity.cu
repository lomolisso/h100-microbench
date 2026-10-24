/*
 * l2_capacity.cu — experiment 08, K4: how much data one SM can keep in L2, and which kind of access (near hit, far
 * hit, miss) serves its chase as the footprint grows past the capacity of its partition and of the whole L2.
 *
 * Usage: bin/l2_capacity {k4|ncu|clockcheck} [--quick] [--timer-pair N]. One invocation = one run_id.
 *
 * Chain: 04's pchase_chain.hpp (Sattolo cycle, kChainSeed, one pointer per 128 B line) over the first F bytes of an
 *   80 MiB buffer; footprints F = 16, 18, …, 80 MiB. The chain for F is written by the touch kernel itself.
 * Configuration (F, probe SM P, touch pattern, rep):
 *   1. CleanL2 (128 MB flush + 128 MB clean read, 05's bw_common.hpp);
 *   2. touch: k_touch_chain, one claimed warp, whole 128 B lines (qword 0 = address of line next[i], rest 0):
 *        pattern "probe": SM P writes every line;
 *        pattern "alternate": SM 0 writes the even lines, then SM 2 the odd lines (an SM of each partition);
 *   3. k_chase_hist on P: 132 blocks × 32 threads with 200 KB smem (1 CTA/SM; the block on P wins the claim).
 *      Lane 0 chases F/128 hops per pass (one full cycle = every line once) with ld.global.cg.u64:
 *        passes 0-2, every hop timed: t0 = %clock64; p = ld [p]; t1 = read_clock64_after(p); hist[pass][min(t1 − t0,
 *        2048)] += 1 in shared memory (no global traffic during the chase); pass 0 is the first read after the touch;
 *        pass 3 untimed per hop (plain chase), timed as a whole.
 *      Every pass records its total %clock64 cycles.
 *   SM P ∈ {0, 2} (one per partition), 3 reps; order rep → F → P → pattern.
 * CSV:
 *   capacity_hist.csv  exact per-pass histograms: touch_pattern, footprint_bytes, probe_smid, rep, pass, cycles (bin;
 *                      2048 = ≥ 2048), count. The order of the samples and the line of each sample are not kept.
 *   capacity.csv       spec columns touch_pattern, footprint_bytes, cycles_per_hop, lts_hit_rate_pct (empty here:
 *                      ncu values are in ncu_capacity.csv) + probe_smid, rep, pass, pass_kind (timed / plain), hops,
 *                      pass_cycles, mean_sample_cycles. cycles_per_hop: plain pass = pass_cycles / hops; timed passes =
 *                      mean sample − timer pair (--timer-pair, from l2_probe calib).
 * ncu mode (profile regex k_ncu): F = 16, 20, …, 80 MiB, P ∈ {0, 2}, both patterns: CleanL2 → touch → k_chase_hist
 *   (2 passes, warm) → k_ncu_chase (1 plain pass, profiled). Kernel order → ncu_manifest.csv (lines = F / 128,
 *   what = touch pattern, first_median = cycles per hop of the profiled pass).
 */
#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "../../04__memory_hierarchy_latency/src/pchase_chain.hpp"
#include "../../05__memory_hierarchy_bandwidth/src/bw_common.hpp"
#include "harness.cuh"
#include "host.hpp"

static const char* EXP = "08__l2_partitions_and_first_touch";
static constexpr uint32_t kBins = 2049, kTimedPasses = 3, kPasses = 4;
static constexpr size_t kSmem = 200 * 1024;  // 1 CTA/SM; the histograms use 3 × 2049 × 4 B of it
static constexpr uint64_t kBufBytes = 80ull << 20;
static constexpr uint32_t kRef0 = 0, kRef1 = 2;
static constexpr int kMaxRetries = 20;

__device__ __forceinline__ uint64_t ld_cg(uint64_t p) {
    uint64_t v;
    asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

// which < 0: every line; 0 / 1: even / odd lines only.
__global__ void __launch_bounds__(32) k_touch_chain(uint64_t* buf, const uint32_t* next, uint32_t n, int which, uint32_t target, uint32_t* claim) {
    if (!claim_target_sm(target, claim)) return;
    const uint64_t base = (uint64_t)buf;
    for (uint64_t q = threadIdx.x; q < (uint64_t)n * chain::kQwordsPerLine; q += 32) {
        const uint64_t line = q / chain::kQwordsPerLine;
        if (which >= 0 && (int)(line & 1) != which) continue;
        buf[q] = (q % chain::kQwordsPerLine) ? 0 : base + (uint64_t)next[line] * chain::kLineBytes;
    }
}

__global__ void __launch_bounds__(32) k_chase_hist(const uint64_t* start, uint32_t hops, uint32_t passes, uint32_t target, uint32_t* claim,
                                                    uint64_t zero, uint32_t* hist_out, uint64_t* pass_cycles, uint64_t* sink) {
    extern __shared__ uint32_t h[];  // [pass][kBins]
    if (!claim_target_sm(target, claim)) return;
    for (uint32_t i = threadIdx.x; i < kTimedPasses * kBins; i += 32) h[i] = 0;
    __syncwarp();
    if (threadIdx.x == 0) {
        uint64_t p = (uint64_t)start;
        for (uint32_t pass = 0; pass < passes; ++pass) {
            const uint64_t c0 = read_clock64_after(p, zero);
            if (pass < kTimedPasses) {
                uint32_t* hp = h + pass * kBins;
#pragma unroll 1
                for (uint32_t k = 0; k < hops; ++k) {
                    const uint64_t t0 = read_clock64();
                    p = ld_cg(p);
                    const uint64_t t1 = read_clock64_after(p, zero);
                    hp[min((uint32_t)(t1 - t0), kBins - 1)] += 1;
                }
            } else {
#pragma unroll 1
                for (uint32_t k = 0; k < hops; ++k) p = ld_cg(p);
            }
            pass_cycles[pass] = read_clock64_after(p, zero) - c0;
        }
        sink[0] = p;
    }
    __syncwarp();
    for (uint32_t i = threadIdx.x; i < kTimedPasses * kBins; i += 32) hist_out[i] = h[i];
}

__global__ void __launch_bounds__(32) k_ncu_chase(const uint64_t* start, uint32_t hops, uint32_t target, uint32_t* claim, uint64_t zero,
                                                   uint64_t* cycles, uint64_t* sink) {
    if (!claim_target_sm(target, claim)) return;
    if (threadIdx.x == 0) {
        uint64_t p = (uint64_t)start;
        const uint64_t c0 = read_clock64();
#pragma unroll 1
        for (uint32_t k = 0; k < hops; ++k) p = ld_cg(p);
        cycles[0] = read_clock64_after(p, zero) - c0;
        sink[0] = p;
    }
}

class Cap {
  public:
    explicit Cap(hb::Run& r) : sm_count_(r.dev().sm_count) {
        CUDA_CHECK(cudaMalloc(&buf_, kBufBytes));
        CUDA_CHECK(cudaMalloc(&d_next_, kBufBytes / chain::kLineBytes * 4));
        CUDA_CHECK(cudaMalloc(&d_hist_, kTimedPasses * kBins * 4));
        CUDA_CHECK(cudaMalloc(&d_cyc_, kPasses * 8));
        CUDA_CHECK(cudaMalloc(&d_sink_, 8));
        CUDA_CHECK(cudaFuncSetAttribute(k_chase_hist, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kSmem));
        CUDA_CHECK(cudaFuncSetAttribute(k_ncu_chase, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kSmem));
        hist_.resize(kTimedPasses * kBins);
        cyc_.resize(kPasses);
    }
    ~Cap() { cudaFree(buf_); cudaFree(d_next_); cudaFree(d_hist_); cudaFree(d_cyc_); cudaFree(d_sink_); }

    uint32_t set_footprint(uint64_t F) {
        n_ = (uint32_t)(F / chain::kLineBytes);
        auto it = chains_.find(n_);
        if (it == chains_.end()) it = chains_.emplace(n_, chain::sattolo_cycle(n_)).first;
        CUDA_CHECK(cudaMemcpy(d_next_, it->second.data(), (size_t)n_ * 4, cudaMemcpyHostToDevice));
        return n_;
    }
    // CleanL2 and the touch pattern.
    void prepare(const std::string& pattern, uint32_t P) {
        clean_();
        if (pattern == "probe") touch(-1, P);
        else { touch(0, kRef0); touch(1, kRef1); }
    }
    void chase(uint32_t P, uint32_t passes) {
        claimed([&] { k_chase_hist<<<sm_count_, 32, kSmem>>>(buf_, n_, passes, P, claim_.ptr(), 0ull, d_hist_, d_cyc_, d_sink_); }, P);
        CUDA_CHECK(cudaMemcpy(hist_.data(), d_hist_, hist_.size() * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(cyc_.data(), d_cyc_, kPasses * 8, cudaMemcpyDeviceToHost));
    }
    uint64_t ncu_chase(uint32_t P) {
        claimed([&] { k_ncu_chase<<<sm_count_, 32, kSmem>>>(buf_, n_, P, claim_.ptr(), 0ull, d_cyc_, d_sink_); }, P);
        uint64_t c = 0;
        CUDA_CHECK(cudaMemcpy(&c, d_cyc_, 8, cudaMemcpyDeviceToHost));
        return c;
    }
    const uint32_t* hist(uint32_t pass) const { return hist_.data() + pass * kBins; }
    uint64_t pass_cycles(uint32_t pass) const { return cyc_[pass]; }
    int retries() const { return retries_; }

  private:
    void touch(int which, uint32_t T) {
        claimed([&] { k_touch_chain<<<hb::SmClaim::blocks(sm_count_), 32>>>(buf_, d_next_, n_, which, T, claim_.ptr()); }, T);
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
    int sm_count_;
    uint64_t* buf_ = nullptr;
    uint32_t* d_next_ = nullptr;
    uint32_t* d_hist_ = nullptr;
    uint64_t* d_cyc_ = nullptr;
    uint64_t* d_sink_ = nullptr;
    uint32_t n_ = 0;
    std::map<uint32_t, std::vector<uint32_t>> chains_;
    hb::SmClaim claim_;
    CleanL2 clean_;
    std::vector<uint32_t> hist_;
    std::vector<uint64_t> cyc_;
    int retries_ = 0;
};

static std::vector<uint64_t> footprints(bool quick, uint64_t step_mib) {
    std::vector<uint64_t> f;
    if (quick) return {16ull << 20, 32ull << 20, 64ull << 20};
    for (uint64_t m = 16; m <= 80; m += step_mib) f.push_back(m << 20);
    return f;
}

static void write_stats(hb::Run& run, const char* mode, long long configs, int retries, int bad) {
    hb::CsvWriter st(hb::Run::results("08__l2__launch_stats.csv"), hb::with_mandatory({"mode", "configs", "claim_retries", "bad_clock_configs"}));
    char b[512];
    std::snprintf(b, sizeof b, "%s,%s,%lld,%d,%d", run.prefix(-1, run.dev().sample()).c_str(), mode, configs, retries, bad);
    st.add(b);
    st.sync();
    std::printf("%s: %lld configurations, %d claim retries, %d with a clock warning\n", mode, configs, retries, bad);
}

static void run_k4(hb::Run& run, bool quick, double timer_pair) {
    Cap cap(run);
    hb::CsvWriter hcsv(hb::Run::results("08__l2__capacity_hist.csv"),
                       hb::with_mandatory({"touch_pattern", "footprint_bytes", "probe_smid", "rep", "pass", "cycles", "count"}));
    hb::CsvWriter ccsv(hb::Run::results("08__l2__capacity.csv"),
                       hb::with_mandatory({"touch_pattern", "footprint_bytes", "cycles_per_hop", "lts_hit_rate_pct", "probe_smid", "rep", "pass",
                                           "pass_kind", "hops", "pass_cycles", "mean_sample_cycles"}));
    const int reps = quick ? 1 : 3;
    long long li = 0;
    int bad = 0;
    char b[512];
    run.lock_clocks_note("k4", "before");
    for (int rep = 0; rep < reps; ++rep)
        for (uint64_t F : footprints(quick, 2)) {
            const uint32_t n = cap.set_footprint(F);
            for (uint32_t P : {kRef0, kRef1})
                for (const char* pat : {"probe", "alternate"}) {
                    cap.prepare(pat, P);
                    cap.chase(P, kPasses);
                    const hb::ClockSample s = run.clock_note("k4", "launch", li);
                    bad += s.bad();
                    const std::string pre = run.prefix(li, s);
                    std::printf("K4 rep %d F %3llu MiB SM %u %-9s:", rep, (unsigned long long)(F >> 20), P, pat);
                    for (uint32_t pass = 0; pass < kPasses; ++pass) {
                        double mean = 0;
                        if (pass < kTimedPasses) {
                            const uint32_t* h = cap.hist(pass);
                            uint64_t cnt = 0;
                            for (uint32_t c = 0; c < kBins; ++c)
                                if (h[c]) {
                                    std::snprintf(b, sizeof b, "%s,%s,%llu,%u,%d,%u,%u,%u", pre.c_str(), pat, (unsigned long long)F, P, rep, pass, c, h[c]);
                                    hcsv.add(b);
                                    mean += (double)c * h[c];
                                    cnt += h[c];
                                }
                            if (cnt != n) hb::die("histogram count differs from hops");
                            mean /= cnt;
                        }
                        const double cph = pass < kTimedPasses ? mean - timer_pair : (double)cap.pass_cycles(pass) / n;
                        std::snprintf(b, sizeof b, "%s,%s,%llu,%.3f,,%u,%d,%u,%s,%u,%llu,%s", pre.c_str(), pat, (unsigned long long)F, cph, P, rep, pass,
                                      pass < kTimedPasses ? "timed" : "plain", n, (unsigned long long)cap.pass_cycles(pass),
                                      pass < kTimedPasses ? std::to_string(mean).c_str() : "");
                        ccsv.add(b);
                        std::printf(" %.1f", cph);
                    }
                    std::printf("\n");
                    std::fflush(stdout);
                    hcsv.flush();
                    ccsv.flush();
                    ++li;
                }
            hcsv.sync();
            ccsv.sync();
        }
    run.lock_clocks_note("k4", "after");
    write_stats(run, "k4", li, cap.retries(), bad);
}

static void run_ncu(hb::Run& run, bool quick) {
    Cap cap(run);
    hb::CsvWriter man(hb::Run::results("08__l2__ncu_manifest.csv"),
                      hb::with_mandatory({"mode", "ncu_order", "kernel", "probe_smid", "first_touch_smid", "lines", "line_offset", "what",
                                          "first_median", "repeat_median"}));
    long long li = 0;
    int order = 0;
    char b[512];
    for (uint64_t F : footprints(quick, 4)) {
        const uint32_t n = cap.set_footprint(F);
        for (uint32_t P : {kRef0, kRef1})
            for (const char* pat : {"probe", "alternate"}) {
                cap.prepare(pat, P);
                cap.chase(P, 2);
                const uint64_t c = cap.ncu_chase(P);
                std::snprintf(b, sizeof b, "%s,ncu_capacity,%d,k_ncu_chase,%u,%d,%u,-1,%s,%.3f,0", run.prefix(li, run.clock_note("ncu_k4", "launch", li)).c_str(),
                              order++, P, std::string(pat) == "probe" ? (int)P : -1, n, pat, (double)c / n);
                man.add(b);
                man.flush();
                ++li;
            }
    }
    man.sync();
    write_stats(run, "ncu_capacity", li, cap.retries(), 0);
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "k4";
    const bool quick = hb::has_flag(argc, argv, "--quick");
    hb::Run run(EXP, argc, argv);
    if (mode == "clockcheck") run.lock_clocks_note(hb::arg_str(argc, argv, "--label", "check"), "check");
    else if (mode == "k4") run_k4(run, quick, std::atof(hb::arg_str(argc, argv, "--timer-pair", "3")));
    else if (mode == "ncu") run_ncu(run, quick);
    else hb::die("unknown mode " + mode);
    run.complete();
    return 0;
}
