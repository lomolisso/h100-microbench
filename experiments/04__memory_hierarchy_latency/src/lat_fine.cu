/*
 * lat_fine.cu — experiment 04, K3: latency of every individual dependent L2/HBM access, to separate latency classes
 * (near/far L2 partition, hit/miss), and the effect of which SM touched a line first.
 *
 * Usage: bin/lat_fine {calib|matrix|raw|clockcheck} [--pairs p:t,...] [--quick] [--label L]. One invocation = one run_id.
 *
 * Chain: as in K2 (pchase_chain.hpp): footprint F = n lines of 128 B, one random cycle (Sattolo, kChainSeed), qword 0
 *   of line i holds the address of line next[i]; hop k reads line order[k]. One buffer per footprint
 *   (8 MiB = 65 536 lines, 256 MiB = 2 Mi lines), reused by every configuration of the invocation.
 * Configuration (toucher T, probe P, F):
 *   1. L2 flush (hb::L2Flusher, 128 MB streamed);
 *   2. k3_touch on SM T writes every line completely (qword 0 = pointer, the rest 0; first touch of every line after
 *      the flush), one warp, claim-pinned (claim_target_sm, 2 × 132 one-warp blocks);
 *   3. k3_probe on SM P: 132 blocks × 32 threads with 200 KB dynamic smem (exactly 1 CTA/SM; the block on P is the
 *      claimed one, all others return). Lane 0 chases 131 072 hops with ld.global.cg.u64 and times EACH hop:
 *        t0 = %clock64; p = ld.global.cg [p]; t1 = read_clock64_after(p) (cannot issue before the load returned);
 *        cycles = t1 − t0 → smem (u32). Every 51 200 samples (smem full) the warp copies them to global, outside
 *      every timed access. 8 MiB: 2 full passes over the 65 536 lines (pass 1 = hops ≥ n: lines already read once by
 *      P); 256 MiB: the first 131 072 hops (one pass, ~6 % of the lines).
 *   cycles include the cost of the timer pair around the load (calib below).
 * Modes:
 *   calib  — per-access timing of a 1024-word shared-memory chain (ld.shared.u32, stride 1) on each probe SM,
 *            4096 samples, 3 launches. Calibration median − K1's LDS latency = timer-pair cost.
 *            → results/04__latency__fine_calib.csv: probe_smid, sample, cycles.
 *   matrix — T, P ∈ first SM of each of the 8 GPCs (03 map) → 64 cells × 2 footprints × 3 launches. Stored as exact
 *            histograms per launch and pass (the per-sample order and line_offset are not kept for the matrix).
 *            → results/04__latency__fine_hist.csv: footprint_bytes, first_touch_smid, probe_smid, pass, cycles, count.
 *   raw    — for each probe P (first SM of each GPC) and footprint: variant A (T = P) and variant B (T = the toucher
 *            given for P in --pairs, chosen by src/pick_toucher.py from the matrix), 1 launch each; every sample.
 *            → results/04__latency__fine.csv: footprint_bytes, first_touch_smid, probe_smid, hop_idx, line_offset, cycles
 *              (line_offset = byte offset of the line from the buffer start).
 *   --quick: 8 MiB only; matrix with 1 launch per cell; calib and raw on 2 probes (GPCs 1, 2) (smoke test).
 */
#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "gpc_map.hpp"
#include "harness.cuh"
#include "host.hpp"
#include "pchase_chain.hpp"

static const char* EXP = "04__memory_hierarchy_latency";
static constexpr uint32_t kHops = 131072, kBatch = 51200, kCalibHops = 4096, kCalibWords = 1024;
static constexpr size_t kProbeSmem = kBatch * sizeof(uint32_t);  // 200 KB → 1 CTA/SM
static constexpr int kMaxRetries = 20;

__global__ void __launch_bounds__(32) k3_touch(uint64_t* buf, const uint32_t* next, uint32_t n, uint32_t target, uint32_t* claim) {
    if (!claim_target_sm(target, claim)) return;
    // Whole 128 B lines (qword 0 = pointer, qwords 1-15 = 0), 2 lines per coalesced warp store: a partial-line write
    // would leave the L2 sectors partially valid, and the probe's first read of each line would then need a DRAM fill.
    const uint64_t base = (uint64_t)buf;
    for (uint64_t q = threadIdx.x; q < (uint64_t)n * chain::kQwordsPerLine; q += 32) {
        const uint64_t line = q / chain::kQwordsPerLine;
        buf[q] = (q % chain::kQwordsPerLine) ? 0 : base + (uint64_t)next[line] * chain::kLineBytes;
    }
}

__device__ __forceinline__ uint64_t ld_cg(uint64_t p) {
    uint64_t v;
    asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

__global__ void __launch_bounds__(32) k3_probe(const uint64_t* start, uint32_t hops, uint32_t target, uint32_t* claim, uint64_t zero,
                                                uint32_t* out_cycles, uint64_t* out_sink) {
    extern __shared__ uint32_t s[];
    if (!claim_target_sm(target, claim)) return;
    uint64_t p = (uint64_t)start;
    for (uint32_t b0 = 0; b0 < hops; b0 += kBatch) {
        const uint32_t m = min(kBatch, hops - b0);
        if (threadIdx.x == 0) {
#pragma unroll 1
            for (uint32_t k = 0; k < m; ++k) {
                const uint64_t t0 = read_clock64();
                p = ld_cg(p);
                const uint64_t t1 = read_clock64_after(p, zero);
                s[k] = (uint32_t)(t1 - t0);
            }
        }
        __syncwarp();
        for (uint32_t k = threadIdx.x; k < m; k += 32) out_cycles[b0 + k] = s[k];
        __syncwarp();
    }
    if (threadIdx.x == 0) out_sink[0] = p;
}

__global__ void __launch_bounds__(32) k3_calib(uint32_t hops, uint32_t target, uint32_t* claim, uint64_t zero, uint32_t* out_cycles,
                                                uint64_t* out_sink) {
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
        out_sink[0] = p;
    }
    __syncwarp();
    for (uint32_t k = threadIdx.x; k < hops; k += 32) out_cycles[k] = samples[k];
}

struct Footprint {
    uint64_t bytes = 0;
    uint32_t n = 0;
    uint64_t* d_buf = nullptr;
    uint32_t* d_next = nullptr;
    std::vector<uint32_t> order;  // line of hop k, k < kHops
};

class Fine {
  public:
    explicit Fine(hb::Run& r) : run_(r), sm_count_(r.dev().sm_count) {
        CUDA_CHECK(cudaMalloc(&d_cycles_, kHops * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_sink_, sizeof(uint64_t)));
        CUDA_CHECK(cudaFuncSetAttribute(k3_probe, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kProbeSmem));
        CUDA_CHECK(cudaFuncSetAttribute(k3_calib, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kProbeSmem));
        cycles_.resize(kHops);
    }
    ~Fine() {
        cudaFree(d_cycles_);
        cudaFree(d_sink_);
        for (auto& f : fps_) { cudaFree(f.d_buf); cudaFree(f.d_next); }
    }

    Footprint& footprint(uint64_t bytes) {
        for (auto& f : fps_) if (f.bytes == bytes) return f;
        Footprint f;
        f.bytes = bytes;
        f.n = (uint32_t)(bytes / chain::kLineBytes);
        const std::vector<uint32_t> next = chain::sattolo_cycle(f.n);
        f.order = chain::chase_order(next, kHops);
        CUDA_CHECK(cudaMalloc(&f.d_buf, bytes));
        CUDA_CHECK(cudaMalloc(&f.d_next, f.n * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(f.d_next, next.data(), f.n * sizeof(uint32_t), cudaMemcpyHostToDevice));
        fps_.push_back(std::move(f));
        return fps_.back();
    }

    // flush → touch on T → probe on P; fills cycles(). Returns retries.
    int configure_and_probe(Footprint& f, uint32_t toucher, uint32_t probe) {
        flusher_();
        int retries = claimed([&] { k3_touch<<<hb::SmClaim::blocks(sm_count_), 32>>>(f.d_buf, f.d_next, f.n, toucher, claim_.ptr()); }, toucher);
        retries += claimed([&] { k3_probe<<<sm_count_, 32, kProbeSmem>>>(f.d_buf, kHops, probe, claim_.ptr(), 0ull, d_cycles_, d_sink_); }, probe);
        CUDA_CHECK(cudaMemcpy(cycles_.data(), d_cycles_, kHops * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        return retries;
    }
    int calib(uint32_t probe) {
        const size_t smem = (kCalibWords + kCalibHops) * sizeof(uint32_t);
        int r = claimed([&] { k3_calib<<<sm_count_, 32, smem>>>(kCalibHops, probe, claim_.ptr(), 0ull, d_cycles_, d_sink_); }, probe);
        CUDA_CHECK(cudaMemcpy(cycles_.data(), d_cycles_, kCalibHops * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        return r;
    }
    const std::vector<uint32_t>& cycles() const { return cycles_; }

  private:
    template <class L>
    int claimed(L launch, uint32_t target) {
        for (int r = 0; r <= kMaxRetries; ++r) {
            claim_.reset();
            launch();
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            if (claim_.ok(target)) return r;
        }
        hb::die("no block ran on SM " + std::to_string(target));
    }
    hb::Run& run_;
    int sm_count_;
    hb::SmClaim claim_;
    hb::L2Flusher flusher_;
    uint32_t* d_cycles_ = nullptr;
    uint64_t* d_sink_ = nullptr;
    std::vector<uint32_t> cycles_;
    std::vector<Footprint> fps_;
};

static std::vector<uint32_t> probe_sms(const std::vector<int>& gpc, bool quick) {
    std::vector<uint32_t> p;
    if (quick) for (int g : {1, 2}) p.push_back((uint32_t)first_sm_of_gpc(gpc, g));
    else for (int g = 0; g < 8; ++g) p.push_back((uint32_t)first_sm_of_gpc(gpc, g));
    return p;
}
static std::vector<uint64_t> fine_footprints(bool quick) {
    if (quick) return {8ull << 20};
    return {8ull << 20, 256ull << 20};
}

static void run_calib(hb::Run& run, bool quick) {
    const auto probes = probe_sms(load_gpc_map(run.dev().sm_count), quick);
    hb::CsvWriter csv(hb::Run::results("04__latency__fine_calib.csv"), hb::with_mandatory({"probe_smid", "sample", "cycles"}));
    Fine fine(run);
    long long li = 0;
    for (uint32_t p : probes)
        for (int l = 0; l < 3; ++l) {
            fine.calib(p);
            const hb::ClockSample s = run.clock_note("calib_sm" + std::to_string(p), "launch", li);
            const std::string pre = run.prefix(li, s);
            char b[320];
            for (uint32_t k = 0; k < kCalibHops; ++k) {
                std::snprintf(b, sizeof b, "%s,%u,%u,%u", pre.c_str(), p, k, fine.cycles()[k]);
                csv.add(b);
            }
            csv.flush();
            ++li;
        }
    csv.sync();
}

static void run_matrix(hb::Run& run, bool quick) {
    const auto sms = probe_sms(load_gpc_map(run.dev().sm_count), false);  // all 8 GPCs, also with --quick
    hb::CsvWriter csv(hb::Run::results("04__latency__fine_hist.csv"),
                      hb::with_mandatory({"footprint_bytes", "first_touch_smid", "probe_smid", "pass", "cycles", "count"}));
    Fine fine(run);
    const int launches = quick ? 1 : 3;
    long long li = 0;
    run.lock_clocks_note("matrix", "before");
    for (uint64_t F : fine_footprints(quick)) {
        Footprint& f = fine.footprint(F);
        for (int l = 0; l < launches; ++l)
            for (uint32_t t : sms)
                for (uint32_t p : sms) {
                    fine.configure_and_probe(f, t, p);
                    const hb::ClockSample s = run.clock_note("matrix_F" + std::to_string(F) + "_t" + std::to_string(t) + "_p" + std::to_string(p), "launch", li);
                    std::map<std::pair<uint32_t, uint32_t>, uint64_t> h;  // (pass, cycles) → count
                    double sum = 0;
                    for (uint32_t k = 0; k < kHops; ++k) { ++h[{k / f.n, fine.cycles()[k]}]; sum += fine.cycles()[k]; }
                    const std::string pre = run.prefix(li, s);
                    char b[320];
                    for (const auto& [key, cnt] : h) {
                        std::snprintf(b, sizeof b, "%s,%llu,%u,%u,%u,%u,%llu", pre.c_str(), (unsigned long long)F, t, p, key.first, key.second,
                                      (unsigned long long)cnt);
                        csv.add(b);
                    }
                    csv.flush();
                    std::printf("matrix F %9llu toucher %3u probe %3u launch %d: mean %.1f cycles\n", (unsigned long long)F, t, p, l, sum / kHops);
                    std::fflush(stdout);
                    ++li;
                }
        csv.sync();
    }
    run.lock_clocks_note("matrix", "after");
}

static std::map<uint32_t, uint32_t> parse_pairs(const char* s) {
    std::map<uint32_t, uint32_t> m;  // probe → toucher
    std::stringstream ss(s ? s : "");
    for (std::string tok; std::getline(ss, tok, ',');) {
        const auto c = tok.find(':');
        if (c == std::string::npos) hb::die("bad --pairs entry " + tok);
        m[(uint32_t)std::stoul(tok.substr(0, c))] = (uint32_t)std::stoul(tok.substr(c + 1));
    }
    return m;
}

static void run_raw(hb::Run& run, bool quick, const char* pairs_arg) {
    const auto probes = probe_sms(load_gpc_map(run.dev().sm_count), quick);
    const auto pairs = parse_pairs(pairs_arg);
    for (uint32_t p : probes)
        if (!pairs.count(p)) hb::die("--pairs gives no toucher for probe " + std::to_string(p));
    hb::CsvWriter csv(hb::Run::results("04__latency__fine.csv"),
                      hb::with_mandatory({"footprint_bytes", "first_touch_smid", "probe_smid", "hop_idx", "line_offset", "cycles"}));
    Fine fine(run);
    long long li = 0;
    run.lock_clocks_note("raw", "before");
    for (uint64_t F : fine_footprints(quick)) {
        Footprint& f = fine.footprint(F);
        for (uint32_t p : probes)
            for (uint32_t t : {p, pairs.at(p)}) {  // variant A, then B
                fine.configure_and_probe(f, t, p);
                const hb::ClockSample s = run.clock_note("raw_F" + std::to_string(F) + "_t" + std::to_string(t) + "_p" + std::to_string(p), "launch", li);
                const std::string pre = run.prefix(li, s);
                char b[320];
                double sum = 0;
                for (uint32_t k = 0; k < kHops; ++k) {
                    std::snprintf(b, sizeof b, "%s,%llu,%u,%u,%u,%llu,%u", pre.c_str(), (unsigned long long)F, t, p, k,
                                  (unsigned long long)f.order[k] * chain::kLineBytes, fine.cycles()[k]);
                    csv.add(b);
                    sum += fine.cycles()[k];
                }
                csv.flush();
                std::printf("raw F %9llu toucher %3u probe %3u: mean %.1f cycles\n", (unsigned long long)F, t, p, sum / kHops);
                std::fflush(stdout);
                ++li;
            }
        csv.sync();
    }
    run.lock_clocks_note("raw", "after");
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "matrix";
    const bool quick = hb::has_flag(argc, argv, "--quick");
    hb::Run run(EXP, argc, argv);
    if (mode == "clockcheck") run.lock_clocks_note(hb::arg_str(argc, argv, "--label", "check"), "check");
    else if (mode == "calib") run_calib(run, quick);
    else if (mode == "matrix") run_matrix(run, quick);
    else if (mode == "raw") run_raw(run, quick, hb::arg_str(argc, argv, "--pairs", nullptr));
    else hb::die("unknown mode " + mode);
    run.complete();
    return 0;
}
