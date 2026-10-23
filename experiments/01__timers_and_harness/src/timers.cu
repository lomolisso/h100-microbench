/*
 * timers.cu — experiment 01: cost and resolution of the GPU timers, the dependent-chain loop overhead,
 * the clock domains (profiled by ncu) and the cudaEvent / launch floor.
 *
 * Usage: bin/timers {k1|k2|k2all|k3|k4|k5|clockcheck} [--label L]. Each invocation is one run (run_id), registered
 * in results/01__timers_and_harness__runs.csv. Every recorded launch is preceded by 2 warm-up launches of
 * the same configuration and followed by a clock sample (NVML) written to the clock log; the sample's SM/
 * memory clock fill the mandatory columns of that launch's rows.
 *
 * K1  timer read cost — k1_timer_cost<<<1, 32>>>, 10 launches. Four sections of 32 groups × 32 reads (1024 reads).
 *     Each group reads the timer 32 times back to back into registers (contiguous CS2R in SASS) and is then
 *     folded into a sink; without the sink ptxas deletes reads whose value is unused.
 *       clock64 / clock32: cost = Σ(last − first within a group) / (32 groups × 31 intervals);
 *       globaltimer / timers_pair: each group bracketed by two %clock64 reads,
 *                                  cost = (mean bracket − one %clock64 read) / 32.
 *     → results/01__timers__timer_cost.csv: timer, reads, cycles_per_read, total_cycles (the Σ above), span_ns
 *       (last − first %globaltimer value of the globaltimer section), smid, lane — one row per lane, timer, launch.
 *
 * K2a %globaltimer resolution — k2_resolution, grid = all SMs × 32 threads, 200 KB dynamic smem so that
 *     exactly one CTA runs per SM (host checks all %smid distinct; otherwise the launch is discarded and
 *     repeated). 8 SMs are chosen evenly over the sorted %smid set seen in the first warm-up. On a chosen SM,
 *     thread 0 reads %globaltimer 2^20 times and, at every change of value, stores (new value, %clock64) to
 *     global memory. All other blocks exit. The host reduces all records exactly (no sampling, no cap):
 *       delta_hist   (smid, delta_ns, delta_cycles, count): every interval between consecutive changes;
 *       window_total (smid, count = intervals, span_ns, span_cycles, sm_mhz_from_globaltimer = cycles / ns)
 *                    from the first to the last change;
 *     → results/01__timers__globaltimer_resolution.csv
 *       and, one row per 1024 consecutive intervals over the whole window (drift over time):
 *     → results/01__timers__globaltimer_trace.csv: smid, window_idx, start_ns, span_ns, span_cycles,
 *       intervals, sm_mhz_from_globaltimer.
 * K2b cross-SM skew — k2_skew, same placement (1 CTA/SM, all SMs), 8 rounds per launch. In round r the block on
 *     the r-th K2a SM is the writer: it waits until every other block is polling, waits a pseudo-random
 *     0–2047 ns (de-phases the round from the timer tick), reads %globaltimer (t_write) and stores r+1 to a
 *     global flag. Every other block polls the flag with ld.relaxed.gpu and reads %globaltimer as soon as it
 *     sees the new value (t_obs). Relaxed accesses are used on purpose: no data is published, and a release
 *     fence between t_write and the store would add to the measured time. skew_ns = t_obs − t_write
 *     = flag propagation + polling period + (timer offset of the observer − timer offset of the writer).
 *     Rotating the writer gives every pair of the 8 SMs both directions: half the difference of the two
 *     directions cancels a symmetric propagation time and leaves the timer offset between the two SMs.
 *     → results/01__timers__globaltimer_skew.csv: round, writer_smid, observer_smid, skew_ns.
 *     (The first run of this experiment used block 0 as the only writer and stored its rows in
 *     globaltimer_resolution.csv as kind = skew / skew_writer.)
 * k2all — K2a on every SM (all 132 chosen), without the trace file: per-SM cycles-per-ns over the window.
 *
 * K3  dependent-chain loop overhead — k3_loop<variant, unroll><<<1, 32>>>; only thread 0 times. A 4 KB
 *     shared array holds a cyclic pointer chain (element i holds the shared address of element i+1).
 *     The timed loop runs `iters` dependent steps p = step(p): an outer loop (#pragma unroll 1) around
 *     `unroll` inlined steps. Steps: ld_shared = `ld.shared.u32 p, [p]` (LDS chain); mov = `mov.u32 p, p`
 *     (ptxas deletes the moves and then the empty loop: the variant measures the bare timed region);
 *     add = `mad.lo.u32 p, p, 1, 0` with runtime operands, one dependent IMAD per step (a plain add.u32
 *     chain is fused pairwise by ptxas). t0 = %clock64 before the loop; t1 = read_clock64_after(p), which
 *     cannot issue before the last step's result is available. cycles_per_iter = (t1 − t0) / iters.
 *     Sweep: iters ∈ {256, 1024, 4096, 16384} × unroll ∈ {1, 4, 16} × 3 variants, 10 launches each.
 *     → results/01__timers__loop_overhead.csv: variant, iters, unroll, cycles_per_iter, total_cycles, smid.
 *
 * K4  clock domains — the kernels that src/run.sh profiles with ncu (--clock-control none, then base):
 *     2 warm-ups, then 5 × k3_loop<ld_shared, 16>(16384) and 5 × k4_spin (all SMs × 128 threads, ~10 ms of
 *     %clock64 spinning). Writes only clock-log rows; src/ncu_clock_domains.py turns ncu's output into
 *     results/01__timers__clock_domains.csv.
 *
 * K5  cudaEvent floor — empty kernel k5_empty<<<1, 32>>> on the default stream:
 *       single:     1000 launches, each between its own event pair (event-measured time per launch);
 *       batch:      10 × (1000 back-to-back launches between one event pair) / 1000; host_api_us_per_launch
 *                   = host time to enqueue the 1000 launches / 1000;
 *       empty_pair: 1000 event pairs with nothing in between (resolution floor of the event timer).
 *     → results/01__timers__event_floor.csv: mode, launches, us_per_launch, host_api_us_per_launch.
 *
 * clockcheck — writes one clock-log row and the nvidia-smi clock report (start/end of the run).
 */
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "harness.cuh"
#include "host.hpp"

static const char* EXP = "01__timers_and_harness";
static constexpr int kRecorded = 10;  // recorded launches per configuration (after 2 warm-ups)

// ================================================================ K1
// ptxas deletes timer reads whose result is unused, so reads are taken in groups of K1_GROUP into registers
// and folded into a sink after each group. Within a group, the reads are back to back in SASS.
constexpr int K1_GROUP = 32, K1_GROUPS = 32, K1_READS = K1_GROUP * K1_GROUPS;
struct K1Out {
    uint64_t c64_sum;    // Σ over groups of (last − first) %clock64 within the group (K1_GROUP − 1 intervals each)
    uint64_t c32_sum;    // same for 32-bit %clock
    uint64_t gt_sum;     // Σ over groups of the %clock64 bracket around K1_GROUP %globaltimer reads
    uint64_t pair_sum;   // Σ over groups of the %clock64 bracket around K1_GROUP read_timers_pair() calls
    uint64_t gt_first, gt_last;  // first and last %globaltimer value of the globaltimer section
    uint64_t sink;
    uint32_t smid, pad;
};

__global__ void k1_timer_cost(K1Out* out) {
    K1Out o{};
    uint64_t sink = 0;
#pragma unroll 1
    for (int g = 0; g < K1_GROUPS; ++g) {
        uint64_t v[K1_GROUP];
#pragma unroll
        for (int i = 0; i < K1_GROUP; ++i) v[i] = read_clock64();
        o.c64_sum += v[K1_GROUP - 1] - v[0];
#pragma unroll
        for (int i = 0; i < K1_GROUP; ++i) sink ^= v[i];
    }
#pragma unroll 1
    for (int g = 0; g < K1_GROUPS; ++g) {
        uint32_t v[K1_GROUP];
#pragma unroll
        for (int i = 0; i < K1_GROUP; ++i) v[i] = read_clock32();
        o.c32_sum += uint32_t(v[K1_GROUP - 1] - v[0]);
#pragma unroll
        for (int i = 0; i < K1_GROUP; ++i) sink ^= v[i];
    }
#pragma unroll 1
    for (int g = 0; g < K1_GROUPS; ++g) {
        uint64_t v[K1_GROUP];
        const uint64_t t0 = read_clock64();
#pragma unroll
        for (int i = 0; i < K1_GROUP; ++i) v[i] = read_globaltimer();
        const uint64_t t1 = read_clock64();
        o.gt_sum += t1 - t0;
        if (g == 0) o.gt_first = v[0];
        o.gt_last = v[K1_GROUP - 1];
#pragma unroll
        for (int i = 0; i < K1_GROUP; ++i) sink ^= v[i];
    }
#pragma unroll 1
    for (int g = 0; g < K1_GROUPS; ++g) {
        TimerPair v[K1_GROUP];
        const uint64_t t0 = read_clock64();
#pragma unroll
        for (int i = 0; i < K1_GROUP; ++i) v[i] = read_timers_pair();
        const uint64_t t1 = read_clock64();
        o.pair_sum += t1 - t0;
#pragma unroll
        for (int i = 0; i < K1_GROUP; ++i) sink ^= v[i].cycles ^ v[i].ns;
    }
    o.sink = sink;
    o.smid = read_smid();
    out[threadIdx.x] = o;
}

static void run_k1(hb::Run& run) {
    hb::CsvWriter csv(hb::Run::results("01__timers__timer_cost.csv"),
                      hb::with_mandatory({"timer", "reads", "cycles_per_read", "total_cycles", "span_ns", "smid", "lane"}));
    K1Out* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, 32 * sizeof(K1Out)));
    std::vector<K1Out> h(32);
    auto launch = [&] { k1_timer_cost<<<1, 32>>>(d_out); };
    run.clock_note("k1", "before");
    hb::warmup(launch);
    for (int l = 0; l < kRecorded; ++l) {
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        const hb::ClockSample s = run.clock_note("k1", "launch", l);
        CUDA_CHECK(cudaMemcpy(h.data(), d_out, 32 * sizeof(K1Out), cudaMemcpyDeviceToHost));
        const std::string pre = run.prefix(l, s);
        for (int lane = 0; lane < 32; ++lane) {
            const K1Out& o = h[lane];
            // per-read costs: intervals inside groups for the cycle counters; bracket minus one %clock64 read
            // (the closing bracket read) per group for %globaltimer and the timer pair
            const double c64 = double(o.c64_sum) / (K1_GROUPS * (K1_GROUP - 1));
            const double c32 = double(o.c32_sum) / (K1_GROUPS * (K1_GROUP - 1));
            const double gt = (double(o.gt_sum) / K1_GROUPS - c64) / K1_GROUP;
            const double pr = (double(o.pair_sum) / K1_GROUPS - c64) / K1_GROUP;
            char b[1024];
            std::snprintf(b, sizeof b,
                          "%s,clock64,%d,%.4f,%llu,,%u,%d\n%s,clock32,%d,%.4f,%llu,,%u,%d\n"
                          "%s,globaltimer,%d,%.4f,%llu,%llu,%u,%d\n%s,timers_pair,%d,%.4f,%llu,,%u,%d",
                          pre.c_str(), K1_READS, c64, (unsigned long long)o.c64_sum, o.smid, lane,
                          pre.c_str(), K1_READS, c32, (unsigned long long)o.c32_sum, o.smid, lane,
                          pre.c_str(), K1_READS, gt, (unsigned long long)o.gt_sum,
                          (unsigned long long)(o.gt_last - o.gt_first), o.smid, lane,
                          pre.c_str(), K1_READS, pr, (unsigned long long)o.pair_sum, o.smid, lane);
            csv.add(b);
            if (l == 0 && lane == 0)
                std::printf("k1 lane0: clock64 %.3f  clock32 %.3f  globaltimer %.3f  pair %.3f cycles/read\n", c64, c32, gt, pr);
        }
        csv.flush();
    }
    csv.sync();
    run.clock_note("k1", "after");
    CUDA_CHECK(cudaFree(d_out));
}

// ================================================================ K2
constexpr uint32_t K2_READS = 1u << 20;
constexpr int K2_NCHOSEN = 8;            // SMs measured by K2a (and writers of K2b)
constexpr int K2_ROUNDS = K2_NCHOSEN;    // K2b: one round per writer SM
constexpr size_t K2_SMEM = 200 * 1024;   // forces one CTA per SM
constexpr uint32_t K2_TRACE_WINDOW = 1024;
struct GtRec { uint64_t ns, cyc; };

__global__ void k2_resolution(const uint32_t* chosen, int nchosen, uint32_t reads, GtRec* buf, uint32_t* nrec,
                              uint32_t* block_smid) {
    const uint32_t smid = read_smid();
    if (threadIdx.x != 0) return;
    block_smid[blockIdx.x] = smid;
    int slot = -1;
    for (int i = 0; i < nchosen; ++i) if (chosen[i] == smid) slot = i;
    if (slot < 0) return;
    GtRec* b = buf + (size_t)slot * (reads + 1);
    uint64_t prev = read_globaltimer();
    uint32_t n = 0;
    b[n++] = GtRec{prev, read_clock64()};  // starting value (mid-tick), not a change
    for (uint32_t i = 0; i < reads; ++i) {
        const uint64_t g = read_globaltimer();
        if (g != prev) {
            b[n++] = GtRec{g, read_clock64()};
            prev = g;
        }
    }
    nrec[slot] = n;
}

// Round r: the block on SM writers[r] writes the flag; all other blocks observe it.
__global__ void k2_skew(const uint32_t* writers, uint32_t* flag, uint32_t* arrived, uint64_t* t_obs, uint64_t* t_write,
                        uint32_t* block_smid, uint32_t* timeout, uint32_t seed) {
    if (threadIdx.x != 0) return;
    const uint32_t smid = read_smid();
    block_smid[blockIdx.x] = smid;
    const uint32_t G = gridDim.x;
    const uint64_t start = read_globaltimer();
    uint32_t rng = seed * 2654435761u + 12345u;
    for (uint32_t r = 0; r < K2_ROUNDS; ++r) {
        if (smid == writers[r]) {
            uint32_t it = 0;
            while (ld_relaxed_gpu_u32(arrived) < (r + 1) * (G - 1)) {  // every observer is polling round r
                if ((++it & 1023u) == 0 && read_globaltimer() - start > 2000000000ull) { atomicExch(timeout, 1u); return; }
            }
            rng = rng * 1664525u + 1013904223u;
            const uint64_t delay = (rng >> 8) & 2047u;
            const uint64_t d0 = read_globaltimer();
            while (read_globaltimer() - d0 < delay) { }
            const uint64_t tw = read_globaltimer();
            st_relaxed_gpu_u32(flag, r + 1);
            t_write[r] = tw;
        } else {
            atomicAdd(arrived, 1u);  // "polling round r from now on"
            uint32_t it = 0;
            while (ld_relaxed_gpu_u32(flag) < r + 1) {
                if ((++it & 1023u) == 0 && read_globaltimer() - start > 2000000000ull) { atomicExch(timeout, 1u); return; }
            }
            t_obs[(size_t)r * G + blockIdx.x] = read_globaltimer();
        }
    }
}

static bool placement_ok(const std::vector<uint32_t>& smid, int sm_count) {
    return (int)smid.size() == sm_count && hb::all_distinct(smid);
}

// K2a. all_sms: measure every SM and skip the trace file. Returns the measured SMs.
static std::vector<uint32_t> run_k2a(hb::Run& run, const hb::ClusterLaunch& cfg, bool all_sms) {
    const int G = run.dev().sm_count;
    const int nch = all_sms ? G : K2_NCHOSEN;
    const std::string label = all_sms ? "k2_resolution_all" : "k2_resolution";
    hb::CsvWriter res(hb::Run::results("01__timers__globaltimer_resolution.csv"),
                      hb::with_mandatory({"kind", "smid", "round", "delta_ns", "delta_cycles", "count", "span_ns",
                                          "span_cycles", "sm_mhz_from_globaltimer", "skew_ns"}));
    std::unique_ptr<hb::CsvWriter> trace;
    if (!all_sms)
        trace = std::make_unique<hb::CsvWriter>(hb::Run::results("01__timers__globaltimer_trace.csv"),
                                                hb::with_mandatory({"smid", "window_idx", "start_ns", "span_ns", "span_cycles",
                                                                    "intervals", "sm_mhz_from_globaltimer"}));
    uint32_t *d_chosen, *d_nrec, *d_smid;
    GtRec* d_buf;
    CUDA_CHECK(cudaMalloc(&d_chosen, nch * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_nrec, nch * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_smid, G * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_buf, (size_t)nch * (K2_READS + 1) * sizeof(GtRec)));
    std::vector<uint32_t> chosen(nch, 0xffffffffu), nrec(nch), smid(G);
    std::vector<GtRec> h_buf((size_t)nch * (K2_READS + 1));
    CUDA_CHECK(cudaMemcpy(d_chosen, chosen.data(), nch * sizeof(uint32_t), cudaMemcpyHostToDevice));

    auto launch = [&] {
        CUDA_CHECK(cudaMemset(d_nrec, 0, nch * sizeof(uint32_t)));
        CUDA_CHECK(cfg.launch(k2_resolution, (const uint32_t*)d_chosen, nch, K2_READS, d_buf, d_nrec, d_smid));
    };
    run.clock_note(label, "before");
    hb::warmup(launch);
    CUDA_CHECK(cudaMemcpy(smid.data(), d_smid, G * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    if (!placement_ok(smid, G)) hb::die("k2: warm-up placement is not one CTA per SM");
    std::vector<uint32_t> sorted = smid;
    std::sort(sorted.begin(), sorted.end());
    for (int k = 0; k < nch; ++k)
        chosen[k] = all_sms ? sorted[k] : sorted[(size_t)((k * (G - 1) + (nch - 1) / 2) / (nch - 1))];
    if (!all_sms) {
        std::printf("k2 chosen smids:");
        for (uint32_t c : chosen) std::printf(" %u", c);
        std::printf("  (smid range %u..%u over %d SMs)\n", sorted.front(), sorted.back(), G);
    }
    CUDA_CHECK(cudaMemcpy(d_chosen, chosen.data(), nch * sizeof(uint32_t), cudaMemcpyHostToDevice));
    hb::warmup(launch, 1);  // one more warm-up with the final SM choice

    for (int l = 0, discarded = 0; l < kRecorded;) {
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        const hb::ClockSample s = run.clock_note(label, "launch", l);
        CUDA_CHECK(cudaMemcpy(smid.data(), d_smid, G * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(nrec.data(), d_nrec, nch * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        bool ok = placement_ok(smid, G);
        for (uint32_t n : nrec) ok = ok && n >= 3;
        if (!ok) {
            run.clock_note(label, "discard_placement", l);
            if (++discarded > 5) hb::die("k2: placement failed repeatedly");
            continue;
        }
        CUDA_CHECK(cudaMemcpy(h_buf.data(), d_buf, h_buf.size() * sizeof(GtRec), cudaMemcpyDeviceToHost));
        const std::string pre = run.prefix(l, s);
        double mhz_min = 1e9, mhz_max = 0;
        for (int k = 0; k < nch; ++k) {
            const GtRec* r = h_buf.data() + (size_t)k * (K2_READS + 1);
            const uint32_t n = nrec[k];  // r[0] = start value, r[1..n-1] = changes
            std::map<std::pair<uint64_t, uint64_t>, uint64_t> hist;
            for (uint32_t i = 2; i < n; ++i) ++hist[{r[i].ns - r[i - 1].ns, r[i].cyc - r[i - 1].cyc}];
            char b[512];
            for (const auto& [key, cnt] : hist) {
                std::snprintf(b, sizeof b, "%s,delta_hist,%u,,%llu,%llu,%llu,,,,", pre.c_str(), chosen[k],
                              (unsigned long long)key.first, (unsigned long long)key.second, (unsigned long long)cnt);
                res.add(b);
            }
            const uint64_t span_ns = r[n - 1].ns - r[1].ns, span_cyc = r[n - 1].cyc - r[1].cyc;
            const double mhz = 1e3 * double(span_cyc) / double(span_ns);
            mhz_min = std::min(mhz_min, mhz);
            mhz_max = std::max(mhz_max, mhz);
            std::snprintf(b, sizeof b, "%s,window_total,%u,,,,%u,%llu,%llu,%.4f,", pre.c_str(), chosen[k], n - 2,
                          (unsigned long long)span_ns, (unsigned long long)span_cyc, mhz);
            res.add(b);
            for (uint32_t w = 0, i0 = 1; trace && i0 < n - 1; ++w, i0 += K2_TRACE_WINDOW) {
                const uint32_t i1 = std::min(i0 + K2_TRACE_WINDOW, n - 1);
                const uint64_t sn = r[i1].ns - r[i0].ns, sc = r[i1].cyc - r[i0].cyc;
                std::snprintf(b, sizeof b, "%s,%u,%u,%llu,%llu,%llu,%u,%.4f", pre.c_str(), chosen[k], w,
                              (unsigned long long)(r[i0].ns - r[1].ns), (unsigned long long)sn, (unsigned long long)sc,
                              i1 - i0, 1e3 * double(sc) / double(sn));
                trace->add(b);
            }
            if (l == 0 && !all_sms)
                std::printf("k2 smid %3u: %u changes, %zu distinct (delta_ns, delta_cycles), %.2f MHz over %.3f ms\n",
                            chosen[k], n - 1, hist.size(), mhz, span_ns * 1e-6);
        }
        std::printf("%s launch %d: cycles/ns over the window %.2f .. %.2f MHz across %d SMs\n", label.c_str(), l, mhz_min,
                    mhz_max, nch);
        res.flush();
        if (trace) trace->flush();
        ++l;
    }
    res.sync();
    if (trace) trace->sync();
    run.clock_note(label, "after");
    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFree(d_chosen));
    CUDA_CHECK(cudaFree(d_nrec));
    CUDA_CHECK(cudaFree(d_smid));
    return chosen;
}

// K2b, writers = the K2a SMs.
static void run_k2b(hb::Run& run, const hb::ClusterLaunch& cfg, const std::vector<uint32_t>& writers) {
    const int G = run.dev().sm_count;
    hb::CsvWriter csv(hb::Run::results("01__timers__globaltimer_skew.csv"),
                      hb::with_mandatory({"round", "writer_smid", "observer_smid", "skew_ns"}));
    uint32_t *d_writers, *d_flag, *d_arrived, *d_timeout, *d_smid;
    uint64_t *d_obs, *d_write;
    CUDA_CHECK(cudaMalloc(&d_writers, K2_ROUNDS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_flag, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_arrived, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_timeout, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_smid, G * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_obs, (size_t)K2_ROUNDS * G * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_write, K2_ROUNDS * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_writers, writers.data(), K2_ROUNDS * sizeof(uint32_t), cudaMemcpyHostToDevice));
    std::vector<uint64_t> obs((size_t)K2_ROUNDS * G), wr(K2_ROUNDS);
    std::vector<uint32_t> smid(G);
    uint32_t seed = 1;
    auto launch = [&] {
        CUDA_CHECK(cudaMemset(d_flag, 0, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_arrived, 0, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_timeout, 0, sizeof(uint32_t)));
        CUDA_CHECK(cfg.launch(k2_skew, (const uint32_t*)d_writers, d_flag, d_arrived, d_obs, d_write, d_smid, d_timeout, seed++));
    };
    run.clock_note("k2_skew", "before");
    hb::warmup(launch);
    for (int l = 0, discarded = 0; l < kRecorded;) {
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        const hb::ClockSample s = run.clock_note("k2_skew", "launch", l);
        uint32_t timeout = 0;
        CUDA_CHECK(cudaMemcpy(&timeout, d_timeout, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(smid.data(), d_smid, G * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if (timeout || !placement_ok(smid, G)) {
            run.clock_note("k2_skew", timeout ? "discard_timeout" : "discard_placement", l);
            if (++discarded > 5) hb::die("k2_skew: launch failed repeatedly");
            continue;
        }
        CUDA_CHECK(cudaMemcpy(obs.data(), d_obs, obs.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(wr.data(), d_write, wr.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        const std::string pre = run.prefix(l, s);
        char b[512];
        for (int r = 0; r < K2_ROUNDS; ++r)
            for (int g = 0; g < G; ++g) {
                if (smid[g] == writers[r]) continue;
                std::snprintf(b, sizeof b, "%s,%d,%u,%u,%lld", pre.c_str(), r, writers[r], smid[g],
                              (long long)obs[(size_t)r * G + g] - (long long)wr[r]);
                csv.add(b);
            }
        csv.flush();
        ++l;
    }
    csv.sync();
    run.clock_note("k2_skew", "after");
    for (void* p : {(void*)d_writers, (void*)d_flag, (void*)d_arrived, (void*)d_timeout, (void*)d_smid, (void*)d_obs, (void*)d_write})
        CUDA_CHECK(cudaFree(p));
}

static void run_k2(hb::Run& run, bool all_sms) {
    hb::ClusterLaunch cfg;
    cfg.grid = dim3(run.dev().sm_count);
    cfg.block = dim3(32);
    cfg.smem = K2_SMEM;
    const std::vector<uint32_t> chosen = run_k2a(run, cfg, all_sms);
    if (!all_sms) run_k2b(run, cfg, chosen);
}

// ================================================================ K3
enum Variant { V_MOV = 0, V_ADD = 1, V_LDS = 2 };
static const char* kVariantName[] = {"mov", "add", "ld_shared"};

// add: mad.lo p, p, 1, 0 with runtime operands. A plain `add.u32 p, p, k` chain is rewritten by ptxas
// (pairs of steps fused into one IMAD p + 2k); mad.lo stays exactly one dependent IMAD per step.
template <int V>
__device__ __forceinline__ void step(uint32_t& p, uint32_t kzero, uint32_t kone) {
    if constexpr (V == V_MOV) asm volatile("mov.u32 %0, %1;" : "=r"(p) : "r"(p) : "memory");
    else if constexpr (V == V_ADD) asm volatile("mad.lo.u32 %0, %1, %2, %3;" : "=r"(p) : "r"(p), "r"(kone), "r"(kzero) : "memory");
    else asm volatile("ld.shared.u32 %0, [%1];" : "=r"(p) : "r"(p) : "memory");
}

template <int V, int U>
__global__ void k3_loop(int iters, uint32_t kzero, uint32_t kone, uint64_t zero64, uint64_t* out_cycles, uint32_t* out_sink, uint32_t* out_smid) {
    __shared__ uint32_t chain[1024];
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) chain[i] = smem_u32(&chain[(i + 1) & 1023]);
    __syncthreads();
    if (threadIdx.x != 0) return;
    uint32_t p = smem_u32(&chain[0]);
    const int outer = iters / U;
    const uint64_t t0 = read_clock64();
#pragma unroll 1
    for (int i = 0; i < outer; ++i) {
#pragma unroll
        for (int u = 0; u < U; ++u) step<V>(p, kzero, kone);
    }
    const uint64_t t1 = read_clock64_after(p, zero64);
    out_cycles[0] = t1 - t0;
    out_sink[0] = p;
    out_smid[0] = read_smid();
}

struct K3Bufs { uint64_t* cyc; uint32_t* sink; uint32_t* smid; };

template <int V, int U>
static void run_k3_config(hb::Run& run, hb::CsvWriter& csv, const K3Bufs& d, int iters) {
    const std::string label = std::string("k3_") + kVariantName[V] + "_u" + std::to_string(U) + "_i" + std::to_string(iters);
    auto launch = [&] { k3_loop<V, U><<<1, 32>>>(iters, 0u, 1u, 0ull, d.cyc, d.sink, d.smid); };
    hb::warmup(launch);
    for (int l = 0; l < kRecorded; ++l) {
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        const hb::ClockSample s = run.clock_note(label, "launch", l);
        uint64_t cyc;
        uint32_t smid;
        CUDA_CHECK(cudaMemcpy(&cyc, d.cyc, sizeof cyc, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&smid, d.smid, sizeof smid, cudaMemcpyDeviceToHost));
        char b[512];
        std::snprintf(b, sizeof b, "%s,%s,%d,%d,%.4f,%llu,%u", run.prefix(l, s).c_str(), kVariantName[V], iters, U,
                      double(cyc) / iters, (unsigned long long)cyc, smid);
        csv.add(b);
        csv.flush();
        if (l == kRecorded - 1) std::printf("k3 %-9s unroll %2d iters %5d: %.3f cycles/iter\n", kVariantName[V], U, iters, double(cyc) / iters);
    }
    csv.sync();
}

template <int V>
static void run_k3_variant(hb::Run& run, hb::CsvWriter& csv, const K3Bufs& d) {
    for (int iters : {256, 1024, 4096, 16384}) {
        run_k3_config<V, 1>(run, csv, d, iters);
        run_k3_config<V, 4>(run, csv, d, iters);
        run_k3_config<V, 16>(run, csv, d, iters);
    }
}

static K3Bufs k3_alloc() {
    K3Bufs d;
    CUDA_CHECK(cudaMalloc(&d.cyc, sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d.sink, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d.smid, sizeof(uint32_t)));
    return d;
}

static void run_k3(hb::Run& run) {
    hb::CsvWriter csv(hb::Run::results("01__timers__loop_overhead.csv"),
                      hb::with_mandatory({"variant", "iters", "unroll", "cycles_per_iter", "total_cycles", "smid"}));
    const K3Bufs d = k3_alloc();
    run.clock_note("k3", "before");
    run_k3_variant<V_MOV>(run, csv, d);
    run_k3_variant<V_ADD>(run, csv, d);
    run_k3_variant<V_LDS>(run, csv, d);
    run.clock_note("k3", "after");
}

// ================================================================ K4 (profiled by ncu)
__global__ void k4_spin(uint64_t cycles, uint32_t* sink) {
    const uint64_t t0 = read_clock64();
    uint32_t n = 0;
    while (read_clock64() - t0 < cycles) ++n;
    if (n == 0xffffffffu) sink[0] = n;
}

static void run_k4(hb::Run& run, const std::string& label) {
    const K3Bufs d = k3_alloc();
    const uint64_t spin_cycles = 10ull * run.dev().prop.clockRate;  // clockRate is in kHz: ~10 ms at the max clock
    auto k3 = [&] { k3_loop<V_LDS, 16><<<1, 32>>>(16384, 0u, 1u, 0ull, d.cyc, d.sink, d.smid); };
    auto spin = [&] { k4_spin<<<run.dev().sm_count, 128>>>(spin_cycles, d.sink); };
    run.lock_clocks_note(label, "before");
    hb::warmup(k3, 1);
    hb::warmup(spin, 1);
    for (int l = 0; l < 5; ++l) hb::warmup(k3, 1), run.clock_note(label + "_k3", "launch", l);
    for (int l = 0; l < 5; ++l) hb::warmup(spin, 1), run.clock_note(label + "_spin", "launch", l);
    run.lock_clocks_note(label, "after");
}

// ================================================================ K5
__global__ void k5_empty() {}

static void run_k5(hb::Run& run) {
    hb::CsvWriter csv(hb::Run::results("01__timers__event_floor.csv"),
                      hb::with_mandatory({"mode", "launches", "us_per_launch", "host_api_us_per_launch"}));
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    auto launch = [] { k5_empty<<<1, 32>>>(); };
    hb::warmup(launch, 12);
    char b[512];

    // single: one event pair around each launch
    run.clock_note("k5_single", "before");
    std::vector<float> ms(1000);
    for (int i = 0; i < 1000; ++i) {
        CUDA_CHECK(cudaEventRecord(e0));
        launch();
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaEventElapsedTime(&ms[i], e0, e1));
    }
    hb::ClockSample s = run.clock_note("k5_single", "after");
    for (int i = 0; i < 1000; ++i) {
        std::snprintf(b, sizeof b, "%s,single,1,%.4f,", run.prefix(i, s).c_str(), ms[i] * 1e3);
        csv.add(b);
    }
    csv.sync();

    // batch: 1000 back-to-back launches between one event pair
    run.clock_note("k5_batch", "before");
    for (int r = 0; r < 10; ++r) {
        CUDA_CHECK(cudaEventRecord(e0));
        const auto h0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 1000; ++i) launch();
        const auto h1 = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float t;
        CUDA_CHECK(cudaEventElapsedTime(&t, e0, e1));
        s = run.clock_note("k5_batch", "launch", r);
        const double host_us = std::chrono::duration<double, std::micro>(h1 - h0).count();
        std::snprintf(b, sizeof b, "%s,batch,1000,%.4f,%.4f", run.prefix(r, s).c_str(), t * 1e3 / 1000, host_us / 1000);
        csv.add(b);
        csv.flush();
        std::printf("k5 batch %d: %.3f us/launch (event), %.3f us/launch (host enqueue)\n", r, t * 1e3 / 1000, host_us / 1000);
    }
    csv.sync();

    // empty_pair: event pair with nothing in between
    run.clock_note("k5_empty_pair", "before");
    for (int i = 0; i < 1000; ++i) {
        CUDA_CHECK(cudaEventRecord(e0));
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaEventElapsedTime(&ms[i], e0, e1));
    }
    s = run.clock_note("k5_empty_pair", "after");
    for (int i = 0; i < 1000; ++i) {
        std::snprintf(b, sizeof b, "%s,empty_pair,0,%.4f,", run.prefix(i, s).c_str(), ms[i] * 1e3);
        csv.add(b);
    }
    csv.sync();
    std::sort(ms.begin(), ms.end());
    std::printf("k5 empty_pair median %.3f us\n", ms[500] * 1e3);
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
}

// ================================================================
int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s {k1|k2|k2all|k3|k4|k5|clockcheck} [--label L]\n", argv[0]);
        return 1;
    }
    const std::string cmd = argv[1];
    const std::string label = hb::arg_str(argc, argv, "--label", cmd.c_str());
    hb::Run run(EXP, argc, argv);
    if (cmd == "k1") run_k1(run);
    else if (cmd == "k2") run_k2(run, false);
    else if (cmd == "k2all") run_k2(run, true);
    else if (cmd == "k3") run_k3(run);
    else if (cmd == "k4") run_k4(run, label);
    else if (cmd == "k5") run_k5(run);
    else if (cmd == "clockcheck") run.lock_clocks_note(label, "check");
    else hb::die("unknown subcommand " + cmd);
    run.complete();
    return 0;
}
