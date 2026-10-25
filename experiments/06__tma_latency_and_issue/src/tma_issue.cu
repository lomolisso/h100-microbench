/*
 * tma_issue.cu — experiment 06, K3: issue gap of back-to-back bulk copies, the number of copies the TMA unit admits
 * before issue stalls, and the completion time of N copies in flight, for 1/2/4 issuing warps, 1/2/4/8 SMs and
 * sources in L2 or HBM.
 *
 * Usage: bin/tma_issue {k3|k3sms|ncu|clockcheck} [--quick] [--label L]. One invocation = one run_id.
 *
 * k3_issue<W, SIZE, N>(src, slots, base_slot, reps, hbm, ctr, start_ns, out): grid = `sms` blocks of W warps, 200 KiB
 *   dynamic smem (1 CTA/SM; %smid of every block checked distinct), destination = copy i → smem i·SIZE (≤ 128 KiB),
 *   one mbarrier.
 *   Sample (rep r):
 *     thread 0: arrive.expect_tx(bar, n·SIZE); all blocks start the sample at the same %globaltimer instant
 *     (start_ns + r·20 µs; start_ns is published by the last block to arrive on a global counter); bar.sync;
 *     lane 0 of warp w: t0_w = clock64, then for every copy i ≡ w (mod W), i < n, in order:
 *       cp.async.bulk [dst + i·SIZE], [src + slot·64·SIZE + i·SIZE], SIZE, [bar]; t_i = clock64
 *       (N is a template parameter and each warp runs its own instantiation issue_warp<…, WARP>: fully unrolled,
 *       compile-time copy offsets, uniform datapath, no guards; one CS2R of ≈ 2 cycles after each issue);
 *     lane 0 of warp 0 then waits (try_wait.parity loop) for the phase: t_done = clock64; bar.sync.
 *   Source: L2 = a 4 MiB buffer cut into slots of 64·SIZE bytes, warmed by each block at kernel start with untimed bulk
 *   copies of every slot; sample r reads slot r mod (number of slots). HBM = a 1 GiB buffer, L2 cleaned before every launch (05's
 *   CleanL2: flush + clean read), sample r of block b reads slot base_slot + r·sms + b (fresh lines; base_slot differs
 *   per launch).
 *   Host per (sample, block): issue_span_cycles = max_w(last t_i) − min_w(t0_w); completion_cycles = t_done − min_w(t0_w);
 *   issue_cycles_list = t_i − min_w(t0_w) for i = 0..n−1 (';'-joined); warp_t0_list = t0_w − min t0;
 *   start_skew_ns = %globaltimer at the sample start minus the earliest block's (multi-SM overlap check).
 * Sweep: n = 1..32 for each of
 *   (issuers 1/2/4, sms 1, SIZE 4 KiB), (issuers 1, sms 1, SIZE 1 KiB), (issuers 1, sms 2/4/8, SIZE 4 KiB),
 *   and n = 1..64 for (issuers 4, sms 1, SIZE 2 KiB) (the fastest issue rate, to look for an admission limit beyond
 *   32 copies), each with source L2 and HBM. 2 unrecorded warm-up launches, then 5 recorded launches × 16 samples.
 *   --quick: n ∈ {1, 8, 16, 17, 24, 32} (64-config: {1, 16, 17, 32, 33, 48, 64}), 2 recorded launches.
 *   → results/06__tma__issue.csv: n_in_flight, issuers, sms, source, copy_bytes, rep, block, smid, issue_span_cycles,
 *     completion_cycles, start_skew_ns, parity (mbarrier phase parity waited on), issue_cycles_list, warp_t0_list
 *     (one row per sample and block).
 *
 * k3sms_issue<SIZE>(src, set, base_region, reps, hbm, ctr, start_ns, out, out_ti): TMA issue and completion under a
 *   sustained load on chosen SMs, to test whether the TMA units of different SMs (TPC mates, other GPCs, other L2
 *   partition) interfere. Grid = sm_count blocks × 32 threads × 200 KiB dynamic smem, so every SM hosts exactly one
 *   block; thread 0 of a block whose %smid is in `set` works, all other blocks and threads return at once.
 *   SMs (kSmsChosen, checked at start against the latest 03 map: TPC mates, one pair per GPC, two pairs per L2
 *   partition): 2,3 (G1) | 0,1 (G0) | 6,7 (G3) | 8,9 (G4); partition 1 = {2,3,6,7}, partition 0 = {0,1,8,9}.
 *   Active subsets: 1 = {2}; 2 = {2,3} (TPC mates); 4 = {2,3,0,1} (both partitions); 8 = all.
 *   Sample = 1 MiB per active SM: 8 rounds of K = 128 KiB / SIZE copies (SIZE 4 KiB → 256 copies, 16 KiB → 64),
 *   one mbarrier per round (expect_tx 128 KiB each, armed before t0; one barrier for 1 MiB would exceed the 2^20 − 1
 *   tx-count limit); every round writes the same 128 KiB destination ring (contents never read; only the transaction
 *   bytes matter). All active SMs start the sample at the same %globaltimer instant (start_ns + r · 50 µs, common
 *   start as in K3), then: t0 = clock64; for each round j (runtime loop), for k < K (unrolled, compile-time offsets):
 *   cp.async.bulk [ring + k·SIZE], [src + j·128 KiB + k·SIZE], SIZE, [bar_j]; ti = clock64 → st.shared (one CS2R +
 *   one STS.64 per copy; otherwise the same per-copy SASS as k3_issue); then try_wait.parity loops on bar_0..bar_7 in
 *   order, tdone_j = clock64 after each. Every sample completes one phase of each barrier (parity = sample count & 1).
 *   Source: L2 = an 8 MiB buffer, one private 1 MiB region per active SM (by its index in the set), read once by an
 *   untimed warm sample in the kernel (the reads then hit a copy in the SM's own L2 partition); HBM = the 1 GiB buffer,
 *   L2 cleaned before every launch, sample r of active SM a reads region base + r · n_active + a (fresh lines).
 *   Sweep: source L2/HBM × active SMs 1/2/4/8 × SIZE 4/16 KiB; 2 unrecorded warm-up launches, then 10 recorded
 *   launches × 16 samples (--quick: 2).
 *   → results/06__tma__issue_sms.csv: active_sms, active_set, source, copy_bytes, copies, rep, block (index in the set),
 *     smid, gpc, partition (03 map), issue_span_cycles (last issue − t0), completion_cycles (tdone_7 − t0),
 *     start_skew_ns (%globaltimer at t0 minus the earliest active SM's), parity, issue_cycles_list (ti − t0 of every
 *     copy, ';'-joined), round_done_cycles_list (tdone_j − t0) — one row per sample and active SM.
 * ncu mode: k3_ncu<1, 4096> (same body) with n = 16 and 32 from L2 on one SM, 16 samples; manifest
 *   bin/ncu_manifest_issue.csv.
 */
#include <algorithm>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../../05__memory_hierarchy_bandwidth/src/bw_common.hpp"
#include "tma_common.hpp"

static const char* EXP = "06__tma_latency_and_issue";
static constexpr int kMaxN = 64, kDstSlots = 32, kReps = 16;  // kMaxN: largest batch (2 KiB config)
static constexpr uint32_t kDynSmem = 200u << 10, kL2Bytes = 4u << 20;
static constexpr uint64_t kHbmBytes = 1ull << 30, kPeriodNs = 20000;

// Number of slots (kMaxN copies of SIZE bytes each) of the 4 MiB L2 source; the L2 warm pass completes one mbarrier
// phase per slot.
template <uint32_t SIZE>
__host__ __device__ constexpr uint32_t l2_slots() { return kL2Bytes / (kMaxN * SIZE); }

// Per (sample, block, warp): t0, t_done (warp 0), globaltimer at start, t_i of this warp's copies (index i/W).
struct WarpRec { uint64_t t0, tdone, g0; uint64_t ti[kMaxN]; uint32_t smid, warpid, parity; };

// Copies i = WARP, WARP + W, … < N of one warp, in order; ti[k] = clock64 after the k-th issue.
template <int W, uint32_t SIZE, int N, int WARP>
__device__ __forceinline__ void issue_warp(uint32_t d, const uint8_t* s0, uint32_t b, uint64_t* ti) {
    constexpr int kCount = N > WARP ? (N - WARP + W - 1) / W : 0;
#pragma unroll
    for (int k = 0; k < kCount; ++k) {
        const uint32_t i = WARP + k * W;
        bulk_g2s(d + i * SIZE, s0 + (size_t)i * SIZE, SIZE, b);
        ti[k] = read_clock64();
    }
}

template <int W, uint32_t SIZE, int N>
__device__ __forceinline__ void issue_body(const uint8_t* src, uint32_t slots, uint32_t base_slot, uint32_t reps,
                                           int hbm, uint32_t* ctr, unsigned long long* start_ns, WarpRec* out) {
    extern __shared__ __align__(1024) uint8_t dyn[];
    __shared__ alignas(8) uint64_t bar;
    __shared__ unsigned long long t_start;
    const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const uint32_t d = smem_u32(dyn), b = smem_u32(&bar);
    const uint32_t slot_bytes = kMaxN * SIZE;
    // Phase-parity bookkeeping. `parity` is a per-thread register and only thread 0 runs the L2 warm pass, which
    // completes one mbarrier phase per slot and flips thread 0's parity each time; the other threads keep parity 0.
    // After the warm pass the barrier's current phase has parity (slots mod 2), and in the sample loop every thread
    // flips its own copy once per sample. The waits of warps 1..W−1 (and their recorded parity) are therefore correct
    // only if the warm pass leaves the barrier on an even phase, i.e. the number of L2 slots is even; with an odd
    // count those warps would wait on the parity of the already completed phase, return at once, and let the next
    // sample overwrite the destination while copies are still landing. HBM launches skip the warm pass (parity 0
    // everywhere). The host derives the slot count from the same expression (l2_slots<SIZE>()).
    static_assert(kL2Bytes % (kMaxN * SIZE) == 0 && l2_slots<SIZE>() % 2 == 0,
                  "the L2 warm pass must complete an even number of mbarrier phases (see parity bookkeeping above)");
    uint32_t parity = 0;
    if (threadIdx.x == 0) {
        mbar_init(b, 1);
        fence_mbarrier_init();
        if (!hbm) {  // untimed warm pass: every slot of the L2 source once, from this SM
#pragma unroll 1
            for (uint32_t s = 0; s < slots; ++s) {
                mbar_arrive_expect_tx(b, slot_bytes);
#pragma unroll 1
                for (uint32_t i = 0; i < kMaxN; ++i)
                    bulk_g2s(d + (i % kDstSlots) * SIZE, src + (size_t)s * slot_bytes + i * SIZE, SIZE, b);
                wait_loop<W_TRY>(b, parity, 0);
                parity ^= 1;
            }
        }
        // common start: the last block to arrive publishes a %globaltimer instant 5 µs ahead
        const uint32_t prev = atomicAdd(ctr, 1u);
        if (prev == gridDim.x - 1) {
            asm volatile("st.release.gpu.global.u64 [%0], %1;" :: "l"(start_ns), "l"(read_globaltimer() + 5000ull) : "memory");
        }
        unsigned long long t;
        do { asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(t) : "l"(start_ns) : "memory"); } while (t == 0);
        t_start = t;
    }
    __syncthreads();
    const uint32_t smid = read_smid();
    uint32_t warpid;
    asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpid));
#pragma unroll 1
    for (uint32_t r = 0; r < reps; ++r) {
        const uint32_t slot = hbm ? base_slot + r * gridDim.x + blockIdx.x : r % slots;
        const uint8_t* s0 = src + (size_t)slot * slot_bytes;
        if (threadIdx.x == 0) {
            mbar_arrive_expect_tx(b, N * SIZE);
            const unsigned long long go = t_start + (unsigned long long)r * kPeriodNs;
            while (read_globaltimer() < go) { }
        }
        __syncthreads();
        if (lane == 0) {
            WarpRec& o = out[((size_t)r * gridDim.x + blockIdx.x) * W + warp];
            const uint64_t g0 = read_globaltimer();
            constexpr int kPerWarp = (N + W - 1) / W;
            uint64_t ti[kPerWarp];
            const uint64_t t0 = read_clock64();
            // warp-specialised: inside each branch the copy offsets are compile-time constants (uniform datapath)
            if (warp == 0) issue_warp<W, SIZE, N, 0>(d, s0, b, ti);
            if constexpr (W > 1) { if (warp == 1) issue_warp<W, SIZE, N, 1>(d, s0, b, ti); }
            if constexpr (W > 2) {
                if (warp == 2) issue_warp<W, SIZE, N, 2>(d, s0, b, ti);
                if (warp == 3) issue_warp<W, SIZE, N, 3>(d, s0, b, ti);
            }
            uint64_t tdone = 0;
            if (warp == 0) {
                wait_loop<W_TRY>(b, parity, 0);
                tdone = read_clock64();
            }
            o.t0 = t0;
            o.tdone = tdone;
            o.g0 = g0;
#pragma unroll
            for (int k = 0; k < kPerWarp; ++k) o.ti[k] = (W == 1 || warp + (uint32_t)k * W < N) ? ti[k] : 0;
            o.smid = smid;
            o.warpid = warpid;
            o.parity = parity;
        }
        if (warp != 0) wait_loop<W_TRY>(b, parity, 0);  // every warp observes the phase before the next sample reuses dst
        parity ^= 1;
        __syncthreads();
    }
}

template <int W, uint32_t SIZE, int N>
__global__ void __launch_bounds__(128) k3_issue(const uint8_t* src, uint32_t slots, uint32_t base_slot, uint32_t reps, int hbm,
                                                 uint32_t* ctr, unsigned long long* start_ns, WarpRec* out) {
    issue_body<W, SIZE, N>(src, slots, base_slot, reps, hbm, ctr, start_ns, out);
}
template <int W, uint32_t SIZE, int N>
__global__ void __launch_bounds__(128) k3_ncu(const uint8_t* src, uint32_t slots, uint32_t base_slot, uint32_t reps, int hbm,
                                               uint32_t* ctr, unsigned long long* start_ns, WarpRec* out) {
    issue_body<W, SIZE, N>(src, slots, base_slot, reps, hbm, ctr, start_ns, out);
}
using KFn = void (*)(const uint8_t*, uint32_t, uint32_t, uint32_t, int, uint32_t*, unsigned long long*, WarpRec*);
// kernel for n = 1..NMAX (index n − 1)
// (separate tables: a conditional inside one table would instantiate both kernels for every configuration)
template <int W, uint32_t SIZE, bool kNcu, int... Is>
static KFn pick_kernel(uint32_t n, std::integer_sequence<int, Is...>) {
    if constexpr (kNcu) {
        static const KFn tab[] = {k3_ncu<W, SIZE, Is + 1>...};
        return tab[n - 1];
    } else {
        static const KFn tab[] = {k3_issue<W, SIZE, Is + 1>...};
        return tab[n - 1];
    }
}

struct Ctx {
    hb::Run& run;
    uint8_t *d_l2 = nullptr, *d_hbm = nullptr;
    uint32_t* d_ctr = nullptr;
    unsigned long long* d_start = nullptr;
    WarpRec* d_out = nullptr;
    CleanL2 clean;
    long long li = 0;
    explicit Ctx(hb::Run& r) : run(r) {
        CUDA_CHECK(cudaMalloc(&d_l2, kL2Bytes));
        CUDA_CHECK(cudaMemset(d_l2, 0x11, kL2Bytes));
        CUDA_CHECK(cudaMalloc(&d_hbm, kHbmBytes));
        CUDA_CHECK(cudaMemset(d_hbm, 0x22, kHbmBytes));
        CUDA_CHECK(cudaMalloc(&d_ctr, sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_start, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&d_out, (size_t)kReps * 8 * 4 * sizeof(WarpRec)));
    }
    ~Ctx() { cudaFree(d_l2); cudaFree(d_hbm); cudaFree(d_ctr); cudaFree(d_start); cudaFree(d_out); }
};

struct Config { int issuers, sms; uint32_t size; int hbm; };

template <int W, uint32_t SIZE, int NMAX, bool kNcu = false>
static std::vector<WarpRec> launch(Ctx& c, int sms, int hbm, uint32_t n, uint32_t base_slot) {
    if (n < 1 || n > NMAX || n * SIZE > kDstSlots * 4096u) hb::die("bad batch size " + std::to_string(n));
    const KFn k = pick_kernel<W, SIZE, kNcu>(n, std::make_integer_sequence<int, NMAX>{});
    CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kDynSmem));
    if (hbm) c.clean();
    CUDA_CHECK(cudaMemset(c.d_ctr, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(c.d_start, 0, sizeof(unsigned long long)));
    const uint32_t slot_bytes = kMaxN * SIZE;
    const uint8_t* src = hbm ? c.d_hbm : c.d_l2;
    const uint32_t slots = hbm ? (uint32_t)(kHbmBytes / slot_bytes) : l2_slots<SIZE>();
    k<<<sms, 32 * W, kDynSmem>>>(src, slots, base_slot, kReps, hbm, c.d_ctr, c.d_start, c.d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<WarpRec> h((size_t)kReps * sms * W);
    CUDA_CHECK(cudaMemcpy(h.data(), c.d_out, h.size() * sizeof(WarpRec), cudaMemcpyDeviceToHost));
    return h;
}

// HBM slot base for launch l: spread over the buffer, leaving room for kReps · sms consecutive slots.
static uint32_t hbm_base(uint32_t size, int sms, long long l) {
    const uint32_t slots = (uint32_t)(kHbmBytes / (kMaxN * size)), need = kReps * sms;
    return (uint32_t)(((uint64_t)(l + 1) * 2654435761ull) % (slots - need));
}

template <int W, uint32_t SIZE, int NMAX>
static void run_config(Ctx& c, hb::CsvWriter& csv, int sms, int hbm, uint32_t n, int launches) {
    const char* srcname = hbm ? "hbm" : "l2";
    const std::string label = "k3_n" + std::to_string(n) + "_w" + std::to_string(W) + "_s" + std::to_string(sms) + "_" + srcname +
                              "_" + std::to_string(SIZE);
    for (int w = 0; w < 2; ++w) launch<W, SIZE, NMAX>(c, sms, hbm, n, hbm_base(SIZE, sms, c.li + 100000 + w));  // warm-ups
    double span_sum = 0, comp_sum = 0;
    int cnt = 0;
    for (int l = 0; l < launches; ++l) {
        const std::vector<WarpRec> h = launch<W, SIZE, NMAX>(c, sms, hbm, n, hbm_base(SIZE, sms, c.li));
        const hb::ClockSample s = c.run.clock_note(label, "launch", c.li);
        std::set<uint32_t> smids;
        for (int b = 0; b < sms; ++b) smids.insert(h[(size_t)b * W].smid);
        if ((int)smids.size() != sms) hb::die(label + ": blocks share an SM");
        for (int r = 0; r < kReps; ++r) {
            uint64_t gmin = ~0ull;
            for (int b = 0; b < sms; ++b) gmin = std::min(gmin, h[((size_t)r * sms + b) * W].g0);
            for (int b = 0; b < sms; ++b) {
                const WarpRec* wr = &h[((size_t)r * sms + b) * W];
                uint64_t t0min = ~0ull, tlast = 0, g0 = ~0ull;
                for (int w = 0; w < W; ++w) { t0min = std::min(t0min, wr[w].t0); g0 = std::min(g0, wr[w].g0); }
                std::string ilist, wlist;
                for (uint32_t i = 0; i < n; ++i) {
                    const uint64_t t = wr[i % W].ti[i / W];
                    tlast = std::max(tlast, t);
                    if (i) ilist += ';';
                    ilist += std::to_string(t - t0min);
                }
                for (int w = 0; w < W; ++w) { if (w) wlist += ';'; wlist += std::to_string(wr[w].t0 - t0min); }
                const uint64_t span = tlast - t0min, comp = wr[0].tdone - t0min;
                span_sum += span;
                comp_sum += comp;
                ++cnt;
                std::string row = c.run.prefix(c.li, s);
                char b2[256];
                std::snprintf(b2, sizeof b2, ",%u,%d,%d,%s,%u,%d,%d,%u,%llu,%llu,%llu,%u,", n, W, sms, srcname, SIZE, r, b, wr[0].smid,
                              (unsigned long long)span, (unsigned long long)comp, (unsigned long long)(g0 - gmin), wr[0].parity);
                row += b2;
                row += ilist + "," + wlist;
                csv.add(row);
            }
        }
        csv.flush();
        ++c.li;
    }
    csv.sync();
    std::printf("k3 %-24s: issue span %8.1f  completion %8.1f cycles (mean)\n", label.c_str(), span_sum / cnt, comp_sum / cnt);
    std::fflush(stdout);
}

template <int W, uint32_t SIZE, int NMAX = 32>
static void sweep(Ctx& c, hb::CsvWriter& csv, int sms, bool quick, int launches) {
    std::vector<uint32_t> ns;
    if (quick) ns = NMAX == 32 ? std::vector<uint32_t>{1, 8, 16, 17, 24, 32} : std::vector<uint32_t>{1, 16, 17, 32, 33, 48, 64};
    else for (uint32_t n = 1; n <= (uint32_t)NMAX; ++n) ns.push_back(n);
    for (int hbm = 0; hbm <= 1; ++hbm)
        for (uint32_t n : ns) run_config<W, SIZE, NMAX>(c, csv, sms, hbm, n, launches);
}

static void run_k3(hb::Run& run, bool quick) {
    hb::CsvWriter csv(hb::Run::results("06__tma__issue.csv"),
                      hb::with_mandatory({"n_in_flight", "issuers", "sms", "source", "copy_bytes", "rep", "block", "smid",
                                          "issue_span_cycles", "completion_cycles", "start_skew_ns", "parity",
                                          "issue_cycles_list", "warp_t0_list"}));
    Ctx c(run);
    const int launches = quick ? 2 : 5;
    run.lock_clocks_note("k3", "before");
    sweep<1, 4096>(c, csv, 1, quick, launches);
    sweep<2, 4096>(c, csv, 1, quick, launches);
    sweep<4, 4096>(c, csv, 1, quick, launches);
    sweep<1, 1024>(c, csv, 1, quick, launches);
    sweep<4, 2048, 64>(c, csv, 1, quick, launches);
    for (int sms : {2, 4, 8}) sweep<1, 4096>(c, csv, sms, quick, launches);
    run.lock_clocks_note("k3", "after");
}

static void run_ncu(hb::Run& run) {
    FILE* man = std::fopen("bin/ncu_manifest_issue.csv", "w");
    if (!man) hb::die("cannot write bin/ncu_manifest_issue.csv");
    std::fprintf(man, "%s,n_in_flight,issuers,sms,source,copy_bytes,samples,mean_completion_cycles_under_ncu,smid\n",
                 hb::join(hb::mandatory_cols()).c_str());
    Ctx c(run);
    run.lock_clocks_note("ncu_issue", "before");
    for (uint32_t n : {16u, 32u}) {
        const std::vector<WarpRec> h = launch<1, 4096, 32, true>(c, 1, 0, n, 0);
        const hb::ClockSample s = run.clock_note("ncu_k3_n" + std::to_string(n), "launch", c.li);
        double sum = 0;
        for (int r = 0; r < kReps; ++r) sum += double(h[r].tdone - h[r].t0);
        std::fprintf(man, "%s,%u,1,1,l2,4096,%d,%.1f,%u\n", run.prefix(c.li, s).c_str(), n, kReps, sum / kReps, h[0].smid);
        std::fflush(man);
        ++c.li;
    }
    run.lock_clocks_note("ncu_issue", "after");
    std::fclose(man);
}

// ---------------------------------------------------------------- k3sms: TMA issue under load on 1/2/4/8 chosen SMs
static constexpr uint32_t kSmsRing = 128u << 10;            // destination ring (bytes), reused by every round
static constexpr uint32_t kSmsBytes = 1u << 20;             // bytes per SM per sample
static constexpr uint32_t kSmsRounds = kSmsBytes / kSmsRing; // rounds of one ring each; one mbarrier per round
static constexpr int kSmsMax = 8;
static constexpr uint64_t kSmsPeriodNs = 50000;             // sample period (≈ 5× the slowest expected completion)
static constexpr uint32_t kSmsL2Bytes = kSmsMax * kSmsBytes; // L2 source: one private 1 MiB region per active SM
static_assert(kSmsRounds * kSmsRing == kSmsBytes && kSmsRing * 1ull <= (1u << 20) - 1, "tx-count per barrier < 2^20");

struct SmsSet { uint32_t n; uint32_t smid[kSmsMax]; };
struct SmsRec { uint64_t t0, g0, tdone[kSmsRounds]; uint32_t smid, parity; };

// One sample of an active block: 1 MiB from `s0` as kSmsRounds rounds of K = kSmsRing/SIZE copies; round j signals
// barrier j (expect_tx(kSmsRing) armed before t0). ti[i] = clock64 after the i-th issue (stored in shared memory).
template <uint32_t SIZE>
__device__ __forceinline__ void sms_sample(uint32_t d, uint32_t b0, const uint8_t* s0, uint32_t parity, uint64_t go,
                                           uint64_t* ti, uint64_t& t0, uint64_t& g0, uint64_t* tdone) {
    constexpr uint32_t K = kSmsRing / SIZE;
#pragma unroll 1
    for (uint32_t j = 0; j < kSmsRounds; ++j) mbar_arrive_expect_tx(b0 + 8 * j, kSmsRing);
    while (read_globaltimer() < go) { }
    g0 = read_globaltimer();
    t0 = read_clock64();
#pragma unroll 1
    for (uint32_t j = 0; j < kSmsRounds; ++j) {
        const uint8_t* sj = s0 + (size_t)j * kSmsRing;
        const uint32_t bj = b0 + 8 * j;
        uint64_t* tj = ti + j * K;
#pragma unroll
        for (uint32_t k = 0; k < K; ++k) {  // compile-time offsets inside a round
            bulk_g2s(d + k * SIZE, sj + k * SIZE, SIZE, bj);
            tj[k] = read_clock64();
        }
    }
#pragma unroll
    for (uint32_t j = 0; j < kSmsRounds; ++j) {
        wait_loop<W_TRY>(b0 + 8 * j, parity, 0);
        tdone[j] = read_clock64();
    }
}

template <uint32_t SIZE>
__global__ void __launch_bounds__(32) k3sms_issue(const uint8_t* src, SmsSet set, uint32_t base_region, uint32_t reps, int hbm,
                                                  uint32_t* ctr, unsigned long long* start_ns, SmsRec* out, uint64_t* out_ti) {
    constexpr uint32_t kCopies = kSmsBytes / SIZE;
    static_assert(kSmsRing % SIZE == 0 && kSmsRing + kCopies * sizeof(uint64_t) <= kDynSmem, "ring + issue times fit");
    extern __shared__ __align__(1024) uint8_t dyn[];
    __shared__ alignas(8) uint64_t bars[kSmsRounds];
    if (threadIdx.x != 0) return;  // one issuing thread per block
    const uint32_t smid = read_smid();
    int a = -1;  // index of this SM in the active set; blocks on other SMs exit at once
    for (uint32_t j = 0; j < set.n; ++j)
        if (set.smid[j] == smid) a = (int)j;
    if (a < 0) return;
    const uint32_t d = smem_u32(dyn), b0 = smem_u32(bars);
    uint64_t* ti = reinterpret_cast<uint64_t*>(dyn + kSmsRing);
    for (uint32_t j = 0; j < kSmsRounds; ++j) mbar_init(b0 + 8 * j, 1);
    fence_mbarrier_init();
    uint32_t phase = 0;  // every sample completes one phase of every barrier: parity = phase & 1
    uint64_t t0, g0, tdone[kSmsRounds];
    if (!hbm) {  // untimed warm sample: the SM's own 1 MiB region once (leaves a copy in this SM's L2 partition)
        sms_sample<SIZE>(d, b0, src + (size_t)a * kSmsBytes, phase & 1, 0, ti, t0, g0, tdone);
        ++phase;
    }
    // common start: the last active block to arrive publishes a %globaltimer instant 5 µs ahead
    if (atomicAdd(ctr, 1u) == set.n - 1)
        asm volatile("st.release.gpu.global.u64 [%0], %1;" :: "l"(start_ns), "l"(read_globaltimer() + 5000ull) : "memory");
    unsigned long long t_start;
    do { asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(t_start) : "l"(start_ns) : "memory"); } while (t_start == 0);
#pragma unroll 1
    for (uint32_t r = 0; r < reps; ++r, ++phase) {
        const uint32_t region = hbm ? base_region + r * set.n + a : (uint32_t)a;
        sms_sample<SIZE>(d, b0, src + (size_t)region * kSmsBytes, phase & 1, t_start + (uint64_t)r * kSmsPeriodNs, ti, t0, g0,
                         tdone);
        SmsRec& o = out[(size_t)r * set.n + a];  // records written after the sample, outside the timed region
        o.t0 = t0;
        o.g0 = g0;
        for (uint32_t j = 0; j < kSmsRounds; ++j) o.tdone[j] = tdone[j];
        o.smid = smid;
        o.parity = phase & 1;
        uint64_t* oti = out_ti + ((size_t)r * set.n + a) * kCopies;
        for (uint32_t i = 0; i < kCopies; ++i) oti[i] = ti[i];
    }
}

// Map of the latest 03 run: gpc, partition, TPC mate per smid.
struct SmInfo { int gpc = -1, partition = -1, mate = -1; };
static std::vector<SmInfo> load_sm_info(int sm_count, std::string& run_out,
                                        const char* path = "../03__sm_gpc_topology/results/03__sm_gpc_topology__map.csv") {
    std::ifstream f(path);
    if (!f) hb::die(std::string("cannot open the 03 map ") + path);
    auto split = [](const std::string& l) {
        std::vector<std::string> v;
        std::stringstream ss(l);
        for (std::string c; std::getline(ss, c, ',');) v.push_back(c);
        if (!l.empty() && l.back() == ',') v.emplace_back();
        return v;
    };
    std::string line;
    std::getline(f, line);
    const auto h = split(line);
    auto col = [&](const char* n) {
        for (int i = 0; i < (int)h.size(); ++i) if (h[i] == n) return i;
        hb::die(std::string("03 map lacks column ") + n);
    };
    const int c_run = col("run_id"), c_smid = col("smid"), c_gpc = col("gpc"), c_part = col("partition"), c_mate = col("tpc_mate_hyp");
    std::vector<std::vector<std::string>> rows;
    std::string latest;
    while (std::getline(f, line)) {
        auto r = split(line);
        if ((int)r.size() <= std::max({c_run, c_smid, c_gpc, c_part, c_mate})) continue;
        if (r[c_run] > latest) latest = r[c_run];  // ISO timestamps with one UTC offset sort lexicographically
        rows.push_back(std::move(r));
    }
    std::vector<SmInfo> m(sm_count);
    for (const auto& r : rows) {
        if (r[c_run] != latest) continue;
        const int s = std::stoi(r[c_smid]);
        if (s < 0 || s >= sm_count) continue;
        if (r[c_part].empty() || r[c_mate].empty()) hb::die("03 map (latest run) lacks partition/tpc_mate_hyp for smid " + r[c_smid]);
        m[s] = {std::stoi(r[c_gpc]), std::stoi(r[c_part]), std::stoi(r[c_mate])};
    }
    for (int s = 0; s < sm_count; ++s)
        if (m[s].gpc < 0) hb::die("03 map has no entry for smid " + std::to_string(s));
    run_out = latest;
    return m;
}

// The 8 SMs: two TPC-mate pairs per L2 partition, each pair in its own GPC. Order = activation order of the subsets
// 1 = {2}, 2 = {2,3} (TPC mates), 4 = {2,3,0,1} (both partitions), 8 = all.
static const uint32_t kSmsChosen[kSmsMax] = {2, 3, 0, 1, 6, 7, 8, 9};

static void check_chosen(const std::vector<SmInfo>& m, const std::string& run) {
    std::set<int> gpcs;
    int per_part[2] = {0, 0};
    for (int p = 0; p < kSmsMax; p += 2) {
        const uint32_t s = kSmsChosen[p], t = kSmsChosen[p + 1];
        if (m[s].mate != (int)t || m[t].mate != (int)s) hb::die("k3sms: SMs " + std::to_string(s) + "," + std::to_string(t) + " are not TPC mates in the 03 map");
        if (m[s].gpc != m[t].gpc || m[s].partition != m[t].partition) hb::die("k3sms: TPC mates in different GPCs/partitions?");
        if (!gpcs.insert(m[s].gpc).second) hb::die("k3sms: two pairs share GPC " + std::to_string(m[s].gpc));
        if (m[s].partition < 0 || m[s].partition > 1) hb::die("k3sms: bad partition label");
        ++per_part[m[s].partition];
    }
    if (per_part[0] != 2 || per_part[1] != 2) hb::die("k3sms: need two pairs per L2 partition");
    if (m[kSmsChosen[0]].partition == m[kSmsChosen[2]].partition) hb::die("k3sms: the 4-SM subset must span both partitions");
    std::printf("k3sms: SMs");
    for (uint32_t s : kSmsChosen) std::printf(" %u(G%d,P%d)", s, m[s].gpc, m[s].partition);
    std::printf(" verified against the 03 map run %s\n", run.c_str());
}

template <uint32_t SIZE>
static void run_sms_config(hb::Run& run, hb::CsvWriter& csv, const std::vector<SmInfo>& m, CleanL2& clean, uint8_t* d_l2,
                           uint8_t* d_hbm, uint32_t* d_ctr, unsigned long long* d_start, SmsRec* d_out, uint64_t* d_ti,
                           uint32_t nact, int hbm, int launches, long long& li) {
    constexpr uint32_t kCopies = kSmsBytes / SIZE;
    const char* srcname = hbm ? "hbm" : "l2";
    SmsSet set{};
    set.n = nact;
    std::string setstr;
    for (uint32_t j = 0; j < nact; ++j) { set.smid[j] = kSmsChosen[j]; setstr += (j ? ";" : "") + std::to_string(kSmsChosen[j]); }
    const std::string label = "k3sms_a" + std::to_string(nact) + "_" + srcname + "_" + std::to_string(SIZE);
    CUDA_CHECK(cudaFuncSetAttribute(k3sms_issue<SIZE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kDynSmem));
    const uint32_t regions = (uint32_t)(kHbmBytes / kSmsBytes), need = kReps * nact;
    double span_sum = 0, comp_sum = 0;
    int cnt = 0;
    for (int l = -2; l < launches; ++l) {  // l < 0: unrecorded warm-up launches
        const long long key = l < 0 ? li + 100000 + l : li;
        const uint32_t base = (uint32_t)(((uint64_t)(key + 1) * 2654435761ull) % (regions - need));
        if (hbm) clean();
        CUDA_CHECK(cudaMemset(d_ctr, 0, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_start, 0, sizeof(unsigned long long)));
        // one block per SM (200 KiB of shared memory each): every SM hosts exactly one block, the chosen ones work
        k3sms_issue<SIZE><<<run.dev().sm_count, 32, kDynSmem>>>(hbm ? d_hbm : d_l2, set, base, kReps, hbm, d_ctr, d_start, d_out, d_ti);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        if (l < 0) continue;
        std::vector<SmsRec> h((size_t)kReps * nact);
        std::vector<uint64_t> ht((size_t)kReps * nact * kCopies);
        CUDA_CHECK(cudaMemcpy(h.data(), d_out, h.size() * sizeof(SmsRec), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ht.data(), d_ti, ht.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        const hb::ClockSample s = run.clock_note(label, "launch", li);
        for (uint32_t a = 0; a < nact; ++a)
            if (h[a].smid != kSmsChosen[a]) hb::die(label + ": record of SM " + std::to_string(kSmsChosen[a]) + " missing");
        for (int r = 0; r < kReps; ++r) {
            uint64_t gmin = ~0ull;
            for (uint32_t a = 0; a < nact; ++a) gmin = std::min(gmin, h[(size_t)r * nact + a].g0);
            for (uint32_t a = 0; a < nact; ++a) {
                const SmsRec& o = h[(size_t)r * nact + a];
                const uint64_t* t = &ht[((size_t)r * nact + a) * kCopies];
                std::string ilist, dlist;
                for (uint32_t i = 0; i < kCopies; ++i) { if (i) ilist += ';'; ilist += std::to_string(t[i] - o.t0); }
                for (uint32_t j = 0; j < kSmsRounds; ++j) { if (j) dlist += ';'; dlist += std::to_string(o.tdone[j] - o.t0); }
                const uint64_t span = t[kCopies - 1] - o.t0, comp = o.tdone[kSmsRounds - 1] - o.t0;
                span_sum += span;
                comp_sum += comp;
                ++cnt;
                char b2[320];
                std::snprintf(b2, sizeof b2, ",%u,%s,%s,%u,%u,%d,%u,%u,%d,%d,%llu,%llu,%llu,%u,", nact, setstr.c_str(), srcname, SIZE,
                              kCopies, r, a, o.smid, m[o.smid].gpc, m[o.smid].partition, (unsigned long long)span,
                              (unsigned long long)comp, (unsigned long long)(o.g0 - gmin), o.parity);
                csv.add(run.prefix(li, s) + b2 + ilist + "," + dlist);
            }
        }
        csv.flush();
        ++li;
    }
    csv.sync();
    std::printf("k3sms %-22s: issue span %9.1f  completion %9.1f cycles (mean per SM, 1 MiB)\n", label.c_str(), span_sum / cnt,
                comp_sum / cnt);
    std::fflush(stdout);
}

static void run_k3sms(hb::Run& run, bool quick) {
    std::string map_run;
    const std::vector<SmInfo> m = load_sm_info(run.dev().sm_count, map_run);
    check_chosen(m, map_run);
    hb::CsvWriter csv(hb::Run::results("06__tma__issue_sms.csv"),
                      hb::with_mandatory({"active_sms", "active_set", "source", "copy_bytes", "copies", "rep", "block", "smid", "gpc",
                                          "partition", "issue_span_cycles", "completion_cycles", "start_skew_ns", "parity",
                                          "issue_cycles_list", "round_done_cycles_list"}));
    uint8_t *d_l2, *d_hbm;
    uint32_t* d_ctr;
    unsigned long long* d_start;
    SmsRec* d_out;
    uint64_t* d_ti;
    CUDA_CHECK(cudaMalloc(&d_l2, kSmsL2Bytes));
    CUDA_CHECK(cudaMemset(d_l2, 0x11, kSmsL2Bytes));
    CUDA_CHECK(cudaMalloc(&d_hbm, kHbmBytes));
    CUDA_CHECK(cudaMemset(d_hbm, 0x22, kHbmBytes));
    CUDA_CHECK(cudaMalloc(&d_ctr, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_start, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)kReps * kSmsMax * sizeof(SmsRec)));
    CUDA_CHECK(cudaMalloc(&d_ti, (size_t)kReps * kSmsMax * (kSmsBytes / 4096) * sizeof(uint64_t)));
    CleanL2 clean;
    const int launches = quick ? 2 : 10;
    long long li = 0;
    run.lock_clocks_note("k3sms", "before");
    for (int hbm = 0; hbm <= 1; ++hbm)
        for (uint32_t nact : {1u, 2u, 4u, 8u}) {
            run_sms_config<4096>(run, csv, m, clean, d_l2, d_hbm, d_ctr, d_start, d_out, d_ti, nact, hbm, launches, li);
            run_sms_config<16384>(run, csv, m, clean, d_l2, d_hbm, d_ctr, d_start, d_out, d_ti, nact, hbm, launches, li);
        }
    run.lock_clocks_note("k3sms", "after");
    cudaFree(d_l2); cudaFree(d_hbm); cudaFree(d_ctr); cudaFree(d_start); cudaFree(d_out); cudaFree(d_ti);
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "k3";
    hb::Run run(EXP, argc, argv);
    if (mode == "clockcheck") run.lock_clocks_note(hb::arg_str(argc, argv, "--label", "check"), "check");
    else if (mode == "k3") run_k3(run, hb::has_flag(argc, argv, "--quick"));
    else if (mode == "k3sms") run_k3sms(run, hb::has_flag(argc, argv, "--quick"));
    else if (mode == "ncu") run_ncu(run);
    else hb::die("unknown mode " + mode);
    run.complete();
    return 0;
}
