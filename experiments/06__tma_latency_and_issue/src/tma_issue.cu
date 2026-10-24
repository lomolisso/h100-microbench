/*
 * tma_issue.cu — experiment 06, K3: issue gap of back-to-back bulk copies, the number of copies the TMA unit admits
 * before issue stalls, and the completion time of N copies in flight, for 1/2/4 issuing warps, 1/2/4/8 SMs and
 * sources in L2 or HBM.
 *
 * Usage: bin/tma_issue {k3|ncu|clockcheck} [--quick] [--label L]. One invocation = one run_id.
 *
 * k3_issue<W, SIZE, N>(src, slots, base_slot, reps, hbm, ctr, start_ns, out): grid = `sms` blocks of W warps, 200 KiB
 *   dynamic smem (1 CTA/SM; %smid of every block checked distinct), destination = copy i → smem i·SIZE (≤ 128 KiB),
 *   one mbarrier.
 *   Sample (rep r):
 *     thread 0: arrive.expect_tx(bar, n·SIZE); all blocks start the sample at the same %globaltimer instant
 *     (start_ns + r·20 µs; start_ns is published by the last block to arrive on a global counter); bar.sync;
 *     lane 0 of warp w: t0_w = clock64, then for every copy i ≡ w (mod W), i < n, in order:
 *       cp.async.bulk [dst + i·SIZE], [src + slot·32·SIZE + i·SIZE], SIZE, [bar]; t_i = clock64
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
 * ncu mode: k3_ncu<1, 4096> (same body) with n = 16 and 32 from L2 on one SM, 16 samples; manifest
 *   bin/ncu_manifest_issue.csv.
 */
#include <algorithm>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "../../05__memory_hierarchy_bandwidth/src/bw_common.hpp"
#include "tma_common.hpp"

static const char* EXP = "06__tma_latency_and_issue";
static constexpr int kMaxN = 64, kDstSlots = 32, kReps = 16;  // kMaxN: largest batch (2 KiB config)
static constexpr uint32_t kDynSmem = 200u << 10, kL2Bytes = 4u << 20;
static constexpr uint64_t kHbmBytes = 1ull << 30, kPeriodNs = 20000;

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
    const uint32_t slots = hbm ? (uint32_t)(kHbmBytes / slot_bytes) : kL2Bytes / slot_bytes;
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

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "k3";
    hb::Run run(EXP, argc, argv);
    if (mode == "clockcheck") run.lock_clocks_note(hb::arg_str(argc, argv, "--label", "check"), "check");
    else if (mode == "k3") run_k3(run, hb::has_flag(argc, argv, "--quick"));
    else if (mode == "ncu") run_ncu(run);
    else hb::die("unknown mode " + mode);
    run.complete();
    return 0;
}
