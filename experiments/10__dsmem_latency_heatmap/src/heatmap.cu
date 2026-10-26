/*
 * heatmap.cu — experiment 10: SM×SM matrix of remote shared-memory (DSMEM) latency inside every GPC, measured with a
 * single active cluster per launch that is steered onto a chosen set of SMs: dependent-load latency (K1, with a
 * per-hop fine pass), latency under load (K1f), and store → visibility round trips (K2) for every ordered SM pair
 * that can share a cluster, at three background-load conditions.
 *
 * Usage: bin/heatmap {steer|sched|ncu|clockcheck} [--schedule A|B|C|D|D8|E] [--quick] [--label L].
 *   One invocation = one run_id (results/10__dsmem_latency_heatmap__runs.csv).
 *
 * Placement (common10.hpp, Steer; CUDA_MODULE_LOADING=EAGER, see there): every measuring kernel is launched as ONE cluster of CS CTAs (1-D, 200 KB dynamic
 *   smem → 1 CTA per SM, explicit cluster attribute, Default policy). Before it, the placeholder kernel k_block occupies
 *   every SM outside a target set T (|T| = CS) and leaves T free, so the cluster can only be placed on T. Each CTA records
 *   %smid, %clusterid.x, %cluster_ctarank; a launch whose ranks are not on distinct SMs of T, or whose NVML sample after
 *   the launch shows a throttle reason or an SM clock below the lock, is discarded and repeated (≤ 3 tries); a cluster
 *   that has not started 1 s after its launch makes the session "unplaceable" (placeholders released, kernel discarded).
 *   Load condition of a session = placeholder mode: idle (sleeping), light (own-smem streaming), heavy (streaming the TPC
 *   mate's smem through DSMEM); "none" = no placeholders at all (schedule E: the hardware's own choice for a lone cluster).
 *   TPC index within a GPC = rank of the TPC among the GPC's TPCs in ascending smid order (03 map); subsets are named by
 *   these indices.
 *
 * K1 — k1_chase(chase, zero, out, fine, place, started): 32 threads per CTA. Each CTA writes a stride-1 chain in its own
 *   smem (1024 × 4 B; element i holds the byte offset of element (i + 1) mod 1024), cluster.sync(). Sources take turns
 *   (src = 0 … CS−1, separated by cluster.sync(), so one SM of the cluster is active at a time); lane 0 of the source, for
 *   every dst (self included): base = mapa(region, dst); 64 warm hops; t0 = read_clock64_after(v); 4096 timed hops
 *   v = ld.shared::cluster.u32 [base + v] (#pragma unroll 1 around 16 inlined hops); t1 = read_clock64_after(v);
 *   cycles_per_hop = (t1 − t0) / 4096 (fixed timer/loop region ≤ 0.05 cycles/hop, not subtracted). Then the fine pass:
 *   256 further hops of the same chain, each timed alone (a = %clock64; v = hop(v); b = read_clock64_after(v)), the
 *   values include the timer pair (≈ 6 cycles). Final offset checked on the host (4160 + 256 hops from element 0).
 *   chase = 0 (steer pilot, ncu baseline): setup and barriers only.
 * K1f — k1f_loaded(zero, samples, place, started): 1024 threads per CTA, each CTA fills its first 192 KiB with w[i] = i.
 *   Same turns; all 1024 threads of the source stream the destination's 192 KiB: thread t, group g (0 … 5) issues 8
 *   independent ld.shared::cluster.u32 of words (8g + j)·1024 + t; t0 = %clock64 after the previous group's values,
 *   t1 = read_clock64_after(sum of the group's 8 values) → sample t1 − t0 (6144 per pair; the sum adds three dependent
 *   IADD3). Only one cluster exists, so no other SM-to-SM traffic is measured at the same time (except heavy placeholders).
 * K2 — k2_all<M, GT>(reps, zero, g, rt, info, place, started): 32 threads per CTA. For every ordered pair (q, p), q ≠ p,
 *   in turn (cluster.sync() before and after; every CTA re-initialises its mbarrier and zeroes its flag/data word per
 *   pair), lane 0 of q and p ping-pong `reps` times (epoch e = i + 1):
 *     q: t0 = %clock64; [ga = %globaltimer]; signal(p, e); wait(own, e); t1 = %clock64 after the value; [gb]
 *     p: wait(own, e); [ga]; (re-arm); [gb]; signal(q, e)
 *   round_trip_cycles = t1 − t0 (SM clock of q); with GT also one_way_ns = ga(p) − ga(q) (q→p) and one_way_back_ns =
 *   gb(q) − gb(p) (p→q), in 32-ns ticks of %globaltimer. Methods M:
 *     st_plain           st.shared::cluster.u32 e → peer flag / spin ld.volatile.shared.u32 on the own flag
 *     st_async_mbarrier  st.async…mbarrier::complete_tx::bytes e → peer data word + complete_tx(4) on the peer's
 *                        mbarrier / loop mbarrier.try_wait.parity.acquire.cluster, then ld.shared of the data word;
 *                        re-arm with arrive.expect_tx(4) after each wait
 *   Polls capped at 2^26 (timeout); a timeout or a data mismatch makes the host die. Variants: <method>_nogt (no
 *   %globaltimer read in the loop: clean round trip) and <method> (with the global-timer reads).
 *
 * Schedules (sessions; each session = placeholders + the listed kernels; 2 unrecorded warm-up sessions per invocation):
 *   A  idle, CS 16: G1–G5 all 16 cluster-capable SMs × 40; G6/G7 (9 capable TPCs) one TPC index k = 0…8 left to the
 *      placeholders × 5 each. K1, K1f, K2 ×4.
 *   B  idle, CS 8: G0's 8 capable SMs × 40 (K1, K1f, K2 ×4); G1–G5 every 4-of-8 capable-TPC subset × 2, G6/G7 every
 *      4-of-9 subset × 1 (K1, K1f).
 *   C  idle, CS 2: TPCs 62–65 (pair only with their mate) × 20. K1, K1f, K2 ×4.
 *   D  idle/light/heavy interleaved, CS 16: G1–G7 × 10 per condition (G6/G7: placeholder TPC index = session mod 9).
 *      K1, K1f, K2 ×4.
 *   D8 idle/light/heavy, CS 8: G1 and G6, TPC-index subsets {0–3}, {4–7}, {0,2,4,6}, {1,3,5,7} × 5. K1, K1f, K2 ×4.
 *   E  none (no placeholders): lone CS-16 × 20 and lone CS-8 × 20. K1.
 *   --quick: a few sessions per schedule, K2 with 64 trials.
 *
 * Results (append; mandatory columns first; per-launch rows flushed per launch, pooled rows written and fsynced at the
 *   end of every (schedule, GPC) block — pooled rows have launch_idx = −1):
 *   10__heatmap__pairs.csv      K1, one row per launch and ordered pair (self included): schedule, load_condition,
 *       session, cs, target_gpc, subset, cluster_id, src_rank, dst_rank, smid_src, smid_dst, gpc, tpc_idx_src,
 *       tpc_idx_dst, relation (self / same_tpc / other_tpc), cycles_per_hop, total_cycles, n_hops, fine_median_cycles,
 *       fine_mad_cycles, fine_mean_cycles, fine_min_cycles, fine_max_cycles (fine = the 256 per-hop values)
 *   10__heatmap__fine_hopstats.csv  per (schedule, load_condition, cs, gpc, smid_src, smid_dst, hop_idx): n, sum,
 *       sumsq, min, max of the per-hop cycles over all launches of the block
 *   10__heatmap__fine_hist.csv  per (schedule, load_condition, cs, gpc, smid_src, smid_dst, cycles): count
 *   10__heatmap__fine_raw.csv   every per-hop value of the first 2 recorded K1 launches of each (schedule, gpc, cs)
 *       block with load condition idle or none: session, cs, smid_src, smid_dst, hop_idx, cycles
 *   10__heatmap__loaded.csv     K1f per launch and pair: …, threads, ilp, mean_cycles, p5, p25, p50, p75, p95, min, max, n
 *   10__heatmap__loaded_hist.csv  per (schedule, load_condition, cs, gpc, smid_src, smid_dst, cycles): count
 *   10__heatmap__visibility_trials.csv  K2 per launch, pair and variant: …, method, n, rt_median_cycles, rt_mad_cycles,
 *       rt_mean_cycles, rt_p5, rt_p95, rt_min, rt_max, ow_mean_ns, ow_median_ns, ob_mean_ns, ob_median_ns (ns: GT only)
 *   10__heatmap__visibility_hist.csv  per (schedule, load_condition, cs, gpc, method, quantity ∈ {rt_cycles, ow_ns,
 *       ob_ns}, smid_src, smid_dst, value): count
 *   10__heatmap__sessions.csv   one row per session: schedule, load_condition, session, cs, target_gpc, subset,
 *       target_smids, placeholders_kept, outcome (ok / unplaceable / failed), n_launches, n_discarded, smids_used
 *   10__heatmap__discards.csv   one row per discarded launch: schedule, session, label, reason
 *
 * ncu mode (profiled by src/run.sh, no placeholders: ncu serialises kernels): k1_chase as a lone CS-16 cluster and a lone
 *   CS-8 cluster, each with the chase and as an all-exit baseline; manifest bin/ncu_manifest_k1.csv.
 */
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "common10.hpp"
#include "../../09__dsmem_latency_vs_cluster_size/src/common09.hpp"

static const char* EXP = "10__dsmem_latency_heatmap";
constexpr uint32_t kChainElems = 1024, kWarmHops = 64, kHops = 4096, kFineHops = 256, kFineOff = 16384;
constexpr int kUnroll = 16, kMaxCs = 16, kMaxPairs = kMaxCs * kMaxCs, kTries = 3, kWarmSessions = 2;
constexpr uint32_t kReps = 1000, kQuickReps = 64, kSpinCap = 1u << 26;
constexpr uint32_t kLThreads = 1024, kLIlp = 8, kLGroups = 6, kLWords = kLThreads * kLIlp * kLGroups, kLSamples = kLThreads * kLGroups;

struct K1Rec { unsigned long long cycles; uint32_t off, valid; };

// ---------------------------------------------------------------- K1
__global__ void __launch_bounds__(32) k1_chase(uint32_t chase, uint64_t zero, K1Rec* out, uint32_t* fine, uint32_t* place, uint32_t* started) {
    extern __shared__ __align__(16) uint8_t dyn[];
    signal_started(started);
    const uint32_t lane = threadIdx.x;
    const ClusterRegs c = read_cluster_regs();
    const uint32_t rank = c.ctarank, cs = c.nctarank;
    const uint32_t region = smem_u32(dyn);
    for (uint32_t i = lane; i < kChainElems; i += 32) reinterpret_cast<uint32_t*>(dyn)[i] = ((i + 1) % kChainElems) * 4;
    record_place(place);
    __syncwarp();
    cluster_sync();
    if (chase) {
        for (uint32_t src = 0; src < cs; ++src) {
            if (rank == src && lane == 0) {
                uint32_t* fb = reinterpret_cast<uint32_t*>(dyn + kFineOff);
                for (uint32_t dst = 0; dst < cs; ++dst) {
                    const uint32_t base = mapa_u32(region, dst);
                    uint32_t v = 0;
#pragma unroll 1
                    for (uint32_t i = 0; i < kWarmHops; ++i) v = ld_dsmem_u32(base + v);
                    const uint64_t t0 = read_clock64_after(v, zero);
#pragma unroll 1
                    for (uint32_t i = 0; i < kHops / kUnroll; ++i) {
#pragma unroll
                        for (int u = 0; u < kUnroll; ++u) v = ld_dsmem_u32(base + v);
                    }
                    const uint64_t t1 = read_clock64_after(v, zero);
#pragma unroll 1
                    for (uint32_t k = 0; k < kFineHops; ++k) {
                        const uint64_t a = read_clock64();
                        v = ld_dsmem_u32(base + v);
                        const uint64_t b = read_clock64_after(v, zero);
                        fb[k] = (uint32_t)(b - a);
                    }
                    const uint32_t idx = src * cs + dst;
                    out[idx] = K1Rec{t1 - t0, v, 1u};
                    for (uint32_t k = 0; k < kFineHops; ++k) fine[(size_t)idx * kFineHops + k] = fb[k];
                }
            }
            __syncwarp();
            cluster_sync();
        }
    } else {
        cluster_sync();
    }
}

// ---------------------------------------------------------------- K1f loaded
__global__ void __launch_bounds__(1024) k1f_loaded(uint64_t zero, uint32_t* samples, uint32_t* place, uint32_t* started) {
    extern __shared__ __align__(16) uint32_t sw[];
    signal_started(started);
    const uint32_t tid = threadIdx.x;
    const ClusterRegs c = read_cluster_regs();
    const uint32_t rank = c.ctarank, cs = c.nctarank;
    for (uint32_t i = tid; i < kLWords; i += kLThreads) sw[i] = i;
    record_place(place);
    __syncthreads();
    cluster_sync();
    const uint32_t region = smem_u32(sw);
    uint32_t sink = 0;
    for (uint32_t src = 0; src < cs; ++src) {
        if (rank == src) {
            for (uint32_t dst = 0; dst < cs; ++dst) {
                __syncthreads();
                const uint32_t base = mapa_u32(region, dst) + 4 * tid;
                uint32_t* out = samples + (size_t)(src * cs + dst) * kLSamples;
                uint32_t acc = 0;
#pragma unroll 1
                for (uint32_t g = 0; g < kLGroups; ++g) {
                    const uint32_t a = base + g * kLIlp * kLThreads * 4;
                    const uint64_t t0 = read_clock64_after(acc, zero);
                    uint32_t x[kLIlp];
#pragma unroll
                    for (uint32_t j = 0; j < kLIlp; ++j) x[j] = ld_dsmem_u32(a + j * kLThreads * 4);
                    const uint32_t sum = x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7];
                    const uint64_t t1 = read_clock64_after(sum, zero);
                    acc += sum;
                    out[g * kLThreads + tid] = (uint32_t)(t1 - t0);
                }
                sink += acc;
            }
        }
        __syncthreads();
        cluster_sync();
    }
    if (sink == 0xffffffffu) place[0] = sink;  // keeps the loads (never true for these data)
}

// ---------------------------------------------------------------- K2 visibility
enum M2 { M_ST = 0, M_STASYNC = 1 };
static const char* kMName[] = {"st_plain", "st_async_mbarrier"};

__device__ __forceinline__ uint32_t lds_cta_u32(uint32_t a) {
    uint32_t v; asm volatile("ld.shared::cta.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory"); return v; }

template <int M>
__device__ __forceinline__ void k2_signal(uint32_t peer_flag, uint32_t peer_data, uint32_t peer_bar, uint32_t e) {
    if constexpr (M == M_ST) st_dsmem_u32(peer_flag, e);
    else st_async_u32(peer_data, e, peer_bar);
}
// false on a timeout; v = the value seen (data word for the mbarrier method)
template <int M>
__device__ __forceinline__ bool k2_wait(uint32_t own_flag, uint32_t own_data, uint32_t own_bar, uint32_t e, uint32_t i, uint32_t& v) {
    if constexpr (M == M_STASYNC) {
        bool done = false;
#pragma unroll 1
        for (uint32_t n = 0; n < kSpinCap && !done; ++n) done = mbar_try_wait_parity_acq_cluster(own_bar, i & 1u);
        v = lds_cta_u32(own_data);
        return done;
    } else {
#pragma unroll 1
        for (uint32_t n = 0; n < kSpinCap; ++n)
            if ((v = lds_vol_u32(own_flag)) == e) return true;
        return false;
    }
}

// g[((pair * 2 + role) * 2 + {0: ga, 1: gb}) * reps + i]; rt[pair * reps + i]; pair = q * (cs − 1) + (p < q ? p : p − 1);
// info[4 * block + {0: smid, 1: timeouts, 2: mismatches}].
template <int M, bool GT>
__global__ void __launch_bounds__(32) k2_all(uint32_t reps, uint32_t zero, uint32_t* g, uint32_t* rt, uint32_t* info, uint32_t* place,
                                            uint32_t* started) {
    extern __shared__ __align__(16) uint8_t dyn[];  // 200 KB, only reserved (1 CTA/SM)
    __shared__ __align__(8) uint64_t s_bar;
    __shared__ uint32_t s_flag, s_data;
    __shared__ uint32_t s_ga[kReps], s_gb[kReps], s_rt[kReps];
    signal_started(started);
    const uint32_t lane = threadIdx.x;
    const ClusterRegs c = read_cluster_regs();
    const uint32_t rank = c.ctarank, cs = c.nctarank;
    const uint32_t own_flag = smem_u32(&s_flag), own_data = smem_u32(&s_data), own_bar = smem_u32(&s_bar);
    record_place(place);
    if (lane == 0) {
        dyn[0] = 0;
        mbar_init(own_bar, 1);
    }
    uint32_t timeouts = 0, mism = 0;
    uint32_t pair = 0;
    for (uint32_t q = 0; q < cs; ++q)
        for (uint32_t p = 0; p < cs; ++p) {
            if (p == q) continue;
            const uint32_t role = rank == q ? 0u : rank == p ? 1u : 2u, peer = rank == q ? p : q;
            const uint32_t peer_flag = mapa_u32(own_flag, peer), peer_data = mapa_u32(own_data, peer), peer_bar = mapa_u32(own_bar, peer);
            if (lane == 0) {  // fresh barrier and flag for every pair: phase parity restarts at 0
                s_flag = 0;
                s_data = 0;
                mbar_inval(own_bar);
                mbar_init(own_bar, 1);
                if constexpr (M == M_STASYNC) if (role < 2) mbar_arrive_expect_tx(own_bar, 4u);
                fence_mbarrier_init();
            }
            __syncwarp();
            cluster_sync();
            if (lane == 0 && role < 2) {
#pragma unroll 1
                for (uint32_t i = 0; i < reps; ++i) {
                    const uint32_t e = i + 1;
                    uint32_t v;
                    if (role == 0) {
                        const uint64_t t0 = read_clock64();
                        const uint32_t ga = GT ? (uint32_t)read_globaltimer() : 0u;
                        k2_signal<M>(peer_flag, peer_data, peer_bar, e);
                        const bool ok = k2_wait<M>(own_flag, own_data, own_bar, e, i, v);
                        const uint64_t t1 = read_clock64_after(v, zero);
                        const uint32_t gb = GT ? (uint32_t)read_globaltimer() : 0u;
                        if (!ok) { ++timeouts; break; }
                        mism += v != e;
                        if constexpr (M == M_STASYNC) mbar_arrive_expect_tx(own_bar, 4u);
                        s_rt[i] = (uint32_t)(t1 - t0);
                        s_ga[i] = ga;
                        s_gb[i] = gb;
                    } else {
                        const bool ok = k2_wait<M>(own_flag, own_data, own_bar, e, i, v);
                        const uint32_t ga = GT ? (uint32_t)read_globaltimer() + (v & zero) : 0u;
                        if (!ok) { ++timeouts; break; }
                        mism += v != e;
                        if constexpr (M == M_STASYNC) mbar_arrive_expect_tx(own_bar, 4u);
                        const uint32_t gb = GT ? (uint32_t)read_globaltimer() : 0u;
                        k2_signal<M>(peer_flag, peer_data, peer_bar, e);
                        s_ga[i] = ga;
                        s_gb[i] = gb;
                    }
                }
            }
            __syncwarp();
            cluster_sync();  // the pair is done; no access to a peer's smem is in flight
            if (role < 2)
                for (uint32_t i = lane; i < reps; i += 32) {
                    g[((pair * 2 + role) * 2 + 0) * reps + i] = s_ga[i];
                    g[((pair * 2 + role) * 2 + 1) * reps + i] = s_gb[i];
                    if (role == 0) rt[pair * reps + i] = s_rt[i];
                }
            __syncwarp();
            ++pair;
        }
    if (lane == 0) {
        uint32_t* inf = info + 4 * blockIdx.x;
        inf[0] = read_smid();
        inf[1] = timeouts;
        inf[2] = mism;
    }
    __syncwarp();
    cluster_sync();
}

using K2Fn = void (*)(uint32_t, uint32_t, uint32_t*, uint32_t*, uint32_t*, uint32_t*, uint32_t*);
struct K2Var { K2Fn k; int m; bool gt; };
static const K2Var kK2[4] = {{k2_all<M_STASYNC, false>, M_STASYNC, false}, {k2_all<M_ST, false>, M_ST, false},
                             {k2_all<M_STASYNC, true>, M_STASYNC, true}, {k2_all<M_ST, true>, M_ST, true}};

// ---------------------------------------------------------------- host: sessions
enum KMask { K_K1 = 1, K_K1F = 2, K_K2 = 4 };

struct Spec {
    std::string schedule;
    int cond = L_IDLE, cs = 16, gpc = -1;  // gpc −1: no placeholders, wherever the hardware puts the cluster
    std::string subset;
    std::vector<int> sms;  // target set T (empty for cond none)
    int kernels = K_K1;
    bool record = true;
    int chase = 1;
};

struct HopStat { unsigned long long n = 0, sum = 0, sumsq = 0; uint32_t mn = ~0u, mx = 0; };
using Hist = std::map<uint32_t, unsigned long long>;
using Key3 = std::tuple<int, uint32_t, uint32_t>;                  // (cond, src, dst)
using KeyV = std::tuple<int, std::string, std::string, uint32_t, uint32_t>;  // (cond, method, quantity, src, dst)

struct Files {
    hb::CsvWriter pairs, hop, fhist, fraw, loaded, lhist, vis, vhist, sess, disc;
    Files()
        : pairs(hb::Run::results("10__heatmap__pairs.csv"),
                hb::with_mandatory({"schedule", "load_condition", "session", "cs", "target_gpc", "subset", "cluster_id", "src_rank", "dst_rank",
                                    "smid_src", "smid_dst", "gpc", "tpc_idx_src", "tpc_idx_dst", "relation", "cycles_per_hop", "total_cycles",
                                    "n_hops", "fine_median_cycles", "fine_mad_cycles", "fine_mean_cycles", "fine_min_cycles", "fine_max_cycles"})),
          hop(hb::Run::results("10__heatmap__fine_hopstats.csv"),
              hb::with_mandatory({"schedule", "load_condition", "cs", "gpc", "smid_src", "smid_dst", "hop_idx", "n", "sum", "sumsq", "min", "max"})),
          fhist(hb::Run::results("10__heatmap__fine_hist.csv"),
                hb::with_mandatory({"schedule", "load_condition", "cs", "gpc", "smid_src", "smid_dst", "cycles", "count"})),
          fraw(hb::Run::results("10__heatmap__fine_raw.csv"),
               hb::with_mandatory({"schedule", "load_condition", "session", "cs", "smid_src", "smid_dst", "hop_idx", "cycles"})),
          loaded(hb::Run::results("10__heatmap__loaded.csv"),
                 hb::with_mandatory({"schedule", "load_condition", "session", "cs", "target_gpc", "subset", "src_rank", "dst_rank", "smid_src",
                                     "smid_dst", "gpc", "relation", "threads", "ilp", "mean_cycles", "p5", "p25", "p50", "p75", "p95", "min",
                                     "max", "n"})),
          lhist(hb::Run::results("10__heatmap__loaded_hist.csv"),
                hb::with_mandatory({"schedule", "load_condition", "cs", "gpc", "smid_src", "smid_dst", "cycles", "count"})),
          vis(hb::Run::results("10__heatmap__visibility_trials.csv"),
              hb::with_mandatory({"schedule", "load_condition", "session", "cs", "target_gpc", "subset", "method", "src_rank", "dst_rank",
                                  "smid_src", "smid_dst", "gpc", "relation", "n", "rt_median_cycles", "rt_mad_cycles", "rt_mean_cycles",
                                  "rt_p5", "rt_p95", "rt_min", "rt_max", "ow_mean_ns", "ow_median_ns", "ob_mean_ns", "ob_median_ns"})),
          vhist(hb::Run::results("10__heatmap__visibility_hist.csv"),
                hb::with_mandatory({"schedule", "load_condition", "cs", "gpc", "method", "quantity", "smid_src", "smid_dst", "value", "count"})),
          sess(hb::Run::results("10__heatmap__sessions.csv"),
               hb::with_mandatory({"schedule", "load_condition", "session", "cs", "target_gpc", "subset", "target_smids", "placeholders_kept",
                                   "outcome", "n_launches", "n_discarded", "smids_used"})),
          disc(hb::Run::results("10__heatmap__discards.csv"), hb::with_mandatory({"schedule", "session", "label", "reason"})) {}
    void flush_all() { for (hb::CsvWriter* w : {&pairs, &hop, &fhist, &fraw, &loaded, &lhist, &vis, &vhist, &sess, &disc}) w->flush(); }
    void sync_all() { for (hb::CsvWriter* w : {&pairs, &hop, &fhist, &fraw, &loaded, &lhist, &vis, &vhist, &sess, &disc}) w->sync(); }
};

struct Ctx {
    hb::Run& run;
    SmMap10 map;
    Steer st;
    Files f;
    bool quick = false;
    uint32_t reps = kReps;
    long long li = 0;
    int session = 0, n_disc = 0, n_unplaceable = 0;
    hb::ClockSample last{};
    // device buffers (allocated once: nothing may synchronise the device while placeholders are resident)
    SBuf<K1Rec> k1out{kMaxPairs};
    SBuf<uint32_t> k1fine{(size_t)kMaxPairs * kFineHops}, place{4 * kMaxCs}, samples{(size_t)kMaxPairs * kLSamples},
        k2g{(size_t)kMaxPairs * 4 * kReps}, k2rt{(size_t)kMaxPairs * kReps}, k2info{4 * kMaxCs};
    // pooled accumulators of the current (schedule, gpc) block
    std::string blk_sched;
    int blk_gpc = -2, blk_cs = 0;
    std::map<Key3, std::array<HopStat, kFineHops>> hop;
    std::map<Key3, Hist> fhist, lhist;
    std::map<KeyV, Hist> vhist;
    std::map<std::tuple<std::string, int, int>, int> raw_written;  // (schedule, gpc, cs) → recorded K1 launches written raw

    Ctx(hb::Run& r, bool q) : run(r), map(load_map10(r.dev().sm_count)), quick(q), reps(q ? kQuickReps : kReps) {
        st.preload();
        std::vector<uint32_t> h;
        for (SBuf<uint32_t>* b : {&k1fine, &place, &samples, &k2g, &k2rt, &k2info}) { b->zero(st.A, 1); b->get(h, st.A, 1); }
    }

    const char* relation(uint32_t a, uint32_t b) const { return a == b ? "self" : a / 2 == b / 2 ? "same_tpc" : "other_tpc"; }

    void flush_block() {
        if (blk_gpc == -2) return;
        const std::string pre = run.prefix(-1, last);
        char b[256];
        for (auto& [k, arr] : hop)
            for (uint32_t h = 0; h < kFineHops; ++h) {
                const HopStat& s = arr[h];
                if (!s.n) continue;
                std::snprintf(b, sizeof b, ",%s,%s,%d,%d,%u,%u,%u,%llu,%llu,%llu,%u,%u", blk_sched.c_str(), cond_name(std::get<0>(k)), blk_cs, blk_gpc,
                              std::get<1>(k), std::get<2>(k), h, s.n, s.sum, s.sumsq, s.mn, s.mx);
                f.hop.add(pre + b);
            }
        auto put = [&](hb::CsvWriter& w, std::map<Key3, Hist>& m) {
            for (auto& [k, hs] : m)
                for (auto& [v, n] : hs) {
                    std::snprintf(b, sizeof b, ",%s,%s,%d,%d,%u,%u,%u,%llu", blk_sched.c_str(), cond_name(std::get<0>(k)), blk_cs, blk_gpc,
                                  std::get<1>(k), std::get<2>(k), v, n);
                    w.add(pre + b);
                }
        };
        put(f.fhist, fhist);
        put(f.lhist, lhist);
        for (auto& [k, hs] : vhist)
            for (auto& [v, n] : hs) {
                std::snprintf(b, sizeof b, ",%s,%s,%d,%d,%s,%s,%u,%u,%d,%llu", blk_sched.c_str(), cond_name(std::get<0>(k)), blk_cs, blk_gpc,
                              std::get<1>(k).c_str(), std::get<2>(k).c_str(), std::get<3>(k), std::get<4>(k), (int)(int32_t)v, n);
                f.vhist.add(pre + b);
            }
        hop.clear();
        fhist.clear();
        lhist.clear();
        vhist.clear();
        f.sync_all();
    }
    void enter_block(const Spec& s) {
        const int g = s.gpc;
        if (s.schedule == blk_sched && g == blk_gpc && s.cs == blk_cs) return;
        flush_block();
        blk_sched = s.schedule;
        blk_gpc = g;
        blk_cs = s.cs;
    }
};

static std::string join_ints(const std::vector<int>& v, char sep) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) { if (i) s += sep; s += std::to_string(v[i]); }
    return s;
}

// Launch one measuring kernel as one cluster on stream A; returns false if the cluster did not start in time.
template <typename... KArgs, typename... Args>
static bool launch_one(Ctx& x, const Spec& s, void (*k)(KArgs...), int threads, Args&&... args) {
    hb::ClusterLaunch L;
    L.grid = dim3(s.cs);
    L.block = dim3(threads);
    L.cluster = dim3(s.cs, 1, 1);
    L.smem = kSmem;
    L.explicit_cluster = true;
    L.policy = cudaClusterSchedulingPolicyDefault;
    L.stream = x.st.A;
    x.st.reset_started();
    DBG("[launch] cs %d threads %d\n", s.cs, threads);
    CUDA_CHECK(L.launch(k, std::forward<Args>(args)...));
    CUDA_CHECK(cudaGetLastError());
    const bool ok = s.cond == L_NONE ? true : x.st.wait_started();
    DBG("[launch] started %d\n", (int)ok);
    if (!ok) return false;
    x.st.sync_or_abort(x.st.A, "measuring kernel");
    DBG("[launch] done\n");
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// placement of the single cluster: sm[rank]; "" if fine, else the reason
static std::string check_place(const Spec& s, const std::vector<uint32_t>& pl, std::vector<uint32_t>& sm) {
    sm.assign(s.cs, 0);
    for (int r = 0; r < s.cs; ++r) {
        const uint32_t* p = &pl[4 * r];
        if (p[1] != 0 || (int)p[2] != r || (int)p[3] != s.cs) return "cluster registers";
        sm[r] = p[0];
    }
    if (!hb::all_distinct(sm)) return "placement";
    if (!s.sms.empty())
        for (uint32_t m : sm)
            if (std::find(s.sms.begin(), s.sms.end(), (int)m) == s.sms.end()) return "outside target";
    return "";
}

struct Outcome { bool ok = true; bool unplaceable = false; int launches = 0, discarded = 0; std::vector<uint32_t> sm; };

// One recorded measuring launch with retries. `run` launches and returns false if the cluster did not start;
// `use(sm, sample)` consumes the outputs.
template <class R, class U>
static void measured(Ctx& x, const Spec& s, Outcome& o, const std::string& label, R&& run, U&& use) {
    if (!o.ok) return;
    for (int t = 0; t < kTries; ++t) {
        x.place.zero(x.st.A);
        if (!run()) {
            o.ok = false;
            o.unplaceable = true;
            return;
        }
        std::vector<uint32_t> pl, sm;
        x.place.get(pl, x.st.A, 4 * s.cs);
        const std::string why = check_place(s, pl, sm);
        const hb::ClockSample cs = x.run.clock_note(label, "launch", x.li);
        std::string reason = why.empty() ? (cs.bad() ? "clock" : "") : why;
        if (!reason.empty()) {
            if (s.record) {
                x.f.disc.add(x.run.prefix(x.li, cs) + "," + s.schedule + "," + std::to_string(x.session) + "," + label + "," + reason);
                x.f.disc.flush();
            }
            ++o.discarded;
            ++x.n_disc;
            ++x.li;
            continue;
        }
        x.last = cs;
        o.sm = sm;
        if (s.record) use(sm, cs);
        ++o.launches;
        ++x.li;
        return;
    }
    o.ok = false;
}

static void do_k1(Ctx& x, const Spec& s, Outcome& o) {
    const int cs = s.cs;
    const uint32_t want = ((kWarmHops + kHops + kFineHops) % kChainElems) * 4;
    const std::string label = s.schedule + "_k1_cs" + std::to_string(cs) + "_g" + std::to_string(s.gpc) + "_" + cond_name(s.cond);
    measured(x, s, o, label,
             [&] {
                 x.k1out.zero(x.st.A, cs * cs);
                 return launch_one(x, s, k1_chase, 32, (uint32_t)s.chase, (uint64_t)0, x.k1out.d, x.k1fine.d, x.place.d, x.st.started());
             },
             [&](const std::vector<uint32_t>& sm, const hb::ClockSample& smp) {
                 if (!s.chase) return;
                 std::vector<K1Rec> o1;
                 std::vector<uint32_t> fh;
                 x.k1out.get(o1, x.st.A, cs * cs);
                 x.k1fine.get(fh, x.st.A, (size_t)cs * cs * kFineHops);
                 const std::string pre = x.run.prefix(x.li, smp);
                 const int g = x.map.gpc[sm[0]];
                 const auto rk = std::make_tuple(s.schedule, g, cs);
                 const bool raw = (s.cond == L_IDLE || s.cond == L_NONE) && x.raw_written[rk] < 2;
                 if (raw) ++x.raw_written[rk];
                 char b[512];
                 for (int sr = 0; sr < cs; ++sr)
                     for (int dr = 0; dr < cs; ++dr) {
                         const K1Rec& k = o1[sr * cs + dr];
                         if (!k.valid || k.off != want)
                             hb::die(label + ": pair " + std::to_string(sr) + "->" + std::to_string(dr) + " final offset " + std::to_string(k.off) +
                                     " expected " + std::to_string(want));
                         const uint32_t a = sm[sr], d = sm[dr];
                         std::vector<uint32_t> v(fh.begin() + (size_t)(sr * cs + dr) * kFineHops, fh.begin() + (size_t)(sr * cs + dr + 1) * kFineHops);
                         const Key3 key{s.cond, a, d};
                         auto& hs = x.hop[key];
                         Hist& hh = x.fhist[key];
                         double sum = 0;
                         for (uint32_t h = 0; h < kFineHops; ++h) {
                             const uint32_t c = v[h];
                             HopStat& q = hs[h];
                             ++q.n;
                             q.sum += c;
                             q.sumsq += (unsigned long long)c * c;
                             q.mn = std::min(q.mn, c);
                             q.mx = std::max(q.mx, c);
                             ++hh[c];
                             sum += c;
                             if (raw) {
                                 std::snprintf(b, sizeof b, ",%s,%s,%d,%d,%u,%u,%u,%u", s.schedule.c_str(), cond_name(s.cond), x.session, cs, a, d, h, c);
                                 x.f.fraw.add(pre + b);
                             }
                         }
                         std::sort(v.begin(), v.end());
                         std::snprintf(b, sizeof b, ",%s,%s,%d,%d,%d,%s,0,%d,%d,%u,%u,%d,%d,%d,%s,%.4f,%llu,%u,%.1f,%.1f,%.3f,%u,%u", s.schedule.c_str(),
                                       cond_name(s.cond), x.session, cs, s.gpc, s.subset.c_str(), sr, dr, a, d, x.map.gpc[a], x.map.tpc_index[a],
                                       x.map.tpc_index[d], x.relation(a, d), (double)k.cycles / kHops, k.cycles, kHops, median_sorted(v),
                                       mad_sorted(v), sum / kFineHops, v.front(), v.back());
                         x.f.pairs.add(pre + b);
                     }
                 x.f.pairs.flush();
                 x.f.fraw.flush();
             });
}

static void do_k1f(Ctx& x, const Spec& s, Outcome& o) {
    const int cs = s.cs;
    const std::string label = s.schedule + "_k1f_cs" + std::to_string(cs) + "_g" + std::to_string(s.gpc) + "_" + cond_name(s.cond);
    measured(x, s, o, label,
             [&] { return launch_one(x, s, k1f_loaded, (int)kLThreads, (uint64_t)0, x.samples.d, x.place.d, x.st.started()); },
             [&](const std::vector<uint32_t>& sm, const hb::ClockSample& smp) {
                 std::vector<uint32_t> all;
                 x.samples.get(all, x.st.A, (size_t)cs * cs * kLSamples);
                 const std::string pre = x.run.prefix(x.li, smp);
                 char b[512];
                 std::vector<uint32_t> v(kLSamples);
                 for (int sr = 0; sr < cs; ++sr)
                     for (int dr = 0; dr < cs; ++dr) {
                         const uint32_t a = sm[sr], d = sm[dr];
                         std::copy_n(all.begin() + (size_t)(sr * cs + dr) * kLSamples, kLSamples, v.begin());
                         Hist& hh = x.lhist[Key3{s.cond, a, d}];
                         double sum = 0;
                         for (uint32_t c : v) { sum += c; ++hh[c]; }
                         std::sort(v.begin(), v.end());
                         std::snprintf(b, sizeof b, ",%s,%s,%d,%d,%d,%s,%d,%d,%u,%u,%d,%s,%u,%u,%.3f,%.1f,%.1f,%.1f,%.1f,%.1f,%u,%u,%u", s.schedule.c_str(),
                                       cond_name(s.cond), x.session, cs, s.gpc, s.subset.c_str(), sr, dr, a, d, x.map.gpc[a], x.relation(a, d), kLThreads,
                                       kLIlp, sum / kLSamples, pct_sorted(v, 0.05), pct_sorted(v, 0.25), pct_sorted(v, 0.5), pct_sorted(v, 0.75),
                                       pct_sorted(v, 0.95), v.front(), v.back(), kLSamples);
                         x.f.loaded.add(pre + b);
                     }
                 x.f.loaded.flush();
             });
}

static void do_k2(Ctx& x, const Spec& s, Outcome& o, const K2Var& kv) {
    const int cs = s.cs, npairs = cs * (cs - 1);
    const uint32_t reps = x.reps;
    const std::string meth = std::string(kMName[kv.m]) + (kv.gt ? "" : "_nogt");
    const std::string label = s.schedule + "_k2_" + meth + "_cs" + std::to_string(cs) + "_g" + std::to_string(s.gpc) + "_" + cond_name(s.cond);
    measured(x, s, o, label,
             [&] {
                 x.k2info.zero(x.st.A, 4 * cs);
                 if (!launch_one(x, s, kv.k, 32, reps, 0u, x.k2g.d, x.k2rt.d, x.k2info.d, x.place.d, x.st.started())) return false;
                 std::vector<uint32_t> inf;
                 x.k2info.get(inf, x.st.A, 4 * cs);
                 for (int r = 0; r < cs; ++r)
                     if (inf[4 * r + 1] || inf[4 * r + 2])
                         hb::die(label + ": rank " + std::to_string(r) + " timeouts " + std::to_string(inf[4 * r + 1]) + " mismatches " +
                                 std::to_string(inf[4 * r + 2]));
                 return true;
             },
             [&](const std::vector<uint32_t>& sm, const hb::ClockSample& smp) {
                 std::vector<uint32_t> gh, rh;
                 x.k2g.get(gh, x.st.A, (size_t)npairs * 4 * reps);
                 x.k2rt.get(rh, x.st.A, (size_t)npairs * reps);
                 const std::string pre = x.run.prefix(x.li, smp);
                 char b[640];
                 int pair = 0;
                 for (int q = 0; q < cs; ++q)
                     for (int p = 0; p < cs; ++p) {
                         if (p == q) continue;
                         const uint32_t a = sm[q], d = sm[p];
                         std::vector<uint32_t> rt(rh.begin() + (size_t)pair * reps, rh.begin() + (size_t)(pair + 1) * reps);
                         Hist& hr = x.vhist[KeyV{s.cond, meth, "rt_cycles", a, d}];
                         double sum = 0;
                         for (uint32_t v : rt) { sum += v; ++hr[v]; }
                         std::sort(rt.begin(), rt.end());
                         std::string ns = ",,,,";
                         if (kv.gt) {
                             const uint32_t* gq = &gh[((size_t)pair * 2 + 0) * 2 * reps];
                             const uint32_t* gp = &gh[((size_t)pair * 2 + 1) * 2 * reps];
                             std::vector<int32_t> ow(reps), ob(reps);
                             Hist& ho = x.vhist[KeyV{s.cond, meth, "ow_ns", a, d}];
                             Hist& hb2 = x.vhist[KeyV{s.cond, meth, "ob_ns", a, d}];
                             double so = 0, sb = 0;
                             for (uint32_t i = 0; i < reps; ++i) {
                                 ow[i] = (int32_t)(gp[i] - gq[i]);                // ga(p) − ga(q): q → p
                                 ob[i] = (int32_t)(gq[reps + i] - gp[reps + i]);  // gb(q) − gb(p): p → q
                                 so += ow[i];
                                 sb += ob[i];
                                 ++ho[(uint32_t)ow[i]];
                                 ++hb2[(uint32_t)ob[i]];
                             }
                             std::sort(ow.begin(), ow.end());
                             std::sort(ob.begin(), ob.end());
                             char nb[128];
                             std::snprintf(nb, sizeof nb, ",%.2f,%.1f,%.2f,%.1f", so / reps, median_sorted(ow), sb / reps, median_sorted(ob));
                             ns = nb;
                         }
                         std::snprintf(b, sizeof b, ",%s,%s,%d,%d,%d,%s,%s,%d,%d,%u,%u,%d,%s,%u,%.1f,%.1f,%.2f,%.1f,%.1f,%u,%u", s.schedule.c_str(),
                                       cond_name(s.cond), x.session, cs, s.gpc, s.subset.c_str(), meth.c_str(), q, p, a, d, x.map.gpc[a], x.relation(a, d),
                                       reps, median_sorted(rt), mad_sorted(rt), sum / reps, pct_sorted(rt, 0.05), pct_sorted(rt, 0.95), rt.front(), rt.back());
                         x.f.vis.add(pre + b + ns);
                         ++pair;
                     }
                 x.f.vis.flush();
             });
}

static Outcome run_session(Ctx& x, const Spec& s) {
    if (s.record) x.enter_block(s);
    Mask132 T;
    for (int m : s.sms) T.set((uint32_t)m);
    int kept = -1;
    if (s.cond != L_NONE) kept = x.st.begin(T, s.cond);
    Outcome o;
    if (s.kernels & K_K1) do_k1(x, s, o);
    if (s.kernels & K_K1F) do_k1f(x, s, o);
    if (s.kernels & K_K2)
        for (const K2Var& kv : kK2) do_k2(x, s, o, kv);
    if (s.cond != L_NONE) x.st.end();
    if (o.unplaceable) {  // the kernel was released by ending the placeholders: let it finish, discard it
        x.st.sync_or_abort(x.st.A, "released measuring kernel");
        ++x.n_unplaceable;
    }
    // placeholders must have occupied every SM outside T exactly once
    if (s.cond != L_NONE && o.ok) {
        std::vector<uint32_t> bs;
        for (uint32_t v : x.st.h_info) bs.push_back(v & 0xffffu);
        if (!hb::all_distinct(bs)) hb::die("placeholder CTAs not on distinct SMs");
    }
    if (!s.record) return o;
    std::vector<int> used(o.sm.begin(), o.sm.end());
    const std::string outcome = o.ok ? "ok" : o.unplaceable ? "unplaceable" : "failed";
    char b[1024];
    std::snprintf(b, sizeof b, ",%s,%s,%d,%d,%d,%s,%s,%d,%s,%d,%d,%s", s.schedule.c_str(), cond_name(s.cond), x.session, s.cs, s.gpc,
                  s.subset.c_str(), join_ints(s.sms, ';').c_str(), kept, outcome.c_str(), o.launches, o.discarded, join_ints(used, ';').c_str());
    x.f.sess.add(x.run.prefix(x.li - 1, x.last) + b);
    x.f.sess.flush();
    if (!o.ok) std::printf("  session %d (%s cs %d G%d %s %s): %s\n", x.session, s.schedule.c_str(), s.cs, s.gpc, s.subset.c_str(), cond_name(s.cond),
                           outcome.c_str());
    ++x.session;
    return o;
}

// ---------------------------------------------------------------- schedules
static std::vector<int> sms_of(const std::vector<int>& tpcs) {
    std::vector<int> v;
    for (int t : tpcs) { v.push_back(2 * t); v.push_back(2 * t + 1); }
    return v;
}
static void combos(int n, int k, int start, std::vector<int>& cur, std::vector<std::vector<int>>& out) {
    if ((int)cur.size() == k) { out.push_back(cur); return; }
    for (int i = start; i < n; ++i) { cur.push_back(i); combos(n, k, i + 1, cur, out); cur.pop_back(); }
}
// TPCs of GPC g at the given TPC indices (indices into the GPC's full TPC list)
static std::vector<int> tpcs_at(const SmMap10& m, int g, const std::vector<int>& idx) {
    std::vector<int> v;
    for (int i : idx) v.push_back(m.tpcs[g][i]);
    return v;
}
static std::vector<int> capable_idx(const SmMap10& m, int g) {
    std::vector<int> v;
    for (int i = 0; i < (int)m.tpcs[g].size(); ++i) if (m.capable(m.tpcs[g][i])) v.push_back(i);
    return v;
}
static Spec mk(const SmMap10& m, const std::string& sch, int cond, int cs, int g, const std::vector<int>& idx, const std::string& label, int kern) {
    Spec s;
    s.schedule = sch;
    s.cond = cond;
    s.cs = cs;
    s.gpc = g;
    s.subset = label;
    s.sms = sms_of(tpcs_at(m, g, idx));
    s.kernels = kern;
    if ((int)s.sms.size() != cs) hb::die("target of " + label + " has " + std::to_string(s.sms.size()) + " SMs, cs " + std::to_string(cs));
    return s;
}
static std::string idx_label(const std::vector<int>& idx) { return join_ints(idx, '-'); }

static std::vector<Spec> build(const SmMap10& m, const std::string& sch, bool quick) {
    std::vector<Spec> v;
    const int ALL = K_K1 | K_K1F | K_K2;
    std::vector<int> g16;  // GPCs that host CS 16
    for (int g = 0; g < m.ngpc; ++g) if ((int)capable_idx(m, g).size() >= 8) g16.push_back(g);
    auto cs16 = [&](int g, int k, int cond, const std::string& schn) {  // k: TPC index left to the placeholders (18-SM GPCs)
        std::vector<int> ci = capable_idx(m, g);
        if (ci.size() == 8) return mk(m, schn, cond, 16, g, ci, "all", ALL);
        std::vector<int> idx;
        for (int i : ci) if (i != ci[k % ci.size()]) idx.push_back(i);
        return mk(m, schn, cond, 16, g, idx, "blk" + std::to_string(ci[k % ci.size()]), ALL);
    };
    if (sch == "A") {
        for (int g : g16) {
            const std::vector<int> ci = capable_idx(m, g);
            if (ci.size() == 8) for (int r = 0; r < (quick ? 2 : 40); ++r) v.push_back(cs16(g, 0, L_IDLE, "A"));
            else
                for (int k = 0; k < (quick ? 2 : (int)ci.size()); ++k)
                    for (int r = 0; r < (quick ? 1 : 5); ++r) v.push_back(cs16(g, k, L_IDLE, "A"));
        }
    } else if (sch == "B") {
        for (int g = 0; g < m.ngpc; ++g) {
            const std::vector<int> ci = capable_idx(m, g);
            if (ci.size() == 4) {  // the 8-SM cluster domain of G0
                for (int r = 0; r < (quick ? 2 : 40); ++r) v.push_back(mk(m, "B", L_IDLE, 8, g, ci, "all", ALL));
                continue;
            }
            std::vector<std::vector<int>> sub;
            std::vector<int> cur;
            combos((int)ci.size(), 4, 0, cur, sub);
            const int reps = quick ? 1 : (ci.size() == 8 ? 2 : 1);
            for (int r = 0; r < reps; ++r)
                for (size_t i = 0; i < sub.size() && (!quick || i < 3); ++i) {
                    std::vector<int> idx;
                    for (int j : sub[i]) idx.push_back(ci[j]);
                    v.push_back(mk(m, "B", L_IDLE, 8, g, idx, idx_label(idx), K_K1 | K_K1F));
                }
        }
    } else if (sch == "C") {
        for (int g = 0; g < m.ngpc; ++g)
            for (int i = 0; i < (int)m.tpcs[g].size(); ++i)
                if (!m.capable(m.tpcs[g][i]))
                    for (int r = 0; r < (quick ? 2 : 20); ++r) v.push_back(mk(m, "C", L_IDLE, 2, g, {i}, "t" + std::to_string(m.tpcs[g][i]), ALL));
    } else if (sch == "D") {
        for (int g : g16)
            for (int r = 0; r < (quick ? 1 : 10); ++r)
                for (int cond : {L_IDLE, L_LIGHT, L_HEAVY}) v.push_back(cs16(g, r, cond, "D"));
    } else if (sch == "D8") {
        const std::vector<std::vector<int>> pos = {{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 2, 4, 6}, {1, 3, 5, 7}};
        std::vector<int> gs;
        for (int g : g16) if (gs.empty() || (capable_idx(m, g).size() != capable_idx(m, gs[0]).size() && gs.size() < 2)) gs.push_back(g);
        for (int g : gs) {  // the first 16-SM and the first 18-SM GPC hosting CS 16 (G1 and G6 on this card)
            const std::vector<int> ci = capable_idx(m, g);
            for (size_t si = 0; si < pos.size() && (!quick || si < 1); ++si) {
                std::vector<int> idx;
                for (int j : pos[si]) idx.push_back(ci[j]);
                for (int r = 0; r < (quick ? 1 : 5); ++r)
                    for (int cond : {L_IDLE, L_LIGHT, L_HEAVY}) v.push_back(mk(m, "D8", cond, 8, g, idx, idx_label(idx), ALL));
            }
        }
    } else if (sch == "E") {
        for (int cs : {16, 8})
            for (int r = 0; r < (quick ? 2 : 20); ++r) {
                Spec s;
                s.schedule = "E";
                s.cond = L_NONE;
                s.cs = cs;
                s.gpc = -1;
                s.subset = "lone";
                s.kernels = K_K1;
                v.push_back(s);
            }
    } else {
        hb::die("unknown schedule " + sch);
    }
    return v;
}

static void run_sched(hb::Run& run, const std::string& sch, bool quick) {
    Ctx x(run, quick);
    std::vector<Spec> v = build(x.map, sch, quick);
    std::printf("schedule %s: %zu sessions\n", sch.c_str(), v.size());
    run.lock_clocks_note("sched_" + sch, "before");
    for (int w = 0; w < kWarmSessions; ++w) {
        Spec s = v.front();
        s.record = false;
        run_session(x, s);
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < v.size(); ++i) {
        // the block key uses the GPC of the target; schedule E has no target: the block is per CS
        run_session(x, v[i]);
        if ((i + 1) % 50 == 0 || i + 1 == v.size()) {
            const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("  %zu/%zu sessions, %.0f s, %d discarded launches, %d unplaceable sessions\n", i + 1, v.size(), el, x.n_disc, x.n_unplaceable);
            std::fflush(stdout);
        }
    }
    x.flush_block();
    x.f.sync_all();
    run.lock_clocks_note("sched_" + sch, "after");
    std::printf("schedule %s done: %d sessions, %d discarded launches, %d unplaceable\n", sch.c_str(), x.session, x.n_disc, x.n_unplaceable);
}

// Steering pilot: K1 without the chase (setup + barriers) on every kind of target; placements printed and logged
// (schedule "steer" in sessions.csv). Also one session per placeholder mode.
static void run_steer(hb::Run& run) {
    Ctx x(run, true);
    std::vector<Spec> v;
    for (const char* sch : {"A", "B", "C"}) {
        std::vector<Spec> b = build(x.map, sch, true);
        std::map<std::tuple<int, int, std::string>, int> seen;
        for (Spec s : b) {
            if (seen[{s.gpc, s.cs, s.subset}]++) continue;
            s.schedule = "steer";
            s.kernels = K_K1;
            s.chase = 0;
            v.push_back(s);
            v.push_back(s);
        }
    }
    for (int cond : {L_LIGHT, L_HEAVY}) {
        Spec s = v.front();
        s.cond = cond;
        v.push_back(s);
    }
    for (const Spec& s : v) {
        const Outcome o = run_session(x, s);
        int kept = 0;
        for (uint32_t i : x.st.h_info) kept += (i >> 16) & 1;
        std::printf("steer cs %2d G%d %-12s %-5s %s: placeholders kept %3d CTAs; cluster on SMs", s.cs, s.gpc, s.subset.c_str(), cond_name(s.cond),
                    o.ok ? "ok" : o.unplaceable ? "UNPLACEABLE" : "FAILED", kept);
        for (uint32_t m : o.sm) std::printf(" %u", m);
        std::printf("\n");
    }
    x.flush_block();
    x.f.sync_all();
    std::printf("steer: %d sessions, %d discarded launches, %d unplaceable\n", x.session, x.n_disc, x.n_unplaceable);
}

static void run_ncu(hb::Run& run) {
    hb::CsvWriter man("bin/ncu_manifest_k1.csv", hb::with_mandatory({"kernel", "cs", "nclusters", "chase", "expected_dshared_ld"}));
    long long li = 0;
    SBuf<K1Rec> out(kMaxPairs);
    SBuf<uint32_t> fine((size_t)kMaxPairs * kFineHops), place(4 * kMaxCs), started(1);
    for (int cs : {16, 8})
        for (uint32_t chase : {1u, 0u}) {
            hb::ClusterLaunch L;
            L.grid = dim3(cs);
            L.block = dim3(32);
            L.cluster = dim3(cs, 1, 1);
            L.smem = kSmem;
            L.explicit_cluster = true;
            CUDA_CHECK(L.launch(k1_chase, chase, (uint64_t)0, out.d, fine.d, place.d, started.d));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            const hb::ClockSample s = run.clock_note("ncu_k1_cs" + std::to_string(cs) + "_chase" + std::to_string(chase), "launch", li);
            // remote pairs only: own-CTA hops through the DSMEM path are not counted as dshared traffic (09)
            const unsigned long long expect = chase ? (unsigned long long)cs * (cs - 1) * (kWarmHops + kHops + kFineHops) : 0ull;
            man.add(run.prefix(li++, s) + ",k1_chase," + std::to_string(cs) + ",1," + std::to_string(chase) + "," + std::to_string(expect));
            man.flush();
        }
    man.sync();
}

int main(int argc, char** argv) {
    // before any CUDA call: lazy loading would make a first launch wait for the resident placeholder kernel (deadlock)
    setenv("CUDA_MODULE_LOADING", "EAGER", 1);
    const std::string mode = argc > 1 ? argv[1] : "sched";
    const bool quick = hb::has_flag(argc, argv, "--quick");
    hb::Run run(EXP, argc, argv);
    if (mode == "clockcheck") run.lock_clocks_note(hb::arg_str(argc, argv, "--label", "check"), "check");
    else if (mode == "steer") run_steer(run);
    else if (mode == "sched") run_sched(run, hb::arg_str(argc, argv, "--schedule", "A"), quick);
    else if (mode == "ncu") run_ncu(run);
    else hb::die("unknown mode " + mode);
    run.complete();
    return 0;
}
