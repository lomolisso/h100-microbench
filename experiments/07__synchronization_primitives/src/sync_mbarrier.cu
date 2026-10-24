/*
 * sync_mbarrier.cu — experiment 07, K3: cost of every mbarrier operation (local and remote), the actual suspension of
 * try_wait with and without suspendTimeHint, the wake-up latency of a waiting warp, and the one-way propagation of a
 * remote arrive between two CTAs of a cluster.
 *
 * Usage: bin/sync_mbarrier {k3|ncu|clockcheck} [--quick] [--label L]. One invocation = one run_id.
 *
 * Local tests (cs = 0 in the CSV: plain launch, no cluster): the measuring block is pinned to the probe SM = smallest
 * smid of GPC 1 (03 map) by a claim: SmClaim::blocks() blocks are launched, the first block on the probe SM wins and
 * runs, every other block returns at once (one-warp blocks: claim_target_sm; larger blocks: thread 0 claims, the
 * result is broadcast through shared memory). A launch whose claim fails is discarded and repeated (counted).
 * Barriers are __shared__ 8-B words (an array of 64 for the unroll64 variants). Per rep: untimed setup by thread 0
 * (listed below), an untimed settle spin of kSettle cycles (earlier SYNCS operations and the previous rep's store
 * drain), then t0 = %clock64; op(s); t1. With several warps (arrive at ≥ 32 threads) the warps are aligned by bar.sync
 * followed by a volatile shared load that t0 depends on (bar.sync is BAR.SYNC.DEFER_BLOCKING: the warps that wait for
 * thread 0's setup would otherwise read t0 up to kSettle cycles before the barrier completed). Ops that return a value (arrival state, wait predicate) end with
 * t1 = read_clock64_after(value, zero): the region lasts until the result is back; ops without a result (init, inval,
 * expect_tx, complete_tx) end with a plain %clock64 read: the region is their issue cost. zero = runtime kernel arg 0.
 * Every rep begins with an untimed fence.acq_rel.gpu, so that no store (the previous rep's record) is pending since the
 * last fence: the release forms (MEMBAR.ALL.GPU) cost ~256 cycles more otherwise (264 vs 520 for one thread).
 * The barrier address is pinned in a register before the loop (pin_u32), and local waits use an immediate parity 0,
 * so that no address or parity computation (S2R SR_CgaCtaId, constant loads) lands inside a timed region.
 * Variants: single (n_ops 1) and unroll64 (64 back-to-back ops between the clocks, n_ops 64, cycles = total; for
 * state-returning ops t1 depends on the 64th state only, the ops themselves are independent).
 *   k3_init<N>              mbarrier.init (count 2^20−1) of N distinct barriers; teardown: inval of the N barriers
 *   k3_inval<N>             mbarrier.inval of N distinct barriers; setup: init of the N barriers (count 1)
 *   k3_arrive<N>            mbarrier.arrive (state returned) on one barrier of count 2^20−1, re-initialised each rep
 *                           (inval + init + bar.sync). block_threads 1 (blockDim 32, lanes 1–31 return after the
 *                           claim), 32 (all lanes of one warp: the coalesced arrive), 128, 512, 1024: every thread
 *                           arrives on the same barrier right after an aligning bar.sync; cycles = warp 0's t1 − t0,
 *                           cycles_warp_max = the slowest warp (lane 0 of each warp times its own region).
 *   k3_arrive_expect_tx<N>  mbarrier.arrive.expect_tx 16 B (state returned), barrier re-initialised each rep
 *   k3_expect_tx<N>         mbarrier.expect_tx.relaxed.cta 16 B (no result), barrier(s) re-initialised each rep.
 *                           unroll64: one expect_tx on each of 64 distinct barriers, because ptxas merges expect_tx
 *                           to one address (64 × 16 B → a single expect_tx of 1024 B, also with register operands)
 *   k3_complete_tx<N>       mbarrier.complete_tx.relaxed.cta 16 B (no result); setup: re-init + expect_tx(16·N) so
 *                           that the tx-count falls from 16·N to 0 (the normal TMA order)
 *   k3_arrive_drop<N>       mbarrier.arrive_drop (state returned), barrier re-initialised each rep
 *   k3_test_wait<N, C>      mbarrier.test_wait.parity on parity 0. C = complete: count 1, one arrive before the loop
 *                           (phase 0 completed, the call returns true); C = incomplete: count 2^20−1 (returns false).
 *                           N = 64: a dependent chain of 64 calls (the address of call i+1 = mad.lo(pred_i, zero,
 *                           bar)), i.e. the cost of one polling-loop iteration incl. the mad.lo glue; variants
 *                           complete, incomplete, chain64_complete, chain64_incomplete.
 *   k3_try_wait<N>          mbarrier.try_wait.parity without hint on a completed phase (single, chain64_complete)
 *   k3_try_wait_inc<H>      try_wait.parity on an incomplete phase (count 2^20−1), single call: time until it returns
 *                           false. H = suspendTimeHint encoded as an immediate (0, 100, 1000, 10000 ns) or no hint.
 *   k3_wakeup_{try,try_hint<H>,test}   block 64 (2 warps), barrier count 1, one phase per rep, parity = rep & 1.
 *                           After an aligning bar.sync, warp 0 lane 0 stamps t_enter and loops on the wait form until
 *                           it returns true, then stamps t_ret; warp 1 lane 0 spins X ∈ {1000, 5000, 20000} cycles,
 *                           stamps t_arr and arrives (same SM clock). cycles = t_ret − t_arr (wake-up latency),
 *                           cycles_warp_max = t_ret − t_enter (the total wait, ≈ X + wake-up) — for this primitive
 *                           (mbar_wakeup) the column does not hold a slowest-warp value. variant <form>_x<X>, form ∈
 *                           try_nohint, try_hint100, try_hint1000, try_hint10000, test_loop. block_threads 64.
 * Remote tests (cs = 2): one cluster of 2 CTAs per launch, 200 KB dynamic smem (1 CTA/SM), one active thread per CTA
 * (non-.aligned cluster barrier). Each rank initialises its own barrier, fence.mbarrier_init.release.cluster,
 * cluster barrier; the ranks must sit on distinct SMs (otherwise the launch is discarded and repeated, counted);
 * trailing cluster barrier before exit.
 *   k3_remote_arrive_release<N>  rank 0: kRemoteSettle-cycle spin; t0; N × mbarrier.arrive.release.cluster.
 *                           shared::cluster (sink) on rank 1's barrier (mapa, count 2^20−1); t1 = plain %clock64 (no
 *                           result: issue cost). k3_remote_arrive_relaxed<N>: the .relaxed.cluster form.
 *   k3_arrive_release_cluster_local<N>  the same on rank 0's own barrier with mbarrier.arrive.release.cluster.
 *                           shared::cta (state returned, t1 after the state). Rows: rank 0 only.
 *   k3_prop<R>              rank 1's barrier count 1, one phase per rep. Per rep: cluster barrier; rank 0 spins 2000
 *                           cycles (rank 1 is already polling), stamps %globaltimer g0 and remote-arrives (R: release,
 *                           else relaxed); rank 1 stamps t_enter, loops on try_wait.parity.acquire.cluster (parity
 *                           rep & 1) until true, stamps %globaltimer g1 and t_ret (both after the predicate).
 * Per configuration: 2 unrecorded warm-up launches, then 5 recorded launches × 1024 reps (--quick: 2 × 64, thread
 * counts {1, 1024}, X {1000}). The SM clock is sampled after every recorded launch (clock log).
 *   → results/07__sync__primitives__k3.csv (kernel k3): one row per rep: cycles, cycles_warp_max, smid, rank, n_ops,
 *     block_threads, variant; remote issue rows: rank 0; primitive mbar_remote_wait: rank 1's t_ret − t_enter in k3_prop
 *     (variant release_spin2000 / relaxed_spin2000). Timer-pair floor rows (kernel k0) from k_timer_floor first.
 *   → results/07__sync__propagation.csv: k3_prop, one row per rep: primitive mbar_remote_arrive_{release,relaxed},
 *     cs 2, smid_src = rank 0, smid_dst = rank 1, one_way_ns = g1 − g0 (32-ns ticks), round_trip empty, variant spin2000.
 * ncu mode: every local and remote configuration once (1024 reps; arrive at all thread counts); of the wake-up kernels
 *   only k3_wakeup_try and k3_wakeup_try_hint<1000> at X = 20000 (the try_wait spin loops for stall sampling); no retry
 *   of failed claims (placement does not matter for the counters); manifest bin/ncu_manifest_k3.csv in launch order.
 */
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "sync_common.hpp"

static const char* EXP = "07__synchronization_primitives";
constexpr uint32_t kNever = (1u << 20) - 1;       // expected count of a barrier that never completes
constexpr uint32_t kSettle = 500, kRemoteSettle = 2000, kPropSpin = 2000;
constexpr uint32_t kNoHint = 0xffffffffu;
constexpr int kProbeGpc = 1;

// ---------------------------------------------------------------- local PTX wrappers (raw predicate as u32)
// test_wait.parity / try_wait.parity. kPar < 0: runtime parity operand; else parity kPar as an immediate (a runtime
// parity would put its constant load and shift into the timed region). H = kNoHint: no suspendTimeHint operand,
// else H ns as an immediate.
template <int kPar>
__device__ __forceinline__ uint32_t test_wait_u32(uint32_t bar, uint32_t parity) {
    uint32_t ok;
    if constexpr (kPar < 0)
        asm volatile("{\n\t.reg .pred p;\n\tmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n\t}"
                     : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    else
        asm volatile("{\n\t.reg .pred p;\n\tmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n\t}"
                     : "=r"(ok) : "r"(bar), "n"(kPar) : "memory");
    return ok;
}
template <int kPar, uint32_t H>
__device__ __forceinline__ uint32_t try_wait_u32(uint32_t bar, uint32_t parity) {
    uint32_t ok;
    if constexpr (kPar < 0 && H == kNoHint)
        asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n\t}"
                     : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    else if constexpr (kPar < 0)
        asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n\tselp.u32 %0, 1, 0, p;\n\t}"
                     : "=r"(ok) : "r"(bar), "r"(parity), "n"(H) : "memory");
    else if constexpr (H == kNoHint)
        asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n\t}"
                     : "=r"(ok) : "r"(bar), "n"(kPar) : "memory");
    else
        asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n\tselp.u32 %0, 1, 0, p;\n\t}"
                     : "=r"(ok) : "r"(bar), "n"(kPar), "n"(H) : "memory");
    return ok;
}
// A shared address that ptxas cannot rematerialise: without this, the cvta'd barrier address (S2R SR_CgaCtaId + LEA)
// is recomputed right before its use, i.e. inside the timed region. v + (%clock & zero): a clock read is never
// repeated, so the value has to stay in a register.
__device__ __forceinline__ uint32_t pin_u32(uint32_t v, uint32_t zero) { return v + (read_clock32() & zero); }
__device__ __forceinline__ uint64_t mbar_arrive_release_cluster_state(uint32_t bar) {
    uint64_t s;
    asm volatile("mbarrier.arrive.release.cluster.shared::cta.b64 %0, [%1];" : "=l"(s) : "r"(bar) : "memory");
    return s;
}

// Pin the block to SM `target` (see the docstring); every thread of a winning block gets true.
__device__ __forceinline__ bool claim_block(uint32_t target, uint32_t* claim) {
    if (blockDim.x == 32) return claim_target_sm(target, claim);
    __shared__ uint32_t s_won;
    if (threadIdx.x == 0) {
        const bool w = read_smid() == target && atomicCAS(claim, 0u, blockIdx.x + 1) == 0u;
        if (w) claim[1] = target;
        s_won = w ? 1u : 0u;
    }
    __syncthreads();
    return s_won != 0;
}

// ---------------------------------------------------------------- local operation kernels
enum P3 { P_INIT, P_INVAL, P_ARRIVE, P_ARRIVE_TX, P_EXPECT_TX, P_COMPLETE_TX, P_DROP, P_TEST, P_TRY };

// The timed region of one rep (thread(s) after setup and alignment); returns t1 − t0.
// expect_tx<64> targets 64 distinct barriers: ptxas merges expect_tx on one address, even interleaved with other
// addresses or with register operands (64 × expect_tx(16) → one expect_tx(1024)).
template <int P, int N, uint32_t H>
__device__ __forceinline__ uint32_t k3_timed(uint32_t b0, uint32_t zero, uint32_t dep) {
    uint64_t t0, t1;
    if constexpr (P == P_INIT || P == P_INVAL || P == P_EXPECT_TX || P == P_COMPLETE_TX) {  // no result: issue cost
        t0 = read_clock64_after(dep, zero);
#pragma unroll
        for (int i = 0; i < N; ++i) {
            const uint32_t b = P == P_INIT || P == P_INVAL || P == P_EXPECT_TX ? b0 + 8u * i : b0;
            if constexpr (P == P_INIT) mbar_init(b, kNever);
            else if constexpr (P == P_INVAL) mbar_inval(b);
            else if constexpr (P == P_EXPECT_TX) mbar_expect_tx(b, 16);
            else mbar_complete_tx(b, 16);
        }
        t1 = read_clock64();
    } else if constexpr (P == P_ARRIVE || P == P_ARRIVE_TX || P == P_DROP) {  // state returned: t1 after the last
        uint64_t st = 0;
        t0 = read_clock64_after(dep, zero);
#pragma unroll
        for (int i = 0; i < N; ++i) {
            if constexpr (P == P_ARRIVE) st = mbar_arrive_state(b0);
            else if constexpr (P == P_ARRIVE_TX) st = mbar_arrive_expect_tx_state(b0, 16);
            else st = mbar_arrive_drop_state(b0);
        }
        t1 = read_clock64_after(st, zero);
    } else {  // P_TEST / P_TRY on parity 0
        t0 = read_clock64_after(dep, zero);
        if constexpr (N == 1) {
            const uint32_t ok = P == P_TEST ? test_wait_u32<0>(b0, 0) : try_wait_u32<0, H>(b0, 0);
            t1 = read_clock64_after(ok, zero);
        } else {
            uint32_t a = b0;
#pragma unroll
            for (int i = 0; i < N; ++i) {
                const uint32_t ok = P == P_TEST ? test_wait_u32<0>(a, 0) : try_wait_u32<0, H>(a, 0);
                a = mad_lo_u32(ok, zero, b0);  // next call waits for this predicate
            }
            t1 = read_clock64_after(a, zero);
        }
    }
    return (uint32_t)(t1 - t0);
}

// param = block_threads for k3_arrive (1 → only thread 0 works); the other kernels use thread 0 only.
// kComplete (waits only): the barrier's phase 0 is completed before the loop.
template <int P, int N, bool kComplete, uint32_t H>
__device__ __forceinline__ void k3_local(uint32_t target, uint32_t* claim, uint32_t reps, uint32_t param, uint32_t zero,
                                         Rec* out) {
    __shared__ alignas(8) uint64_t s_bar[64];
    __shared__ uint32_t s_w[64], s_probe;
    if (!claim_block(target, claim)) return;
    const bool one = P != P_ARRIVE || param == 1;
    if (threadIdx.x == 0) s_probe = 0;
    if (one && threadIdx.x != 0) return;
    const uint32_t smid = read_smid(), b0 = pin_u32(smem_u32(&s_bar[0]), zero);
    const bool reinit = P == P_ARRIVE || P == P_ARRIVE_TX || P == P_EXPECT_TX || P == P_COMPLETE_TX || P == P_DROP;
    constexpr int kBars = P == P_EXPECT_TX ? N : 1;  // barriers re-initialised per rep
    if (threadIdx.x == 0) {
        if (reinit)
            for (int i = 0; i < kBars; ++i) mbar_init(b0 + 8u * i, kNever);
        if constexpr (P == P_TEST || P == P_TRY) {
            mbar_init(b0, kComplete ? 1u : kNever);
            if (kComplete) mbar_arrive(b0);  // phase 0 completes: a wait on parity 0 returns true from now on
        }
    }
    if (!one) bar_sync0();
#pragma unroll 1
    for (uint32_t r = 0; r < reps; ++r) {
        fence_acq_rel_gpu();  // untimed drain: a .gpu fence costs ~256 cycles more if any store was issued since the last fence
        if (threadIdx.x == 0) {  // untimed setup
            if (reinit)
                for (int i = 0; i < kBars; ++i) { mbar_inval(b0 + 8u * i); mbar_init(b0 + 8u * i, kNever); }
            if constexpr (P == P_COMPLETE_TX) mbar_expect_tx(b0, 16u * N);
            if constexpr (P == P_INVAL)
                for (int i = 0; i < N; ++i) mbar_init(b0 + 8u * i, 1u);
            spin_cycles(kSettle);
        }
        // aligns all arriving threads (and publishes the re-initialised barrier); t0 depends on a load after the
        // barrier, because an early warp would otherwise read t0 before the deferred bar.sync completed
        const uint32_t dep = one ? 0u : bar_sync0_then_lds(smem_u32(&s_probe));
        const uint32_t cyc = k3_timed<P, N, H>(b0, zero, dep);
        block_record(s_w, cyc, 0u, smid, out, r, one);
        if constexpr (P == P_INIT)
            if (threadIdx.x == 0)
                for (int i = 0; i < N; ++i) mbar_inval(b0 + 8u * i);
    }
}

#define K3_LARGS uint32_t target, uint32_t* claim, uint32_t reps, uint32_t param, uint32_t zero, Rec* out
#define K3_LOCAL(name, P)                                                                       \
    template <int N>                                                                            \
    __global__ void __launch_bounds__(1024) name(K3_LARGS) {                                    \
        k3_local<P, N, false, kNoHint>(target, claim, reps, param, zero, out);                  \
    }
K3_LOCAL(k3_init, P_INIT)
K3_LOCAL(k3_inval, P_INVAL)
K3_LOCAL(k3_arrive, P_ARRIVE)
K3_LOCAL(k3_arrive_expect_tx, P_ARRIVE_TX)
K3_LOCAL(k3_expect_tx, P_EXPECT_TX)
K3_LOCAL(k3_complete_tx, P_COMPLETE_TX)
K3_LOCAL(k3_arrive_drop, P_DROP)
template <int N, bool C>
__global__ void __launch_bounds__(1024) k3_test_wait(K3_LARGS) { k3_local<P_TEST, N, C, kNoHint>(target, claim, reps, param, zero, out); }
template <int N>
__global__ void __launch_bounds__(1024) k3_try_wait(K3_LARGS) { k3_local<P_TRY, N, true, kNoHint>(target, claim, reps, param, zero, out); }
template <uint32_t H>
__global__ void __launch_bounds__(1024) k3_try_wait_inc(K3_LARGS) { k3_local<P_TRY, 1, false, H>(target, claim, reps, param, zero, out); }

// ---------------------------------------------------------------- wake-up of a waiting warp (block 64, param = X)
enum WForm { WF_TRY, WF_TEST };
template <int F, uint32_t H>
__device__ __forceinline__ void k3_wakeup_body(K3_LARGS) {
    __shared__ alignas(8) uint64_t s_bar;
    __shared__ uint64_t s_tarr;
    if (!claim_block(target, claim)) return;
    const uint32_t smid = read_smid(), b = pin_u32(smem_u32(&s_bar), zero);
    if (threadIdx.x == 0) mbar_init(b, 1);
    bar_sync0();
#pragma unroll 1
    for (uint32_t r = 0; r < reps; ++r) {
        uint64_t t_enter = 0, t_ret = 0;
        fence_acq_rel_gpu();  // untimed drain: a .gpu fence costs ~256 cycles more if any store was issued since the last fence
        bar_sync0();  // align the two warps
        if (threadIdx.x == 0) {
            const uint32_t par = (r & 1u) + zero;
            uint32_t ok;
            t_enter = read_clock64();
            do {
                ok = F == WF_TRY ? try_wait_u32<-1, H>(b, par) : test_wait_u32<-1>(b, par);
            } while (!ok);
            t_ret = read_clock64_after(ok, zero);
        } else if (threadIdx.x == 32) {
            spin_cycles(param);
            const uint64_t t_arr = read_clock64();
            mbar_arrive(b);   // completes phase r
            s_tarr = t_arr;   // read by thread 0 after the next bar.sync
        }
        bar_sync0();
        if (threadIdx.x == 0) out[r] = Rec{(uint32_t)(t_ret - s_tarr), 0u, (uint32_t)(t_ret - t_enter), smid};
    }
}
__global__ void __launch_bounds__(64) k3_wakeup_try(K3_LARGS) { k3_wakeup_body<WF_TRY, kNoHint>(target, claim, reps, param, zero, out); }
template <uint32_t H>
__global__ void __launch_bounds__(64) k3_wakeup_try_hint(K3_LARGS) { k3_wakeup_body<WF_TRY, H>(target, claim, reps, param, zero, out); }
__global__ void __launch_bounds__(64) k3_wakeup_test(K3_LARGS) { k3_wakeup_body<WF_TEST, kNoHint>(target, claim, reps, param, zero, out); }

// ---------------------------------------------------------------- remote (CS 2, one thread per CTA)
__device__ __forceinline__ uint32_t ctarank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ void csync_na() { cluster_arrive_release_na(); cluster_wait_acquire_na(); }

enum R3 { R_REL, R_RLX, R_LOCAL };
// rec[rank * reps + r]; rank 1 writes only its smid (rec[reps]).
template <int V, int N>
__device__ __forceinline__ void k3_remote_issue_body(uint32_t reps, uint32_t zero, Rec* rec) {
    extern __shared__ __align__(16) uint8_t dyn[];  // 200 KB, only reserved (1 CTA/SM)
    __shared__ alignas(8) uint64_t s_bar;
    if (threadIdx.x != 0) return;
    dyn[0] = 0;
    const uint32_t rank = ctarank(), smid = read_smid(), b = smem_u32(&s_bar);
    mbar_init(b, kNever);
    fence_mbarrier_init();
    csync_na();
    if (rank == 0) {
        const uint32_t rb = mapa_u32(b, 1);
#pragma unroll 1
        for (uint32_t r = 0; r < reps; ++r) {
            fence_acq_rel_gpu();  // untimed drain: a .gpu fence costs ~256 cycles more if any store was issued since the last fence
            spin_cycles(kRemoteSettle);
            uint64_t t0, t1;
            if constexpr (V == R_LOCAL) {
                uint64_t st = 0;
                t0 = read_clock64();
#pragma unroll
                for (int i = 0; i < N; ++i) st = mbar_arrive_release_cluster_state(b);
                t1 = read_clock64_after(st, zero);
            } else {
                t0 = read_clock64();
#pragma unroll
                for (int i = 0; i < N; ++i) {
                    if constexpr (V == R_REL) mbar_remote_arrive(rb);
                    else mbar_remote_arrive_relaxed(rb);
                }
                t1 = read_clock64();
            }
            const uint32_t c = (uint32_t)(t1 - t0);
            rec[r] = Rec{c, 0u, c, smid};
        }
    } else {
        rec[reps] = Rec{0u, 0u, 0u, smid};
    }
    csync_na();  // rank 1 stays resident until every remote arrive has been issued
}
#define K3_RARGS uint32_t reps, uint32_t zero, Rec* rec, uint64_t* gt
template <int N> __global__ void __launch_bounds__(32) k3_remote_arrive_release(K3_RARGS) { k3_remote_issue_body<R_REL, N>(reps, zero, rec); }
template <int N> __global__ void __launch_bounds__(32) k3_remote_arrive_relaxed(K3_RARGS) { k3_remote_issue_body<R_RLX, N>(reps, zero, rec); }
template <int N> __global__ void __launch_bounds__(32) k3_arrive_release_cluster_local(K3_RARGS) { k3_remote_issue_body<R_LOCAL, N>(reps, zero, rec); }

// One-way propagation of a remote arrive; gt[rank * reps + r] = g0 (rank 0) / g1 (rank 1).
template <bool kRelease>
__global__ void __launch_bounds__(32) k3_prop(K3_RARGS) {
    extern __shared__ __align__(16) uint8_t dyn[];
    __shared__ alignas(8) uint64_t s_bar;
    if (threadIdx.x != 0) return;
    dyn[0] = 0;
    const uint32_t rank = ctarank(), smid = read_smid(), b = smem_u32(&s_bar);
    mbar_init(b, 1);
    fence_mbarrier_init();
    csync_na();
    const uint32_t rb = mapa_u32(b, 1);
#pragma unroll 1
    for (uint32_t r = 0; r < reps; ++r) {
        fence_acq_rel_gpu();  // untimed drain: a .gpu fence costs ~256 cycles more if any store was issued since the last fence
        csync_na();
        if (rank == 0) {
            spin_cycles(kPropSpin);
            const uint64_t g0 = read_globaltimer();
            if constexpr (kRelease) mbar_remote_arrive(rb);
            else mbar_remote_arrive_relaxed(rb);
            gt[r] = g0;
            rec[r] = Rec{0u, 0u, 0u, smid};
        } else {
            const uint32_t par = (r & 1u) + zero;
            const uint64_t t_enter = read_clock64();
            uint32_t ok;
            do { ok = mbar_try_wait_parity_acq_cluster(b, par) ? 1u : 0u; } while (!ok);
            const TimerPair p = read_timers_after(ok, zero);
            gt[reps + r] = p.ns;
            const uint32_t c = (uint32_t)(p.cycles - t_enter);
            rec[reps + r] = Rec{c, 0u, c, smid};
        }
    }
    csync_na();
}

// ---------------------------------------------------------------- host
using LFn = void (*)(uint32_t, uint32_t*, uint32_t, uint32_t, uint32_t, Rec*);
using RFn = void (*)(uint32_t, uint32_t, Rec*, uint64_t*);

struct LCase { LFn k; std::string primitive, variant; int n_ops, bt, block; uint32_t param; };

static std::vector<LCase> local_cases(bool quick, bool ncu) {
    std::vector<LCase> v;
    auto add2 = [&](LFn k1, LFn k64, const char* prim) {
        v.push_back({k1, prim, "single", 1, 1, 32, 1});
        v.push_back({k64, prim, "unroll64", 64, 1, 32, 1});
    };
    add2(k3_init<1>, k3_init<64>, "mbar_init");
    add2(k3_inval<1>, k3_inval<64>, "mbar_inval");
    const std::vector<int> bts = quick ? std::vector<int>{1, 1024} : std::vector<int>{1, 32, 128, 512, 1024};
    for (int bt : bts) {
        v.push_back({k3_arrive<1>, "mbar_arrive", "single", 1, bt, std::max(32, bt), (uint32_t)bt});
        v.push_back({k3_arrive<64>, "mbar_arrive", "unroll64", 64, bt, std::max(32, bt), (uint32_t)bt});
    }
    add2(k3_arrive_expect_tx<1>, k3_arrive_expect_tx<64>, "mbar_arrive_expect_tx");
    add2(k3_expect_tx<1>, k3_expect_tx<64>, "mbar_expect_tx");
    add2(k3_complete_tx<1>, k3_complete_tx<64>, "mbar_complete_tx");
    add2(k3_arrive_drop<1>, k3_arrive_drop<64>, "mbar_arrive_drop");
    v.push_back({k3_test_wait<1, true>, "mbar_test_wait", "complete", 1, 1, 32, 1});
    v.push_back({k3_test_wait<1, false>, "mbar_test_wait", "incomplete", 1, 1, 32, 1});
    v.push_back({k3_test_wait<64, true>, "mbar_test_wait", "chain64_complete", 64, 1, 32, 1});
    v.push_back({k3_test_wait<64, false>, "mbar_test_wait", "chain64_incomplete", 64, 1, 32, 1});
    v.push_back({k3_try_wait<1>, "mbar_try_wait", "complete", 1, 1, 32, 1});
    v.push_back({k3_try_wait<64>, "mbar_try_wait", "chain64_complete", 64, 1, 32, 1});
    v.push_back({k3_try_wait_inc<kNoHint>, "mbar_try_wait", "incomplete_nohint", 1, 1, 32, 1});
    v.push_back({k3_try_wait_inc<0>, "mbar_try_wait", "incomplete_hint0", 1, 1, 32, 1});
    v.push_back({k3_try_wait_inc<100>, "mbar_try_wait", "incomplete_hint100", 1, 1, 32, 1});
    v.push_back({k3_try_wait_inc<1000>, "mbar_try_wait", "incomplete_hint1000", 1, 1, 32, 1});
    v.push_back({k3_try_wait_inc<10000>, "mbar_try_wait", "incomplete_hint10000", 1, 1, 32, 1});
    struct WF { LFn k; const char* name; };
    const std::vector<WF> forms = ncu ? std::vector<WF>{{k3_wakeup_try, "try_nohint"}, {k3_wakeup_try_hint<1000>, "try_hint1000"}}
                                      : std::vector<WF>{{k3_wakeup_try, "try_nohint"},
                                                        {k3_wakeup_try_hint<100>, "try_hint100"},
                                                        {k3_wakeup_try_hint<1000>, "try_hint1000"},
                                                        {k3_wakeup_try_hint<10000>, "try_hint10000"},
                                                        {k3_wakeup_test, "test_loop"}};
    const std::vector<uint32_t> xs = ncu ? std::vector<uint32_t>{20000} : quick ? std::vector<uint32_t>{1000}
                                                                                 : std::vector<uint32_t>{1000, 5000, 20000};
    for (const WF& f : forms)
        for (uint32_t x : xs) v.push_back({f.k, "mbar_wakeup", std::string(f.name) + "_x" + std::to_string(x), 1, 64, 64, x});
    return v;
}

struct Local {
    hb::SmClaim claim;
    uint32_t target;
    int blocks;
};

static bool local_launch(Local& L, const LCase& c, uint32_t reps, DevBuf<Rec>& rec, std::vector<Rec>& h) {
    L.claim.reset();
    c.k<<<L.blocks, c.block>>>(L.target, L.claim.ptr(), reps, c.param, 0u, rec.d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (!L.claim.ok(L.target)) return false;
    h = rec.get(reps);
    return true;
}

static void local_point(Out07& o, Local& L, const LCase& c, uint32_t reps, int launches) {
    DevBuf<Rec> rec(reps);
    std::vector<Rec> h;
    for (int w = 0; w < kWarmups; ++w) local_launch(L, c, reps, rec, h);
    const std::string label = "k3_" + c.primitive + "_" + c.variant + "_t" + std::to_string(c.bt);
    double sum = 0, sum_max = 0;
    for (int l = 0, tries = 0; l < launches;) {
        if (!local_launch(L, c, reps, rec, h)) {
            ++o.discarded;
            if (++tries > 20) hb::die(label + ": block never ran on the probe SM");
            continue;
        }
        const hb::ClockSample s = o.run.clock_note(label, "launch", o.li);
        o.prim_rows(s, "k3", c.primitive, 0, c.bt, c.variant, h, reps, 1, c.n_ops);
        o.flush();
        ++o.li;
        ++l;
        for (uint32_t r = 0; r < reps; ++r) { sum += h[r].c0; sum_max += h[r].cmax; }
    }
    o.sync();
    const double n = (double)launches * reps;
    std::printf("k3 %-22s %-22s threads %4d: %9.1f cycles/op (warp 0), %9.1f (cycles_warp_max)\n", c.primitive.c_str(),
                c.variant.c_str(), c.bt, sum / n / c.n_ops, sum_max / n / c.n_ops);
}

// Same format as Out07::prim_rows, for rows of one given rank.
static void prim_rows_rank(Out07& o, const hb::ClockSample& s, const std::string& primitive, int cs, int bt,
                           const std::string& variant, const Rec* recs, uint32_t reps, int rank, int n_ops) {
    const std::string pre = o.run.prefix(o.li, s);
    char b[256];
    for (uint32_t r = 0; r < reps; ++r) {
        const Rec& x = recs[r];
        std::snprintf(b, sizeof b, ",%s,%s,%d,%d,%s,%u,%u,%d,%u,%d,%u", "k3", primitive.c_str(), cs, bt, variant.c_str(),
                      r, x.c0, rank, x.smid, n_ops, x.cmax);
        o.prim.add(pre + b);
    }
}

struct RCase { RFn k; std::string primitive, variant; int n_ops; const char* prop; };  // prop: propagation primitive
static std::vector<RCase> remote_cases() {
    return {{k3_remote_arrive_release<1>, "mbar_remote_arrive_release", "single", 1, nullptr},
            {k3_remote_arrive_release<64>, "mbar_remote_arrive_release", "unroll64", 64, nullptr},
            {k3_remote_arrive_relaxed<1>, "mbar_remote_arrive_relaxed", "single", 1, nullptr},
            {k3_remote_arrive_relaxed<64>, "mbar_remote_arrive_relaxed", "unroll64", 64, nullptr},
            {k3_arrive_release_cluster_local<1>, "mbar_arrive_release_cluster_local", "single", 1, nullptr},
            {k3_arrive_release_cluster_local<64>, "mbar_arrive_release_cluster_local", "unroll64", 64, nullptr},
            {k3_prop<true>, "mbar_remote_wait", "release_spin2000", 1, "mbar_remote_arrive_release"},
            {k3_prop<false>, "mbar_remote_wait", "relaxed_spin2000", 1, "mbar_remote_arrive_relaxed"}};
}

static bool remote_launch(const RCase& c, uint32_t reps, DevBuf<Rec>& rec, DevBuf<uint64_t>& gt, std::vector<Rec>& h) {
    ClusterOnce co(2, 32);
    CUDA_CHECK(co.L.launch(c.k, reps, 0u, rec.d, gt.d));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    h = rec.get(2 * (size_t)reps);
    return ranks_distinct(h, reps, 2);
}

static void remote_point(Out07& o, const RCase& c, uint32_t reps, int launches) {
    DevBuf<Rec> rec(2 * (size_t)reps);
    DevBuf<uint64_t> gt(2 * (size_t)reps);
    std::vector<Rec> h;
    for (int w = 0; w < kWarmups; ++w) remote_launch(c, reps, rec, gt, h);
    const std::string label = "k3_" + c.primitive + "_" + c.variant + "_cs2";
    double sum = 0, sum_ns = 0;
    for (int l = 0, tries = 0; l < launches;) {
        if (!remote_launch(c, reps, rec, gt, h)) {
            ++o.discarded;
            if (++tries > 20) hb::die(label + ": ranks never on distinct SMs");
            continue;
        }
        const hb::ClockSample s = o.run.clock_note(label, "launch", o.li);
        if (c.prop) {  // rank 1's wait rows + propagation rows
            prim_rows_rank(o, s, c.primitive, 2, 1, c.variant, h.data() + reps, reps, 1, 1);
            const std::vector<uint64_t> g = gt.get(2 * (size_t)reps);
            const std::string pre = o.run.prefix(o.li, s);
            for (uint32_t r = 0; r < reps; ++r) {
                const long long d = (long long)g[reps + r] - (long long)g[r];
                o.prop_row(s, c.prop, 2, h[0].smid, h[reps].smid, d, -1, 1, "spin2000", r, pre);
                sum_ns += d;
                sum += h[reps + r].c0;
            }
        } else {
            o.prim_rows(s, "k3", c.primitive, 2, 1, c.variant, h, reps, 1, c.n_ops);
            for (uint32_t r = 0; r < reps; ++r) sum += h[r].c0;
        }
        o.flush();
        ++o.li;
        ++l;
    }
    o.sync();
    const double n = (double)launches * reps;
    if (c.prop)
        std::printf("k3 %-34s %-18s cs 2: rank-1 wait %8.1f cycles, one-way %7.1f ns\n", c.primitive.c_str(),
                    c.variant.c_str(), sum / n, sum_ns / n);
    else
        std::printf("k3 %-34s %-18s cs 2: %8.1f cycles/op\n", c.primitive.c_str(), c.variant.c_str(), sum / n / c.n_ops);
}

static Local* make_local(hb::Run& run) {
    Local* L = new Local;
    const int sm_count = run.dev().sm_count;
    L->target = (uint32_t)first_sm_of_gpc(load_gpc_map(sm_count), kProbeGpc);
    L->blocks = hb::SmClaim::blocks(sm_count);
    std::printf("probe SM %u (GPC %d)\n", L->target, kProbeGpc);
    return L;
}

static void run_k3(hb::Run& run, bool quick) {
    Out07 o(run, "k3");
    const uint32_t reps = quick ? kQuickReps : kReps;
    const int launches = quick ? kQuickLaunches : kLaunches;
    timer_floor(o, reps, launches);
    Local* L = make_local(run);
    for (const LCase& c : local_cases(quick, false)) local_point(o, *L, c, reps, launches);
    delete L;
    for (const RCase& c : remote_cases()) remote_point(o, c, reps, launches);
    std::printf("k3: %d launches discarded (claim failed or ranks not on distinct SMs)\n", o.discarded);
}

// One launch per configuration (no retries); manifest rows in launch order for post_ncu.py.
static void run_ncu(hb::Run& run) {
    hb::CsvWriter man("bin/ncu_manifest_k3.csv", hb::with_mandatory({"kernel", "primitive", "cs", "block_threads", "variant"}));
    long long li = 0;
    auto row = [&](const std::string& prim, int cs, int bt, const std::string& var) {
        const hb::ClockSample s = run.clock_note("ncu_k3_" + prim + "_" + var + "_t" + std::to_string(bt), "launch", li);
        man.add(run.prefix(li++, s) + ",k3," + prim + "," + std::to_string(cs) + "," + std::to_string(bt) + "," + var);
        man.flush();
    };
    Local* L = make_local(run);
    for (const LCase& c : local_cases(false, true)) {
        DevBuf<Rec> rec(kReps);
        std::vector<Rec> h;
        local_launch(*L, c, kReps, rec, h);
        row(c.primitive, 0, c.bt, c.variant);
    }
    delete L;
    for (const RCase& c : remote_cases()) {
        DevBuf<Rec> rec(2 * (size_t)kReps);
        DevBuf<uint64_t> gt(2 * (size_t)kReps);
        std::vector<Rec> h;
        remote_launch(c, kReps, rec, gt, h);
        row(c.primitive, 2, 1, c.variant);
    }
    man.sync();
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
