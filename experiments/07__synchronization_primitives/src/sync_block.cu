/*
 * sync_block.cu — experiment 07, K1: cost of the intra-block synchronisation primitives (CTA barriers, warp sync,
 * memory and proxy fences, nanosleep) versus threads per block, issued once or 64 times back to back.
 *
 * Usage: bin/sync_block {k1|ncu|clockcheck} [--quick] [--label L]. One invocation = one run_id.
 *
 * Placement: one CTA on the probe SM = the first SM of GPC 1 of the 03 map (as in experiment 06). Every launch has
 * hb::SmClaim::blocks() = 2 × 132 blocks (all resident in one wave); thread 0 of the first block that runs on the
 * probe SM wins an atomicCAS claim and broadcasts it through shared memory, every other block returns at once. A
 * launch whose claim failed (no block on the probe SM) is discarded and repeated (count printed). No dynamic smem.
 * Threads per block ("block_threads") bt ∈ {32, 128, 512, 1024}, all threads active. Every rep starts with an untimed
 * bar.sync 0 that aligns the warps, followed by a volatile shared load that t0 depends on (bar.sync is
 * BAR.SYNC.DEFER_BLOCKING: an early warp would otherwise read t0 before the barrier completed); then each warp times
 * its own region with %clock64 (lane 0's value is used):
 *   t0; op (single, n_ops = 1) or 64 × op (unroll64, fully unrolled, n_ops = 64; cycles = the whole 64); t1.
 * Every rep begins with an untimed fence.acq_rel.gpu: the record store of the previous rep would otherwise make every
 * .gpu/.cluster fence pay the "store since the last fence" cost (measured: 264 vs 520 cycles for one thread, however
 * long ago the store was issued); single = no store since the last fence, after_st_global = one store just before.
 * For the CTA barriers (bar_sync0, bar_sync_named, and the consumer of the split) t1 depends on a volatile shared load
 * issued after the barrier(s), because a clock read can issue before a deferred barrier completes; k1_lds_volatile
 * times that load alone (primitive lds_volatile, the baseline to subtract).
 * Kernels (one per primitive, template <U = 1|64, ST = after_st_global>):
 *   k1_bar_sync0            bar.sync 0 (__syncthreads)
 *   k1_bar_sync_named       bar.sync 1, N with N ∈ {32, 128, 512, 1024} ≤ bt, executed by warps w < N/32 only; the
 *                           other warps skip the timed region (their cycles = 0). Variant single_n<N> / unroll64_n<N>.
 *   k1_bar_arrive_split     bt = 64 (2 warps), single only: warp 1: t0; bar.arrive 1, 64; t1 (producer);
 *                           warp 0: spin 2000 cycles (warp 1 has arrived), t0; bar.sync 1, 64; t1 (consumer).
 *   k1_syncwarp             bar.warp.sync -1 (__syncwarp)
 *   k1_fence_*              fence.acq_rel.cta, fence.sc.cta, fence.acq_rel.cluster, fence.sc.cluster,
 *                           fence.acq_rel.gpu, fence.sc.gpu, __threadfence(), fence.proxy.async (all state spaces),
 *                           fence.proxy.async.shared::cta, fence.proxy.async.global, fence.mbarrier_init.release.cluster
 *                           Extra variant after_st_global (single; acq_rel/sc cta, acq_rel cluster, acq_rel/sc gpu,
 *                           threadfence): every thread stores one u32 (st.global.u32) to its own word of its warp's
 *                           128-B line right before t0 → the fence behind one pending store per thread.
 *   k1_nanosleep            nanosleep.u32 ns, ns ∈ {0, 100, 1000}: variant single_ns<ns> / unroll64_ns<ns>.
 *   k1_lds_volatile         t0; ld.volatile.shared; t1 after its value (baseline of the barrier rows), single only
 *   k1_alu_chain            alu_chain(512): 512 dependent mad.lo.u32 (64 × 8 unrolled, the same code as K2's
 *                           split-phase work), t1 read after the chain's result; single, n_ops = 512, variant n512.
 *   Per (primitive, variant, bt): 2 unrecorded warm-up launches, then 5 recorded launches × 1024 reps.
 *   → results/07__sync__primitives__k1.csv: one row per rep: kernel k1, cs 1, rank 0, cycles = warp 0's t1 − t0
 *     (k1_bar_arrive_split: primitive bar_arrive = warp 1's value, bar_sync_after_arrive = warp 0's, both from the
 *     same launch, bt 64), cycles_warp_max = the slowest warp of the block, smid, n_ops. Timer-pair floor rows
 *     (kernel k0) from k_timer_floor.
 * ncu mode: every K1 configuration at bt = 128 (split at 64) once (1024 reps); manifest bin/ncu_manifest_k1.csv (one
 *   row per profiled launch, launch order). A failed claim is not repeated there (every launch is profiled).
 * Quick mode (--quick): 64 reps, 2 recorded launches, bt ∈ {32, 1024}, every primitive.
 * SASS between the clock reads (results/sass/sync_block.sass): one instruction (group) per op, plus at most one
 * IMAD.MOV of t0; bar.warp.sync and fence.mbarrier_init.release.cluster compile to NOP, the .cluster fences to the
 * same MEMBAR.*.GPU + ERRBAR + CGAERRBAR + CCTL.IVALL as the .gpu fences.
 */
#include <string>
#include <utility>
#include <vector>

#include "sync_common.hpp"

static const char* EXP = "07__synchronization_primitives";
constexpr uint32_t kAluN = 512, kSplitSpin = 2000, kGbufWords = 1024;  // gbuf: 32 warps × one 128-B line

// PTX forms not in the harness (all with a "memory" clobber so that nothing moves across them).
static __device__ __forceinline__ void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
static __device__ __forceinline__ void nanosleep_mem(uint32_t ns) { asm volatile("nanosleep.u32 %0;" :: "r"(ns) : "memory"); }
static __device__ __forceinline__ void st_global_u32(uint32_t* a, uint32_t v) {
    asm volatile("st.global.u32 [%0], %1;" :: "l"(a), "r"(v) : "memory");
}

// Block-wide claim of SM `target` (claim_target_sm is per warp): thread 0 claims, the result reaches every thread.
static __device__ __forceinline__ bool claim_block(uint32_t target, uint32_t* claim, uint32_t* s_won) {
    if (threadIdx.x == 0) {
        uint32_t won = 0;
        if (read_smid() == target && atomicCAS(claim, 0u, blockIdx.x + 1) == 0u) {
            claim[1] = target;
            won = 1;
        }
        *s_won = won;
    }
    bar_sync0();
    return *s_won != 0;
}

enum K1P {
    P_BAR0, P_BARN, P_SYNCWARP, P_F_AR_CTA, P_F_SC_CTA, P_F_AR_CLU, P_F_SC_CLU, P_F_AR_GPU, P_F_SC_GPU, P_THREADFENCE,
    P_F_PROXY_ASYNC, P_F_PROXY_SH, P_F_PROXY_GL, P_F_MBAR_INIT, P_NANOSLEEP, P_ALU, P_LDS
};
// Barriers are timed up to a dependent ld.volatile.shared after them (P_LDS: the same load alone, the baseline).
template <int P> constexpr bool kEndsWithLoad = P == P_BAR0 || P == P_BARN || P == P_LDS;

// The timed operation; `param` = N (named barrier) or ns (nanosleep).
template <int P>
__device__ __forceinline__ void k1_op(uint32_t param) {
    if constexpr (P == P_BAR0) bar_sync0();
    else if constexpr (P == P_BARN) bar_sync_n(1, param);
    else if constexpr (P == P_SYNCWARP) bar_warp_sync();
    else if constexpr (P == P_F_AR_CTA) fence_acq_rel_cta();
    else if constexpr (P == P_F_SC_CTA) fence_sc_cta();
    else if constexpr (P == P_F_AR_CLU) fence_acq_rel_cluster();
    else if constexpr (P == P_F_SC_CLU) fence_sc_cluster();
    else if constexpr (P == P_F_AR_GPU) fence_acq_rel_gpu();
    else if constexpr (P == P_F_SC_GPU) fence_sc_gpu();
    else if constexpr (P == P_THREADFENCE) __threadfence();
    else if constexpr (P == P_F_PROXY_ASYNC) fence_proxy_async_all();
    else if constexpr (P == P_F_PROXY_SH) fence_proxy_async_shared_cta();
    else if constexpr (P == P_F_PROXY_GL) fence_proxy_async_global();
    else if constexpr (P == P_F_MBAR_INIT) fence_mbarrier_init();
    else if constexpr (P == P_NANOSLEEP) nanosleep_mem(param);
}

// rec[r]: one Rec per rep (warp 0's cycles, slowest warp, smid).
template <int P, int U, bool ST>
__device__ __forceinline__ void k1_body(uint32_t reps, uint32_t target, uint32_t* claim, uint32_t param, uint32_t zero,
                                        uint32_t* gbuf, Rec* rec) {
    __shared__ uint32_t s_w[64], s_won, s_probe;
    if (!claim_block(target, claim, &s_won)) return;
    const uint32_t smid = read_smid(), warp = threadIdx.x >> 5, probe = smem_u32(&s_probe);
    if (threadIdx.x == 0) s_probe = 0;
    // named barrier: only the N/32 first warps take part. The shuffled warp index makes the branch provably
    // warp-uniform (on a plain threadIdx-derived condition ptxas puts WARPSYNC.ALL + NOP between t0 and the BAR).
    const bool active = P != P_BARN || __shfl_sync(0xffffffffu, warp, 0) < param / 32;
    uint32_t* const my = gbuf + threadIdx.x;  // after_st_global: own word of the warp's 128-B line
    // N / ns in a register computed once (ptxas otherwise re-loads the kernel parameter with an LDC inside the timed
    // region); zero = 0 at run time.
    const uint32_t p = param + (smid & zero);
    bar_sync0();
#pragma unroll 1
    for (uint32_t r = 0; r < reps; ++r) {
        fence_acq_rel_gpu();  // untimed drain: a .gpu fence costs ~256 cycles more if any store was issued since the last fence
        // align the warps (untimed); t0 after the barrier has completed (bar_sync0_then_lds, sync_common.hpp)
        const uint32_t dep = bar_sync0_then_lds(probe);
        uint32_t cyc = 0;
        if (active) {
            if constexpr (ST) st_global_u32(my, r);
            uint64_t t0, t1;
            if constexpr (P == P_ALU) {
                uint32_t x = r + threadIdx.x;
                t0 = read_clock64_after(dep, zero);
                x = alu_chain(kAluN, x, zero);
                t1 = read_clock64_after(x, zero);
            } else {
                t0 = read_clock64_after(dep, zero);
#pragma unroll
                for (int u = 0; u < U; ++u) k1_op<P>(p);
                if constexpr (kEndsWithLoad<P>) t1 = read_clock64_after(lds_volatile_u32(probe), zero);
                else t1 = read_clock64();
            }
            cyc = (uint32_t)(t1 - t0);
        }
        block_record(s_w, cyc, 0u, smid, rec, r, false);
    }
}

#define K1_ARGS uint32_t reps, uint32_t target, uint32_t* claim, uint32_t param, uint32_t zero, uint32_t* gbuf, Rec* rec
#define K1_KERNEL(name, P)                                                                    \
    template <int U, bool ST>                                                                 \
    __global__ void __launch_bounds__(1024) name(K1_ARGS) {                                   \
        k1_body<P, U, ST>(reps, target, claim, param, zero, gbuf, rec);                       \
    }
K1_KERNEL(k1_bar_sync0, P_BAR0)
K1_KERNEL(k1_bar_sync_named, P_BARN)
K1_KERNEL(k1_syncwarp, P_SYNCWARP)
K1_KERNEL(k1_fence_acq_rel_cta, P_F_AR_CTA)
K1_KERNEL(k1_fence_sc_cta, P_F_SC_CTA)
K1_KERNEL(k1_fence_acq_rel_cluster, P_F_AR_CLU)
K1_KERNEL(k1_fence_sc_cluster, P_F_SC_CLU)
K1_KERNEL(k1_fence_acq_rel_gpu, P_F_AR_GPU)
K1_KERNEL(k1_fence_sc_gpu, P_F_SC_GPU)
K1_KERNEL(k1_threadfence, P_THREADFENCE)
K1_KERNEL(k1_fence_proxy_async, P_F_PROXY_ASYNC)
K1_KERNEL(k1_fence_proxy_async_shared_cta, P_F_PROXY_SH)
K1_KERNEL(k1_fence_proxy_async_global, P_F_PROXY_GL)
K1_KERNEL(k1_fence_mbarrier_init, P_F_MBAR_INIT)
K1_KERNEL(k1_nanosleep, P_NANOSLEEP)
K1_KERNEL(k1_alu_chain, P_ALU)
K1_KERNEL(k1_lds_volatile, P_LDS)

// Producer/consumer split on named barrier 1 (bt = 64). Each warp passes its own cycles also as c1, so that
// Rec::c0 = warp 0 (bar.sync after the arrive) and Rec::c1 = warp 1 (bar.arrive).
__global__ void __launch_bounds__(1024) k1_bar_arrive_split(K1_ARGS) {
    __shared__ uint32_t s_w[64], s_won, s_probe;
    if (!claim_block(target, claim, &s_won)) return;
    const uint32_t smid = read_smid(), warp = threadIdx.x >> 5, probe = smem_u32(&s_probe);
    if (threadIdx.x == 0) s_probe = 0;
    bar_sync0();
#pragma unroll 1
    for (uint32_t r = 0; r < reps; ++r) {
        fence_acq_rel_gpu();  // untimed drain: a .gpu fence costs ~256 cycles more if any store was issued since the last fence
        const uint32_t dep = bar_sync0_then_lds(probe);
        uint64_t t0, t1;
        // shuffled warp index: warp-uniform branch, no WARPSYNC in the timed region (see k1_body)
        if (__shfl_sync(0xffffffffu, warp, 0) == 1) {
            t0 = read_clock64_after(dep, zero); bar_arrive_n(1, 64); t1 = read_clock64();
        } else {
            spin_cycles(kSplitSpin);
            t0 = read_clock64_after(dep, zero); bar_sync_n(1, 64); t1 = read_clock64_after(lds_volatile_u32(probe), zero);
        }
        const uint32_t cyc = (uint32_t)(t1 - t0);
        block_record(s_w, cyc, cyc, smid, rec, r, false);
    }
    (void)param; (void)gbuf;
}

using K1Fn = void (*)(K1_ARGS);

// ---------------------------------------------------------------- configurations
enum ParamKind { PK_NONE, PK_NAMED, PK_NS };
struct K1Prim { const char* primitive; K1Fn single, unroll, after_st; ParamKind pk; };
#define K1_FNS(k) k<1, false>, k<64, false>
static const K1Prim kPrims[] = {
    {"bar_sync0", K1_FNS(k1_bar_sync0), nullptr, PK_NONE},
    {"bar_sync_named", K1_FNS(k1_bar_sync_named), nullptr, PK_NAMED},
    {"syncwarp", K1_FNS(k1_syncwarp), nullptr, PK_NONE},
    {"fence_acq_rel_cta", K1_FNS(k1_fence_acq_rel_cta), k1_fence_acq_rel_cta<1, true>, PK_NONE},
    {"fence_sc_cta", K1_FNS(k1_fence_sc_cta), k1_fence_sc_cta<1, true>, PK_NONE},
    {"fence_acq_rel_cluster", K1_FNS(k1_fence_acq_rel_cluster), k1_fence_acq_rel_cluster<1, true>, PK_NONE},
    {"fence_sc_cluster", K1_FNS(k1_fence_sc_cluster), nullptr, PK_NONE},
    {"fence_acq_rel_gpu", K1_FNS(k1_fence_acq_rel_gpu), k1_fence_acq_rel_gpu<1, true>, PK_NONE},
    {"fence_sc_gpu", K1_FNS(k1_fence_sc_gpu), k1_fence_sc_gpu<1, true>, PK_NONE},
    {"threadfence", K1_FNS(k1_threadfence), k1_threadfence<1, true>, PK_NONE},
    {"fence_proxy_async", K1_FNS(k1_fence_proxy_async), nullptr, PK_NONE},
    {"fence_proxy_async_shared_cta", K1_FNS(k1_fence_proxy_async_shared_cta), nullptr, PK_NONE},
    {"fence_proxy_async_global", K1_FNS(k1_fence_proxy_async_global), nullptr, PK_NONE},
    {"fence_mbarrier_init", K1_FNS(k1_fence_mbarrier_init), nullptr, PK_NONE},
    {"nanosleep", K1_FNS(k1_nanosleep), nullptr, PK_NS},
};

struct K1Cfg { std::string primitive, variant; K1Fn fn; int bt; uint32_t param; int n_ops; bool split; };

static std::vector<K1Cfg> k1_configs(const std::vector<int>& bts) {
    std::vector<K1Cfg> v;
    for (int bt : bts) {
        for (const K1Prim& p : kPrims) {
            std::vector<std::pair<uint32_t, std::string>> params = {{0u, ""}};
            if (p.pk == PK_NAMED) {
                params.clear();
                for (uint32_t n : {32u, 128u, 512u, 1024u})
                    if ((int)n <= bt) params.push_back({n, "_n" + std::to_string(n)});
            } else if (p.pk == PK_NS) {
                params = {{0u, "_ns0"}, {100u, "_ns100"}, {1000u, "_ns1000"}};
            }
            for (const auto& q : params) {
                v.push_back({p.primitive, "single" + q.second, p.single, bt, q.first, 1, false});
                v.push_back({p.primitive, "unroll64" + q.second, p.unroll, bt, q.first, 64, false});
            }
            if (p.after_st) v.push_back({p.primitive, "after_st_global", p.after_st, bt, 0u, 1, false});
        }
        v.push_back({"alu_chain", "n512", k1_alu_chain<1, false>, bt, 0u, (int)kAluN, false});
        v.push_back({"lds_volatile", "single", k1_lds_volatile<1, false>, bt, 0u, 1, false});
    }
    v.push_back({"bar_arrive_split", "single", k1_bar_arrive_split, 64, 0u, 1, true});
    return v;
}

struct K1Ctx {
    hb::Run& run;
    hb::SmClaim claim;
    DevBuf<uint32_t> gbuf;
    DevBuf<Rec> rec;
    uint32_t target;
    int blocks;
    K1Ctx(hb::Run& r, uint32_t reps)
        : run(r), gbuf(kGbufWords), rec(reps), target((uint32_t)first_sm_of_gpc(load_gpc_map(r.dev().sm_count), 1)),
          blocks(hb::SmClaim::blocks(r.dev().sm_count)) {}

    // One launch; returns false if no block claimed the probe SM.
    bool launch(const K1Cfg& c, uint32_t reps, std::vector<Rec>& h) {
        claim.reset();
        c.fn<<<blocks, c.bt>>>(reps, target, claim.ptr(), c.param, 0u, gbuf.d, rec.d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        if (!claim.ok(target)) return false;
        h = rec.get(reps);
        if (h[0].smid != target) hb::die("claimed block reported smid " + std::to_string(h[0].smid));
        return true;
    }
};

static void k1_point(Out07& o, K1Ctx& x, const K1Cfg& c, uint32_t reps, int launches) {
    std::vector<Rec> h;
    for (int w = 0; w < kWarmups; ++w) x.launch(c, reps, h);
    const std::string label = "k1_" + c.primitive + "_" + c.variant + "_t" + std::to_string(c.bt);
    double sum = 0, sum1 = 0;
    for (int l = 0, tries = 0; l < launches;) {
        if (!x.launch(c, reps, h)) {
            ++o.discarded;
            if (++tries > 20) hb::die(label + ": no block on the probe SM");
            continue;
        }
        const hb::ClockSample s = o.run.clock_note(label, "launch", o.li);
        if (c.split) {
            o.prim_rows(s, "k1", "bar_arrive", 1, c.bt, c.variant, h, reps, 1, c.n_ops, true);
            o.prim_rows(s, "k1", "bar_sync_after_arrive", 1, c.bt, c.variant, h, reps, 1, c.n_ops, false);
        } else {
            o.prim_rows(s, "k1", c.primitive, 1, c.bt, c.variant, h, reps, 1, c.n_ops);
        }
        o.flush();
        ++o.li;
        ++l;
        for (uint32_t r = 0; r < reps; ++r) { sum += h[r].c0; sum1 += h[r].c1; }
    }
    o.sync();
    const double n = (double)launches * reps;
    if (c.split)
        std::printf("k1 %-28s %-16s threads %4d: arrive %8.1f, sync after arrive %8.1f cycles (mean)\n",
                    c.primitive.c_str(), c.variant.c_str(), c.bt, sum1 / n, sum / n);
    else
        std::printf("k1 %-28s %-16s threads %4d: %9.1f cycles (warp 0 mean), %8.2f per op\n", c.primitive.c_str(),
                    c.variant.c_str(), c.bt, sum / n, sum / n / c.n_ops);
}

static void run_k1(hb::Run& run, bool quick) {
    Out07 o(run, "k1");
    const uint32_t reps = quick ? kQuickReps : kReps;
    const int launches = quick ? kQuickLaunches : kLaunches;
    timer_floor(o, reps, launches);
    K1Ctx x(run, reps);
    std::printf("k1: probe SM %u (first SM of GPC 1)\n", x.target);
    const std::vector<int> bts = quick ? std::vector<int>{32, 1024} : std::vector<int>{32, 128, 512, 1024};
    for (const K1Cfg& c : k1_configs(bts)) k1_point(o, x, c, reps, launches);
    std::printf("k1: %d launches discarded (no block on the probe SM)\n", o.discarded);
}

// One launch per configuration at 128 threads (split: 64); manifest rows in launch order for post_ncu.py.
static void run_ncu(hb::Run& run) {
    hb::CsvWriter man("bin/ncu_manifest_k1.csv", hb::with_mandatory({"kernel", "primitive", "cs", "block_threads", "variant"}));
    K1Ctx x(run, kReps);
    long long li = 0;
    for (const K1Cfg& c : k1_configs({128})) {
        std::vector<Rec> h;
        if (!x.launch(c, kReps, h)) std::printf("ncu: claim failed for %s %s (launch kept)\n", c.primitive.c_str(), c.variant.c_str());
        const hb::ClockSample s = run.clock_note("ncu_k1_" + c.primitive + "_" + c.variant, "launch", li);
        man.add(run.prefix(li++, s) + ",k1," + c.primitive + ",1," + std::to_string(c.bt) + "," + c.variant);
        man.flush();
    }
    man.sync();
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "k1";
    hb::Run run(EXP, argc, argv);
    if (mode == "clockcheck") run.lock_clocks_note(hb::arg_str(argc, argv, "--label", "check"), "check");
    else if (mode == "k1") run_k1(run, hb::has_flag(argc, argv, "--quick"));
    else if (mode == "ncu") run_ncu(run);
    else hb::die("unknown mode " + mode);
    run.complete();
    return 0;
}
