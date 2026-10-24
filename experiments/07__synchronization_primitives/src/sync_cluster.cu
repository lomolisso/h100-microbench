/*
 * sync_cluster.cu — experiment 07, K2: components of the cluster barrier (barrier.cluster.arrive / .wait) versus
 * cluster size and threads per CTA, and the one-way propagation of an arrive to the waiting CTAs.
 *
 * Usage: bin/sync_cluster {k2|ncu|clockcheck} [--quick] [--label L]. One invocation = one run_id.
 *
 * Placement: one cluster of CS ∈ {1, 2, 4, 8, 16} CTAs per launch (explicit cluster attribute also for CS 1, Default
 * scheduling policy), 200 KB dynamic smem so that every CTA has its own SM; rank r = %cluster_ctarank = blockIdx.x.
 * A launch whose ranks do not sit on distinct SMs is discarded and repeated (count printed and kept in the log).
 * Threads per CTA ("block_threads"): 1 (blockDim 32, lanes 1–31 exit at once, non-.aligned barrier forms), 32, 128,
 * 512, 1024 (all threads execute the .aligned forms). Every rep starts with an untimed cluster.sync() that aligns the
 * CTAs; then each warp times its own region with %clock64 (lane 0):
 *   k2_arrive_release   t0; barrier.cluster.arrive.release.aligned; t1; (untimed) wait.acquire
 *   k2_arrive_relaxed   the same with .relaxed (no MEMBAR/ERRBAR/CGAERRBAR before the arrive)
 *   k2_arrive_rel_st    every thread first stores one u32 to its own word of a global scratch buffer (st.global,
 *                       still in flight), then t0; arrive.release; t1 → the release arrive behind pending stores
 *   k2_wait_all         arrive.release; spin 8000 cycles (every CTA has arrived ≥ 5000 cycles earlier, given the
 *                       < 3000-cycle skew of the CTAs); t0; wait.acquire; t1 → the pure wait cost
 *   k2_wait_delay_rel   rank CS−1 spins D ∈ {0, 500, 2000} cycles, stamps %globaltimer, arrives; the other ranks
 *                       arrive at once; every rank: t0; wait.acquire; t1; non-delayed ranks stamp %globaltimer after
 *                       the wait returned. wait ≈ D + propagation − own arrive time.
 *   k2_wait_delay_rlx   the same with relaxed arrives, D = 2000 (propagation without the release fence)
 *   k2_sync             t0; arrive.release; wait.acquire; t1 (= cluster.sync())
 *   k2_split            t0; arrive.release; 512 dependent mad.lo (≈ 2000 cycles; alone: K1 primitive alu_chain);
 *                       wait.acquire; t1 → overlap of the arrive latency with independent work
 *   Per (CS, threads, variant): 2 unrecorded warm-up launches, then 5 recorded launches × 1024 reps.
 *   → results/07__sync__primitives__k2.csv (arrive, wait all_arrived, sync, split) and …__k2d.csv (the delay-D waits;
 *     split only to keep each file below ~600 MB): one row per (rep, rank): cycles = warp 0's t1 − t0,
 *     cycles_warp_max = the slowest warp of that CTA, smid, n_ops = 1. Timer-pair floor rows (kernel k0) in k2.csv.
 *   → results/07__sync__propagation.csv (k2_wait_delay_* with D = 2000, CS ≥ 2): one row per (rep, non-delayed rank):
 *     one_way_ns = %globaltimer after its wait − %globaltimer of rank CS−1 before its arrive (32-ns ticks),
 *     smid_src = rank CS−1, smid_dst = the waiting rank; primitive cluster_barrier_{release,relaxed}_arrive.
 * ncu mode: every variant once (1024 reps) at CS 2 and CS 16, 128 threads, for the stall sampling and opcode counts;
 *   manifest bin/ncu_manifest_k2.csv (one row per profiled launch, launch order).
 */
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "sync_common.hpp"

static const char* EXP = "07__synchronization_primitives";
constexpr uint32_t kWaitAllSpin = 8000, kAluN = 512;

enum K2V { V_ARRIVE_REL, V_ARRIVE_REL_ST, V_ARRIVE_RLX, V_WAIT_ALL, V_WAIT_DELAY_REL, V_WAIT_DELAY_RLX, V_SYNC, V_SPLIT };

template <bool A> __device__ __forceinline__ void arrive_rel() { if constexpr (A) cluster_arrive_release(); else cluster_arrive_release_na(); }
template <bool A> __device__ __forceinline__ void arrive_rlx() { if constexpr (A) cluster_arrive_relaxed(); else cluster_arrive_relaxed_na(); }
template <bool A> __device__ __forceinline__ void wait_acq()   { if constexpr (A) cluster_wait_acquire();   else cluster_wait_acquire_na(); }
template <bool A> __device__ __forceinline__ void csync()      { arrive_rel<A>(); wait_acq<A>(); }
__device__ __forceinline__ void st_global_u32(uint32_t* a, uint32_t v) { asm volatile("st.global.u32 [%0], %1;" :: "l"(a), "r"(v) : "memory"); }

// rec[rank * reps + r], gt[rank * reps + r] (propagation stamps; 0 where unused).
template <int V, bool A>
__device__ __forceinline__ void k2_body(uint32_t reps, uint32_t delay, uint32_t zero, Rec* rec, uint64_t* gt, uint32_t* scratch) {
    extern __shared__ __align__(16) uint8_t dyn[];  // 200 KB, only reserved (1 CTA/SM)
    __shared__ uint32_t s_w[64];
    if (!A && threadIdx.x != 0) return;
    uint32_t rank, cs;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(cs));
    const uint32_t smid = read_smid(), last = cs - 1;
    if (threadIdx.x == 0) dyn[0] = 0;
    csync<A>();
#pragma unroll 1
    for (uint32_t r = 0; r < reps; ++r) {
        csync<A>();
        uint64_t t0, t1, g = 0;
        uint32_t x = rank;
        if constexpr (V == V_ARRIVE_REL) {
            t0 = read_clock64(); arrive_rel<A>(); t1 = read_clock64(); wait_acq<A>();
        } else if constexpr (V == V_ARRIVE_REL_ST) {
            st_global_u32(scratch + rank * 1024u + threadIdx.x, r);
            t0 = read_clock64(); arrive_rel<A>(); t1 = read_clock64(); wait_acq<A>();
        } else if constexpr (V == V_ARRIVE_RLX) {
            t0 = read_clock64(); arrive_rlx<A>(); t1 = read_clock64(); wait_acq<A>();
        } else if constexpr (V == V_WAIT_ALL) {
            arrive_rel<A>(); spin_cycles(delay);
            t0 = read_clock64(); wait_acq<A>(); t1 = read_clock64();
        } else if constexpr (V == V_WAIT_DELAY_REL || V == V_WAIT_DELAY_RLX) {
            if (rank == last) { spin_cycles(delay); g = read_globaltimer(); }
            if constexpr (V == V_WAIT_DELAY_REL) arrive_rel<A>(); else arrive_rlx<A>();
            t0 = read_clock64(); wait_acq<A>(); t1 = read_clock64();
            if (rank != last) g = read_globaltimer();
        } else if constexpr (V == V_SYNC) {
            t0 = read_clock64(); arrive_rel<A>(); wait_acq<A>(); t1 = read_clock64();
        } else {  // V_SPLIT
            t0 = read_clock64(); arrive_rel<A>(); x = alu_chain(kAluN, x, zero); wait_acq<A>(); t1 = read_clock64();
        }
        block_record(s_w, (uint32_t)(t1 - t0), 0u, smid, rec, rank * reps + r, !A);
        if (threadIdx.x == 0) gt[rank * reps + r] = g + (x & zero);
    }
    csync<A>();  // no CTA exits while a peer may still be in the last barrier phase
}

#define K2_KERNEL(name, V)                                                                                       \
    template <bool A>                                                                                            \
    __global__ void __launch_bounds__(1024) name(uint32_t reps, uint32_t delay, uint32_t zero, Rec* rec, uint64_t* gt, \
                                                 uint32_t* scratch) {                                            \
        k2_body<V, A>(reps, delay, zero, rec, gt, scratch);                                                      \
    }
K2_KERNEL(k2_arrive_release, V_ARRIVE_REL)
K2_KERNEL(k2_arrive_rel_st, V_ARRIVE_REL_ST)
K2_KERNEL(k2_arrive_relaxed, V_ARRIVE_RLX)
K2_KERNEL(k2_wait_all, V_WAIT_ALL)
K2_KERNEL(k2_wait_delay_rel, V_WAIT_DELAY_REL)
K2_KERNEL(k2_wait_delay_rlx, V_WAIT_DELAY_RLX)
K2_KERNEL(k2_sync, V_SYNC)
K2_KERNEL(k2_split, V_SPLIT)

using K2Fn = void (*)(uint32_t, uint32_t, uint32_t, Rec*, uint64_t*, uint32_t*);
template <bool A>
static K2Fn k2_kernel(int v) {
    switch (v) {
        case V_ARRIVE_REL: return k2_arrive_release<A>;
        case V_ARRIVE_REL_ST: return k2_arrive_rel_st<A>;
        case V_ARRIVE_RLX: return k2_arrive_relaxed<A>;
        case V_WAIT_ALL: return k2_wait_all<A>;
        case V_WAIT_DELAY_REL: return k2_wait_delay_rel<A>;
        case V_WAIT_DELAY_RLX: return k2_wait_delay_rlx<A>;
        case V_SYNC: return k2_sync<A>;
        default: return k2_split<A>;
    }
}

struct K2Case { int v; const char* primitive; const char* variant; uint32_t delay; const char* prop; bool alt = false; };
static const K2Case kCases[] = {
    {V_ARRIVE_REL, "cluster_arrive_release", "single", 0, nullptr},
    {V_ARRIVE_REL_ST, "cluster_arrive_release", "after_st_global", 0, nullptr},
    {V_ARRIVE_RLX, "cluster_arrive_relaxed", "single", 0, nullptr},
    {V_WAIT_ALL, "cluster_wait_acquire", "all_arrived", kWaitAllSpin, nullptr},
    {V_WAIT_DELAY_REL, "cluster_wait_acquire", "delay0", 0, nullptr, true},
    {V_WAIT_DELAY_REL, "cluster_wait_acquire", "delay500", 500, nullptr, true},
    {V_WAIT_DELAY_REL, "cluster_wait_acquire", "delay2000", 2000, "cluster_barrier_release_arrive", true},
    {V_WAIT_DELAY_RLX, "cluster_wait_acquire", "delay2000_relaxed_arrive", 2000, "cluster_barrier_relaxed_arrive", true},
    {V_SYNC, "cluster_sync", "single", 0, nullptr},
    {V_SPLIT, "cluster_split_phase", "alu512", 0, nullptr},
};

// One launch; returns false if the ranks did not land on distinct SMs.
static bool k2_launch(const K2Case& c, int cs, int bt, uint32_t reps, DevBuf<Rec>& rec, DevBuf<uint64_t>& gt, std::vector<Rec>& h) {
    static DevBuf<uint32_t> scratch(16u * 1024u);  // one word per thread of a CS-16 cluster
    ClusterOnce co(cs, std::max(32, bt));
    const K2Fn k = bt == 1 ? k2_kernel<false>(c.v) : k2_kernel<true>(c.v);
    CUDA_CHECK(co.L.launch(k, reps, c.delay, 0u, rec.d, gt.d, scratch.d));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    h = rec.get((size_t)cs * reps);
    return ranks_distinct(h, reps, cs);
}

static void k2_point(Out07& o, const K2Case& c, int cs, int bt, uint32_t reps, int launches) {
    DevBuf<Rec> rec((size_t)cs * reps);
    DevBuf<uint64_t> gt((size_t)cs * reps);
    std::vector<Rec> h;
    for (int w = 0; w < kWarmups; ++w) k2_launch(c, cs, bt, reps, rec, gt, h);
    const std::string label = std::string("k2_") + c.primitive + "_" + c.variant + "_cs" + std::to_string(cs) + "_t" + std::to_string(bt);
    double sum = 0;
    for (int l = 0, tries = 0; l < launches;) {
        if (!k2_launch(c, cs, bt, reps, rec, gt, h)) {
            ++o.discarded;
            if (++tries > 20) hb::die(label + ": ranks never on distinct SMs");
            continue;
        }
        const hb::ClockSample s = o.run.clock_note(label, "launch", o.li);
        o.prim_rows(s, "k2", c.primitive, cs, bt, c.variant, h, reps, cs, 1, false, c.alt);
        if (c.prop && cs > 1) {
            const std::vector<uint64_t> g = gt.get((size_t)cs * reps);
            const std::string pre = o.run.prefix(o.li, s);
            const uint32_t last = cs - 1;
            for (uint32_t r = 0; r < reps; ++r)
                for (int k = 0; k < (int)last; ++k)
                    o.prop_row(s, c.prop, cs, h[(size_t)last * reps].smid, h[(size_t)k * reps].smid,
                               (long long)g[(size_t)k * reps + r] - (long long)g[(size_t)last * reps + r], -1, bt, c.variant, r, pre);
        }
        o.flush();
        ++o.li;
        ++l;
        for (uint32_t r = 0; r < reps; ++r) sum += h[r].c0;
    }
    o.sync();
    std::printf("k2 %-24s %-26s cs %2d threads %4d: %8.1f cycles (rank 0 mean)\n", c.primitive, c.variant, cs, bt,
                sum / ((double)launches * reps));
}

static void run_k2(hb::Run& run, bool quick) {
    Out07 o(run, "k2");
    o.open_alt("k2d");
    const uint32_t reps = quick ? kQuickReps : kReps;
    const int launches = quick ? kQuickLaunches : kLaunches;
    timer_floor(o, reps, launches);
    const std::vector<int> css = quick ? std::vector<int>{2, 16} : std::vector<int>{1, 2, 4, 8, 16};
    const std::vector<int> bts = quick ? std::vector<int>{1, 1024} : std::vector<int>{1, 32, 128, 512, 1024};
    for (int cs : css)
        for (int bt : bts)
            for (const K2Case& c : kCases) k2_point(o, c, cs, bt, reps, launches);
    std::printf("k2: %d launches discarded (ranks not on distinct SMs)\n", o.discarded);
}

// One launch per (variant, CS ∈ {2, 16}) at 128 threads; manifest rows in launch order for post_ncu.py.
static void run_ncu(hb::Run& run) {
    hb::CsvWriter man("bin/ncu_manifest_k2.csv", hb::with_mandatory({"kernel", "primitive", "cs", "block_threads", "variant"}));
    long long li = 0;
    for (int cs : {2, 16})
        for (const K2Case& c : kCases) {
            DevBuf<Rec> rec((size_t)cs * kReps);
            DevBuf<uint64_t> gt((size_t)cs * kReps);
            std::vector<Rec> h;
            k2_launch(c, cs, 128, kReps, rec, gt, h);
            const hb::ClockSample s = run.clock_note(std::string("ncu_k2_") + c.primitive + "_" + c.variant, "launch", li);
            man.add(run.prefix(li++, s) + ",k2," + c.primitive + "," + std::to_string(cs) + ",128," + c.variant);
            man.flush();
        }
    man.sync();
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
