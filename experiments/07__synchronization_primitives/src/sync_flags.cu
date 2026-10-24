/*
 * sync_flags.cu — experiment 07, K4: round trip and one-way latency of every way of signalling "ready" between the
 * two CTAs of a cluster (flag ping-pong matrix), with a global-memory flag and cluster.sync() as references.
 *
 * Usage: bin/sync_flags {k4|ncu|clockcheck} [--quick] [--label L]. One invocation = one run_id.
 *
 * Placement: one cluster of CS CTAs per launch (ClusterOnce: explicit cluster attribute, Default policy, 200 KB
 * dynamic smem → 1 CTA/SM), one warp per CTA. The active pair is rank 0 and rank p (kernel argument); placement sweep
 * (CS, p) ∈ {(2, 1), (4, 1), (4, 2)}: a lone CS-2 cluster lands on SMs 124/125 (the special TPC that never joins larger
 * clusters); in CS 4 (regular GPCs) ranks 0/1 are TPC mates and ranks 0/2 sit on different TPCs of one GPC. Ranks not in
 * {0, p} take part only in the setup/teardown cluster.sync() and, for cluster_sync, in every barrier of the loop (the
 * same number of barriers as the pair). A launch whose CS ranks do not sit on distinct SMs is discarded and repeated.
 * Lane 0 does all signalling (lanes 1–31 wait at __syncwarp); only cluster_sync runs on the whole warp (aligned barrier).
 * Setup: each CTA zeroes its flag/data words, inits its mbarrier (count 1; st.async/red.async: pre-armed with
 * arrive.expect_tx(4)), fence.mbarrier_init, cluster.sync(). Teardown: cluster.sync() before any CTA exits.
 * Push model: the signaller writes into the PEER's smem; the waiter polls its OWN word through the .shared::cluster
 * address of its own rank (mapa(addr, own rank)) with the method's load form.
 * Iteration i (epoch e = i + 1), reps = 1024 (64 with --quick):
 *   rank 0: t0 = %clock64; g_send = %globaltimer; signal(rank p, e); wait(own, e); t1 = %clock64
 *   rank p: wait(own, e); g_recv = %globaltimer; [re-arm]; signal(rank 0, e)
 *   round trip = t1 − t0 (rank 0's SM clock; includes one %globaltimer read), one-way = g_recv − g_send (32-ns ticks).
 * Methods (one named kernel each; primitive name in the CSV):
 *   k4_st_ld_volatile     flag_st_ld_volatile             st.shared::cluster e → peer flag; spin ld.volatile.shared::cluster own flag
 *   k4_pull_ld_volatile   flag_pull_ld_volatile ("pull")  st.volatile.shared::cta e → OWN flag; spin ld.volatile.shared::cluster PEER flag
 *   k4_release_acquire    flag_release_acquire            st.release.cluster; spin ld.acquire.cluster
 *   k4_relaxed_fence      flag_relaxed_fence              fence.acq_rel.cluster; st.relaxed.cluster | spin ld.relaxed.cluster; fence.acq_rel.cluster
 *   k4_red_release        flag_red_release                red.release.cluster.shared::cluster.add 1 → peer counter; spin ld.acquire.cluster == e
 *   k4_atom_exch          flag_atom_exch                  atom.acq_rel.cluster.shared::cluster.exch e (old value unused); spin ld.acquire.cluster
 *   k4_st_async           st_async_mbarrier               st.async e → peer data, complete_tx(4) on the peer's mbarrier;
 *                                                         loop try_wait.parity.acquire.cluster (parity i & 1), then ld.shared data;
 *                                                         t1 waits for the data load (rank 0); re-arm arrive.expect_tx(4)
 *                                                         after the wait (rank p: before signalling back; rank 0: after t1)
 *   k4_red_async          red_async_mbarrier              red.async add 1 → peer data counter (== e), same barrier protocol
 *   k4_arrive_release     remote_mbarrier_arrive_release  mbarrier.arrive.release.cluster.shared::cluster on the peer's
 *                                                         barrier (count 1, no tx); try_wait.parity.acquire.cluster loop
 *   k4_arrive_relaxed     remote_mbarrier_arrive_relaxed  the same with .relaxed.cluster
 *   k4_gmem_flag          gmem_flag_release_acquire       st.release.gpu / ld.acquire.gpu spin on global flags; rank 0's flag at
 *                                                         base + off, rank p's at base + off + 128, variants off0 / off4096 (base 8-KiB aligned: the
 *                                                         two 4-KiB pages have opposite L2 home partitions); flags reset per launch
 *   k4_cluster_sync       cluster_sync (32 threads)       rank 0: t0; g_send; cluster.sync(); cluster.sync(); t1;
 *                                                         rank p: cluster.sync(); g_recv; cluster.sync()
 * Every poll loop is capped at 2^26 polls: a timeout sets an error count and ends the launch; data mismatches
 * (st.async/red.async word ≠ e after the wait) are counted too; the host dies on either.
 * Per (placement, method): 2 unrecorded warm-up launches, then 5 recorded launches (2 with --quick).
 *   → results/07__sync__propagation.csv: one row per iteration: primitive, cs (2 or 4), smid_src = rank 0, smid_dst =
 *     rank p, one_way_ns, round_trip_cycles, block_threads (1; 32 for cluster_sync), variant = method variant (push /
 *     pull / off0 / off4096) + "_r0<p>" (e.g. push_r01, off4096_r02), rep = i. Per-iteration values are buffered in __shared__ and copied to global memory after the teardown.
 *   → results/07__sync__primitives__k4.csv: the k0 timer-floor rows of this run (k_timer_floor).
 * ncu mode: every (placement, method) once (1024 reps); manifest bin/ncu_manifest_k4.csv (one row per profiled launch, in order).
 */
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "sync_common.hpp"

static const char* EXP = "07__synchronization_primitives";
constexpr uint32_t kSpinCap = 1u << 26;

enum K4M { M_ST_LD_VOL, M_PULL_VOL, M_REL_ACQ, M_RLX_FENCE, M_RED_REL, M_ATOM_EXCH, M_ST_ASYNC, M_RED_ASYNC,
           M_ARRIVE_REL, M_ARRIVE_RLX, M_GMEM, M_CSYNC };

// Local forms not in the harness.
__device__ __forceinline__ void st_volatile_cta_u32(uint32_t a, uint32_t v) {
    asm volatile("st.volatile.shared::cta.u32 [%0], %1;" :: "r"(a), "r"(v) : "memory"); }
__device__ __forceinline__ uint32_t ld_shared_cta_u32(uint32_t a) {
    uint32_t v; asm volatile("ld.shared::cta.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory"); return v; }

// Addresses of one CTA: own / peer words (.shared::cluster unless noted) and global flags.
struct K4Addr {
    uint32_t own_flag, peer_flag;   // flag (or red counter) word, own rank / peer rank
    uint32_t own_flag_cta;          // own flag, .shared::cta (pull signal)
    uint32_t own_data_cta, peer_data;
    uint32_t own_bar, peer_bar;     // own: .shared::cta; peer: .shared::cluster
    uint32_t *gown, *gpeer;         // global flags
};

template <int M> constexpr bool kHasData = (M == M_ST_ASYNC || M == M_RED_ASYNC);
template <int M> constexpr bool kBarWait = (M == M_ST_ASYNC || M == M_RED_ASYNC || M == M_ARRIVE_REL || M == M_ARRIVE_RLX);

template <int M>
__device__ __forceinline__ void k4_signal(const K4Addr& a, uint32_t e) {
    if constexpr (M == M_ST_LD_VOL) st_dsmem_u32(a.peer_flag, e);
    else if constexpr (M == M_PULL_VOL) st_volatile_cta_u32(a.own_flag_cta, e);
    else if constexpr (M == M_REL_ACQ) st_release_cluster_u32(a.peer_flag, e);
    else if constexpr (M == M_RLX_FENCE) { fence_acq_rel_cluster(); st_relaxed_cluster_u32(a.peer_flag, e); }
    else if constexpr (M == M_RED_REL) red_release_cluster_add_u32(a.peer_flag, 1u);
    else if constexpr (M == M_ATOM_EXCH) (void)atom_exch_acq_rel_cluster_u32(a.peer_flag, e);
    else if constexpr (M == M_ST_ASYNC) st_async_u32(a.peer_data, e, a.peer_bar);
    else if constexpr (M == M_RED_ASYNC) red_async_add_u32(a.peer_data, 1u, a.peer_bar);
    else if constexpr (M == M_ARRIVE_REL) mbar_remote_arrive(a.peer_bar);
    else if constexpr (M == M_ARRIVE_RLX) mbar_remote_arrive_relaxed(a.peer_bar);
    else if constexpr (M == M_GMEM) st_release_gpu_u32(a.gpeer, e);
}

// Returns false on a timeout (kSpinCap polls). `v` = the data word (st.async / red.async), else e.
template <int M>
__device__ __forceinline__ bool k4_wait(const K4Addr& a, uint32_t e, uint32_t i, uint32_t& v) {
    v = e;
    if constexpr (kBarWait<M>) {
        bool done = false;
#pragma unroll 1
        for (uint32_t n = 0; n < kSpinCap && !done; ++n) done = mbar_try_wait_parity_acq_cluster(a.own_bar, i & 1u);
        if (!done) return false;
        if constexpr (kHasData<M>) v = ld_shared_cta_u32(a.own_data_cta);
        return true;
    } else {
#pragma unroll 1
        for (uint32_t n = 0; n < kSpinCap; ++n) {
            uint32_t x;
            if constexpr (M == M_ST_LD_VOL) x = ld_volatile_cluster_u32(a.own_flag);
            else if constexpr (M == M_PULL_VOL) x = ld_volatile_cluster_u32(a.peer_flag);
            else if constexpr (M == M_RLX_FENCE) x = ld_relaxed_cluster_u32(a.own_flag);
            else if constexpr (M == M_GMEM) x = ld_acquire_gpu_u32(a.gown);
            else x = ld_acquire_cluster_u32(a.own_flag);  // release_acquire, red_release, atom_exch
            if (x == e) {
                if constexpr (M == M_RLX_FENCE) fence_acq_rel_cluster();
                return true;
            }
        }
        return false;
    }
}

template <int M>
__device__ __forceinline__ void k4_rearm(const K4Addr& a) {
    if constexpr (kHasData<M>) mbar_arrive_expect_tx(a.own_bar, 4u);
}

// Active pair: rank 0 and rank p (role 0 / 1); other ranks (role 2) only join the cluster barriers.
// g[role * reps + i]: rank 0 g_send, rank p g_recv; rt[i] (rank 0); info[rank * 4 + {0: smid, 1: timeouts, 2: mismatches}].
template <int M>
__device__ __forceinline__ void k4_body(uint32_t reps, uint32_t zero, uint32_t p, uint32_t* gflag, uint64_t* g, uint32_t* rt,
                                        uint32_t* info) {
    extern __shared__ __align__(16) uint8_t dyn[];  // 200 KB, only reserved (1 CTA/SM)
    __shared__ __align__(8) uint64_t s_bar;
    __shared__ uint32_t s_flag, s_data;
    __shared__ uint64_t s_g[kReps];
    __shared__ uint32_t s_rt[kReps];
    const uint32_t lane = threadIdx.x & 31;
    uint32_t rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    const uint32_t role = rank == 0 ? 0u : rank == p ? 1u : 2u, peer = rank == 0 ? p : 0u, smid = read_smid();
    K4Addr a;
    a.own_flag_cta = smem_u32(&s_flag);
    a.own_flag = mapa_u32(a.own_flag_cta, rank);
    a.peer_flag = mapa_u32(a.own_flag_cta, peer);
    a.own_data_cta = smem_u32(&s_data);
    a.peer_data = mapa_u32(a.own_data_cta, peer);
    a.own_bar = smem_u32(&s_bar);
    a.peer_bar = mapa_u32(a.own_bar, peer);
    a.gown = gflag + 32 * role;  // 128 B apart, both in the same 4-KiB page
    a.gpeer = gflag + 32 * (role ^ 1u);
    if (lane == 0) {
        dyn[0] = 0;
        s_flag = 0;
        s_data = 0;
        mbar_init(a.own_bar, 1);
        if constexpr (kHasData<M>) if (role < 2) mbar_arrive_expect_tx(a.own_bar, 4u);  // arm phase 0
        fence_mbarrier_init();
    }
    __syncwarp();
    cluster_sync();
    uint32_t timeouts = 0, mism = 0;
    if constexpr (M == M_CSYNC) {
#pragma unroll 1
        for (uint32_t i = 0; i < reps; ++i) {
            if (role == 0) {
                const uint64_t t0 = read_clock64();
                const uint64_t gs = read_globaltimer();
                cluster_sync();
                cluster_sync();
                const uint64_t t1 = read_clock64();
                if (lane == 0) { s_rt[i] = (uint32_t)(t1 - t0); s_g[i] = gs; }
            } else {  // rank p records; the other ranks execute the same two barriers
                cluster_sync();
                const uint64_t gr = read_globaltimer();
                cluster_sync();
                if (lane == 0) s_g[i] = gr;
            }
        }
    } else {
        if (lane == 0 && role < 2) {
            if (role == 0) {
#pragma unroll 1
                for (uint32_t i = 0; i < reps; ++i) {
                    const uint32_t e = i + 1;
                    uint32_t v;
                    const uint64_t t0 = read_clock64();
                    const uint64_t gs = read_globaltimer();
                    k4_signal<M>(a, e);
                    const bool ok = k4_wait<M>(a, e, i, v);
                    // st.async / red.async: t1 after the data word has arrived in a register.
                    const uint64_t t1 = kHasData<M> ? read_clock64_after(v, zero) : read_clock64();
                    if (!ok) { ++timeouts; break; }
                    mism += v != e;
                    k4_rearm<M>(a);
                    s_rt[i] = (uint32_t)(t1 - t0);
                    s_g[i] = gs;
                }
            } else {
#pragma unroll 1
                for (uint32_t i = 0; i < reps; ++i) {
                    const uint32_t e = i + 1;
                    uint32_t v;
                    const bool ok = k4_wait<M>(a, e, i, v);
                    const uint64_t gr = read_globaltimer();
                    if (!ok) { ++timeouts; break; }
                    mism += v != e;
                    k4_rearm<M>(a);
                    k4_signal<M>(a, e);
                    s_g[i] = gr;
                }
            }
        }
        __syncwarp();
    }
    cluster_sync();  // no CTA exits while its peer may still access its smem
    if (role < 2)
        for (uint32_t i = lane; i < reps; i += 32) {
            g[role * reps + i] = s_g[i];
            if (role == 0) rt[i] = s_rt[i];
        }
    if (lane == 0) { info[rank * 4 + 0] = smid; info[rank * 4 + 1] = timeouts; info[rank * 4 + 2] = mism; }
}

#define K4_KERNEL(name, M)                                                                                         \
    __global__ void __launch_bounds__(32) name(uint32_t reps, uint32_t zero, uint32_t p, uint32_t* gflag,           \
                                               uint64_t* g, uint32_t* rt, uint32_t* info) {                        \
        k4_body<M>(reps, zero, p, gflag, g, rt, info);                                                             \
    }
K4_KERNEL(k4_st_ld_volatile, M_ST_LD_VOL)
K4_KERNEL(k4_pull_ld_volatile, M_PULL_VOL)
K4_KERNEL(k4_release_acquire, M_REL_ACQ)
K4_KERNEL(k4_relaxed_fence, M_RLX_FENCE)
K4_KERNEL(k4_red_release, M_RED_REL)
K4_KERNEL(k4_atom_exch, M_ATOM_EXCH)
K4_KERNEL(k4_st_async, M_ST_ASYNC)
K4_KERNEL(k4_red_async, M_RED_ASYNC)
K4_KERNEL(k4_arrive_release, M_ARRIVE_REL)
K4_KERNEL(k4_arrive_relaxed, M_ARRIVE_RLX)
K4_KERNEL(k4_gmem_flag, M_GMEM)
K4_KERNEL(k4_cluster_sync, M_CSYNC)

using K4Fn = void (*)(uint32_t, uint32_t, uint32_t, uint32_t*, uint64_t*, uint32_t*, uint32_t*);
struct K4Case { K4Fn k; const char* kname; const char* primitive; const char* variant; uint32_t goff; int bt; };
static const K4Case kCases[] = {
    {k4_st_ld_volatile, "k4_st_ld_volatile", "flag_st_ld_volatile", "push", 0, 1},
    {k4_pull_ld_volatile, "k4_pull_ld_volatile", "flag_pull_ld_volatile", "pull", 0, 1},
    {k4_release_acquire, "k4_release_acquire", "flag_release_acquire", "push", 0, 1},
    {k4_relaxed_fence, "k4_relaxed_fence", "flag_relaxed_fence", "push", 0, 1},
    {k4_red_release, "k4_red_release", "flag_red_release", "push", 0, 1},
    {k4_atom_exch, "k4_atom_exch", "flag_atom_exch", "push", 0, 1},
    {k4_st_async, "k4_st_async", "st_async_mbarrier", "push", 0, 1},
    {k4_red_async, "k4_red_async", "red_async_mbarrier", "push", 0, 1},
    {k4_arrive_release, "k4_arrive_release", "remote_mbarrier_arrive_release", "push", 0, 1},
    {k4_arrive_relaxed, "k4_arrive_relaxed", "remote_mbarrier_arrive_relaxed", "push", 0, 1},
    {k4_gmem_flag, "k4_gmem_flag", "gmem_flag_release_acquire", "off0", 0, 1},
    {k4_gmem_flag, "k4_gmem_flag", "gmem_flag_release_acquire", "off4096", 4096, 1},
    {k4_cluster_sync, "k4_cluster_sync", "cluster_sync", "push", 0, 32},
};

// Placement sweep: (cluster size, peer rank of rank 0).
struct K4Pair { int cs; uint32_t p; };
static const K4Pair kPairs[] = {{2, 1}, {4, 1}, {4, 2}};
static std::string pair_variant(const K4Case& c, const K4Pair& q) { return std::string(c.variant) + "_r0" + std::to_string(q.p); }

struct K4Bufs {
    DevBuf<uint32_t> gflag{(2u << 20) / 4};  // 2 MiB: base 8-KiB aligned (checked)
    DevBuf<uint64_t> g;
    DevBuf<uint32_t> rt, info{16};  // 4 words per rank, cs ≤ 4
    explicit K4Bufs(uint32_t reps) : g(2 * (size_t)reps), rt(reps) {
        if ((uintptr_t)gflag.d % 8192 != 0) hb::die("global flag buffer is not 8-KiB aligned");
    }
};

// One launch; returns false if the cs ranks did not land on distinct SMs. Dies on timeouts / data mismatches.
static bool k4_launch(const K4Case& c, const K4Pair& q, uint32_t reps, K4Bufs& b, std::vector<uint32_t>& info) {
    if (reps > kReps) hb::die("reps > kReps");
    b.gflag.zero();
    b.info.zero();
    ClusterOnce co(q.cs, 32);
    CUDA_CHECK(co.L.launch(c.k, reps, 0u, q.p, b.gflag.d + c.goff / 4, b.g.d, b.rt.d, b.info.d));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    info = b.info.get();
    std::vector<uint32_t> sm(q.cs);
    for (int r = 0; r < q.cs; ++r) {
        sm[r] = info[r * 4];
        if (info[r * 4 + 1] || info[r * 4 + 2])
            hb::die(std::string(c.kname) + " " + pair_variant(c, q) + " cs " + std::to_string(q.cs) + ": rank " +
                    std::to_string(r) + " timeouts " + std::to_string(info[r * 4 + 1]) + " data mismatches " +
                    std::to_string(info[r * 4 + 2]));
    }
    return hb::all_distinct(sm);
}

static double median(std::vector<long long> v) {
    if (v.empty()) return 0;
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return (double)v[v.size() / 2];
}

static void k4_point(Out07& o, const K4Case& c, const K4Pair& q, uint32_t reps, int launches) {
    K4Bufs b(reps);
    std::vector<uint32_t> info;
    for (int w = 0; w < kWarmups; ++w) k4_launch(c, q, reps, b, info);
    const std::string variant = pair_variant(c, q);
    const std::string label = std::string("k4_") + c.primitive + "_" + variant + "_cs" + std::to_string(q.cs);
    std::vector<long long> all_rt, all_ow;
    for (int l = 0, tries = 0; l < launches;) {
        if (!k4_launch(c, q, reps, b, info)) {
            ++o.discarded;
            if (++tries > 20) hb::die(label + ": ranks never on distinct SMs");
            continue;
        }
        const hb::ClockSample s = o.run.clock_note(label, "launch", o.li);
        const std::vector<uint64_t> g = b.g.get();
        const std::vector<uint32_t> rt = b.rt.get();
        const std::string pre = o.run.prefix(o.li, s);
        for (uint32_t i = 0; i < reps; ++i) {
            const long long ow = (long long)g[reps + i] - (long long)g[i];
            o.prop_row(s, c.primitive, q.cs, info[0], info[q.p * 4], ow, rt[i], c.bt, variant, i, pre);
            all_rt.push_back(rt[i]);
            all_ow.push_back(ow);
        }
        o.flush();
        ++o.li;
        ++l;
    }
    o.sync();
    std::printf("k4 %-32s %-12s cs %d smid %3u -> %3u: round trip %7.0f cycles, one-way %5.0f ns (medians)\n",
                c.primitive, variant.c_str(), q.cs, info[0], info[q.p * 4], median(all_rt), median(all_ow));
}

static void run_k4(hb::Run& run, bool quick) {
    Out07 o(run, "k4");
    const uint32_t reps = quick ? kQuickReps : kReps;
    const int launches = quick ? kQuickLaunches : kLaunches;
    timer_floor(o, reps, launches);
    for (const K4Pair& q : kPairs)
        for (const K4Case& c : kCases) k4_point(o, c, q, reps, launches);
    std::printf("k4: %d launches discarded (ranks not on distinct SMs)\n", o.discarded);
}

// One launch per (placement, method) (1024 reps); manifest rows in launch order for post_ncu.py.
static void run_ncu(hb::Run& run) {
    hb::CsvWriter man("bin/ncu_manifest_k4.csv", hb::with_mandatory({"kernel", "primitive", "cs", "block_threads", "variant"}));
    long long li = 0;
    K4Bufs b(kReps);
    std::vector<uint32_t> info;
    for (const K4Pair& q : kPairs)
        for (const K4Case& c : kCases) {
            k4_launch(c, q, kReps, b, info);
            const std::string variant = pair_variant(c, q);
            const hb::ClockSample s = run.clock_note(std::string("ncu_k4_") + c.primitive + "_" + variant + "_cs" + std::to_string(q.cs), "launch", li);
            man.add(run.prefix(li++, s) + ",k4," + c.primitive + "," + std::to_string(q.cs) + "," + std::to_string(c.bt) + "," + variant);
            man.flush();
        }
    man.sync();
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "k4";
    hb::Run run(EXP, argc, argv);
    if (mode == "clockcheck") run.lock_clocks_note(hb::arg_str(argc, argv, "--label", "check"), "check");
    else if (mode == "k4") run_k4(run, hb::has_flag(argc, argv, "--quick"));
    else if (mode == "ncu") run_ncu(run);
    else hb::die("unknown mode " + mode);
    run.complete();
    return 0;
}
