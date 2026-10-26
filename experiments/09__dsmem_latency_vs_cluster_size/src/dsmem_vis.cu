/*
 * dsmem_vis.cu — experiment 09, K2: remote store → visibility latency between two CTAs of a cluster (one way, from
 * %globaltimer) and the ping-pong round trip (from %clock64 of the initiating SM), for five ways of writing into the
 * peer CTA's shared memory.
 *
 * Usage: bin/dsmem_vis {k2|clockcheck} [--quick] [--label L]. One invocation = one run_id.
 *
 * Placement: Grid09 (common09.hpp) with 8 clusters of CS CTAs per launch (all ping-pong at the same time), 32 threads and
 *   200 KB dynamic smem per CTA (→ 1 CTA per SM), Default policy, explicit cluster attribute. Active pair: initiator
 *   rank q = 0 and peer rank p (kernel argument): p = 1 (the spec's ranks 0/1; in a first-wave cluster they share a TPC)
 *   and, for CS ≥ 4, p = CS − 1 (another TPC). The other ranks only take part in the setup/teardown cluster.sync().
 *   Launches whose cluster ranks are not on distinct SMs or whose clock sample is bad are discarded and repeated.
 * Setup: each CTA zeroes its flag and data words, inits its mbarrier (count 1; st.async / red.async: pre-armed with
 *   arrive.expect_tx(4)), fence.mbarrier_init, cluster.sync(). Teardown: cluster.sync() before any CTA exits.
 * Lane 0 does all signalling (lanes 1–31 wait at __syncwarp). Iteration i (epoch e = i + 1), reps = 1024 (64 --quick):
 *   rank q: t0 = %clock64; ga = %globaltimer; signal(peer p, e); wait(own, e); t1 = %clock64; gb = %globaltimer
 *   rank p: wait(own, e); ga = %globaltimer; [re-arm]; gb = %globaltimer; signal(peer q, e)
 *   one_way_ns = ga(p) − ga(q) (q → p), one_way_back_ns = gb(q) − gb(p) (p → q) in 32-ns ticks of the global timer
 *   (offsets between SMs ≤ 1 tick, 01); round_trip_cycles = t1 − t0 on rank q's SM (includes one %globaltimer read).
 * Every method also runs as <method>_nogt: the same kernel without any %globaltimer read (one_way columns empty), so the
 *   round trip contains no global-timer access (with the reads, rank q's round trip includes its own read and rank p's two).
 * Methods (signal / wait):
 *   st_plain            st.shared::cluster.u32 e → peer flag / spin ld.volatile.shared.u32 on the own flag (.shared::cta)
 *   st_release          st.release.cluster.shared::cluster.u32 e → peer flag / same local spin
 *   atom_exch           atom.shared::cluster.exch.b32 e → peer flag (old value consumed only after the loop) / same spin
 *   st_async_mbarrier   st.async.shared::cluster.mbarrier::complete_tx::bytes e → peer data word, complete_tx(4) on the
 *                       peer's mbarrier / loop mbarrier.try_wait.parity.acquire.cluster (parity i & 1), then ld.shared of
 *                       the data word (t1 waits for it on rank q); re-arm arrive.expect_tx(4) after the wait
 *   red_async_mbarrier  red.async.relaxed.cluster…add.u32 1 → peer data counter (== e after the wait), same barrier protocol
 * Every poll loop is capped at 2^26 polls: a timeout or a data mismatch is counted and the host dies on either.
 * Per (method, CS, pair): 2 warm-up launches, 5 recorded launches (2 with --quick). Per-trial values are buffered in
 *   __shared__ and copied out after the teardown.
 *   → results/09__dsmem_latency__visibility.csv: cs, variant, cluster_id, src_rank (q), dst_rank (p), smid_src, smid_dst,
 *     trial, one_way_ns, one_way_back_ns, round_trip_cycles (one row per trial and cluster)
 */
#include <algorithm>
#include <string>
#include <vector>

#include "common09.hpp"

static const char* EXP = "09__dsmem_latency_vs_cluster_size";
constexpr uint32_t kReps = 1024, kQuickReps = 64, kSpinCap = 1u << 26;
constexpr int kVisClusters = 8;

enum M2 { M_ST, M_REL, M_ATOM, M_STASYNC, M_REDASYNC };
static const char* kMName[] = {"st_plain", "st_release", "atom_exch", "st_async_mbarrier", "red_async_mbarrier"};
template <int M> constexpr bool kBar = (M == M_STASYNC || M == M_REDASYNC);

__device__ __forceinline__ uint32_t lds_volatile_u32(uint32_t a) {
    uint32_t v; asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory"); return v; }
__device__ __forceinline__ uint32_t lds_u32(uint32_t a) {
    uint32_t v; asm volatile("ld.shared::cta.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory"); return v; }
__device__ __forceinline__ uint32_t atom_exch_cluster_u32(uint32_t a, uint32_t v) {
    uint32_t o; asm volatile("atom.shared::cluster.exch.b32 %0, [%1], %2;" : "=r"(o) : "r"(a), "r"(v) : "memory"); return o; }

struct K2Addr { uint32_t own_flag_cta, peer_flag, own_data_cta, peer_data, own_bar, peer_bar; };

template <int M>
__device__ __forceinline__ void k2_signal(const K2Addr& a, uint32_t e, uint32_t& sink) {
    if constexpr (M == M_ST) st_dsmem_u32(a.peer_flag, e);
    else if constexpr (M == M_REL) st_release_cluster_u32(a.peer_flag, e);
    else if constexpr (M == M_ATOM) sink += atom_exch_cluster_u32(a.peer_flag, e);
    else if constexpr (M == M_STASYNC) st_async_u32(a.peer_data, e, a.peer_bar);
    else red_async_add_u32(a.peer_data, 1u, a.peer_bar);
}

// false on a timeout; v = the value seen (data word for the mbarrier methods).
template <int M>
__device__ __forceinline__ bool k2_wait(const K2Addr& a, uint32_t e, uint32_t i, uint32_t& v) {
    if constexpr (kBar<M>) {
        bool done = false;
#pragma unroll 1
        for (uint32_t n = 0; n < kSpinCap && !done; ++n) done = mbar_try_wait_parity_acq_cluster(a.own_bar, i & 1u);
        v = lds_u32(a.own_data_cta);
        return done;
    } else {
#pragma unroll 1
        for (uint32_t n = 0; n < kSpinCap; ++n)
            if ((v = lds_volatile_u32(a.own_flag_cta)) == e) return true;
        return false;
    }
}

// g[((cluster * 2 + role) * 2 + {0: ga, 1: gb}) * reps + i]; rt[cluster * reps + i];
// info[4 * block + {0: smid, 1: timeouts, 2: mismatches}].
// GT = false: no %globaltimer read anywhere in the loop (round trip only; the one-way columns are left empty).
template <int M, bool GT>
__global__ void __launch_bounds__(32) k2_pingpong(uint32_t reps, uint32_t zero, uint32_t p, uint32_t* g, uint32_t* rt, uint32_t* info,
                                                 uint32_t* place) {
    extern __shared__ __align__(16) uint8_t dyn[];  // 200 KB, only reserved (1 CTA/SM)
    __shared__ __align__(8) uint64_t s_bar;
    __shared__ uint32_t s_flag, s_data;
    __shared__ uint32_t s_ga[kReps], s_gb[kReps], s_rt[kReps];
    const uint32_t lane = threadIdx.x;
    const ClusterRegs c = read_cluster_regs();
    const uint32_t rank = c.ctarank, cid = c.clusterid_x;
    const uint32_t q = 0, role = rank == q ? 0u : rank == p ? 1u : 2u, peer = rank == q ? p : q;
    K2Addr a;
    a.own_flag_cta = smem_u32(&s_flag);
    a.peer_flag = mapa_u32(a.own_flag_cta, peer);
    a.own_data_cta = smem_u32(&s_data);
    a.peer_data = mapa_u32(a.own_data_cta, peer);
    a.own_bar = smem_u32(&s_bar);
    a.peer_bar = mapa_u32(a.own_bar, peer);
    record_place(place);
    if (lane == 0) {
        dyn[0] = 0;
        s_flag = 0;
        s_data = 0;
        mbar_init(a.own_bar, 1);
        if constexpr (kBar<M>) if (role < 2) mbar_arrive_expect_tx(a.own_bar, 4u);  // arm phase 0
        fence_mbarrier_init();
    }
    __syncwarp();
    cluster_sync();
    uint32_t timeouts = 0, mism = 0, sink = 0;
    if (lane == 0 && role < 2) {
#pragma unroll 1
        for (uint32_t i = 0; i < reps; ++i) {
            const uint32_t e = i + 1;
            uint32_t v;
            if (role == 0) {
                const uint64_t t0 = read_clock64();
                const uint32_t ga = GT ? (uint32_t)read_globaltimer() : 0u;
                k2_signal<M>(a, e, sink);
                const bool ok = k2_wait<M>(a, e, i, v);
                const uint64_t t1 = read_clock64_after(v, zero);  // after the flag / data value is in a register
                const uint32_t gb = GT ? (uint32_t)read_globaltimer() : 0u;
                if (!ok) { ++timeouts; break; }
                mism += v != e;
                if constexpr (kBar<M>) mbar_arrive_expect_tx(a.own_bar, 4u);
                s_rt[i] = (uint32_t)(t1 - t0);
                s_ga[i] = ga;
                s_gb[i] = gb;
            } else {
                const bool ok = k2_wait<M>(a, e, i, v);
                const uint32_t ga = GT ? (uint32_t)read_globaltimer() + (v & zero) : 0u;
                if (!ok) { ++timeouts; break; }
                mism += v != e;
                if constexpr (kBar<M>) mbar_arrive_expect_tx(a.own_bar, 4u);
                const uint32_t gb = GT ? (uint32_t)read_globaltimer() : 0u;
                k2_signal<M>(a, e, sink);
                s_ga[i] = ga;
                s_gb[i] = gb;
            }
        }
    }
    __syncwarp();
    cluster_sync();  // no CTA exits while its peer may still access its smem
    if (role < 2)
        for (uint32_t i = lane; i < reps; i += 32) {
            g[((cid * 2 + role) * 2 + 0) * reps + i] = s_ga[i];
            g[((cid * 2 + role) * 2 + 1) * reps + i] = s_gb[i];
            if (role == 0) rt[cid * reps + i] = s_rt[i];
        }
    if (lane == 0) {
        uint32_t* inf = info + 4 * blockIdx.x;
        inf[0] = read_smid();
        inf[1] = timeouts;
        inf[2] = mism;
        inf[3] = sink & zero;
    }
}

using K2Fn = void (*)(uint32_t, uint32_t, uint32_t, uint32_t*, uint32_t*, uint32_t*, uint32_t*);
static const K2Fn kKernels[2][5] = {
    {k2_pingpong<M_ST, true>, k2_pingpong<M_REL, true>, k2_pingpong<M_ATOM, true>, k2_pingpong<M_STASYNC, true>, k2_pingpong<M_REDASYNC, true>},
    {k2_pingpong<M_ST, false>, k2_pingpong<M_REL, false>, k2_pingpong<M_ATOM, false>, k2_pingpong<M_STASYNC, false>,
     k2_pingpong<M_REDASYNC, false>}};

static double median_of(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static void run_k2(hb::Run& run, bool quick) {
    Discards disc(run);
    hb::CsvWriter csv(hb::Run::results("09__dsmem_latency__visibility.csv"),
                      hb::with_mandatory({"cs", "variant", "cluster_id", "src_rank", "dst_rank", "smid_src", "smid_dst", "trial",
                                          "one_way_ns", "one_way_back_ns", "round_trip_cycles"}));
    const uint32_t reps = quick ? kQuickReps : kReps;
    const int launches = quick ? kQuickLaunches : kLaunches;
    long long li = 0;
    run.lock_clocks_note("k2", "before");
    for (int cs : {2, 4, 8, 16}) {
        std::vector<uint32_t> peers{1};
        if (cs >= 4) peers.push_back((uint32_t)cs - 1);
        for (uint32_t p : peers)
            for (int gt = 0; gt < 2; ++gt)
            for (int m = 0; m < 5; ++m) {
                const bool nogt = gt == 1;
                const K2Fn k = kKernels[gt][m];
                const std::string vname = std::string(kMName[m]) + (nogt ? "_nogt" : "");
                Grid09 g(k, cs, 32, cudaClusterSchedulingPolicyDefault, kVisClusters);
                const int ncl = g.nclusters;
                DevBuf<uint32_t> d_g((size_t)ncl * 4 * reps), d_rt((size_t)ncl * reps), info(4 * (size_t)g.blocks()), place(4 * (size_t)g.blocks());
                const std::string label = "k2_" + vname + "_cs" + std::to_string(cs) + "_r0" + std::to_string(p);
                auto launch = [&] {
                    info.zero();
                    place.zero();
                    CUDA_CHECK(g.L.launch(k, reps, 0u, p, d_g.d, d_rt.d, info.d, place.d));
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaDeviceSynchronize());
                    const std::vector<uint32_t> inf = info.get();
                    for (int b = 0; b < g.blocks(); ++b)
                        if (inf[4 * b + 1] || inf[4 * b + 2])
                            hb::die(label + ": block " + std::to_string(b) + " timeouts " + std::to_string(inf[4 * b + 1]) + " mismatches " +
                                    std::to_string(inf[4 * b + 2]));
                    return Placement(place.get(), cs, ncl);
                };
                for (int w = 0; w < kWarmups; ++w) launch();
                std::vector<double> rts, ows, obs;
                for (int l = 0; l < launches; ++l) {
                    auto r = recorded_launch(run, disc, label, li, launch);
                    const Placement& pl = r.second;
                    const std::string pre = run.prefix(li, r.first);
                    const std::vector<uint32_t> gh = d_g.get(), rh = d_rt.get();
                    char b[256];
                    for (int c = 0; c < ncl; ++c) {
                        const uint32_t* gq = &gh[((size_t)c * 2 + 0) * 2 * reps];
                        const uint32_t* gp = &gh[((size_t)c * 2 + 1) * 2 * reps];
                        for (uint32_t i = 0; i < reps; ++i) {
                            const long long ow = (long long)(int32_t)(gp[i] - gq[i]);                 // ga(p) − ga(q)
                            const long long ob = (long long)(int32_t)(gq[reps + i] - gp[reps + i]);   // gb(q) − gb(p)
                            const uint32_t rt = rh[(size_t)c * reps + i];
                            rts.push_back(rt);
                            ows.push_back((double)ow);
                            obs.push_back((double)ob);
                            if (nogt)
                                std::snprintf(b, sizeof b, ",%d,%s,%d,0,%u,%u,%u,%u,,,%u", cs, vname.c_str(), c, p, pl.at(c, 0), pl.at(c, (int)p), i, rt);
                            else
                                std::snprintf(b, sizeof b, ",%d,%s,%d,0,%u,%u,%u,%u,%lld,%lld,%u", cs, vname.c_str(), c, p, pl.at(c, 0),
                                              pl.at(c, (int)p), i, ow, ob, rt);
                            csv.add(pre + b);
                        }
                    }
                    csv.flush();
                    ++li;
                }
                csv.sync();
                double mo = 0, mb = 0;
                for (double v : ows) mo += v;
                for (double v : obs) mb += v;
                std::printf("k2 %-24s cs %2d ranks 0->%-2u: round trip %6.0f cycles (median)", vname.c_str(), cs, p, median_of(rts));
                if (nogt) std::printf("\n");
                else std::printf(", one-way mean %6.1f ns (q->p) %6.1f ns (p->q)\n", mo / ows.size(), mb / obs.size());
            }
    }
    run.lock_clocks_note("k2", "after");
    std::printf("k2: %d launches discarded\n", disc.n);
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "k2";
    hb::Run run(EXP, argc, argv);
    if (mode == "clockcheck") run.lock_clocks_note(hb::arg_str(argc, argv, "--label", "check"), "check");
    else if (mode == "k2") run_k2(run, hb::has_flag(argc, argv, "--quick"));
    else hb::die("unknown mode " + mode);
    run.complete();
    return 0;
}
