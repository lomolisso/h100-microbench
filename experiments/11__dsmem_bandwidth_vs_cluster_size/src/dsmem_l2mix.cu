/*
 * dsmem_l2mix.cu — experiment 11, K5: interference between SM-to-SM (DSMEM) traffic and SM→L2 traffic inside one GPC.
 *
 * Usage: bin/dsmem_l2mix {k5|clockcheck} [--quick]. One invocation = one run_id (results/11__…__runs.csv).
 *
 * Placement: one cluster of 16 CTAs per launch (1024 threads, 200 KB dynamic smem → 1 CTA per SM, Default policy, the
 *   hardware places it; a lone CS-16 cluster fills a whole 16-SM GPC). Roles are given per physical SM, from the
 *   latest 03 map: TPC index i = rank of the SM's TPC in its GPC (ascending smid; TPC t = SMs 2t, 2t+1), parity =
 *   smid & 1, CPC hypothesis group = i mod 3 (A = {0,3,6}, B = {1,4,7}, C = {2,5}; no CPC level was found in DSMEM
 *   latency, so this grouping is a hypothesis under test). Roles:
 *     DSMEM sources  the even SM of the TPCs 1, 4, 7 (group B; 3 SMs);
 *     servers        dest same_cpc: the odd SM of the next B TPC (1→4, 4→7, 7→1);
 *                    dest other_cpc: TPC 1 → odd SM of TPC 2, 4 → odd SM of TPC 5, 7 → even SM of TPC 2 (group C);
 *                    a server does nothing but hold the 96 KiB tile its source reads (one reader per server);
 *     L2 streamers   role map l2A: both SMs of the TPCs 0, 3, 6 (group A; 6 SMs);
 *                    role map l2all: every SM that is neither a source nor a server (10 SMs);
 *     role map dsm_ring (heavy DSMEM traffic, dest "ring"): L2 streamers = group A (6 SMs); the 10 SMs of groups B and C
 *                    form one ring ordered by (TPC index, parity) — TPC 1 even, 1 odd, 2 even, …, 7 odd — and every
 *                    one of them reads the next one's tile (each is source and server; about half of the hops are
 *                    TPC mates);
 *     the remaining SMs are idle. A launch whose cluster does not cover a 16-SM GPC with all roles present is discarded.
 *   The CTAs exchange their %smid through DSMEM after the first cluster.sync() and find their source/server ranks.
 * Kernel k_mix<IL, ID>(M11): L2 streamer j of n reads its 1/n slice of a 16 MiB buffer (ld.global.cg.v4.u32), DSMEM
 *   source reads its server's 96 KiB tile (ld.shared::cluster.v4.u32); thread t issues groups of IL (or ID) independent
 *   16-B loads, elements e, e + 1024, … (each warp access = 512 contiguous bytes); per group a = %clock64 before the
 *   first load, b = read_clock64_after(sum of the group) → loaded latency of the group (the next group starts after
 *   b, so IL / ID is exactly the number of loads in flight per thread). L2 lines: one untimed pass by the streamers
 *   before the timed region (reads leave a copy in the reader's L2 partition: every later read is a near hit).
 *   Both groups start after one cluster.sync(); iteration counts (passes over the slice / tile) come from a
 *   calibration launch of each group alone at the same ILP, sized for ≈ 20 ms; "both" runs use the same counts.
 *   Per CTA: cycles = max t1 − min t0 over its threads (%clock64; ns with %globaltimer), bytes, mean group latency.
 * Configurations: (role map, dest) ∈ {l2A, l2all} × {same_cpc, other_cpc} + (dsm_ring, ring); × {alone L2 at IL 1/2/4/8, alone DSMEM at
 *   ID 1/2/4/8, both with IL 8 × ID 1/2/4/8 and ID 8 × IL 1/2/4}; 2 warm-up + 30 recorded launches each.
 *   → results/11__dsmem_bw__interference.csv: role_map, dsmem_dest, ilp_l2, ilp_dsmem, group (l2 / dsmem),
 *     alone_or_both, cluster_id, rank, smid, gpc, tpc_index, cpc (hypothesis group 0/1/2 = A/B/C), sample (launch),
 *     iters, cycles, ns, bytes_moved, bpc, loaded_latency_cycles (one row per active CTA and launch)
 */
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "../../09__dsmem_latency_vs_cluster_size/src/common09.hpp"
#include "../../10__dsmem_latency_heatmap/src/common10.hpp"
#include "common11.hpp"

static const char* EXP = "11__dsmem_bandwidth_vs_cluster_size";
constexpr uint32_t kTileElems = 96u * 1024u / 16u, kThreads = 1024, kCS = 16;
constexpr size_t kL2Bytes = 16u << 20;
constexpr int kMixLaunches = 30, kMixQuickLaunches = 3;
constexpr double kTargetCycles = 20e-3 * 1635e6;  // ≈ 20 ms at the locked clock

enum Role { R_IDLE = 0, R_L2 = 1, R_SRC = 2, R_SRV = 3 };

struct MixRec { unsigned long long cycles, ns, bytes, lat_sum, lat_n; uint32_t role, valid; };
struct M11 {
    const uint8_t* role;      // role[smid]
    const int16_t* dest;      // server smid of a source, by smid
    const uint16_t* l2slot;   // index of an L2 streamer among the L2 streamers of its GPC
    uint32_t n_l2, iters_l2, iters_dsm, run_l2, run_dsm;
    const uint4* l2buf;
    uint64_t zero;
    MixRec* out;
    uint32_t* place;
    uint32_t* err;
};

__device__ __forceinline__ uint4 ld_cg_v4(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
    return v;
}

template <int IL, int ID>
__global__ void __launch_bounds__(1024, 1) k_mix(M11 p) {
    extern __shared__ __align__(1024) uint8_t dyn[];
    __shared__ uint32_t s_me, s_smid[kCS], s_dest;
    __shared__ unsigned long long s_c0, s_c1, s_n0, s_n1, s_lat, s_latn;
    const uint32_t tid = threadIdx.x;
    const ClusterRegs c = read_cluster_regs();
    const uint32_t rank = c.ctarank, zero = (uint32_t)p.zero;
    uint4* src = reinterpret_cast<uint4*>(dyn);
    for (uint32_t e = tid; e < kTileElems; e += blockDim.x) src[e] = make_uint4(e, rank + 1u, e ^ 0x5a5a5a5au, rank);
    record_place(p.place);
    const uint32_t me = read_smid();
    if (tid == 0) { s_me = me; s_dest = 0xffffffffu; s_c0 = ~0ull; s_c1 = 0; s_n0 = ~0ull; s_n1 = 0; s_lat = 0; s_latn = 0; }
    __syncthreads();
    cluster_sync();
    if (tid < kCS) s_smid[tid] = ld_dsmem_u32(mapa_u32(smem_u32(&s_me), tid));
    __syncthreads();
    const uint32_t role = p.role[me];
    if (tid < kCS && role == R_SRC && s_smid[tid] == (uint32_t)p.dest[me]) s_dest = tid;
    __syncthreads();
    if (role == R_SRC && s_dest == 0xffffffffu && tid == 0) atomicOr(p.err, 1u);
    const bool l2 = role == R_L2 && p.run_l2, dsm = role == R_SRC && p.run_dsm && s_dest != 0xffffffffu;
    const uint32_t per = (uint32_t)(kL2Bytes / 16u) / p.n_l2;
    const uint4* slice = p.l2buf + (size_t)(role == R_L2 ? p.l2slot[me] : 0u) * per;
    uint32_t sx = 0;
    if (l2)  // warm pass: leaves a copy of every line in this partition
        for (uint32_t e = tid; e < per; e += blockDim.x) { const uint4 x = ld_cg_v4(slice + e); sx ^= x.x ^ x.y ^ x.z ^ x.w; }
    __syncthreads();
    cluster_sync();
    const TimerPair a = read_timers_pair();
    TimerPair b = a;
    unsigned long long lat = 0, latn = 0, bytes = 0;
    if (l2) {
#pragma unroll 1
        for (uint32_t q = 0; q < p.iters_l2; ++q) {
            const uint4* s = slice + (q & zero);
#pragma unroll 1
            for (uint32_t e = tid; e < per; e += IL * kThreads) {
                const uint64_t t0 = read_clock64();
                uint4 x[IL];
#pragma unroll
                for (int j = 0; j < IL; ++j) { const uint32_t ej = e + j * kThreads; x[j] = ej < per ? ld_cg_v4(s + ej) : make_uint4(0, 0, 0, 0); }
                uint32_t g = 0;
#pragma unroll
                for (int j = 0; j < IL; ++j) g += x[j].x ^ x[j].y ^ x[j].z ^ x[j].w;
                const uint64_t t1 = read_clock64_after(g, p.zero);
                lat += t1 - t0; ++latn;
                sx ^= g;
            }
        }
        bytes = (unsigned long long)per * 16ull * p.iters_l2;
        b = read_timers_after((uint64_t)sx, p.zero);
    } else if (dsm) {
        const uint32_t base = mapa_u32(smem_u32(dyn), s_dest);
#pragma unroll 1
        for (uint32_t q = 0; q < p.iters_dsm; ++q) {
            const uint32_t pb = base + (q & zero);
#pragma unroll 1
            for (uint32_t e = tid; e < kTileElems; e += ID * kThreads) {
                const uint64_t t0 = read_clock64();
                uint4 x[ID];
#pragma unroll
                for (int j = 0; j < ID; ++j) { const uint32_t ej = e + j * kThreads; x[j] = ej < kTileElems ? ld_dsmem_v4_u32(pb + ej * 16u) : make_uint4(0, 0, 0, 0); }
                uint32_t g = 0;
#pragma unroll
                for (int j = 0; j < ID; ++j) g += x[j].x ^ x[j].y ^ x[j].z ^ x[j].w;
                const uint64_t t1 = read_clock64_after(g, p.zero);
                lat += t1 - t0; ++latn;
                sx ^= g;
            }
        }
        bytes = (unsigned long long)kTileElems * 16ull * p.iters_dsm;
        b = read_timers_after((uint64_t)sx, p.zero);
    }
    if (l2 || dsm) {
        atomicMin(&s_c0, a.cycles); atomicMax(&s_c1, b.cycles); atomicMin(&s_n0, a.ns); atomicMax(&s_n1, b.ns);
        atomicAdd(&s_lat, lat); atomicAdd(&s_latn, latn);
    }
    if (sx == 0x9e3779b9u) atomicOr(p.err, 2u);
    __syncthreads();
    if (tid == 0 && (l2 || dsm))
        p.out[blockIdx.x] = MixRec{s_c1 - s_c0, s_n1 - s_n0, bytes, s_lat, s_latn, role, 1u};
    cluster_sync();
}

using MFn = void (*)(M11);
template <int IL>
static MFn pick_d(int id) {
    switch (id) {
    case 1: return k_mix<IL, 1>;
    case 2: return k_mix<IL, 2>;
    case 4: return k_mix<IL, 4>;
    case 8: return k_mix<IL, 8>;
    }
    hb::die("bad ILP");
}
static MFn pick_mix(int il, int id) {
    switch (il) {
    case 1: return pick_d<1>(id);
    case 2: return pick_d<2>(id);
    case 4: return pick_d<4>(id);
    case 8: return pick_d<8>(id);
    }
    hb::die("bad ILP");
}

// Role tables of one (role map, dest) for every 16-SM GPC of the map.
struct Roles {
    std::vector<uint8_t> role;
    std::vector<int16_t> dest;
    std::vector<uint16_t> l2slot;
    uint32_t n_l2 = 0, n_src = 0, n_srv = 0;  // per 16-SM GPC
};
// map 0 = l2A, 1 = l2all, 2 = dsm_ring; other_cpc selects the servers of maps 0 and 1
static Roles make_roles(const SmMap10& m, int map, bool other_cpc) {
    Roles r;
    r.role.assign(m.n, R_IDLE);
    r.dest.assign(m.n, -1);
    r.l2slot.assign(m.n, 0);
    auto sm_at = [&](int g, int i, int par) { return 2 * m.tpcs[g][i] + par; };
    for (int g = 0; g < m.ngpc; ++g) {
        if (m.tpcs[g].size() != 8) continue;  // 16-SM GPCs only
        if (map == 2) {
            std::vector<int> ring;
            for (int i = 0; i < 8; ++i)
                if (i % 3 != 0)
                    for (int par = 0; par < 2; ++par) ring.push_back(sm_at(g, i, par));
            for (size_t k = 0; k < ring.size(); ++k) {
                r.role[ring[k]] = R_SRC;
                r.dest[ring[k]] = (int16_t)ring[(k + 1) % ring.size()];
            }
            r.n_src = (uint32_t)ring.size();
            r.n_srv = 0;
        } else {
            const int next_same[8] = {-1, 4, -1, -1, 7, -1, -1, 1};
            for (int i : {1, 4, 7}) {
                const int s = sm_at(g, i, 0);
                const int d = other_cpc ? (i == 1 ? sm_at(g, 2, 1) : i == 4 ? sm_at(g, 5, 1) : sm_at(g, 2, 0)) : sm_at(g, next_same[i], 1);
                r.role[s] = R_SRC;
                r.role[d] = R_SRV;
                r.dest[s] = (int16_t)d;
            }
            r.n_src = 3;
            r.n_srv = 3;
        }
        uint32_t n = 0;
        for (int i = 0; i < 8; ++i)
            for (int par = 0; par < 2; ++par) {
                const int s = sm_at(g, i, par);
                if (r.role[s] != R_IDLE) continue;
                if (map == 1 || i % 3 == 0) { r.role[s] = R_L2; r.l2slot[s] = (uint16_t)n++; }
            }
        r.n_l2 = n;
    }
    return r;
}

struct MixCfg { int map, dest, il, id, run_l2, run_dsm; };
static const char* kMapName[] = {"l2A", "l2all", "dsm_ring"};
static const char* kDestName[] = {"same_cpc", "other_cpc", "ring"};

static void run_k5(hb::Run& run, bool quick) {
    const SmMap10 m = load_map10(run.dev().sm_count);
    Discards11 disc(run);
    hb::CsvWriter csv(hb::Run::results("11__dsmem_bw__interference.csv"),
                      hb::with_mandatory({"role_map", "dsmem_dest", "ilp_l2", "ilp_dsmem", "group", "alone_or_both", "cluster_id", "rank", "smid",
                                          "gpc", "tpc_index", "cpc", "sample", "iters", "cycles", "ns", "bytes_moved", "bpc", "loaded_latency_cycles"}));
    DevBuf<uint4> l2buf(kL2Bytes / 16);
    DevBuf<uint8_t> d_role(m.n);
    DevBuf<int16_t> d_dest(m.n);
    DevBuf<uint16_t> d_slot(m.n);
    DevBuf<MixRec> out(kCS);
    DevBuf<uint32_t> place(4 * kCS), err(1);
    long long li = 0;
    run.lock_clocks_note("k5", "before");
    const int launches = quick ? kMixQuickLaunches : kMixLaunches;
    const std::pair<int, int> combos[] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 2}};
    for (const auto& [map, dest] : combos) {
            const Roles R = make_roles(m, map, dest == 1);
            CUDA_CHECK(cudaMemcpy(d_role.d, R.role.data(), m.n, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_dest.d, R.dest.data(), m.n * 2, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_slot.d, R.l2slot.data(), m.n * 2, cudaMemcpyHostToDevice));
            std::vector<MixCfg> cfgs;
            const std::vector<int> ilps = quick ? std::vector<int>{1, 8} : std::vector<int>{1, 2, 4, 8};
            for (int i : ilps) cfgs.push_back({map, dest, i, 8, 1, 0});
            for (int i : ilps) cfgs.push_back({map, dest, 8, i, 0, 1});
            for (int i : ilps) cfgs.push_back({map, dest, 8, i, 1, 1});
            for (int i : ilps) if (i != 8) cfgs.push_back({map, dest, i, 8, 1, 1});
            // calibration: cycles per pass of each group alone, per ILP
            std::vector<double> cyc_l2(9, 0), cyc_dsm(9, 0);
            auto launch = [&](const MixCfg& c, uint32_t il2, uint32_t idsm) {
                hb::ClusterLaunch L;
                L.block = dim3(kThreads);
                L.cluster = dim3(kCS, 1, 1);
                L.smem = 200u * 1024u;
                L.explicit_cluster = true;
                L.grid = dim3(kCS);
                M11 p{d_role.d, d_dest.d, d_slot.d, R.n_l2, il2, idsm, (uint32_t)c.run_l2, (uint32_t)c.run_dsm, l2buf.d, 0, out.d, place.d, err.d};
                out.zero();
                place.zero();
                CUDA_CHECK(L.launch(pick_mix(c.il, c.id), p));
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaGetLastError());
                if (err.get()[0] & 1u) hb::die("k5: a DSMEM source did not find its server in the cluster");
                return Placement(place.get(), kCS, 1);
            };
            auto placed_ok = [&](const Placement& pl) {
                if (!pl.ok()) return false;
                int cnt[4] = {0, 0, 0, 0};
                for (int r = 0; r < (int)kCS; ++r) ++cnt[R.role[pl.at(0, r)]];
                return cnt[R_SRC] == (int)R.n_src && cnt[R_SRV] == (int)R.n_srv && cnt[R_L2] == (int)R.n_l2;
            };
            auto median_rate = [&](bool l2g) {  // cycles per pass, median over the group's CTAs
                const std::vector<MixRec> o = out.get();
                std::vector<double> v;
                for (const MixRec& q : o)
                    if (q.valid && (q.role == R_L2) == l2g) v.push_back((double)q.cycles);
                std::sort(v.begin(), v.end());
                return v.empty() ? NAN : v[v.size() / 2];
            };
            for (int i : ilps) {
                for (int t = 0;; ++t) {
                    if (t > kMaxTries) hb::die("k5 calibration: no placement on a 16-SM GPC");
                    if (placed_ok(launch({map, dest, i, 8, 1, 0}, 20, 1))) break;
                }
                cyc_l2[i] = median_rate(true) / 20.0;
                for (int t = 0;; ++t) {
                    if (t > kMaxTries) hb::die("k5 calibration: no placement on a 16-SM GPC");
                    if (placed_ok(launch({map, dest, 8, i, 0, 1}, 1, 200))) break;
                }
                cyc_dsm[i] = median_rate(false) / 200.0;
                std::printf("k5 %s %s calibration ILP %d: L2 %.0f cycles/pass (%u streamers), DSMEM %.0f cycles/pass\n", kMapName[map],
                            kDestName[dest], i, cyc_l2[i], R.n_l2, cyc_dsm[i]);
            }
            for (const MixCfg& c : cfgs) {
                const uint32_t il2 = (uint32_t)std::max(1.0, std::round(kTargetCycles / cyc_l2[c.il]));
                const uint32_t idsm = (uint32_t)std::max(1.0, std::round(kTargetCycles / cyc_dsm[c.id]));
                const char* ab = c.run_l2 && c.run_dsm ? "both" : "alone";
                char lb[128];
                std::snprintf(lb, sizeof lb, "k5_%s_%s_il%d_id%d_%s%s", kMapName[map], kDestName[dest], c.il, c.id, ab,
                              c.run_l2 && c.run_dsm ? "" : c.run_l2 ? "_l2" : "_dsm");
                for (int w = 0; w < kWarmups; ++w) launch(c, il2, idsm);
                std::vector<double> bl2, bds;
                for (int l = 0; l < launches; ++l) {
                    Placement pl = launch(c, il2, idsm);
                    hb::ClockSample s = run.clock_note(lb, "launch", li);
                    int tries = 0;
                    while (!placed_ok(pl) || s.bad()) {
                        disc.add(lb, !placed_ok(pl) ? "placement" : "clock", li, s);
                        if (++tries > kMaxTries) hb::die(std::string(lb) + ": no clean launch");
                        pl = launch(c, il2, idsm);
                        s = run.clock_note(lb, "launch", li);
                    }
                    const std::vector<MixRec> o = out.get();
                    const std::string pre = run.prefix(li, s);
                    char b[400];
                    for (int r = 0; r < (int)kCS; ++r) {
                        const MixRec& q = o[r];
                        if (!q.valid) continue;
                        const uint32_t sm = pl.at(0, r);
                        const bool isl2 = q.role == R_L2;
                        const double bpc = (double)q.bytes / q.cycles;
                        (isl2 ? bl2 : bds).push_back(bpc);
                        std::snprintf(b, sizeof b, ",%s,%s,%d,%d,%s,%s,0,%d,%u,%d,%d,%d,%d,%u,%llu,%llu,%llu,%.4f,%.3f", kMapName[map], kDestName[dest],
                                      c.il, c.id, isl2 ? "l2" : "dsmem", ab, r, sm, m.gpc[sm], m.tpc_index[sm], m.tpc_index[sm] % 3, l, isl2 ? il2 : idsm,
                                      q.cycles, q.ns, q.bytes, bpc, q.lat_n ? (double)q.lat_sum / q.lat_n : NAN);
                        csv.add(pre + b);
                    }
                    csv.flush();
                    ++li;
                }
                csv.sync();
                auto med = [](std::vector<double>& v) -> double { if (v.empty()) return NAN; std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
                std::printf("%-36s L2 %6.2f  DSMEM %6.2f B/cycle per CTA (median)\n", lb, med(bl2), med(bds));
                std::fflush(stdout);
            }
        }
    run.lock_clocks_note("k5", "after");
    std::printf("k5: %d launches discarded\n", disc.n);
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "k5";
    const bool quick = hb::has_flag(argc, argv, "--quick");
    hb::Run run(EXP, argc, argv);
    if (mode == "clockcheck") run.lock_clocks_note(hb::arg_str(argc, argv, "--label", "check"), "check");
    else if (mode == "k5") run_k5(run, quick);
    else hb::die("unknown mode " + mode);
    run.complete();
    return 0;
}
