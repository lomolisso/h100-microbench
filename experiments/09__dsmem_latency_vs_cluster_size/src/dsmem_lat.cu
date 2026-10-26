/*
 * dsmem_lat.cu — experiment 09: latency of a dependent remote shared-memory (DSMEM) load between the CTAs of a
 * thread-block cluster, for every (source rank, destination rank) pair, versus cluster size (CS) and scheduling policy;
 * the same chase under load (variant f); same-binary local / L2 / HBM reference chases.
 *
 * Usage: bin/dsmem_lat {k1|loaded|k3|ncu|clockcheck} [--quick] [--label L]. One invocation = one run_id
 *   (results/09__dsmem_latency_vs_cluster_size__runs.csv). Launches whose cluster ranks do not sit on distinct SMs, or
 *   whose NVML sample after the launch shows a throttle reason or an SM clock below the lock, are discarded, logged in
 *   results/09__dsmem_latency__discards.csv and repeated (common09.hpp).
 *
 * Placement (k1, loaded, ncu): Grid09 — 1-D clusters of CS CTAs, 200 KB dynamic smem (→ exactly 1 CTA per SM),
 *   explicit cluster attribute (also for CS 1, the own-CTA "loopback" through the DSMEM path), explicit policy.
 *   Clusters per launch = max(20, cudaOccupancyMaxActiveClusters) = 132 / 66 / 30 / 20 / 20 for CS 1 / 2 / 4 / 8 / 16:
 *   one full wave (every cluster-capable SM hosts a CTA), and 2–3 waves of different SM sets at CS 8 / 16. Every CTA
 *   records %smid, %clusterid.x, %cluster_ctarank (record_place); the host maps smid → GPC with the latest 03 map.
 *
 * K1 — kernel k1_chase<V>(perm, n_elem, random, fine_clusters, chase, zero, out, fine, place): 32 threads per CTA.
 *   Chain (each CTA, in its own smem, before a cluster.sync()): element i (E bytes, E = 4; 8 for ld_u64; 16 for ld_v4)
 *   holds in its first word the BYTE OFFSET of element next(i) from the start of the region; the other words are 0.
 *   chain "stride1": next(i) = (i + 1) mod 1024 (1024 elements, bank-conflict-free); chain "random16k": a uniformly
 *   random single cycle over the 16 KiB / E elements (Sattolo, 04's pchase_chain.hpp, seed kChainSeed).
 *   Variant ld_cluster_addr stores absolute .shared::cluster addresses instead (mapa of the CTA's own rank + offset).
 *   Turns: for src = 0 … CS−1, separated by cluster.sync() (all 32 lanes of all CTAs; only one source SM of a cluster
 *   is active at a time), lane 0 of rank src, for dst = 0 … CS−1 (self included): base = mapa(region, dst);
 *   v = 0 (element 0); 64 warm hops; t0 = read_clock64_after(v); 4096 timed hops (01's loop: #pragma unroll 1 around
 *   16 inlined hops); t1 = read_clock64_after(v) (cannot issue before the last load has returned). The value read is
 *   the next offset, so every hop depends on the previous one. Variants V (one hop):
 *     ld_cluster       ld.shared::cluster.u32 v, [base + v]                     (a: explicit 32-bit address; one add)
 *     ld_cluster_addr  ld.shared::cluster.u32 v, [v]                            (absolute address chain: no ALU op)
 *     ld_generic       v = *(uint32_t*)(p + v), p = cluster.map_shared_rank(region, dst)   (b: CUDA C++, generic ld)
 *     ld_volatile      ld.volatile.shared::cluster.u32                          (c)
 *     ld_relaxed       ld.relaxed.cluster.shared::cluster.u32                   (d)
 *     ld_acquire       ld.acquire.cluster.shared::cluster.u32                   (d)
 *     ld_u64           ld.shared::cluster.u64, v = low word                     (e: 8-B elements)
 *     ld_v4            ld.shared::cluster.v4.u32, v = .x                        (e: 16-B elements)
 *   After the timed hops the source stores (t1 − t0, final offset) to out[(cluster · CS + src) · CS + dst]; the host
 *   checks the final offset against the host-computed position after 4160 hops (proves that the chase followed the
 *   destination's chain, also for absolute addresses computed by another CTA). With 4096 hops the fixed timer/loop
 *   region (≤ 192 cycles, 01) is ≤ 0.05 cycles per hop: cycles_per_hop = (t1 − t0) / 4096, nothing subtracted.
 *   Fine pass (K1-fine; clusters with id < fine_clusters): right after the timed hops, 256 more hops of the same chain,
 *   each timed alone: a = %clock64; v = hop(v); b = read_clock64_after(v); buffered in the source's smem (after the
 *   chain region), copied to fine[] afterwards. Per-hop values include the timer pair (K3 level timer_pair).
 *   Sweep: chain stride1 × all 8 variants and chain random16k × {ld_cluster, ld_cluster_addr}; × policy {Default,
 *   Spread, LoadBalancing} × CS {1, 2, 4, 8, 16}; 2 warm-up + 5 recorded launches each. Fine pass only for
 *   (stride1, ld_cluster, Default), first 4 clusters of every launch (--quick: Default, stride1, ld_cluster and
 *   ld_cluster_addr, 2 recorded launches).
 *   → results/09__dsmem_latency__chase.csv: cs, policy, variant, chain, cluster_id, src_rank, dst_rank, smid_src,
 *     smid_dst, gpc_src, gpc_dst, same_sm, cycles_per_hop, total_cycles, hops (one row per pair and launch)
 *   → results/09__dsmem_latency__fine.csv: cs, policy, variant, cluster_id, src_rank, dst_rank, smid_src, smid_dst,
 *     hop_idx, cycles (one row per timed hop)
 *
 * K1f loaded (variant f) — kernel k1f_loaded(mutex, zero, samples, place): 1024 threads per CTA; every CTA fills its
 *   first 192 KiB with sw[i] = i. Same turns (src, dst) as K1, but all 1024 threads of the source CTA stream the
 *   destination's 192 KiB: thread t, group g (0 … 5) issues 8 independent ld.shared::cluster.u32 of words
 *   (8g + j) · 1024 + t, j = 0 … 7 (stride 1 across the threads: each warp load is one contiguous 128 B); t0 =
 *   %clock64 after all values of the previous group have returned, t1 = read_clock64_after(sum of the group's 8 values:
 *   ptxas reorders the independent loads, so "after the last value" would wait for one load only; the sum adds three dependent
 *   three-input integer adds (IADD3 in the SASS), a few cycles, to every sample);
 *   sample = t1 − t0 (6 groups × 1024 threads = 6144 samples per pair). A chip-wide mutex (global atomicCAS by thread 0)
 *   lets only one source CTA on the whole GPU stream at a time, so every sample is one loaded pair without traffic from
 *   other clusters. CS {1, 2, 4, 8, 16}, Default policy, 2 warm-up + 5 recorded launches.
 *   Exact reductions on the host (all 6144 samples of a pair):
 *   → results/09__dsmem_latency__loaded.csv: cs, policy, cluster_id, src_rank, dst_rank, smid_src, smid_dst, threads,
 *     ilp, cta_mean_cycles_per_group, cta_p5, cta_p50, cta_p95, cta_min, cta_max, n_groups (one row per pair and launch;
 *     percentiles with linear interpolation)
 *   → results/09__dsmem_latency__loaded_hist.csv: cs, same_sm, same_tpc, cycles, count (every sample of the launch,
 *     pooled per (same_sm, same_tpc); TPC = smid / 2)
 *
 * K3 baselines — one-warp kernels pinned to an SM with claim_target_sm (2 × 132 blocks, the first block on the target
 *   SM works, the others exit; re-launched if no block reached it), on the first SM of every GPC of the 03 map:
 *     lds (k3_lds<F>, 16 KiB dynamic smem): the K1 chains (stride1, random16k; E = 4) in the CTA's own smem, read with
 *       ld.shared.u32 — form "idx" = [base + v] (as ld_cluster), form "addr" = [v] (as ld_cluster_addr); 64 warm +
 *       4096 timed hops; levels lds_idx / lds_addr.
 *     timer_pair (k3_timer): 256 × 32 pairs a = %clock64; b = read_clock64_after(x) with x already available — the
 *       timer overhead contained in every K1-fine per-hop value.
 *     l2_chase / dram_chase (k3_gchase): 04's global chain (one 128-B line per hop, random single cycle, seed
 *       kChainSeed), ld.global.cg.u64 p = [p]; 4 MiB: every launch first chases all 32768 lines untimed, then 2^20
 *       timed hops (L2, near: lines touched by this SM); 2 GiB: 2^22 timed hops per launch (HBM).
 *   lds and timer: 2 warm-up + 5 recorded launches; chases: 1 warm-up + 5 recorded launches.
 *   → results/09__dsmem_latency__baselines.csv: level, chain, smid, gpc, footprint_bytes, hops, cycles_per_hop,
 *     total_cycles (one row per launch; timer_pair: hops = number of pairs)
 *
 * ncu mode (profiled by src/run.sh): k1_chase<ld_cluster>, chain stride1, no fine pass, 2 clusters, CS 2 and CS 8,
 *   Default policy: one launch with the chase and one all-exit launch (chase = 0: setup and barriers only) of the same
 *   grid; manifest bin/ncu_manifest_k1.csv (one row per launch, in order, with the expected DSMEM load count).
 */
#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "common09.hpp"
#include "../../04__memory_hierarchy_latency/src/pchase_chain.hpp"

namespace cg = cooperative_groups;

static const char* EXP = "09__dsmem_latency_vs_cluster_size";
constexpr uint32_t kChainBytes = 16384, kStrideElems = 1024, kWarmHops = 64, kHops = 4096, kFineHops = 256;
constexpr uint32_t kFineClusters = 4, kFineOff = kChainBytes;  // fine buffer after the chain region
constexpr int kUnroll = 16;

enum Var { V_IDX, V_ADDR, V_GEN, V_VOL, V_RLX, V_ACQ, V_U64, V_V4 };
static const char* kVarName[] = {"ld_cluster", "ld_cluster_addr", "ld_generic", "ld_volatile",
                                 "ld_relaxed", "ld_acquire", "ld_u64", "ld_v4"};
template <int V> constexpr uint32_t kElem = V == V_U64 ? 8u : V == V_V4 ? 16u : 4u;

struct K1Rec { unsigned long long cycles; uint32_t off, valid; };

// ---------------------------------------------------------------- K1
// One dependent hop; the wide forms fold their other words into `sink` (off the dependency chain) so that ptxas keeps the
// full-width load (with only the chain word used it narrows them to a 32-bit LD).
template <int V>
__device__ __forceinline__ uint32_t hop(uint32_t base, const uint8_t* gbase, uint32_t v, uint32_t& sink) {
    if constexpr (V == V_IDX) return ld_dsmem_u32(base + v);
    else if constexpr (V == V_ADDR) return ld_dsmem_u32(v);
    else if constexpr (V == V_GEN) return *reinterpret_cast<const uint32_t*>(gbase + v);
    else if constexpr (V == V_VOL) return ld_volatile_cluster_u32(base + v);
    else if constexpr (V == V_RLX) return ld_relaxed_cluster_u32(base + v);
    else if constexpr (V == V_ACQ) return ld_acquire_cluster_u32(base + v);
    else if constexpr (V == V_U64) {
        const uint64_t x = ld_dsmem_u64(base + v);
        sink ^= (uint32_t)(x >> 32);
        return (uint32_t)x;
    } else {
        const uint4 x = ld_dsmem_v4_u32(base + v);
        sink ^= x.y ^ x.z ^ x.w;
        return x.x;
    }
}

template <int V>
__global__ void __launch_bounds__(32) k1_chase(const uint32_t* perm, uint32_t n_elem, uint32_t random, uint32_t fine_clusters,
                                              uint32_t chase, uint64_t zero, K1Rec* out, uint32_t* fine, uint32_t* place) {
    extern __shared__ __align__(16) uint8_t dyn[];
    constexpr uint32_t E = kElem<V>;
    const uint32_t lane = threadIdx.x;
    const ClusterRegs c = read_cluster_regs();
    const uint32_t rank = c.ctarank, cs = c.nctarank, cid = c.clusterid_x;
    const uint32_t region = smem_u32(dyn);
    const uint32_t own = mapa_u32(region, rank);
    for (uint32_t i = lane; i < n_elem; i += 32) {
        const uint32_t off = (random ? perm[i] : (i + 1) % n_elem) * E;
        uint32_t* w = reinterpret_cast<uint32_t*>(dyn + i * E);
        w[0] = V == V_ADDR ? own + off : off;
        for (uint32_t k = 1; k < E / 4; ++k) w[k] = 0;
    }
    record_place(place);
    __syncwarp();
    cluster_sync();  // chains written and visible; every CTA of the cluster exists
    if (chase) {
        for (uint32_t src = 0; src < cs; ++src) {
            if (rank == src && lane == 0) {
                for (uint32_t dst = 0; dst < cs; ++dst) {
                    const uint32_t base = mapa_u32(region, dst);
                    const uint8_t* gbase = nullptr;
                    if constexpr (V == V_GEN) gbase = cg::this_cluster().map_shared_rank(dyn, dst);
                    uint32_t v = V == V_ADDR ? base : 0u, sink = 0;
#pragma unroll 1
                    for (uint32_t i = 0; i < kWarmHops; ++i) v = hop<V>(base, gbase, v, sink);
                    const uint64_t t0 = read_clock64_after(v, zero);
#pragma unroll 1
                    for (uint32_t i = 0; i < kHops / kUnroll; ++i) {
#pragma unroll
                        for (int u = 0; u < kUnroll; ++u) v = hop<V>(base, gbase, v, sink);
                    }
                    const uint64_t t1 = read_clock64_after(v, zero);
                    const uint32_t idx = (cid * cs + src) * cs + dst;
                    out[idx] = K1Rec{t1 - t0, V == V_ADDR ? v - base : v, 1u + (uint32_t)(sink & zero)};
                    if (cid < fine_clusters) {
                        uint32_t* fb = reinterpret_cast<uint32_t*>(dyn + kFineOff);
#pragma unroll 1
                        for (uint32_t k = 0; k < kFineHops; ++k) {
                            const uint64_t a = read_clock64();
                            v = hop<V>(base, gbase, v, sink);
                            const uint64_t b = read_clock64_after(v, zero);
                            fb[k] = (uint32_t)(b - a);
                        }
                        for (uint32_t k = 0; k < kFineHops; ++k) fine[(size_t)idx * kFineHops + k] = fb[k];
                    }
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
constexpr uint32_t kLThreads = 1024, kLIlp = 8, kLGroups = 6, kLWords = kLThreads * kLIlp * kLGroups;  // 192 KiB
constexpr uint32_t kLSamples = kLThreads * kLGroups;

__global__ void __launch_bounds__(1024) k1f_loaded(uint32_t* mutex, uint64_t zero, uint32_t* samples, uint32_t* place) {
    extern __shared__ __align__(16) uint32_t sw[];
    const uint32_t tid = threadIdx.x;
    const ClusterRegs c = read_cluster_regs();
    const uint32_t rank = c.ctarank, cs = c.nctarank, cid = c.clusterid_x;
    for (uint32_t i = tid; i < kLWords; i += kLThreads) sw[i] = i;
    record_place(place);
    __syncthreads();
    cluster_sync();
    const uint32_t region = smem_u32(sw);
    uint32_t sink = 0;
    for (uint32_t src = 0; src < cs; ++src) {
        if (rank == src) {
            for (uint32_t dst = 0; dst < cs; ++dst) {
                if (tid == 0) {
                    while (atomicCAS(mutex, 0u, 1u) != 0u) nanosleep_ns(256);
                    __threadfence();
                }
                __syncthreads();
                const uint32_t base = mapa_u32(region, dst) + 4 * tid;
                uint32_t* out = samples + (size_t)((cid * cs + src) * cs + dst) * kLSamples;
                uint32_t acc = 0;
#pragma unroll 1
                for (uint32_t g = 0; g < kLGroups; ++g) {
                    const uint32_t a = base + g * kLIlp * kLThreads * 4;
                    const uint64_t t0 = read_clock64_after(acc, zero);
                    uint32_t x[kLIlp];
#pragma unroll
                    for (uint32_t j = 0; j < kLIlp; ++j) x[j] = ld_dsmem_u32(a + j * kLThreads * 4);
                    // ptxas reorders the 8 independent loads (it issued the last one first), so t1 waits for their sum:
                    // all 8 values (3 dependent IADD3) and acc carries the group into the next t0.
                    const uint32_t sum = x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7];
                    const uint64_t t1 = read_clock64_after(sum, zero);
                    acc += sum;
                    out[g * kLThreads + tid] = (uint32_t)(t1 - t0);
                }
                sink += acc;
                __syncthreads();
                if (tid == 0) {
                    __threadfence();
                    atomicExch(mutex, 0u);
                }
            }
        }
        cluster_sync();
    }
    if (sink == 0xffffffffu) place[0] = sink;  // keeps the loads (never true for these data)
}

// ---------------------------------------------------------------- K3 baselines
struct K3Out { unsigned long long cycles, sink; uint32_t smid; };

// F = 0: [base + v] (offset chain), F = 1: [v] (address chain).
template <int F>
__global__ void __launch_bounds__(32) k3_lds(const uint32_t* perm, uint32_t n_elem, uint32_t random, uint32_t target, uint32_t* claim,
                                            uint64_t zero, K3Out* out) {
    extern __shared__ __align__(16) uint32_t s[];
    if (!claim_target_sm(target, claim)) return;
    const uint32_t base = smem_u32(s);
    for (uint32_t i = threadIdx.x; i < n_elem; i += 32) {
        const uint32_t off = (random ? perm[i] : (i + 1) % n_elem) * 4;
        s[i] = F ? base + off : off;
    }
    __syncwarp();
    if (threadIdx.x != 0) return;
    uint32_t v = F ? base : 0u;
    auto step = [&](uint32_t x) {
        uint32_t r;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r) : "r"(F ? x : base + x) : "memory");
        return r;
    };
#pragma unroll 1
    for (uint32_t i = 0; i < kWarmHops; ++i) v = step(v);
    const uint64_t t0 = read_clock64_after(v, zero);
#pragma unroll 1
    for (uint32_t i = 0; i < kHops / kUnroll; ++i) {
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) v = step(v);
    }
    const uint64_t t1 = read_clock64_after(v, zero);
    *out = K3Out{t1 - t0, F ? v - base : v, read_smid()};
}

constexpr uint32_t kTimerReps = 256, kTimerPerRep = 32;
__global__ void __launch_bounds__(32) k3_timer(uint32_t target, uint32_t* claim, uint64_t zero, K3Out* out) {
    if (!claim_target_sm(target, claim)) return;
    if (threadIdx.x != 0) return;
    unsigned long long s = 0;
    uint64_t x = zero;
#pragma unroll 1
    for (uint32_t r = 0; r < kTimerReps; ++r) {
#pragma unroll
        for (uint32_t u = 0; u < kTimerPerRep; ++u) {
            const uint64_t a = read_clock64();
            const uint64_t b = read_clock64_after(x, zero);
            s += b - a;
            x = b & zero;
        }
    }
    *out = K3Out{s, x, read_smid()};
}

__device__ __forceinline__ uint64_t ld_cg_u64(uint64_t p) {
    uint64_t v; asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory"); return v; }
__global__ void __launch_bounds__(32) k3_gchase(const uint64_t* start, uint64_t warm, uint64_t hops, uint32_t target, uint32_t* claim,
                                               uint64_t zero, K3Out* out) {
    if (!claim_target_sm(target, claim)) return;
    if (threadIdx.x != 0) return;
    uint64_t p = (uint64_t)start;
#pragma unroll 1
    for (uint64_t i = 0; i < warm; ++i) p = ld_cg_u64(p);
    const uint64_t t0 = read_clock64_after(p, zero);
#pragma unroll 1
    for (uint64_t i = 0; i < hops / kUnroll; ++i) {
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) p = ld_cg_u64(p);
    }
    const uint64_t t1 = read_clock64_after(p, zero);
    *out = K3Out{t1 - t0, p, read_smid()};
}
// line i (qword 16·i) = address of line next[i]
__global__ void k3_fill(uint64_t* buf, const uint32_t* next, uint64_t n) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x)
        buf[i * chain::kQwordsPerLine] = (uint64_t)(buf + (uint64_t)next[i] * chain::kQwordsPerLine);
}

// ---------------------------------------------------------------- host
struct Ctx {
    hb::Run& run;
    Discards disc;
    std::vector<int> gpc;
    long long li = 0;
    explicit Ctx(hb::Run& r) : run(r), disc(r), gpc(load_gpc_map(r.dev().sm_count)) {}
};

static double pct(const std::vector<uint32_t>& sorted, double q) {
    const double x = q * (sorted.size() - 1);
    const size_t i = (size_t)x;
    const double f = x - i;
    return i + 1 < sorted.size() ? sorted[i] * (1 - f) + sorted[i + 1] * f : sorted[i];
}
static double median_of(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Host-side position after `hops` hops from element 0 (byte offset).
static uint32_t expected_off(bool random, const std::vector<uint32_t>& perm, uint32_t E, uint32_t hops) {
    if (!random) return (hops % kStrideElems) * E;
    uint32_t e = 0;
    for (uint32_t k = 0; k < hops; ++k) e = perm[e];
    return e * E;
}

struct K1Files {
    hb::CsvWriter chase, fine;
    K1Files()
        : chase(hb::Run::results("09__dsmem_latency__chase.csv"),
                hb::with_mandatory({"cs", "policy", "variant", "chain", "cluster_id", "src_rank", "dst_rank", "smid_src", "smid_dst",
                                    "gpc_src", "gpc_dst", "same_sm", "cycles_per_hop", "total_cycles", "hops"})),
          fine(hb::Run::results("09__dsmem_latency__fine.csv"),
               hb::with_mandatory({"cs", "policy", "variant", "cluster_id", "src_rank", "dst_rank", "smid_src", "smid_dst", "hop_idx", "cycles"})) {}
};

template <int V>
static void k1_point(Ctx& x, K1Files& f, int cs, cudaClusterSchedulingPolicy pol, bool random, bool fine, int launches) {
    constexpr uint32_t E = kElem<V>;
    const uint32_t n_elem = random ? kChainBytes / E : kStrideElems;
    const std::vector<uint32_t> perm = random ? chain::sattolo_cycle(n_elem) : std::vector<uint32_t>(1, 0);
    const uint32_t want = expected_off(random, perm, E, kWarmHops + kHops);
    Grid09 g(k1_chase<V>, cs, 32, pol);
    const int ncl = g.nclusters, npairs = ncl * cs * cs;
    const uint32_t nfine = fine ? (uint32_t)std::min(ncl, (int)kFineClusters) : 0u;
    DevBuf<uint32_t> d_perm(perm.size());
    CUDA_CHECK(cudaMemcpy(d_perm.d, perm.data(), perm.size() * 4, cudaMemcpyHostToDevice));
    DevBuf<K1Rec> out(npairs);
    DevBuf<uint32_t> place(4 * (size_t)g.blocks()), d_fine(std::max<size_t>(1, (size_t)nfine * cs * cs * kFineHops));
    const char* chain_name = random ? "random16k" : "stride1";
    const std::string label = std::string("k1_") + kVarName[V] + "_" + chain_name + "_" + policy_name(pol) + "_cs" + std::to_string(cs);
    auto launch = [&] {
        out.zero();
        place.zero();
        CUDA_CHECK(g.L.launch(k1_chase<V>, (const uint32_t*)d_perm.d, n_elem, random ? 1u : 0u, nfine, 1u, (uint64_t)0, out.d, d_fine.d, place.d));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return Placement(place.get(), cs, ncl);
    };
    for (int w = 0; w < kWarmups; ++w) launch();
    std::vector<double> all;
    for (int l = 0; l < launches; ++l) {
        auto r = recorded_launch(x.run, x.disc, label, x.li, launch);
        const hb::ClockSample& s = r.first;
        const Placement& p = r.second;
        const std::vector<K1Rec> o = out.get();
        const std::string pre = x.run.prefix(x.li, s);
        char b[320];
        for (int c = 0; c < ncl; ++c)
            for (int sr = 0; sr < cs; ++sr)
                for (int dr = 0; dr < cs; ++dr) {
                    const K1Rec& k = o[((size_t)c * cs + sr) * cs + dr];
                    if (!k.valid || k.off != want)
                        hb::die(label + ": cluster " + std::to_string(c) + " pair " + std::to_string(sr) + "->" + std::to_string(dr) +
                                " final offset " + std::to_string(k.off) + " expected " + std::to_string(want) + " (valid " + std::to_string(k.valid) + ")");
                    const uint32_t a = p.at(c, sr), d = p.at(c, dr);
                    const double cph = (double)k.cycles / kHops;
                    if (sr != dr) all.push_back(cph);
                    std::snprintf(b, sizeof b, ",%d,%s,%s,%s,%d,%d,%d,%u,%u,%d,%d,%d,%.4f,%llu,%u", cs, policy_name(pol), kVarName[V], chain_name, c,
                                  sr, dr, a, d, x.gpc[a], x.gpc[d], (int)(a == d), cph, k.cycles, kHops);
                    f.chase.add(pre + b);
                }
        if (nfine) {
            const std::vector<uint32_t> fh = d_fine.get();
            for (uint32_t c = 0; c < nfine; ++c)
                for (int sr = 0; sr < cs; ++sr)
                    for (int dr = 0; dr < cs; ++dr) {
                        const size_t idx = ((size_t)c * cs + sr) * cs + dr;
                        for (uint32_t h = 0; h < kFineHops; ++h) {
                            std::snprintf(b, sizeof b, ",%d,%s,%s,%u,%d,%d,%u,%u,%u,%u", cs, policy_name(pol), kVarName[V], c, sr, dr,
                                          p.at(c, sr), p.at(c, dr), h, fh[idx * kFineHops + h]);
                            f.fine.add(pre + b);
                        }
                    }
        }
        f.chase.flush();
        f.fine.flush();
        ++x.li;
    }
    f.chase.sync();
    f.fine.sync();
    std::printf("k1 %-16s %-9s %-13s cs %2d: %3d clusters (max active %3d), remote median %.2f cycles/hop%s\n", kVarName[V], chain_name,
                policy_name(pol), cs, ncl, g.max_active, cs > 1 ? median_of(all) : 0.0, cs == 1 ? " (cs 1: self only)" : "");
    if (cs == 1) {
        // print the loopback value too
        const std::vector<K1Rec> o = out.get();
        std::vector<double> v;
        for (const K1Rec& k : o) v.push_back((double)k.cycles / kHops);
        std::printf("    cs 1 own-CTA median %.3f cycles/hop\n", median_of(v));
    }
}

static void run_k1(hb::Run& run, bool quick) {
    Ctx x(run);
    K1Files f;
    const int launches = quick ? kQuickLaunches : kLaunches;
    const int css[] = {1, 2, 4, 8, 16};
    run.lock_clocks_note("k1", "before");
    for (cudaClusterSchedulingPolicy pol : kPolicies) {
        if (quick && pol != cudaClusterSchedulingPolicyDefault) continue;
        const bool def = pol == cudaClusterSchedulingPolicyDefault;
        for (int cs : css) k1_point<V_IDX>(x, f, cs, pol, false, def, launches);
        for (int cs : css) k1_point<V_ADDR>(x, f, cs, pol, false, false, launches);
        if (quick) continue;
        for (int cs : css) k1_point<V_IDX>(x, f, cs, pol, true, false, launches);
        for (int cs : css) k1_point<V_ADDR>(x, f, cs, pol, true, false, launches);
        for (int cs : css) k1_point<V_GEN>(x, f, cs, pol, false, false, launches);
        for (int cs : css) k1_point<V_VOL>(x, f, cs, pol, false, false, launches);
        for (int cs : css) k1_point<V_RLX>(x, f, cs, pol, false, false, launches);
        for (int cs : css) k1_point<V_ACQ>(x, f, cs, pol, false, false, launches);
        for (int cs : css) k1_point<V_U64>(x, f, cs, pol, false, false, launches);
        for (int cs : css) k1_point<V_V4>(x, f, cs, pol, false, false, launches);
    }
    run.lock_clocks_note("k1", "after");
    std::printf("k1: %d launches discarded\n", x.disc.n);
}

static void run_loaded(hb::Run& run, bool quick) {
    Ctx x(run);
    hb::CsvWriter csv(hb::Run::results("09__dsmem_latency__loaded.csv"),
                      hb::with_mandatory({"cs", "policy", "cluster_id", "src_rank", "dst_rank", "smid_src", "smid_dst", "threads", "ilp",
                                          "cta_mean_cycles_per_group", "cta_p5", "cta_p50", "cta_p95", "cta_min", "cta_max", "n_groups"}));
    hb::CsvWriter hist(hb::Run::results("09__dsmem_latency__loaded_hist.csv"),
                       hb::with_mandatory({"cs", "same_sm", "same_tpc", "cycles", "count"}));
    const int launches = quick ? kQuickLaunches : kLaunches;
    const auto pol = cudaClusterSchedulingPolicyDefault;
    DevBuf<uint32_t> mutex(1);
    run.lock_clocks_note("loaded", "before");
    for (int cs : {1, 2, 4, 8, 16}) {
        Grid09 g(k1f_loaded, cs, kLThreads, pol);
        const int ncl = g.nclusters, npairs = ncl * cs * cs;
        DevBuf<uint32_t> samples((size_t)npairs * kLSamples), place(4 * (size_t)g.blocks());
        const std::string label = "loaded_cs" + std::to_string(cs);
        auto launch = [&] {
            mutex.zero();
            place.zero();
            CUDA_CHECK(g.L.launch(k1f_loaded, mutex.d, (uint64_t)0, samples.d, place.d));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            return Placement(place.get(), cs, ncl);
        };
        for (int w = 0; w < kWarmups; ++w) launch();
        std::vector<double> remote_means;
        for (int l = 0; l < launches; ++l) {
            auto r = recorded_launch(x.run, x.disc, label, x.li, launch);
            const Placement& p = r.second;
            const std::string pre = x.run.prefix(x.li, r.first);
            const std::vector<uint32_t> sm = samples.get();
            std::vector<unsigned long long> h[2][2];  // h[same_sm][same_tpc][cycles] = count
            char b[320];
            std::vector<uint32_t> v(kLSamples);
            for (int c = 0; c < ncl; ++c)
                for (int sr = 0; sr < cs; ++sr)
                    for (int dr = 0; dr < cs; ++dr) {
                        const uint32_t a = p.at(c, sr), d = p.at(c, dr);
                        const int same_sm = a == d, same_tpc = !same_sm && a / 2 == d / 2;
                        std::copy_n(sm.begin() + (((size_t)c * cs + sr) * cs + dr) * kLSamples, kLSamples, v.begin());
                        double sum = 0;
                        std::vector<unsigned long long>& hc = h[same_sm][same_tpc];
                        for (uint32_t s : v) {
                            sum += s;
                            if (s >= hc.size()) hc.resize((size_t)s + 1, 0ull);
                            ++hc[s];
                        }
                        std::sort(v.begin(), v.end());
                        const double mean = sum / kLSamples;
                        if (!same_sm) remote_means.push_back(mean);
                        std::snprintf(b, sizeof b, ",%d,%s,%d,%d,%d,%u,%u,%u,%u,%.3f,%.1f,%.1f,%.1f,%u,%u,%u", cs, policy_name(pol), c, sr, dr, a, d,
                                      kLThreads, kLIlp, mean, pct(v, 0.05), pct(v, 0.5), pct(v, 0.95), v.front(), v.back(), kLSamples);
                        csv.add(pre + b);
                    }
            for (int ss = 0; ss < 2; ++ss)
                for (int st = 0; st < 2; ++st)
                    for (size_t cyc = 0; cyc < h[ss][st].size(); ++cyc)
                        if (h[ss][st][cyc]) {
                            std::snprintf(b, sizeof b, ",%d,%d,%d,%zu,%llu", cs, ss, st, cyc, h[ss][st][cyc]);
                            hist.add(pre + b);
                        }
            csv.flush();
            hist.flush();
            ++x.li;
        }
        csv.sync();
        hist.sync();
        std::printf("loaded cs %2d: %3d clusters, median of remote pair means %.1f cycles per 8-load group\n", cs, ncl, median_of(remote_means));
    }
    run.lock_clocks_note("loaded", "after");
    std::printf("loaded: %d launches discarded\n", x.disc.n);
}

// One claimed launch on SM `target`; re-launched if no block ran there (counted).
template <class L>
static int claimed(L launch, hb::SmClaim& claim, uint32_t target) {
    for (int r = 0; r <= kMaxTries; ++r) {
        claim.reset();
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        if (claim.ok(target)) return r;
    }
    hb::die("no block ran on SM " + std::to_string(target) + " after retries");
}

static void run_k3(hb::Run& run, bool quick) {
    Ctx x(run);
    hb::CsvWriter csv(hb::Run::results("09__dsmem_latency__baselines.csv"),
                      hb::with_mandatory({"level", "chain", "smid", "gpc", "footprint_bytes", "hops", "cycles_per_hop", "total_cycles"}));
    const int launches = quick ? kQuickLaunches : kLaunches;
    const int nb = hb::SmClaim::blocks(run.dev().sm_count);
    int ngpc = 0;
    for (int g : x.gpc) ngpc = std::max(ngpc, g + 1);
    std::vector<uint32_t> sms;
    for (int g = 0; g < ngpc; ++g) sms.push_back((uint32_t)first_sm_of_gpc(x.gpc, g));
    if (quick) sms.resize(1);
    hb::SmClaim claim;
    DevBuf<K3Out> out(1);
    // chains
    const std::vector<uint32_t> perm_s = chain::sattolo_cycle(kChainBytes / 4);
    DevBuf<uint32_t> d_perm(perm_s.size());
    CUDA_CHECK(cudaMemcpy(d_perm.d, perm_s.data(), perm_s.size() * 4, cudaMemcpyHostToDevice));
    const uint64_t n_l2 = (4ull << 20) / chain::kLineBytes, n_dram = (2ull << 30) / chain::kLineBytes;
    DevBuf<uint64_t> l2buf(n_l2 * chain::kQwordsPerLine), drambuf(n_dram * chain::kQwordsPerLine);
    const std::pair<uint64_t*, uint64_t> bufs[] = {{l2buf.d, n_l2}, {drambuf.d, n_dram}};
    for (const auto& [buf, n] : bufs) {
        const std::vector<uint32_t> next = chain::sattolo_cycle((uint32_t)n);
        DevBuf<uint32_t> d_next(n);
        CUDA_CHECK(cudaMemcpy(d_next.d, next.data(), n * 4, cudaMemcpyHostToDevice));
        k3_fill<<<1024, 256>>>(buf, d_next.d, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    run.lock_clocks_note("k3", "before");
    auto record = [&](const std::string& level, const char* chain_name, uint32_t sm, uint64_t fp, uint64_t hops, int warm, auto&& launch) {
        int retries = 0;
        for (int w = 0; w < warm; ++w) retries += claimed(launch, claim, sm);
        std::vector<double> v;
        for (int l = 0; l < launches; ++l) {
            hb::ClockSample s;
            for (int t = 0;; ++t) {
                if (t >= kMaxTries) hb::die(level + ": no clean launch");
                retries += claimed(launch, claim, sm);
                s = run.clock_note("k3_" + level, "launch", x.li);
                if (!s.bad()) break;
                x.disc.add("k3_" + level + "_sm" + std::to_string(sm), "clock", x.li, s);
            }
            const K3Out o = out.get()[0];
            if (o.smid != sm) hb::die("k3: claimed block reported smid " + std::to_string(o.smid));
            const double cph = (double)o.cycles / hops;
            v.push_back(cph);
            char b[256];
            std::snprintf(b, sizeof b, ",%s,%s,%u,%d,%llu,%llu,%.4f,%llu", level.c_str(), chain_name, sm, x.gpc[sm], (unsigned long long)fp,
                          (unsigned long long)hops, cph, o.cycles);
            csv.add(run.prefix(x.li, s) + b);
            csv.flush();
            ++x.li;
        }
        csv.sync();
        std::printf("k3 %-10s %-9s sm %3u (G%d): %.3f cycles/hop (median), re-launches %d\n", level.c_str(), chain_name, sm, x.gpc[sm],
                    median_of(v), retries);
    };
    for (uint32_t sm : sms) {
        for (int random = 0; random < 2; ++random) {
            const uint32_t n = random ? kChainBytes / 4 : kStrideElems;
            const char* cn = random ? "random16k" : "stride1";
            const uint32_t want = expected_off(random, perm_s, 4, kWarmHops + kHops);
            for (int F = 0; F < 2; ++F) {
                auto launch = [&] {
                    hb::ClusterLaunch cl;
                    cl.grid = dim3(nb);
                    cl.block = dim3(32);
                    cl.smem = kChainBytes;
                    if (F) CUDA_CHECK(cl.launch(k3_lds<1>, (const uint32_t*)d_perm.d, n, (uint32_t)random, sm, claim.ptr(), (uint64_t)0, out.d));
                    else CUDA_CHECK(cl.launch(k3_lds<0>, (const uint32_t*)d_perm.d, n, (uint32_t)random, sm, claim.ptr(), (uint64_t)0, out.d));
                };
                record(F ? "lds_addr" : "lds_idx", cn, sm, (uint64_t)n * 4, kHops, kWarmups, launch);
                if (out.get()[0].sink != want) hb::die("k3 lds: wrong final offset");
            }
        }
        record("timer_pair", "-", sm, 0, (uint64_t)kTimerReps * kTimerPerRep, kWarmups,
               [&] { k3_timer<<<nb, 32>>>(sm, claim.ptr(), 0, out.d); });
        record("l2_chase", "random", sm, n_l2 * chain::kLineBytes, 1ull << 20, 1,
               [&] { k3_gchase<<<nb, 32>>>(l2buf.d, n_l2, 1ull << 20, sm, claim.ptr(), 0, out.d); });
        record("dram_chase", "random", sm, n_dram * chain::kLineBytes, quick ? 1ull << 18 : 1ull << 22, 1,
               [&] { k3_gchase<<<nb, 32>>>(drambuf.d, 0, quick ? 1ull << 18 : 1ull << 22, sm, claim.ptr(), 0, out.d); });
    }
    run.lock_clocks_note("k3", "after");
}

static void run_ncu(hb::Run& run) {
    hb::CsvWriter man("bin/ncu_manifest_k1.csv", hb::with_mandatory({"kernel", "cs", "nclusters", "chase", "expected_dshared_ld"}));
    long long li = 0;
    const std::vector<uint32_t> perm(1, 0);
    DevBuf<uint32_t> d_perm(1);
    for (int cs : {2, 8}) {
        Grid09 g(k1_chase<V_IDX>, cs, 32, cudaClusterSchedulingPolicyDefault, 2);
        DevBuf<K1Rec> out((size_t)g.nclusters * cs * cs);
        DevBuf<uint32_t> place(4 * (size_t)g.blocks()), fine(1);
        for (uint32_t chase : {1u, 0u}) {
            CUDA_CHECK(g.L.launch(k1_chase<V_IDX>, (const uint32_t*)d_perm.d, kStrideElems, 0u, 0u, chase, (uint64_t)0, out.d, fine.d, place.d));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            const hb::ClockSample s = run.clock_note("ncu_k1_cs" + std::to_string(cs) + "_chase" + std::to_string(chase), "launch", li);
            const unsigned long long expect = chase ? (unsigned long long)g.nclusters * cs * cs * (kWarmHops + kHops) : 0ull;
            man.add(run.prefix(li++, s) + ",k1_chase_ld_cluster," + std::to_string(cs) + "," + std::to_string(g.nclusters) + "," +
                    std::to_string(chase) + "," + std::to_string(expect));
            man.flush();
        }
    }
    man.sync();
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "k1";
    const bool quick = hb::has_flag(argc, argv, "--quick");
    hb::Run run(EXP, argc, argv);
    if (mode == "clockcheck") run.lock_clocks_note(hb::arg_str(argc, argv, "--label", "check"), "check");
    else if (mode == "k1") run_k1(run, quick);
    else if (mode == "loaded") run_loaded(run, quick);
    else if (mode == "k3") run_k3(run, quick);
    else if (mode == "ncu") run_ncu(run);
    else hb::die("unknown mode " + mode);
    run.complete();
    return 0;
}
