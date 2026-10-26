/*
 * dsmem_bw.cu — experiment 11: bandwidth of the SM-to-SM network (distributed shared memory, DSMEM) inside thread-block
 * clusters, for remote loads, remote stores, st.async and TMA shared→shared puts / reduce-puts, versus cluster size,
 * traffic pattern, threads × ILP, TMA command size, scheduling policy and the number of clusters running at once;
 * the local-SRAM pressure control (K4) and a many-block launch in the style of a published ring benchmark (p1).
 *
 * Usage: bin/dsmem_bw {A|S|B|C|D|E|F|P1|K4|ncu|plan|clockcheck} [--quick] [--label L]. One invocation = one run_id
 *   (results/11__dsmem_bandwidth_vs_cluster_size__runs.csv). A launch whose cluster ranks do not sit on distinct SMs, or
 *   whose NVML sample after the launch shows a throttle reason or an SM clock below the lock, is discarded, logged in
 *   results/11__dsmem_bw__discards.csv and repeated (≤ 20 tries).
 *
 * Placement: 1-D clusters of CS CTAs, 200 KB dynamic smem (→ exactly 1 CTA per SM), explicit cluster attribute and
 *   scheduling policy. launch_mode "single": 1 cluster per launch (idle chip; the hardware chooses its SMs);
 *   "full": cudaOccupancyMaxActiveClusters clusters (66 / 30 / 15 / 7 for CS 2 / 4 / 8 / 16), one wave. Every CTA
 *   records %smid, %clusterid.x, %cluster_ctarank; the host maps smid → GPC with the latest 03 map.
 *
 * K1–K3 — kernel k_stream<OP, ILP, UNI>(P11): CTA smem = 96 KiB source tile (element e = {e, rank + 1, e ^ 0x5a5a5a5a,
 *   rank}) + 96 KiB destination tile + one mbarrier per (source rank, pass) + scratch. The issuing work of a CTA is a
 *   list of warp groups (common11.hpp plan_groups: target rank, element range, warps, operation) given by the pattern.
 *   A group moves payload_bytes per pass, every 16 B element once per pass; element order per thread
 *   e = e0 + (i·ILP + j)·Tg + tg (Tg = threads of the group): each warp access is 512 contiguous bytes; elements past
 *   e1 are skipped (at 1024 threads one step of ILP 4 covers 64 KiB, so the payload limits the loads in flight).
 *   Each launch runs `samples` samples; one sample:
 *     arm: the target CTA arrives with expect_tx(bytes that source s sends in pass q) on barrier (s, q) — computed by
 *          evaluating every source's plan — then cluster.sync() (all CTAs armed, every CTA exists);
 *     timed region (every thread: t0 = %clock64 + %globaltimer right after the barrier):
 *       ld        ILP independent ld.shared::cluster.v4.u32 per step, summed (.y) / XOR-ed (other words) into
 *                 registers; t1 = read_timers_after(sum ^ xor): after the thread's last load has returned.
 *       st        st.shared::cluster.v4.u32 of {e, pass, rank + 1, sample}; then fence.acq_rel.cluster (MEMBAR: waits
 *                 until the thread's remote stores are performed — the release part of cluster.sync()); t1 after it.
 *       st_async  st.async.v4.b32 with complete_tx(16 B) on the target's barrier (rank, pass); t1 after the last issue.
 *       tma_put / tma_red  one thread (lane 0 of the first warp of the CTA's first TMA group) issues
 *                 cp.async.bulk.shared::cluster.shared::cta (or cp.reduce.async.bulk ….add.u32) commands of
 *                 chunk_bytes from its source tile into the target's destination tile at the same offset, group after
 *                 group, completing on the target's barrier (rank, pass); t1 after the last issue.
 *       target    for st_async / TMA: lane 0 of the CTA's last warp waits (mbarrier.try_wait.parity.acquire.cluster
 *                 loop, timeout 4·10⁹ cycles → error) on every armed barrier; t1 after the last completed wait.
 *     Per CTA and sample up to three records (warp-reduced, then smem atomics, outside the timed region):
 *       side issuer (slot 0; slot 1 = the store/TMA stream of the mixed kernels): cycles = max t1 − min t0 over the
 *       participating threads, ns the same with %globaltimer, bytes = bytes the CTA's groups moved;
 *       side target: cycles from the waiter's t0 to its last completed wait, bytes = bytes received.
 *     ld check: sum of the loaded .y words = Σ elements × (owner rank + 1) is compared with the host's value.
 *   UNI = uniform_random: target per 16 KiB chunk from a table filled before the sample.
 *   Address of a re-read element: base + (pass & zero) (zero = 0 at run time) so that ptxas cannot hoist loads.
 *
 * Steps (threads 1024, ILP 4, Default policy, chunk 16 KiB unless swept; payload 16/32/48/64/80/96 KiB, passes 1;
 *   single: 2 warm-up + 5 recorded launches × 6 samples; full: × 2 samples):
 *   A  ring (+1) × {ld, st, st_async, tma_put, tma_red} × CS {2, 4, 8, 16}, single and full.
 *   S  full, ring (+1) × the 5 operations × CS, payload 96 KiB × passes {4, 8, 16, 32}, 1 sample (GPU-wide TB/s from
 *      the slope of the cudaEvent time over the bytes).
 *   B  threads {128, 256, 512, 1024} × ILP {1, 2, 4, 8} × CS × {pair_uni, ring} for ld and st; st_async × threads.
 *   C  every pattern × {ld, st, tma_put} (+ mixed with st or TMA), CS {2, 4, 8, 16} single, CS 8 full.
 *   D  CS 16: broadcast_in with k = 1 … 15 readers (ld), broadcast_out to k = 1 … 15 targets (st, tma_put).
 *   E  TMA chunk {128 B … 16 KiB} × {tma_put, tma_red} × {pair_uni, ring} at CS 8.
 *   F  policies Spread, LoadBalancing × ring {ld, st, tma_put} × CS, single and full.
 *   → results/11__dsmem_bw__samples.csv: kernel, pattern, pattern_k, cs, policy, threads, ilp, chunk_bytes,
 *     payload_bytes, passes, cluster_id, rank, smid, peer_smid (−1: several), gpc, side, stream, sample, cycles, ns,
 *     t0_gt_ns (relative to the launch's earliest start), bytes_moved, launch_mode, step (one row per record)
 *   → results/11__dsmem_bw__launches.csv: per launch: configuration, nclusters, max_active_clusters, active_ctas,
 *     total_bytes (issuer side, all samples), samples, event_ns (cudaEvent pair around the launch), makespan_ns.
 *
 * K4 — kernel k_victim<OPR>(V11), CS 2, 1024 threads: rank 0 (reader) streams rank 1's source tile 32 times with
 *   reader_threads threads (ld, ILP 4) or stores into rank 1's destination tile (st + fence); then one reader thread
 *   sets a flag in rank 1's smem (st.volatile.shared::cluster). Rank 1 (victim) runs victim_warps warps, each streaming
 *   its 1/W share of its own source tile with local ld.shared.v4 (ILP 2, no tail) pass after pass, polling the flag
 *   (local ld.volatile.shared) after every pass; victim bytes = completed passes × share. reader_on = 0: rank 0 sets the flag after 160 000 cycles instead
 *   (victim alone). Configurations: {ld, st} × reader_threads {256, 1024} × W {0, 4, 8, 16, 32} + victim alone ×
 *   W {4, 8, 16, 32}; single and full (66 clusters); 2 warm-up + 5 launches × 6 samples.
 *   → results/11__dsmem_bw__victim.csv: kernel, reader_threads, victim_warps, reader_on, launch_mode, cluster_id, rank,
 *     smid, peer_smid, side (reader / victim), sample, cycles, ns, bytes_moved, bpc
 *
 * P1 — kernel k_p1<OP>: 50 160 blocks × 1024 threads, 16 KiB dynamic smem, __launch_bounds__(1024, 2) (2 blocks per SM),
 *   clusters of CS {2, 4, 8, 16}, ring (+1): every thread moves one 16 B element of the peer's 16 KiB tile per pass,
 *   passes {1, 4, 16}; ld (XOR), st, tma_put (thread 0, 16 KiB per pass, target waits on its barrier). Timed only by
 *   the cudaEvent pair (no per-block number); 2 warm-up + 5 launches.
 *   → 11__dsmem_bw__launches.csv rows (step P1, launch_mode p1_style); results/11__dsmem_bw__p1_blocks.csv: blocks
 *     run by every SM in the launch (exact placement histogram).
 *
 * ncu mode (profiled by src/run.sh): every operation × pattern of step C at CS 8 single, 96 KiB, 1 sample, one launch
 *   each, plus ring at CS 8 full; manifest bin/ncu_manifest.csv (launch order, configuration, expected bytes).
 * plan mode: prints the warp groups of every pattern (for review; no GPU work besides the device query).
 */
#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "../../09__dsmem_latency_vs_cluster_size/src/common09.hpp"
#include "common11.hpp"

static const char* EXP = "11__dsmem_bandwidth_vs_cluster_size";
constexpr uint32_t kTile = 96u * 1024u, kTileElems = kTile / 16u;
constexpr uint32_t kDstOff = kTile, kBarOff = 2u * kTile, kMaxBars = 512, kFlagOff = kBarOff + kMaxBars * 8u;
constexpr size_t kSmem = 200u * 1024u;
constexpr uint32_t kMaxCt = 256;  // uniform_random table entries (passes × chunks)
constexpr int kSamplesSingle = 6, kSamplesFull = 2;
constexpr unsigned long long kWaitTimeout = 4000000000ull;

struct Rec11 { unsigned long long cycles, ns, t0_gt, bytes; uint32_t peer, valid, chk, stream; };
struct P11 {
    uint32_t pat, k, payload, chunk, samples, passes, seed, pad;
    uint64_t zero;
    Perm16 perm;
    Rec11* out;
    uint32_t* place;
    uint32_t* err;
};

// ---------------------------------------------------------------- device: streams of one warp group
template <int ILP, bool UNI>
__device__ __forceinline__ void ld_stream(const Group11& g, uint32_t warp, uint32_t lane, uint32_t passes, uint32_t src_u32,
                                          const uint8_t* ct, uint32_t zero, uint32_t& sy, uint32_t& sx) {
    const uint32_t Tg = (g.w1 - g.w0) * 32u, tg = (warp - g.w0) * 32u + lane;
    const uint32_t base = UNI ? 0u : mapa_u32(src_u32, g.target);
#pragma unroll 1
    for (uint32_t pass = 0; pass < passes; ++pass) {
        const uint32_t pb = base + (pass & zero);
#pragma unroll 1
        for (uint32_t e = g.e0 + tg; e < g.e1; e += ILP * Tg) {
            uint4 x[ILP];
#pragma unroll
            for (int j = 0; j < ILP; ++j) {
                const uint32_t ej = e + j * Tg;
                uint32_t a;
                if constexpr (UNI) a = mapa_u32(src_u32, ct[pass * 6u + ej / kChunkElems]) + ej * 16u + (pass & zero);
                else a = pb + ej * 16u;
                x[j] = ej < g.e1 ? ld_dsmem_v4_u32(a) : make_uint4(0, 0, 0, 0);
            }
#pragma unroll
            for (int j = 0; j < ILP; ++j) { sy += x[j].y; sx ^= x[j].x ^ x[j].z ^ x[j].w; }
        }
    }
}

template <int ILP, bool UNI, bool ASY>
__device__ __forceinline__ void st_stream(const Group11& g, uint32_t warp, uint32_t lane, uint32_t passes, uint32_t dst_u32,
                                          uint32_t bar_u32, uint32_t rank, uint32_t sample, const uint8_t* ct, uint32_t zero) {
    const uint32_t Tg = (g.w1 - g.w0) * 32u, tg = (warp - g.w0) * 32u + lane;
#pragma unroll 1
    for (uint32_t pass = 0; pass < passes; ++pass) {
        const uint32_t pb = UNI ? 0u : mapa_u32(dst_u32, g.target) + (pass & zero);
        const uint32_t rb = UNI ? 0u : mapa_u32(bar_u32 + (rank * passes + pass) * 8u, g.target);
#pragma unroll 1
        for (uint32_t e = g.e0 + tg; e < g.e1; e += ILP * Tg) {
#pragma unroll
            for (int j = 0; j < ILP; ++j) {
                const uint32_t ej = e + j * Tg;
                if (ej < g.e1) {
                    uint32_t a, r = rb;
                    if constexpr (UNI) {
                        const uint32_t t = ct[pass * 6u + ej / kChunkElems];
                        a = mapa_u32(dst_u32, t) + ej * 16u;
                        if constexpr (ASY) r = mapa_u32(bar_u32 + (rank * passes + pass) * 8u, t);
                    } else {
                        a = pb + ej * 16u;
                    }
                    const uint4 d = make_uint4(ej, pass, rank + 1u, sample);
                    if constexpr (ASY) st_async_v4_b32(a, d, r);
                    else st_dsmem_v4_u32(a, d);
                }
            }
        }
    }
}

// One thread issues the bulk commands of all TMA groups of its CTA: per pass, group after group, each group's
// commands back to back with the destination and barrier addresses in registers and 8 unguarded commands per loop
// iteration (uniform_random: target of every command from the per-chunk table).
template <bool RED>
__device__ __forceinline__ void tma_group(uint32_t d0, uint32_t src_u32, uint32_t off, uint32_t end, uint32_t chunk, uint32_t rb) {
#pragma unroll 1
    for (; off + 8u * chunk <= end; off += 8u * chunk) {  // 8 unguarded commands per iteration
#pragma unroll
        for (uint32_t u = 0; u < 8u; ++u) {
            if constexpr (RED) bulk_red_s2s_add_u32(d0 + off + u * chunk, src_u32 + off + u * chunk, chunk, rb);
            else bulk_s2s_dsmem(d0 + off + u * chunk, src_u32 + off + u * chunk, chunk, rb);
        }
    }
#pragma unroll 1
    for (; off < end; off += chunk) {
        const uint32_t bytes = min(chunk, end - off);
        if constexpr (RED) bulk_red_s2s_add_u32(d0 + off, src_u32 + off, bytes, rb);
        else bulk_s2s_dsmem(d0 + off, src_u32 + off, bytes, rb);
    }
}
template <bool RED, bool UNI>
__device__ __forceinline__ void tma_issue(const Group11* gs, int ng, uint32_t passes, uint32_t chunk, uint32_t src_u32,
                                          uint32_t dst_u32, uint32_t bar_u32, uint32_t rank, const uint8_t* ct) {
#pragma unroll 1
    for (uint32_t pass = 0; pass < passes; ++pass) {
#pragma unroll 1
        for (int i = 0; i < ng; ++i) {
            const Group11 g = gs[i];
            if (g.op != OP_TMA && g.op != OP_TRED) continue;
            const uint32_t end = g.e1 * 16u;
            if constexpr (UNI) {
#pragma unroll 1
                for (uint32_t off = g.e0 * 16u; off < end; off += chunk) {
                    const uint32_t t = ct[pass * 6u + off / 16384u], bytes = min(chunk, end - off);
                    const uint32_t d = mapa_u32(dst_u32, t) + off, rb = mapa_u32(bar_u32 + (rank * passes + pass) * 8u, t);
                    if constexpr (RED) bulk_red_s2s_add_u32(d, src_u32 + off, bytes, rb);
                    else bulk_s2s_dsmem(d, src_u32 + off, bytes, rb);
                }
            } else {
                tma_group<RED>(mapa_u32(dst_u32, g.target), src_u32, g.e0 * 16u, end, chunk,
                               mapa_u32(bar_u32 + (rank * passes + pass) * 8u, g.target));
            }
        }
    }
}

// warp reduction of one record slot (slot warp-uniform), then shared atomics by lane 0
__device__ __forceinline__ void reduce_slot(bool part, TimerPair a, TimerPair b, unsigned long long* c0, unsigned long long* c1,
                                            unsigned long long* n0, unsigned long long* n1, uint32_t sy, uint32_t* chk) {
    unsigned long long x0 = part ? a.cycles : ~0ull, x1 = part ? b.cycles : 0ull, y0 = part ? a.ns : ~0ull, y1 = part ? b.ns : 0ull;
    uint32_t s = part ? sy : 0u;
    for (int o = 16; o; o >>= 1) {
        x0 = min(x0, __shfl_xor_sync(0xffffffffu, x0, o));
        x1 = max(x1, __shfl_xor_sync(0xffffffffu, x1, o));
        y0 = min(y0, __shfl_xor_sync(0xffffffffu, y0, o));
        y1 = max(y1, __shfl_xor_sync(0xffffffffu, y1, o));
        s += __shfl_xor_sync(0xffffffffu, s, o);
    }
    if ((threadIdx.x & 31) == 0 && x1 != 0ull) {  // some lane of the warp participated
        atomicMin(c0, x0); atomicMax(c1, x1); atomicMin(n0, y0); atomicMax(n1, y1);
        if (chk) atomicAdd(chk, s);
    }
}

// ---------------------------------------------------------------- K1–K3 kernel
template <int OP, int ILP, bool UNI>
__global__ void __launch_bounds__(1024, 1) k_stream(P11 p) {
    extern __shared__ __align__(1024) uint8_t dyn[];
    __shared__ Group11 s_g[kMaxGroups];
    __shared__ int s_ng;
    __shared__ uint32_t s_exp[kMaxBars];      // bytes expected on barrier (source, pass) in this sample
    __shared__ uint32_t s_ph[kMaxBars / 32];  // phase parity per barrier
    __shared__ uint8_t s_ct[kMaxCt];          // uniform_random: target of (pass, chunk)
    __shared__ unsigned long long s_c0[3], s_c1[3], s_n0[3], s_n1[3];
    __shared__ uint32_t s_chk[2];
    constexpr bool ASYNC = OP == OP_STA || OP == OP_TMA || OP == OP_TRED || OP == OP_MIX_TMA;
    constexpr bool MIXED = OP == OP_MIX_ST || OP == OP_MIX_TMA;
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nwarps = blockDim.x >> 5;
    const ClusterRegs c = read_cluster_regs();
    const uint32_t rank = c.ctarank, cs = c.nctarank;
    const uint32_t nelem = p.payload / 16u, passes = p.passes, nbars = cs * passes;
    const uint32_t src_u32 = smem_u32(dyn), dst_u32 = src_u32 + kDstOff, bar_u32 = src_u32 + kBarOff;
    const uint32_t zero = (uint32_t)p.zero;
    uint4* src = reinterpret_cast<uint4*>(dyn);
    for (uint32_t e = tid; e < kTileElems; e += blockDim.x) src[e] = make_uint4(e, rank + 1u, e ^ 0x5a5a5a5au, rank);
    record_place(p.place);
    if (tid == 0) s_ng = plan_groups(OP, p.pat, cs, p.k, rank, nwarps, nelem, p.perm, s_g);
    if (ASYNC) {
        for (uint32_t b = tid; b < nbars; b += blockDim.x) mbar_init(bar_u32 + b * 8u, 1);
        for (uint32_t w = tid; w < kMaxBars / 32; w += blockDim.x) s_ph[w] = 0;
        fence_mbarrier_init();
    }
    fence_proxy_async_shared_cta();  // source tile written by generic stores, read by the TMA unit
    __syncthreads();
    // my group (warp-uniform) and the TMA issuer
    int my = -1, first_tma = -1;
    const int ng = s_ng;
    for (int i = 0; i < ng; ++i) {
        if (warp >= s_g[i].w0 && warp < s_g[i].w1) my = i;
        if (first_tma < 0 && (s_g[i].op == OP_TMA || s_g[i].op == OP_TRED)) first_tma = i;
    }
    const bool tma_issuer = first_tma >= 0 && lane == 0 && warp == s_g[first_tma].w0;
    const bool waiter = ASYNC && warp == nwarps - 1 && lane == 0;
    cluster_sync();
#pragma unroll 1
    for (uint32_t sample = 0; sample < p.samples; ++sample) {
        // ---- arm (untimed)
        if (tid < 3) { s_c0[tid] = ~0ull; s_c1[tid] = 0; s_n0[tid] = ~0ull; s_n1[tid] = 0; }
        if (tid < 2) s_chk[tid] = 0;
        if (ASYNC && tid < cs) {
            const uint32_t s = tid;
            if (UNI) {
                for (uint32_t q = 0; q < passes; ++q)
                    s_exp[s * passes + q] = incoming_bytes(OP, p.pat, cs, p.k, s, rank, nwarps, nelem, p.perm, p.seed, sample, q);
            } else {
                const uint32_t b = incoming_bytes(OP, p.pat, cs, p.k, s, rank, nwarps, nelem, p.perm, p.seed, sample, 0);
                for (uint32_t q = 0; q < passes; ++q) s_exp[s * passes + q] = b;
            }
        }
        if (UNI)
            for (uint32_t i = tid; i < passes * 6u && i < kMaxCt; i += blockDim.x)
                s_ct[i] = (uint8_t)chunk_target(p.seed, sample, i / 6u, rank, i % 6u, cs);
        __syncthreads();
        if (ASYNC)
            for (uint32_t b = tid; b < nbars; b += blockDim.x)
                if (s_exp[b]) mbar_arrive_expect_tx(bar_u32 + b * 8u, s_exp[b]);
        __syncthreads();
        cluster_sync();
        // ---- timed region
        const TimerPair a = read_timers_pair();
        TimerPair b = a;
        bool part = false;
        uint32_t sy = 0, sx = 0;
        int slot = 0;
        if (my >= 0) {
            const Group11 g = s_g[my];
            slot = (MIXED && g.op != OP_LD) ? 1 : 0;
            if (g.op == OP_LD) {
                if constexpr (OP == OP_LD || MIXED) {
                    ld_stream<ILP, UNI>(g, warp, lane, passes, src_u32, s_ct, zero, sy, sx);
                    b = read_timers_after((uint64_t)(sy ^ sx), p.zero);
                    part = true;
                }
            } else if (g.op == OP_ST) {
                if constexpr (OP == OP_ST || OP == OP_MIX_ST) {
                    st_stream<ILP, UNI, false>(g, warp, lane, passes, dst_u32, bar_u32, rank, sample, s_ct, zero);
                    fence_acq_rel_cluster();
                    b = read_timers_pair();
                    part = true;
                }
            } else if (g.op == OP_STA) {
                if constexpr (OP == OP_STA) {
                    st_stream<ILP, UNI, true>(g, warp, lane, passes, dst_u32, bar_u32, rank, sample, s_ct, zero);
                    b = read_timers_pair();
                    part = true;
                }
            } else if (tma_issuer) {
                if constexpr (OP == OP_TMA || OP == OP_TRED || OP == OP_MIX_TMA) {
                    tma_issue<OP == OP_TRED, UNI>(s_g, ng, passes, p.chunk, src_u32, dst_u32, bar_u32, rank, s_ct);
                    b = read_timers_pair();
                    part = true;
                }
            }
        }
        if (waiter) {
            uint32_t done = 1;
            for (uint32_t bi = 0; bi < nbars; ++bi) {
                if (!s_exp[bi]) continue;
                const uint32_t ph = (s_ph[bi >> 5] >> (bi & 31)) & 1u;
                while (!(done = mbar_try_wait_parity_acq_cluster(bar_u32 + bi * 8u, ph)))
                    if (read_clock64() - a.cycles > kWaitTimeout) { atomicOr(p.err, 1u); break; }
                s_ph[bi >> 5] ^= 1u << (bi & 31);
            }
            const TimerPair t = read_timers_after((uint64_t)done, p.zero);
            s_c0[2] = a.cycles; s_c1[2] = t.cycles; s_n0[2] = a.ns; s_n1[2] = t.ns;
        }
        // ---- records (untimed)
        reduce_slot(part, a, b, &s_c0[slot], &s_c1[slot], &s_n0[slot], &s_n1[slot], sy, &s_chk[slot]);
        if (sx == 0x9e3779b9u && sy == 0x7f4a7c15u) atomicOr(p.err, 2u);  // keeps every loaded word live
        __syncthreads();
        if (tid == 0) {
            const size_t base = ((size_t)sample * gridDim.x + blockIdx.x) * 3;
            for (int sl = 0; sl < 2; ++sl) {
                unsigned long long bytes = 0;
                uint32_t peer = 0xffffffffu, n = 0, op = 0;
                for (int i = 0; i < ng; ++i) {
                    const int gsl = (MIXED && s_g[i].op != OP_LD) ? 1 : 0;
                    if (gsl != sl) continue;
                    bytes += (unsigned long long)(s_g[i].e1 - s_g[i].e0) * 16ull * passes;
                    peer = ++n == 1 && s_g[i].target != kChunkTarget ? s_g[i].target : 0xffffffffu;
                    op = s_g[i].op;
                }
                if (n && s_c1[sl])
                    p.out[base + sl] = Rec11{s_c1[sl] - s_c0[sl], s_n1[sl] - s_n0[sl], s_n0[sl], bytes, peer, 1u, s_chk[sl], op};
            }
            if (ASYNC) {
                unsigned long long bytes = 0;
                uint32_t peer = 0xffffffffu, n = 0;
                for (uint32_t s = 0; s < cs; ++s) {
                    unsigned long long bs = 0;
                    for (uint32_t q = 0; q < passes; ++q) bs += s_exp[s * passes + q];
                    if (bs) { bytes += bs; peer = ++n == 1 ? s : 0xffffffffu; }
                }
                if (bytes)
                    p.out[base + 2] = Rec11{s_c1[2] - s_c0[2], s_n1[2] - s_n0[2], s_n0[2], bytes, peer, 1u, 0u,
                                            (uint32_t)(OP == OP_MIX_TMA ? OP_TMA : OP)};
            }
        }
        __syncthreads();
    }
    cluster_sync();  // no CTA exits while a peer may still access its smem
}

// ---------------------------------------------------------------- K4 victim kernel
struct V11 {
    uint32_t reader_on, reader_threads, victim_warps, passes, spin_cycles, samples;
    uint64_t zero;
    Rec11* out;
    uint32_t* place;
    uint32_t* err;
};

template <int OPR>
__global__ void __launch_bounds__(1024, 1) k_victim(V11 p) {
    extern __shared__ __align__(1024) uint8_t dyn[];
    __shared__ unsigned long long s_c0[2], s_c1[2], s_n0[2], s_n1[2], s_bytes;
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const ClusterRegs c = read_cluster_regs();
    const uint32_t rank = c.ctarank, zero = (uint32_t)p.zero;
    const uint32_t src_u32 = smem_u32(dyn), dst_u32 = src_u32 + kDstOff, flag_u32 = src_u32 + kFlagOff;
    uint4* src = reinterpret_cast<uint4*>(dyn);
    for (uint32_t e = tid; e < kTileElems; e += blockDim.x) src[e] = make_uint4(e, rank + 1u, e ^ 0x5a5a5a5au, rank);
    record_place(p.place);
    __syncthreads();
    cluster_sync();
#pragma unroll 1
    for (uint32_t sample = 0; sample < p.samples; ++sample) {
        if (tid < 2) { s_c0[tid] = ~0ull; s_c1[tid] = 0; s_n0[tid] = ~0ull; s_n1[tid] = 0; }
        if (tid == 0) { s_bytes = 0; if (rank == 1) st_volatile_cluster_u32(mapa_u32(flag_u32, 1), 0); }
        __syncthreads();
        cluster_sync();
        const TimerPair a = read_timers_pair();
        TimerPair b = a;
        bool part = false;
        uint32_t sy = 0, sx = 0;
        if (rank == 0) {
            if (p.reader_on) {
                if (tid < p.reader_threads) {
                    const Group11 g{(uint32_t)OPR, 1u, 0u, kTileElems, 0u, p.reader_threads / 32u};
                    if constexpr (OPR == OP_LD) {
                        ld_stream<4, false>(g, warp, lane, p.passes, src_u32, nullptr, zero, sy, sx);
                        b = read_timers_after((uint64_t)(sy ^ sx), p.zero);
                    } else {
                        st_stream<4, false, false>(g, warp, lane, p.passes, dst_u32, 0u, rank, sample, nullptr, zero);
                        fence_acq_rel_cluster();
                        b = read_timers_pair();
                    }
                    part = true;
                    bar_sync_n(1, p.reader_threads);
                    if (tid == 0) st_volatile_cluster_u32(mapa_u32(flag_u32, 1), 1);
                }
            } else if (tid == 0) {
                while (read_clock64() - a.cycles < p.spin_cycles) { }
                st_volatile_cluster_u32(mapa_u32(flag_u32, 1), 1);
            }
            reduce_slot(part, a, b, &s_c0[0], &s_c1[0], &s_n0[0], &s_n1[0], 0u, nullptr);
        } else if (warp < p.victim_warps) {
            const uint32_t share = kTileElems / p.victim_warps, e0 = warp * share, e1 = e0 + share;
            uint32_t q = 0, stop = 0;
            do {
                // share = 6144 / W elements is a multiple of 64 for W ≤ 32: two unguarded 512-B warp loads per step
#pragma unroll 1
                for (uint32_t e = e0 + lane; e < e1; e += 64u) {
                    const uint4 x0 = ld_shared_v4_u32(src_u32 + e * 16u + (q & zero));
                    const uint4 x1 = ld_shared_v4_u32(src_u32 + (e + 32u) * 16u + (q & zero));
                    sy += x0.y + x1.y;
                    sx ^= x0.x ^ x0.z ^ x0.w ^ x1.x ^ x1.z ^ x1.w;
                }
                ++q;
                uint32_t f = 0;
                if (lane == 0) asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(f) : "r"(flag_u32) : "memory");
                stop = __shfl_sync(0xffffffffu, f, 0);
            } while (!stop && q < (1u << 24));
            b = read_timers_after((uint64_t)(sy ^ sx), p.zero);
            part = true;
            if (lane == 0) atomicAdd(&s_bytes, (unsigned long long)q * share * 16ull);
            reduce_slot(part, a, b, &s_c0[1], &s_c1[1], &s_n0[1], &s_n1[1], 0u, nullptr);
        }
        if (sx == 0x9e3779b9u && sy == 0x7f4a7c15u) atomicOr(p.err, 2u);
        __syncthreads();
        if (tid == 0) {
            const size_t base = ((size_t)sample * gridDim.x + blockIdx.x) * 3;
            const int sl = rank == 0 ? 0 : 1;
            const unsigned long long bytes = rank == 0 ? (unsigned long long)kTile * p.passes : s_bytes;
            if (s_c1[sl] && (rank == 1 || p.reader_on))
                p.out[base + sl] = Rec11{s_c1[sl] - s_c0[sl], s_n1[sl] - s_n0[sl], s_n0[sl], bytes, rank ^ 1u, 1u, 0u, (uint32_t)OPR};
        }
        __syncthreads();
    }
    cluster_sync();
}

// ---------------------------------------------------------------- P1-style kernel
constexpr uint32_t kP1Tile = 16384, kP1Blocks = 50160;
template <int OP>
__global__ void __launch_bounds__(1024, 2) k_p1(uint32_t passes, uint64_t zero, uint32_t* hist, uint32_t* err) {
    extern __shared__ __align__(128) uint8_t t1[];
    __shared__ __align__(8) uint64_t bar;
    const uint32_t tid = threadIdx.x;
    const ClusterRegs c = read_cluster_regs();
    const uint32_t rank = c.ctarank, target = (rank + 1u) % c.nctarank, z = (uint32_t)zero;
    reinterpret_cast<uint4*>(t1)[tid] = make_uint4(tid, rank + 1u, tid ^ 0x5a5a5a5au, rank);
    const uint32_t tile = smem_u32(t1), bar_u32 = smem_u32(&bar);
    if (OP == OP_TMA && tid == 0) {
        mbar_init(bar_u32, 1);
        fence_mbarrier_init();
        mbar_arrive_expect_tx(bar_u32, passes * kP1Tile);
    }
    fence_proxy_async_shared_cta();
    __syncthreads();
    cluster_sync();
    uint32_t sx = 0;
    const uint32_t peer = mapa_u32(tile, target);
    if constexpr (OP == OP_LD) {
#pragma unroll 4
        for (uint32_t q = 0; q < passes; ++q) {
            const uint4 x = ld_dsmem_v4_u32(peer + tid * 16u + (q & z));
            sx ^= x.x ^ x.y ^ x.z ^ x.w;
        }
    } else if constexpr (OP == OP_ST) {
#pragma unroll 4
        for (uint32_t q = 0; q < passes; ++q) st_dsmem_v4_u32(peer + tid * 16u + (q & z), make_uint4(tid, q, rank, 0));
    } else {
        if (tid == 0) {
            for (uint32_t q = 0; q < passes; ++q) bulk_s2s_dsmem(peer, tile, kP1Tile, mapa_u32(bar_u32, target));
            const unsigned long long t0 = read_clock64();
            while (!mbar_try_wait_parity_acq_cluster(bar_u32, 0))
                if (read_clock64() - t0 > kWaitTimeout) { atomicOr(err, 1u); break; }
        }
    }
    __syncthreads();
    cluster_sync();
    if (tid == 0) atomicAdd(&hist[read_smid()], 1u);
    if (sx == 0x9e3779b9u) atomicOr(err, 2u);
}

// ---------------------------------------------------------------- host
using KFn = void (*)(P11);
// Instantiated variants (the sweeps use no others): ld / st at ILP 1, 2, 4, 8 (uniform_random at ILP 4 only), every other
// operation at ILP 4 (TMA ignores the ILP), mixed kernels without uniform_random.
template <int OP>
static KFn pick_ilp(int ilp, bool uni) {
    if (uni) {
        if (ilp != 4) hb::die("uniform_random is instantiated at ILP 4 only");
        return k_stream<OP, 4, true>;
    }
    switch (ilp) {
    case 1: return k_stream<OP, 1, false>;
    case 2: return k_stream<OP, 2, false>;
    case 4: return k_stream<OP, 4, false>;
    case 8: return k_stream<OP, 8, false>;
    }
    hb::die("unsupported ILP " + std::to_string(ilp));
}
static KFn pick(int op, int ilp, bool uni) {
    if (op != OP_LD && op != OP_ST && ilp != 4 && !(op == OP_TMA || op == OP_TRED))
        hb::die(std::string(kOpName[op]) + " is instantiated at ILP 4 only");
    switch (op) {
    case OP_LD: return pick_ilp<OP_LD>(ilp, uni);
    case OP_ST: return pick_ilp<OP_ST>(ilp, uni);
    case OP_STA: return uni ? k_stream<OP_STA, 4, true> : k_stream<OP_STA, 4, false>;
    case OP_TMA: return uni ? k_stream<OP_TMA, 4, true> : k_stream<OP_TMA, 4, false>;
    case OP_TRED: return uni ? k_stream<OP_TRED, 4, true> : k_stream<OP_TRED, 4, false>;
    case OP_MIX_ST: if (uni) break; return k_stream<OP_MIX_ST, 4, false>;
    case OP_MIX_TMA: if (uni) break; return k_stream<OP_MIX_TMA, 4, false>;
    }
    hb::die("no kernel instantiated for this operation");
}

enum Mode11 { M_SINGLE = 0, M_FULL = 1 };
static const char* kModeName[] = {"single", "full"};

struct Cfg {
    std::string step;
    int op = OP_LD, pat = PT_RING, k = 1, cs = 2;
    cudaClusterSchedulingPolicy pol = cudaClusterSchedulingPolicyDefault;
    int threads = 1024, ilp = 4;
    uint32_t chunk = 16384, payload = kTile;
    int passes = 1, mode = M_SINGLE, samples = 0, launches = kLaunches;
};
static bool is_tma(int op) { return op == OP_TMA || op == OP_TRED; }

static Perm16 derangement(std::mt19937& rng, int cs) {
    Perm16 p{};
    std::vector<int> v(cs);
    for (;;) {
        for (int i = 0; i < cs; ++i) v[i] = i;
        std::shuffle(v.begin(), v.end(), rng);
        bool ok = true;
        for (int i = 0; i < cs; ++i) ok &= v[i] != i;
        if (ok) break;
    }
    for (int i = 0; i < cs; ++i) p.p[i] = (uint8_t)v[i];
    return p;
}

// Host value of the ld check word of CTA `rank`, slot 0, sample `sample`.
static uint32_t expected_chk(const Cfg& c, const Perm16& perm, uint32_t seed, int rank, int sample) {
    Group11 g[kMaxGroups];
    const int n = plan_groups(c.op, c.pat, c.cs, c.k, rank, c.threads / 32, c.payload / 16, perm, g);
    uint32_t s = 0;
    for (int i = 0; i < n; ++i) {
        if (g[i].op != OP_LD) continue;
        for (int q = 0; q < c.passes; ++q) {
            if (g[i].target == kChunkTarget) {
                for (uint32_t e = g[i].e0; e < g[i].e1; ++e) s += chunk_target(seed, sample, q, rank, e / kChunkElems, c.cs) + 1u;
            } else {
                s += (g[i].e1 - g[i].e0) * (g[i].target + 1u);
            }
        }
    }
    return s;
}

struct Ctx {
    hb::Run& run;
    Discards11 disc;
    std::vector<int> gpc;
    long long li = 0;
    bool quick = false;
    std::mt19937 rng{20261026u};
    hb::CsvWriter samples, launches;
    DevBuf<Rec11> out{(size_t)kSamplesSingle * 132 * 3 * 2};
    DevBuf<uint32_t> place{4 * 132 * 2}, err{1};
    cudaEvent_t e0{}, e1{};
    explicit Ctx(hb::Run& r, bool q)
        : run(r), disc(r), gpc(load_gpc_map(r.dev().sm_count)), quick(q),
          samples(hb::Run::results("11__dsmem_bw__samples.csv"),
                  hb::with_mandatory({"kernel", "pattern", "pattern_k", "cs", "policy", "threads", "ilp", "chunk_bytes", "payload_bytes", "passes",
                                      "cluster_id", "rank", "smid", "peer_smid", "gpc", "side", "stream", "sample", "cycles", "ns", "t0_gt_ns",
                                      "bytes_moved", "launch_mode", "step"})),
          launches(hb::Run::results("11__dsmem_bw__launches.csv"),
                   hb::with_mandatory({"kernel", "pattern", "pattern_k", "cs", "policy", "threads", "ilp", "chunk_bytes", "payload_bytes",
                                       "passes", "launch_mode", "step", "nclusters", "max_active_clusters", "active_ctas", "total_bytes",
                                       "samples", "event_ns", "makespan_ns"})) {
        CUDA_CHECK(cudaEventCreate(&e0));
        CUDA_CHECK(cudaEventCreate(&e1));
    }
};

template <class F>
static std::pair<hb::ClockSample, Placement> recorded11(Ctx& x, const std::string& label, F&& launch) {
    for (int t = 0; t < kMaxTries; ++t) {
        Placement p = launch();
        const hb::ClockSample s = x.run.clock_note(label, "launch", x.li);
        if (!p.ok()) { x.disc.add(label, "placement", x.li, s); continue; }
        if (s.bad()) { x.disc.add(label, "clock", x.li, s); continue; }
        return {s, std::move(p)};
    }
    hb::die(label + ": no clean launch after " + std::to_string(kMaxTries) + " tries");
}

static std::string label_of(const Cfg& c) {
    char b[200];
    std::snprintf(b, sizeof b, "%s_%s_%s%d_cs%d_%s_t%d_i%d_c%u_p%u_x%d_%s", c.step.c_str(), kOpName[c.op], kPatName[c.pat], c.k, c.cs,
                  policy_name(c.pol), c.threads, c.ilp, c.chunk, c.payload, c.passes, kModeName[c.mode]);
    return b;
}

static void check_cfg(const Cfg& c) {
    const int nw = c.threads / 32;
    int maxg = 1;
    if (c.pat == PT_BCAST_OUT) maxg = c.k;
    if (c.pat == PT_A2A || c.pat == PT_A2A_FULL) maxg = c.cs - 1;
    if (nw < maxg || (c.pat >= PT_MIX_SAME && nw < 2)) hb::die(label_of(c) + ": fewer warps than groups");
    if (op_async(c.op) && c.cs * c.passes > (int)kMaxBars) hb::die(label_of(c) + ": too many barriers");
    if (c.pat == PT_UNIFORM && c.passes * 6 > (int)kMaxCt) hb::die(label_of(c) + ": uniform_random table too small");
    if (c.payload % 16 || c.payload > kTile || c.payload == 0) hb::die(label_of(c) + ": bad payload");
    if ((c.pat == PT_BCAST_IN || c.pat == PT_BCAST_OUT) && (c.k < 1 || c.k >= c.cs)) hb::die(label_of(c) + ": bad k");
    if (c.pat == PT_PACKED && (c.cs % (2 * c.k))) hb::die(label_of(c) + ": bad stride");
}

// Run one configuration: warm-ups, recorded launches, CSV rows. Returns the median B/cycle of the bandwidth side.
static double run_point(Ctx& x, Cfg c) {
    check_cfg(c);
    if (!c.samples) c.samples = c.mode == M_SINGLE ? kSamplesSingle : kSamplesFull;
    if (x.quick) c.launches = std::min(c.launches, kQuickLaunches);
    const KFn k = pick(c.op, c.ilp, c.pat == PT_UNIFORM);
    hb::ClusterLaunch L;
    L.block = dim3(c.threads);
    L.cluster = dim3(c.cs, 1, 1);
    L.smem = kSmem;
    L.explicit_cluster = true;
    L.policy = c.pol;
    L.grid = dim3(c.cs);
    const int maxact = L.max_active_clusters(k);
    const int ncl = c.mode == M_SINGLE ? 1 : maxact;
    L.grid = dim3(ncl * c.cs);
    const int nb = ncl * c.cs;
    const size_t nrec = (size_t)c.samples * nb * 3;
    if (nrec > x.out.n) hb::die("record buffer too small");
    hb::ClusterLaunch::Prepared prep;
    L.prepare(k, prep);
    P11 p{};
    p.pat = c.pat; p.k = c.k; p.payload = c.payload; p.chunk = c.chunk; p.samples = c.samples; p.passes = c.passes;
    p.zero = 0; p.out = x.out.d; p.place = x.place.d; p.err = x.err.d;
    float ms = 0;
    auto launch = [&] {
        p.seed = x.rng();
        p.perm = derangement(x.rng, c.cs);
        CUDA_CHECK(cudaMemset(x.out.d, 0, nrec * sizeof(Rec11)));
        CUDA_CHECK(cudaMemset(x.place.d, 0, 4 * (size_t)nb * sizeof(uint32_t)));
        CUDA_CHECK(cudaEventRecord(x.e0));
        CUDA_CHECK(cudaLaunchKernelEx(&prep.cfg, k, p));
        CUDA_CHECK(cudaEventRecord(x.e1));
        CUDA_CHECK(cudaEventSynchronize(x.e1));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventElapsedTime(&ms, x.e0, x.e1));
        const uint32_t e = x.err.get()[0];
        if (e & 1u) hb::die(label_of(c) + ": mbarrier wait timed out (expected bytes never arrived)");
        return Placement(x.place.get(4 * (size_t)nb), c.cs, ncl);
    };
    const std::string label = label_of(c);
    for (int w = 0; w < kWarmups; ++w) launch();
    std::vector<double> bpc;
    char b[512];
    for (int l = 0; l < c.launches; ++l) {
        auto r = recorded11(x, label, launch);
        const Placement& pl = r.second;
        const std::vector<Rec11> o = x.out.get(nrec);
        const std::string pre = x.run.prefix(x.li, r.first);
        unsigned long long gmin = ~0ull, gmax = 0, total = 0;
        for (const Rec11& q : o)
            if (q.valid) { gmin = std::min(gmin, q.t0_gt); gmax = std::max(gmax, q.t0_gt + q.ns); }
        std::vector<char> active(nb, 0);
        for (int s = 0; s < c.samples; ++s)
            for (int blk = 0; blk < nb; ++blk)
                for (int sl = 0; sl < 3; ++sl) {
                    const Rec11& q = o[((size_t)s * nb + blk) * 3 + sl];
                    if (!q.valid) continue;
                    const int cl = blk / c.cs, rk = blk % c.cs;
                    if (sl == 0 && (c.op == OP_LD || c.op >= OP_MIX_ST)) {
                        const uint32_t want = expected_chk(c, p.perm, p.seed, rk, s);
                        if (q.chk != want)
                            hb::die(label + ": ld check of cluster " + std::to_string(cl) + " rank " + std::to_string(rk) + " is " +
                                    std::to_string(q.chk) + ", expected " + std::to_string(want));
                    }
                    if (sl < 2) { total += q.bytes; active[blk] = 1; }
                    const uint32_t sm = pl.at(cl, rk);
                    const int peer = q.peer == 0xffffffffu ? -1 : (int)pl.at(cl, (int)q.peer);
                    std::snprintf(b, sizeof b, ",%s,%s,%d,%d,%s,%d,%d,%u,%u,%d,%d,%d,%u,%d,%d,%s,%s,%d,%llu,%llu,%lld,%llu,%s,%s", kOpName[c.op],
                                  kPatName[c.pat], c.k, c.cs, policy_name(c.pol), c.threads, is_tma(c.op) ? 0 : c.ilp, is_tma(c.op) || c.op == OP_MIX_TMA ? c.chunk : 0u,
                                  c.payload, c.passes, cl, rk, sm, peer, x.gpc[sm], sl == 2 ? "target" : "issuer", kOpName[q.stream], s, q.cycles, q.ns,
                                  (long long)(q.t0_gt - gmin), q.bytes, kModeName[c.mode], c.step.c_str());
                    x.samples.add(pre + b);
                    const bool bw_side = op_async(c.op) || c.op == OP_MIX_TMA ? sl == 2 : sl == 0;
                    if (bw_side) bpc.push_back((double)q.bytes / q.cycles);
                }
        int nact = 0;
        for (char a : active) nact += a;
        std::snprintf(b, sizeof b, ",%s,%s,%d,%d,%s,%d,%d,%u,%u,%d,%s,%s,%d,%d,%d,%llu,%d,%.0f,%llu", kOpName[c.op], kPatName[c.pat], c.k, c.cs,
                      policy_name(c.pol), c.threads, is_tma(c.op) ? 0 : c.ilp, is_tma(c.op) || c.op == OP_MIX_TMA ? c.chunk : 0u, c.payload, c.passes,
                      kModeName[c.mode], c.step.c_str(), ncl, maxact, nact, total, c.samples, ms * 1e6, gmax - gmin);
        x.launches.add(pre + b);
        x.samples.flush();
        x.launches.flush();
        ++x.li;
    }
    x.samples.sync();
    x.launches.sync();
    std::sort(bpc.begin(), bpc.end());
    const double med = bpc.empty() ? NAN : bpc[bpc.size() / 2];
    std::printf("%-58s %3d cl  %7.2f B/cycle/CTA (median)\n", label.c_str(), ncl, med);
    std::fflush(stdout);
    return med;
}

static std::vector<uint32_t> payloads(bool quick) {
    if (quick) return {16384u, kTile};
    return {16384u, 32768u, 49152u, 65536u, 81920u, kTile};
}
static const int kCS[] = {2, 4, 8, 16};

static void step_A(Ctx& x) {
    for (int mode : {M_SINGLE, M_FULL})
        for (int op : {OP_LD, OP_ST, OP_STA, OP_TMA, OP_TRED})
            for (int cs : kCS) {
                if (x.quick && cs != 2 && cs != 16) continue;
                for (uint32_t pl : payloads(x.quick)) {
                    Cfg c; c.step = "A"; c.op = op; c.cs = cs; c.payload = pl; c.mode = mode;
                    run_point(x, c);
                }
            }
}

static void step_S(Ctx& x) {
    const std::vector<int> passes = x.quick ? std::vector<int>{4, 16} : std::vector<int>{4, 8, 16, 32};
    for (int op : {OP_LD, OP_ST, OP_STA, OP_TMA, OP_TRED})
        for (int cs : kCS)
            for (int q : passes) {
                Cfg c; c.step = "S"; c.op = op; c.cs = cs; c.passes = q; c.mode = M_FULL; c.samples = 1;
                run_point(x, c);
            }
}

static void step_B(Ctx& x) {
    for (int op : {OP_LD, OP_ST})
        for (int pat : {PT_PAIR_UNI, PT_RING})
            for (int cs : kCS) {
                if (x.quick && cs != 8) continue;
                for (int t : {128, 256, 512, 1024})
                    for (int ilp : {1, 2, 4, 8}) {
                        if (x.quick && ilp != 1 && ilp != 8) continue;
                        for (uint32_t pl : payloads(x.quick)) {
                            Cfg c; c.step = "B"; c.op = op; c.pat = pat; c.cs = cs; c.threads = t; c.ilp = ilp; c.payload = pl;
                            run_point(x, c);
                        }
                    }
            }
    for (int pat : {PT_PAIR_UNI, PT_RING})
        for (int cs : kCS) {
            if (x.quick && cs != 8) continue;
            for (int t : {128, 256, 512, 1024})
                for (uint32_t pl : payloads(x.quick)) {
                    Cfg c; c.step = "B"; c.op = OP_STA; c.pat = pat; c.cs = cs; c.threads = t; c.payload = pl;
                    run_point(x, c);
                }
        }
}

// (pattern, k) list of step C for one operation and cluster size
static std::vector<std::pair<int, int>> patterns_C(int op, int cs) {
    std::vector<std::pair<int, int>> v = {{PT_PAIR_UNI, 1}, {PT_PAIR_BI, 1}, {PT_RING, 1}};
    if (cs > 2) v.push_back({PT_RING, cs / 2});
    v.push_back({PT_PACKED, 1});
    if (cs > 2) v.push_back({PT_PACKED, cs / 2});
    v.push_back({PT_BCAST_IN, cs - 1});
    if (op != OP_LD) v.push_back({PT_BCAST_OUT, cs - 1});
    if (cs > 2) v.push_back({PT_A2A, 1});
    if (op == OP_LD && cs > 2) v.push_back({PT_A2A_FULL, 1});
    if (cs > 2) v.push_back({PT_RANDPERM, 1});
    if (cs > 2) v.push_back({PT_UNIFORM, 1});
    return v;
}

static void step_C(Ctx& x) {
    for (int mode : {M_SINGLE, M_FULL})
        for (int cs : kCS) {
            if (mode == M_FULL && cs != 8) continue;
            if (x.quick && cs != 8) continue;
            for (int op : {OP_LD, OP_ST, OP_TMA})
                for (auto [pat, k] : patterns_C(op, cs))
                    for (uint32_t pl : payloads(x.quick)) {
                        Cfg c; c.step = "C"; c.op = op; c.pat = pat; c.k = k; c.cs = cs; c.payload = pl; c.mode = mode;
                        run_point(x, c);
                    }
            for (int op : {OP_MIX_ST, OP_MIX_TMA})
                for (int pat : {PT_MIX_SAME, PT_MIX_OPP})
                    for (uint32_t pl : payloads(x.quick)) {
                        Cfg c; c.step = "C"; c.op = op; c.pat = pat; c.cs = cs; c.payload = pl; c.mode = mode;
                        run_point(x, c);
                    }
        }
}

static void step_D(Ctx& x) {
    for (int n = 1; n <= 15; ++n) {
        if (x.quick && n != 1 && n != 15) continue;
        for (uint32_t pl : payloads(x.quick)) {
            Cfg c; c.step = "D"; c.op = OP_LD; c.pat = PT_BCAST_IN; c.k = n; c.cs = 16; c.payload = pl;
            run_point(x, c);
            for (int op : {OP_ST, OP_TMA}) {
                Cfg d = c; d.op = op; d.pat = PT_BCAST_OUT;
                run_point(x, d);
            }
        }
    }
}

static void step_E(Ctx& x) {
    for (int op : {OP_TMA, OP_TRED})
        for (int pat : {PT_PAIR_UNI, PT_RING})
            for (uint32_t ch : {128u, 256u, 512u, 1024u, 2048u, 4096u, 8192u, 16384u}) {
                if (x.quick && ch != 128u && ch != 16384u) continue;
                for (uint32_t pl : payloads(x.quick)) {
                    Cfg c; c.step = "E"; c.op = op; c.pat = pat; c.cs = 8; c.chunk = ch; c.payload = pl;
                    run_point(x, c);
                }
            }
}

static void step_F(Ctx& x) {
    for (cudaClusterSchedulingPolicy pol : {cudaClusterSchedulingPolicySpread, cudaClusterSchedulingPolicyLoadBalancing})
        for (int mode : {M_SINGLE, M_FULL})
            for (int op : {OP_LD, OP_ST, OP_TMA})
                for (int cs : kCS) {
                    if (x.quick && cs != 8) continue;
                    for (uint32_t pl : payloads(x.quick)) {
                        Cfg c; c.step = "F"; c.op = op; c.cs = cs; c.payload = pl; c.mode = mode; c.pol = pol;
                        run_point(x, c);
                    }
                }
}

// ---------------------------------------------------------------- P1 style
static void step_P1(Ctx& x) {
    hb::CsvWriter blocks(hb::Run::results("11__dsmem_bw__p1_blocks.csv"), hb::with_mandatory({"kernel", "cs", "passes", "smid", "gpc", "blocks"}));
    DevBuf<uint32_t> hist(132);
    for (int op : {OP_LD, OP_ST, OP_TMA})
        for (int cs : kCS) {
            if (x.quick && cs != 2) continue;
            for (uint32_t passes : {1u, 4u, 16u}) {
                void (*k)(uint32_t, uint64_t, uint32_t*, uint32_t*) = op == OP_LD ? k_p1<OP_LD> : op == OP_ST ? k_p1<OP_ST> : k_p1<OP_TMA>;
                hb::ClusterLaunch L;
                L.block = dim3(1024);
                L.cluster = dim3(cs, 1, 1);
                L.smem = kP1Tile;
                L.explicit_cluster = true;
                L.grid = dim3(cs);
                const int maxact = L.max_active_clusters(k);
                L.grid = dim3(kP1Blocks);
                hb::ClusterLaunch::Prepared prep;
                L.prepare(k, prep);
                float ms = 0;
                auto launch = [&] {
                    hist.zero();
                    CUDA_CHECK(cudaEventRecord(x.e0));
                    CUDA_CHECK(cudaLaunchKernelEx(&prep.cfg, k, passes, (uint64_t)0, hist.d, x.err.d));
                    CUDA_CHECK(cudaEventRecord(x.e1));
                    CUDA_CHECK(cudaEventSynchronize(x.e1));
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaEventElapsedTime(&ms, x.e0, x.e1));
                    if (x.err.get()[0] & 1u) hb::die("p1: mbarrier wait timed out");
                };
                const std::string label = std::string("P1_") + kOpName[op] + "_cs" + std::to_string(cs) + "_x" + std::to_string(passes);
                for (int w = 0; w < kWarmups; ++w) launch();
                std::vector<double> tb;
                for (int l = 0; l < (x.quick ? kQuickLaunches : kLaunches); ++l) {
                    hb::ClockSample s;
                    for (int t = 0;; ++t) {
                        if (t >= kMaxTries) hb::die(label + ": no clean launch");
                        launch();
                        s = x.run.clock_note(label, "launch", x.li);
                        if (!s.bad()) break;
                        x.disc.add(label, "clock", x.li, s);
                    }
                    const std::vector<uint32_t> h = hist.get();
                    const std::string pre = x.run.prefix(x.li, s);
                    char b[320];
                    unsigned long long nblk = 0;
                    int nsm = 0;
                    for (int sm = 0; sm < 132; ++sm) {
                        nblk += h[sm];
                        nsm += h[sm] > 0;
                        std::snprintf(b, sizeof b, ",%s,%d,%u,%d,%d,%u", kOpName[op], cs, passes, sm, x.gpc[sm], h[sm]);
                        blocks.add(pre + b);
                    }
                    if (nblk != kP1Blocks) hb::die(label + ": block histogram sums to " + std::to_string(nblk));
                    const unsigned long long bytes = (unsigned long long)kP1Blocks * passes * kP1Tile;
                    std::snprintf(b, sizeof b, ",%s,ring,1,%d,Default,1024,%d,%u,%u,%u,p1_style,P1,%u,%d,%d,%llu,1,%.0f,0", kOpName[op], cs,
                                  op == OP_TMA ? 0 : 1, op == OP_TMA ? kP1Tile : 0u, kP1Tile, passes, kP1Blocks / cs, maxact, nsm, bytes, ms * 1e6);
                    x.launches.add(pre + b);
                    blocks.flush();
                    x.launches.flush();
                    tb.push_back(bytes / (ms * 1e-3) / 1e12);
                    ++x.li;
                }
                blocks.sync();
                x.launches.sync();
                std::sort(tb.begin(), tb.end());
                std::printf("%-30s %5.3f TB/s (event, median)\n", label.c_str(), tb[tb.size() / 2]);
                std::fflush(stdout);
            }
        }
}

// ---------------------------------------------------------------- K4
static void step_K4(Ctx& x) {
    hb::CsvWriter csv(hb::Run::results("11__dsmem_bw__victim.csv"),
                      hb::with_mandatory({"kernel", "reader_threads", "victim_warps", "reader_on", "launch_mode", "cluster_id", "rank", "smid",
                                          "peer_smid", "gpc", "side", "sample", "cycles", "ns", "bytes_moved", "bpc"}));
    struct VC { int op, rt, w, on; };
    std::vector<VC> cfgs;
    for (int op : {OP_LD, OP_ST})
        for (int rt : {256, 1024})
            for (int w : {0, 4, 8, 16, 32}) cfgs.push_back({op, rt, w, 1});
    for (int w : {4, 8, 16, 32}) cfgs.push_back({OP_LD, 0, w, 0});
    for (int mode : {M_SINGLE, M_FULL})
        for (const VC& v : cfgs) {
            if (x.quick && v.w != 0 && v.w != 32) continue;
            void (*k)(V11) = v.op == OP_LD ? k_victim<OP_LD> : k_victim<OP_ST>;
            hb::ClusterLaunch L;
            L.block = dim3(1024);
            L.cluster = dim3(2, 1, 1);
            L.smem = kSmem;
            L.explicit_cluster = true;
            L.grid = dim3(2);
            const int ncl = mode == M_SINGLE ? 1 : L.max_active_clusters(k), nb = 2 * ncl;
            L.grid = dim3(nb);
            hb::ClusterLaunch::Prepared prep;
            L.prepare(k, prep);
            V11 p{(uint32_t)v.on, (uint32_t)v.rt, (uint32_t)v.w, 32u, 160000u, (uint32_t)kSamplesSingle, 0, x.out.d, x.place.d, x.err.d};
            const size_t nrec = (size_t)kSamplesSingle * nb * 3;
            auto launch = [&] {
                CUDA_CHECK(cudaMemset(x.out.d, 0, nrec * sizeof(Rec11)));
                CUDA_CHECK(cudaMemset(x.place.d, 0, 4 * (size_t)nb * sizeof(uint32_t)));
                CUDA_CHECK(cudaLaunchKernelEx(&prep.cfg, k, p));
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaGetLastError());
                return Placement(x.place.get(4 * (size_t)nb), 2, ncl);
            };
            char lb[96];
            std::snprintf(lb, sizeof lb, "K4_%s_r%d_w%d_on%d_%s", kOpName[v.op], v.rt, v.w, v.on, kModeName[mode]);
            for (int w = 0; w < kWarmups; ++w) launch();
            std::vector<double> rd, vi;
            for (int l = 0; l < (x.quick ? kQuickLaunches : kLaunches); ++l) {
                auto r = recorded11(x, lb, launch);
                const std::vector<Rec11> o = x.out.get(nrec);
                const std::string pre = x.run.prefix(x.li, r.first);
                char b[320];
                for (int s = 0; s < kSamplesSingle; ++s)
                    for (int blk = 0; blk < nb; ++blk)
                        for (int sl = 0; sl < 2; ++sl) {
                            const Rec11& q = o[((size_t)s * nb + blk) * 3 + sl];
                            if (!q.valid) continue;
                            const int cl = blk / 2, rk = blk % 2;
                            const uint32_t sm = r.second.at(cl, rk);
                            const double bpc = (double)q.bytes / q.cycles;
                            (sl == 0 ? rd : vi).push_back(bpc);
                            std::snprintf(b, sizeof b, ",%s,%d,%d,%d,%s,%d,%d,%u,%u,%d,%s,%d,%llu,%llu,%llu,%.4f", v.on ? kOpName[v.op] : "none", v.rt,
                                          v.w, v.on, kModeName[mode], cl, rk, sm, r.second.at(cl, rk ^ 1), x.gpc[sm], sl == 0 ? "reader" : "victim", s,
                                          q.cycles, q.ns, q.bytes, bpc);
                            csv.add(pre + b);
                        }
                csv.flush();
                ++x.li;
            }
            csv.sync();
            auto med = [](std::vector<double>& v) -> double { if (v.empty()) return NAN; std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
            std::printf("%-34s reader %7.2f  victim %7.2f B/cycle (median)\n", lb, med(rd), med(vi));
            std::fflush(stdout);
        }
}

// ---------------------------------------------------------------- ncu
static void step_ncu(Ctx& x) {
    hb::CsvWriter man("bin/ncu_manifest.csv", hb::with_mandatory({"kernel", "pattern", "pattern_k", "cs", "launch_mode", "nclusters", "threads",
                                                                   "ilp", "chunk_bytes", "payload_bytes", "issuer_bytes"}));
    std::vector<Cfg> cfgs;
    for (int op : {OP_LD, OP_ST, OP_STA, OP_TMA, OP_TRED})
        for (auto [pat, k] : patterns_C(op == OP_STA || op == OP_TRED ? OP_ST : op, 8)) {
            Cfg c; c.op = op; c.pat = pat; c.k = k; c.cs = 8;
            cfgs.push_back(c);
        }
    for (int op : {OP_MIX_ST, OP_MIX_TMA})
        for (int pat : {PT_MIX_SAME, PT_MIX_OPP}) { Cfg c; c.op = op; c.pat = pat; c.cs = 8; cfgs.push_back(c); }
    for (int op : {OP_LD, OP_ST, OP_STA, OP_TMA, OP_TRED}) { Cfg c; c.op = op; c.cs = 8; c.mode = M_FULL; cfgs.push_back(c); }
    for (Cfg& c : cfgs) {
        c.step = "ncu";
        c.samples = 1;
        check_cfg(c);
        const KFn k = pick(c.op, c.ilp, c.pat == PT_UNIFORM);
        hb::ClusterLaunch L;
        L.block = dim3(c.threads);
        L.cluster = dim3(c.cs, 1, 1);
        L.smem = kSmem;
        L.explicit_cluster = true;
        L.grid = dim3(c.cs);
        const int ncl = c.mode == M_SINGLE ? 1 : L.max_active_clusters(k);
        L.grid = dim3(ncl * c.cs);
        P11 p{};
        p.pat = c.pat; p.k = c.k; p.payload = c.payload; p.chunk = c.chunk; p.samples = 1; p.passes = 1;
        p.seed = x.rng(); p.perm = derangement(x.rng, c.cs); p.out = x.out.d; p.place = x.place.d; p.err = x.err.d;
        CUDA_CHECK(cudaMemset(x.out.d, 0, (size_t)ncl * c.cs * 3 * sizeof(Rec11)));
        CUDA_CHECK(L.launch(k, p));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        const std::vector<Rec11> o = x.out.get((size_t)ncl * c.cs * 3);
        unsigned long long bytes = 0;
        for (size_t i = 0; i < o.size(); ++i) if (o[i].valid && i % 3 < 2) bytes += o[i].bytes;
        const hb::ClockSample s = x.run.clock_note(label_of(c), "launch", x.li);
        char b[256];
        std::snprintf(b, sizeof b, ",%s,%s,%d,%d,%s,%d,%d,%d,%u,%u,%llu", kOpName[c.op], kPatName[c.pat], c.k, c.cs, kModeName[c.mode], ncl, c.threads,
                      is_tma(c.op) ? 0 : c.ilp, c.chunk, c.payload, bytes);
        man.add(x.run.prefix(x.li++, s) + b);
        man.flush();
    }
    man.sync();
}

// ---------------------------------------------------------------- plan (review of the pattern decomposition)
static void step_plan() {
    Perm16 perm{};
    std::mt19937 rng(1);
    for (int cs : {2, 8, 16}) {
        perm = derangement(rng, cs);
        for (int op : {OP_LD, OP_ST, OP_TMA, OP_MIX_ST}) {
            std::vector<std::pair<int, int>> pats = op == OP_MIX_ST ? std::vector<std::pair<int, int>>{{PT_MIX_SAME, 1}, {PT_MIX_OPP, 1}} : patterns_C(op, cs);
            for (auto [pat, k] : pats) {
                std::printf("cs %d %s %s k %d:\n", cs, kOpName[op], kPatName[pat], k);
                for (int r = 0; r < cs; ++r) {
                    Group11 g[kMaxGroups];
                    const int n = plan_groups(op, pat, cs, k, r, 32, kTileElems, perm, g);
                    std::printf("  r%-2d:", r);
                    for (int i = 0; i < n; ++i) std::printf(" [%s->%d e%u-%u w%u-%u]", kOpName[g[i].op], g[i].target == kChunkTarget ? -1 : (int)g[i].target,
                                                            g[i].e0, g[i].e1, g[i].w0, g[i].w1);
                    if (op_async(op)) {
                        std::printf("  in:");
                        for (int s = 0; s < cs; ++s) {
                            const uint32_t bi = incoming_bytes(op, pat, cs, k, s, r, 32, kTileElems, perm, 7, 0, 0);
                            if (bi) std::printf(" %d:%u", s, bi);
                        }
                    }
                    std::printf("\n");
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "A";
    if (mode == "plan") { step_plan(); return 0; }
    const bool quick = hb::has_flag(argc, argv, "--quick");
    hb::Run run(EXP, argc, argv);
    if (mode == "clockcheck") {
        run.lock_clocks_note(hb::arg_str(argc, argv, "--label", "check"), "check");
    } else {
        Ctx x(run, quick);
        run.lock_clocks_note(mode, "before");
        if (mode == "A") step_A(x);
        else if (mode == "S") step_S(x);
        else if (mode == "B") step_B(x);
        else if (mode == "C") step_C(x);
        else if (mode == "D") step_D(x);
        else if (mode == "E") step_E(x);
        else if (mode == "F") step_F(x);
        else if (mode == "P1") step_P1(x);
        else if (mode == "K4") step_K4(x);
        else if (mode == "ncu") step_ncu(x);
        else hb::die("unknown mode " + mode);
        run.lock_clocks_note(mode, "after");
        std::printf("%s: %d launches discarded\n", mode.c_str(), x.disc.n);
    }
    run.complete();
    return 0;
}
