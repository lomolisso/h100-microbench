/*
 * dsmem_async.cu — experiment 12, K4–K7: DSMEM fire-and-forget throughput (red.shared::cluster, atom.shared::cluster,
 * red.async, st.async), contention from N sources into one CTA, store-visibility latency per signalling primitive for
 * every SM pair of one GPC, and the width independence of remote stores.
 *
 * Usage: bin/dsmem_async {k4|k4ring|k5|k6|k7|ncu1|clockcheck} [--quick] [--cfg N]. One invocation = one run_id.
 * Placement: 200 KB dynamic smem (1 CTA per SM), explicit cluster attribute, Default policy; %smid of every CTA recorded;
 *   a launch whose ranks do not sit on distinct SMs or whose NVML sample shows a throttle reason / clock below the lock is
 *   discarded (results/12__atomics__discards.csv) and repeated. Lone clusters: CS 2 lands on SMs 124/125, CS 16 on GPC 1.
 * Shared layout per CTA: 128 KiB data region (zeroed), then mbarriers and flags.
 *
 * K4 — k4_issue<OP>: CS 2 (lone) or a full wave of CS 2/4/8/16 (k4ring: every rank → rank + 1, all ranks issue).
 *   1024 threads per issuing CTA, 4096 operations per thread (1024 steps × 4 independent operations; atom results are
 *   XORed by volatile PTX after the step's last issue). OP: red.shared::cluster.add.u32 (RED.E), atom.shared::cluster
 *   .add.u32 (ATOM.E), red.async.add.u32 (REDAS), st.async.b32 (STAS), st.async.v4.b32 (STAS.128). Patterns: distinct
 *   (word (k·1024 + t) mod 32 768: 32 consecutive words per warp instruction, conflict-free), per32 (the 32 lanes of a
 *   warp on one word, word = k·32 + warp), one (every thread on word 0); st.async: distinct only.
 *   .async completion: before the first cluster.sync() the destination arms one mbarrier per chunk of 512 KiB of
 *   incoming payload (arrive.expect_tx; count 1; the tx-count limit is 2^20 − 1); every operation names the barrier of
 *   its chunk (mapa to the destination). Destination thread 0 waits (try_wait.parity.acquire.cluster) on every chunk
 *   barrier in order; completion = destination %clock64 at the last barrier − its %clock64 after the cluster.sync()
 *   (+ %globaltimer at both ends). Issuer: t0 after the cluster.sync(), t1 after every thread's fence.acq_rel.cluster
 *   (red, atom: the operations are performed; .async: issue only) and a bar.sync. Memory sums are checked against the
 *   issued operations (add forms; st.async: every word written).
 * K5 — k5_contend<OP>: lone CS 16 (GPC 1); sources = ranks 1 … N, N ∈ {1, 2, 4, 8, 15}; each source CTA (1024 threads)
 *   issues 1024 operations per thread into rank 0's region: same word (word 0), same bank (word 32·((s·1024 + t) mod
 *   1024)), spread (word (s·1024 + t + k·N·1024) mod 32 768). OP: red, atom (ILP 4), red.async (barriers per source and
 *   chunk). After its fence every source adds 1 to a counter in rank 0 (red.release.cluster); rank 0's thread 0 waits for
 *   N (or for every red.async barrier) → completion. dest_busy = 1: warps 1–31 of rank 0 run local atom.shared.add.u32 on
 *   the same words until thread 0 has seen completion; they report their own operations and cycles.
 * K6 — k6_pingpong<M, GT>: lone CS 16 (GPC 1), all 120 unordered rank pairs (a < b) in turn, separated by cluster.sync();
 *   200 (GT: 100) trials per pair and launch, epoch e = trial + 1. a: t0, signal(b, e), wait(e), t1 = read_clock64_after;
 *   b: wait(e), signal(a, e). Signals M: st_plain (st.shared::cluster flag = e), st_release (st.release.cluster), st_async
 *   (st.async data = e + complete_tx on the peer's barrier for this pair), red_async (red.async.add 1 + complete_tx),
 *   red_cluster (red.shared::cluster.add 1 on the flag), atom_cluster (atom.shared::cluster.add 1 on the flag; the
 *   returned value is consumed before polling). Waits: flag methods poll ld.volatile.shared of the own flag ≥ e; mbarrier
 *   methods try_wait.parity.acquire.cluster on the own barrier of this pair (re-armed with expect_tx(4) after each
 *   observation), then read the data word. GT = 1: %globaltimer at send and observation on both sides (one-way); GT = 0:
 *   no %globaltimer read in the loop (round trip only). Mode k6poll1: the same with the wait loops kept rolled (#pragma unroll
 *   1: one poll and one YIELD per iteration, the loop shape of experiment 09); in k6, ptxas unrolls the flag-poll loop
 *   (several polls per YIELD). variant gets the suffix _poll1.
 * K7 — k7_width<P>: lone CS 2, rank 0, 32 threads stream 256 KB into rank 1's 128 KiB region (wraps twice) with
 *   st.shared::cluster.{u8, u16, u32, u64, v2.u32, v4.u32} or st.async.{b32, b64, v2.b32, v4.b32} (256 KB = one barrier);
 *   address (k·32 + lane)·width mod 128 KiB. Issuer cycles: t0 after the cluster.sync() → stores → fence.acq_rel.cluster
 *   → __syncwarp → t1 (plain); st.async: issue cycles and the destination's completion (as K4).
 *
 * → results/12__dsmem_atomics__throughput.csv (one row per CTA role and launch): kernel (k4 | k4ring | k5), op, pattern,
 *   threads, cs, n_sources, dest_busy, role (src | dst | dst_local), rank, smid, ops, cycles, ops_per_cycle, bpc,
 *   completion_cycles, completion_ns, ms_event, check_ok.
 * → results/12__dsmem_atomics__visibility.csv (shared with atom_lat flags): primitive, scope (cluster), variant (gt | nogt),
 *   smid_src, smid_dst, trial, one_way_ns, one_way_back_ns, round_trip_cycles.
 * → results/12__dsmem_atomics__width.csv: primitive, width_bytes, bytes, cycles, bpc, side (issuer | dest), smid_src,
 *   smid_dst.
 */
#include <cstring>
#include <memory>

#include "common12.hpp"

constexpr uint32_t kRegion = 128 * 1024, kRegionWords = kRegion / 4;
constexpr uint32_t kChunkBytes = 512 * 1024;  // payload per completion barrier
constexpr uint32_t kMaxBars = 256;
constexpr uint32_t kSpin = 1u << 26;

// smem layout after the 128 KiB region
struct Ctl {
    uint64_t bar[kMaxBars];
    uint32_t flag[32], data[32], counter, done, err;
    unsigned long long sum;
};
__device__ __forceinline__ Ctl* ctl_of(unsigned char* dyn) { return reinterpret_cast<Ctl*>(dyn + kRegion); }

__device__ __forceinline__ void zero_region(unsigned char* dyn) {
    for (uint32_t i = threadIdx.x; i < kRegionWords; i += blockDim.x) reinterpret_cast<uint32_t*>(dyn)[i] = 0;
}
__device__ __forceinline__ unsigned long long region_sum(unsigned char* dyn) {  // block-wide; valid in thread 0
    __shared__ unsigned long long s;
    if (threadIdx.x == 0) s = 0;
    __syncthreads();
    unsigned long long p = 0;
    for (uint32_t i = threadIdx.x; i < kRegionWords; i += blockDim.x) p += reinterpret_cast<uint32_t*>(dyn)[i];
    for (int o = 16; o; o >>= 1) p += __shfl_xor_sync(0xffffffffu, p, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(&s, p);
    __syncthreads();
    return s;
}
// wait for barriers [b0, b0 + n) (phase 0) in order; false on a timeout
__device__ __forceinline__ bool wait_bars(uint32_t bars, uint32_t b0, uint32_t n) {
    for (uint32_t b = b0; b < b0 + n; ++b) {
        bool ok = false;
        for (uint32_t k = 0; k < kSpin && !ok; ++k) ok = mbar_try_wait_parity_acq_cluster(bars + 8 * b, 0);
        if (!ok) return false;
    }
    return true;
}

// ---------------------------------------------------------------- K4
enum Op4 { O_RED = 0, O_ATOM, O_REDASYNC, O_STASYNC, O_STASYNC4 };
static const char* kOp4Name[] = {"red.shared::cluster.add.u32", "atom.shared::cluster.add.u32", "red.async.add.u32", "st.async.b32",
                                 "st.async.v4.b32"};
enum Pat4 { P_DISTINCT = 0, P_PER32, P_ONE, P_BANK };
static const char* kPat4Name[] = {"distinct", "per32", "one", "same_bank"};
__host__ __device__ constexpr uint32_t op4_bytes(int op) { return op == O_STASYNC4 ? 16 : 4; }

struct K4Rec { uint64_t c0, c1, n0, n1, ops, comp_c, comp_n, sum; uint32_t smid, rank, err, pad; };
struct K4P {
    uint32_t pattern, steps, ring, issuers_mask;  // ring: dst = rank + 1 (else rank 0 → 1); issuers_mask: ranks that issue
    uint64_t zero;
    uint32_t* place;
    K4Rec* rec;
};
template <int OP>
__device__ __forceinline__ uint32_t op4(uint32_t a, uint32_t rbar, uint32_t v) {
    if constexpr (OP == O_RED) { asm volatile("red.shared::cluster.add.u32 [%0], %1;" :: "r"(a), "r"(v) : "memory"); return 0; }
    else if constexpr (OP == O_ATOM) { uint32_t o; asm volatile("atom.shared::cluster.add.u32 %0, [%1], %2;" : "=r"(o) : "r"(a), "r"(v) : "memory"); return o; }
    else if constexpr (OP == O_REDASYNC) { red_async_add_u32(a, v, rbar); return 0; }
    else if constexpr (OP == O_STASYNC) { st_async_u32(a, v, rbar); return 0; }
    else { st_async_v4_b32(a, make_uint4(v, v, v, v), rbar); return 0; }
}

template <int OP>
__global__ void __launch_bounds__(1024, 1) k4_issue(K4P p) {
    extern __shared__ __align__(16) unsigned char dyn[];
    constexpr int ILP = 4;
    constexpr bool ASYNC = OP >= O_REDASYNC;
    Ctl* ctl = ctl_of(dyn);
    const ClusterRegs cr = read_cluster_regs();
    const uint32_t rank = cr.ctarank, cs = cr.nctarank, t = threadIdx.x, T = blockDim.x, lane = t & 31, warp = t / 32;
    record_place(p.place);
    zero_region(dyn);
    // who sends to me: ring → rank − 1; else rank 1 receives from rank 0
    const uint32_t dst = p.ring ? (rank + 1) % cs : 1u;
    const uint32_t src = p.ring ? (rank + cs - 1) % cs : 0u;
    const bool issuer = (p.issuers_mask >> rank) & 1u;
    const bool receiver = (p.issuers_mask >> src) & 1u && (p.ring || rank == 1);
    const uint64_t payload = (uint64_t)T * p.steps * ILP * op4_bytes(OP);
    const uint32_t nchunks = (uint32_t)((payload + kChunkBytes - 1) / kChunkBytes);
    if (t == 0) {
        ctl->err = 0;
        if (ASYNC && receiver)
            for (uint32_t b = 0; b < nchunks; ++b) {
                mbar_init(smem_u32(&ctl->bar[b]), 1);
                const uint64_t rest = payload - (uint64_t)b * kChunkBytes;
                mbar_arrive_expect_tx(smem_u32(&ctl->bar[b]), (uint32_t)(rest < kChunkBytes ? rest : kChunkBytes));
            }
        fence_mbarrier_init();
    }
    __syncthreads();
    cluster_sync();
    const TimerPair s0 = read_timers_pair();
    TimerPair s1{};
    uint32_t acc = 0;
    if (issuer) {
        const uint32_t rbase = mapa_u32(smem_u32(dyn), dst), rbars = mapa_u32(smem_u32(&ctl->bar[0]), dst);
        const uint32_t ops_per_chunk = kChunkBytes / op4_bytes(OP) / T;  // operations of one thread per chunk (steps·ILP units)
        const uint32_t zl = lane & (uint32_t)p.zero;
#pragma unroll 1
        for (uint32_t s = 0; s < p.steps; ++s) {
            uint32_t r[ILP];
#pragma unroll
            for (int j = 0; j < ILP; ++j) {
                const uint32_t k = s * ILP + j;
                uint32_t w;
                if (OP == O_STASYNC4) w = ((k * T + t) * 4) % kRegionWords;
                else if (p.pattern == P_DISTINCT) w = (k * T + t) % kRegionWords;
                else if (p.pattern == P_PER32) w = (k * (T / 32) + warp) % kRegionWords;
                else w = 0;
                w += zl;
                r[j] = op4<OP>(rbase + 4 * w, rbars + 8 * (k / ops_per_chunk), 1u);
            }
            if constexpr (OP == O_ATOM) {
#pragma unroll
                for (int j = 0; j < ILP; ++j) asm volatile("xor.b32 %0, %0, %1;" : "+r"(acc) : "r"(r[j]));
            }
        }
        fence_acq_rel_cluster();
        __syncthreads();
        s1 = read_timers_after(acc, p.zero);
    } else {
        __syncthreads();
    }
    uint64_t cc = 0, cn = 0;
    if (t == 0 && receiver) {
        if (ASYNC) {
            if (!wait_bars(smem_u32(&ctl->bar[0]), 0, nchunks)) ctl->err = 1;
            const TimerPair e = read_timers_pair();
            cc = e.cycles - s0.cycles; cn = e.ns - s0.ns;
        }
    }
    __syncthreads();
    cluster_sync();  // every writer done: the region sums are final
    const unsigned long long sum = region_sum(dyn);
    if (t == 0) {
        K4Rec& q = p.rec[blockIdx.x];
        q = {s0.cycles, s1.cycles, s0.ns, s1.ns, issuer ? (uint64_t)T * p.steps * ILP : 0, cc, cn, sum, read_smid(), rank, ctl->err | (acc & (uint32_t)p.zero), 0};
    }
}

// ---------------------------------------------------------------- K5
struct K5P {
    uint32_t pattern, n_src, ops_per_thread, dest_busy;
    uint64_t zero;
    uint32_t* place;
    K4Rec* rec;
    uint64_t* local;  // [2]: rank 0's local ops, cycles (dest_busy)
};
template <int OP>
__global__ void __launch_bounds__(1024, 1) k5_contend(K5P p) {
    extern __shared__ __align__(16) unsigned char dyn[];
    constexpr int ILP = 4;
    constexpr bool ASYNC = OP == O_REDASYNC;
    Ctl* ctl = ctl_of(dyn);
    const ClusterRegs cr = read_cluster_regs();
    const uint32_t rank = cr.ctarank, t = threadIdx.x, T = blockDim.x, lane = t & 31;
    record_place(p.place);
    zero_region(dyn);
    const uint64_t payload = (uint64_t)T * p.ops_per_thread * 4;  // per source
    const uint32_t nch = (uint32_t)((payload + kChunkBytes - 1) / kChunkBytes);
    if (t == 0) {
        ctl->counter = 0; ctl->done = 0; ctl->err = 0;
        if (rank == 0 && ASYNC)
            for (uint32_t s = 1; s <= p.n_src; ++s)
                for (uint32_t b = 0; b < nch; ++b) {
                    const uint32_t bar = smem_u32(&ctl->bar[(s - 1) * nch + b]);
                    mbar_init(bar, 1);
                    const uint64_t rest = payload - (uint64_t)b * kChunkBytes;
                    mbar_arrive_expect_tx(bar, (uint32_t)(rest < kChunkBytes ? rest : kChunkBytes));
                }
        fence_mbarrier_init();
    }
    __syncthreads();
    cluster_sync();
    const TimerPair s0 = read_timers_pair();
    TimerPair s1{};
    uint64_t lops = 0, lcyc = 0, cc = 0, cn = 0;
    uint32_t acc = 0;
    const uint32_t zl = lane & (uint32_t)p.zero;
    auto word = [&](uint32_t s, uint32_t k) -> uint32_t {  // word of source s, thread t, operation k
        if (p.pattern == P_ONE) return 0;
        if (p.pattern == P_BANK) return 32 * ((s * 1024 + t) % 1024);
        return (s * 1024 + t + k * p.n_src * 1024) % kRegionWords;
    };
    if (rank >= 1 && rank <= p.n_src) {
        const uint32_t rbase = mapa_u32(smem_u32(dyn), 0), rbars = mapa_u32(smem_u32(&ctl->bar[(rank - 1) * nch]), 0);
        const uint32_t per_chunk = kChunkBytes / 4 / T;
#pragma unroll 1
        for (uint32_t k0 = 0; k0 < p.ops_per_thread; k0 += ILP) {
            uint32_t r[ILP];
#pragma unroll
            for (int j = 0; j < ILP; ++j) {
                const uint32_t k = k0 + j;
                r[j] = op4<OP>(rbase + 4 * (word(rank, k) + zl), rbars + 8 * (k / per_chunk), 1u);
            }
            if constexpr (OP == O_ATOM) {
#pragma unroll
                for (int j = 0; j < ILP; ++j) asm volatile("xor.b32 %0, %0, %1;" : "+r"(acc) : "r"(r[j]));
            }
        }
        fence_acq_rel_cluster();
        __syncthreads();
        s1 = read_timers_after(acc, p.zero);
        if (t == 0 && !ASYNC) red_release_cluster_add_u32(mapa_u32(smem_u32(&ctl->counter), 0), 1u);
    } else if (rank == 0) {
        if (t == 0) {  // completion: every source's counter increment (red, atom) or every barrier (red.async)
            if (ASYNC) { if (!wait_bars(smem_u32(&ctl->bar[0]), 0, p.n_src * nch)) ctl->err = 1; }
            else {
                uint32_t k = 0;
                while (ld_acquire_cluster_u32(mapa_u32(smem_u32(&ctl->counter), 0)) < p.n_src)
                    if (++k > kSpin) { ctl->err = 1; break; }
            }
            const TimerPair e = read_timers_pair();
            cc = e.cycles - s0.cycles; cn = e.ns - s0.ns;
            asm volatile("st.volatile.shared.u32 [%0], %1;" :: "r"(smem_u32(&ctl->done)), "r"(1u) : "memory");
        } else if (p.dest_busy && t >= 32) {  // local atomics on the same words until thread 0 has seen completion
            const uint32_t base = smem_u32(dyn);
            const uint64_t c0 = read_clock64();
            uint32_t k = 0;
            while (!ld_volatile_shared_u32(smem_u32(&ctl->done))) {
#pragma unroll
                for (int j = 0; j < 16; ++j, ++k) {
                    uint32_t o;
                    asm volatile("atom.shared::cta.add.u32 %0, [%1], %2;" : "=r"(o) : "r"(base + 4 * (word(1 + (t & 7) % p.n_src, k) + zl)), "r"(1u) : "memory");
                    acc ^= o;
                }
                lops += 16;
            }
            lcyc = read_clock64_after(acc, p.zero) - c0;
            atomicAdd((unsigned long long*)&p.local[0], (unsigned long long)lops);
            atomicMax((unsigned long long*)&p.local[1], (unsigned long long)lcyc);
        }
    }
    __syncthreads();
    cluster_sync();
    const unsigned long long sum = region_sum(dyn);
    if (t == 0) {
        const bool src = rank >= 1 && rank <= p.n_src;
        p.rec[blockIdx.x] = {s0.cycles, s1.cycles, s0.ns, s1.ns, src ? (uint64_t)T * p.ops_per_thread : 0, cc, cn, sum, read_smid(), rank,
                             ctl->err | (acc & (uint32_t)p.zero), 0};
    }
}

// ---------------------------------------------------------------- K6
enum M6 { M_ST = 0, M_REL, M_STASYNC, M_REDASYNC, M_RED, M_ATOM };
static const char* kM6Name[] = {"st_plain", "st_release", "st_async_mbarrier", "red_async_mbarrier", "red_cluster", "atom_cluster"};
template <int M> constexpr bool kBar6 = M == M_STASYNC || M == M_REDASYNC;
constexpr uint32_t kTrials = 200, kTrialsGt = 100;

// POLL1 = 1: the wait loops are kept rolled (#pragma unroll 1: one poll, one YIELD per iteration, as in experiment 09);
// POLL1 = 0: ptxas unrolls them (several polls per YIELD).
template <int M, int GT, int POLL1>
__global__ void __launch_bounds__(32, 1) k6_pingpong(uint32_t trials, uint64_t zero, uint32_t* place, uint32_t* rt, uint32_t* g, uint32_t* info) {
    extern __shared__ __align__(16) unsigned char dyn[];
    __shared__ uint32_t s_rt[kTrials], s_g[2][kTrials];
    Ctl* ctl = ctl_of(dyn);
    const ClusterRegs cr = read_cluster_regs();
    const uint32_t rank = cr.ctarank, cs = cr.nctarank, lane = threadIdx.x;
    record_place(place);
    if (lane == 0) {
        for (uint32_t r = 0; r < cs; ++r) {
            ctl->flag[r] = 0; ctl->data[r] = 0;
            mbar_init(smem_u32(&ctl->bar[r]), 1);
            if (kBar6<M>) mbar_arrive_expect_tx(smem_u32(&ctl->bar[r]), 4u);  // phase 0 armed
        }
        fence_mbarrier_init();
    }
    __syncwarp();
    cluster_sync();
    uint32_t timeouts = 0, mism = 0, sink = 0, pair = 0;
    for (uint32_t a = 0; a < cs; ++a)
        for (uint32_t b = a + 1; b < cs; ++b, ++pair) {
            const bool on = rank == a || rank == b;
            const uint32_t peer = rank == a ? b : a;
            if (lane == 0 && on) {
                // own slots for this peer; the peer's slots for me
                const uint32_t own_flag = smem_u32(&ctl->flag[peer]), own_data = smem_u32(&ctl->data[peer]), own_bar = smem_u32(&ctl->bar[peer]);
                const uint32_t pflag = mapa_u32(smem_u32(&ctl->flag[rank]), peer), pdata = mapa_u32(smem_u32(&ctl->data[rank]), peer);
                const uint32_t pbar = mapa_u32(smem_u32(&ctl->bar[rank]), peer);
                auto signal = [&](uint32_t e) {
                    if constexpr (M == M_ST) st_dsmem_u32(pflag, e);
                    else if constexpr (M == M_REL) st_release_cluster_u32(pflag, e);
                    else if constexpr (M == M_STASYNC) st_async_u32(pdata, e, pbar);
                    else if constexpr (M == M_REDASYNC) red_async_add_u32(pdata, 1u, pbar);
                    else if constexpr (M == M_RED) asm volatile("red.shared::cluster.add.u32 [%0], %1;" :: "r"(pflag), "r"(1u) : "memory");
                    else { uint32_t o; asm volatile("atom.shared::cluster.add.u32 %0, [%1], %2;" : "=r"(o) : "r"(pflag), "r"(1u) : "memory"); sink += o; }
                };
                auto wait = [&](uint32_t e, uint32_t i, uint32_t& v) -> bool {
                    if constexpr (kBar6<M>) {
                        bool ok = false;
                        if constexpr (POLL1) {
#pragma unroll 1
                            for (uint32_t k = 0; k < kSpin && !ok; ++k) ok = mbar_try_wait_parity_acq_cluster(own_bar + (sink & (uint32_t)zero), i & 1u);
                        } else {
                            for (uint32_t k = 0; k < kSpin && !ok; ++k) ok = mbar_try_wait_parity_acq_cluster(own_bar + (sink & (uint32_t)zero), i & 1u);
                        }
                        asm volatile("ld.shared::cta.u32 %0, [%1];" : "=r"(v) : "r"(own_data) : "memory");
                        if (ok) mbar_arrive_expect_tx(own_bar, 4u);  // arm the next phase
                        return ok;
                    } else {
                        if constexpr (POLL1) {
#pragma unroll 1
                            for (uint32_t k = 0; k < kSpin; ++k)
                                if ((v = ld_volatile_shared_u32(own_flag + (sink & (uint32_t)zero))) >= e) return true;
                        } else {
                            for (uint32_t k = 0; k < kSpin; ++k)
                                if ((v = ld_volatile_shared_u32(own_flag + (sink & (uint32_t)zero))) >= e) return true;
                        }
                        return false;
                    }
                };
#pragma unroll 1
                for (uint32_t i = 0; i < trials; ++i) {
                    const uint32_t e = i + 1;
                    uint32_t v = 0;
                    if (rank == a) {
                        const uint64_t t0 = read_clock64();
                        const uint32_t ga = GT ? (uint32_t)read_globaltimer() : 0u;
                        signal(e);
                        const bool ok = wait(e, i, v);
                        const uint64_t t1 = read_clock64_after(v, zero);
                        const uint32_t gb = GT ? (uint32_t)read_globaltimer() + (v & (uint32_t)zero) : 0u;
                        if (!ok) { ++timeouts; break; }
                        mism += v != e;
                        s_rt[i] = (uint32_t)(t1 - t0); s_g[0][i] = ga; s_g[1][i] = gb;
                    } else {
                        const bool ok = wait(e, i, v);
                        const uint32_t ga = GT ? (uint32_t)read_globaltimer() + (v & (uint32_t)zero) : 0u;
                        const uint32_t gb = GT ? (uint32_t)read_globaltimer() : 0u;
                        if (!ok) { ++timeouts; break; }
                        mism += v != e;
                        signal(e);
                        s_g[0][i] = ga; s_g[1][i] = gb;
                    }
                }
            }
            __syncwarp();
            if (on)  // out: rt[pair][i]; g[((pair · 2 + (rank == b)) · 2 + k)][i]
                for (uint32_t i = lane; i < trials; i += 32) {
                    const uint32_t side = rank == b;
                    if (!side) rt[pair * trials + i] = s_rt[i];
                    g[((pair * 2 + side) * 2 + 0) * trials + i] = s_g[0][i];
                    g[((pair * 2 + side) * 2 + 1) * trials + i] = s_g[1][i];
                }
            __syncwarp();
            cluster_sync();
        }
    if (lane == 0) {
        uint32_t* inf = info + 4 * blockIdx.x;
        inf[0] = read_smid(); inf[1] = timeouts; inf[2] = mism; inf[3] = sink & (uint32_t)zero;
    }
}

// ---------------------------------------------------------------- K7
enum P7 { W_U8 = 0, W_U16, W_U32, W_U64, W_V2, W_V4, A_B32, A_B64, A_V2, A_V4 };
static const char* kP7Name[] = {"st.shared::cluster.u8", "st.shared::cluster.u16", "st.shared::cluster.u32", "st.shared::cluster.u64",
                                "st.shared::cluster.v2.u32", "st.shared::cluster.v4.u32", "st.async.b32", "st.async.b64", "st.async.v2.b32",
                                "st.async.v4.b32"};
__host__ __device__ constexpr uint32_t p7_bytes(int p) {
    return p == W_U8 ? 1 : p == W_U16 ? 2 : (p == W_U32 || p == A_B32) ? 4 : (p == W_U64 || p == W_V2 || p == A_B64 || p == A_V2) ? 8 : 16;
}
constexpr uint32_t kK7Bytes = 256 * 1024;
template <int P>
__global__ void __launch_bounds__(32, 1) k7_width(uint64_t zero, uint32_t* place, K4Rec* rec) {
    extern __shared__ __align__(16) unsigned char dyn[];
    constexpr uint32_t W = p7_bytes(P);
    constexpr bool ASYNC = P >= A_B32;
    Ctl* ctl = ctl_of(dyn);
    const uint32_t rank = read_cluster_regs().ctarank, lane = threadIdx.x;
    record_place(place);
    if (lane == 0) {
        ctl->err = 0;
        if (ASYNC && rank == 1) { mbar_init(smem_u32(&ctl->bar[0]), 1); mbar_arrive_expect_tx(smem_u32(&ctl->bar[0]), kK7Bytes); }
        fence_mbarrier_init();
    }
    __syncwarp();
    cluster_sync();
    const TimerPair s0 = read_timers_pair();
    TimerPair s1{};
    uint64_t cc = 0, cn = 0;
    if (rank == 0) {
        const uint32_t rbase = mapa_u32(smem_u32(dyn), 1), rbar = mapa_u32(smem_u32(&ctl->bar[0]), 1);
        const uint32_t n = kK7Bytes / (32 * W), zl = lane & (uint32_t)zero;
#pragma unroll 4
        for (uint32_t k = 0; k < n; ++k) {
            const uint32_t a = rbase + ((k * 32 + lane) * W) % kRegion + zl;
            if constexpr (P == W_U8) st_dsmem_u8(a, (uint16_t)k);
            else if constexpr (P == W_U16) st_dsmem_u16(a, (uint16_t)k);
            else if constexpr (P == W_U32) st_dsmem_u32(a, k);
            else if constexpr (P == W_U64) st_dsmem_u64(a, k);
            else if constexpr (P == W_V2) st_dsmem_v2_u32(a, k, k);
            else if constexpr (P == W_V4) st_dsmem_v4_u32(a, make_uint4(k, k, k, k));
            else if constexpr (P == A_B32) st_async_u32(a, k, rbar);
            else if constexpr (P == A_B64) st_async_b64(a, k, rbar);
            else if constexpr (P == A_V2) st_async_v2_b32(a, k, k, rbar);
            else st_async_v4_b32(a, make_uint4(k, k, k, k), rbar);
        }
        if constexpr (!ASYNC) fence_acq_rel_cluster();
        __syncwarp();
        s1 = read_timers_pair();
    } else if (ASYNC && lane == 0) {
        if (!wait_bars(smem_u32(&ctl->bar[0]), 0, 1)) ctl->err = 1;
        const TimerPair e = read_timers_pair();
        cc = e.cycles - s0.cycles; cn = e.ns - s0.ns;
    }
    __syncwarp();
    cluster_sync();
    if (lane == 0) rec[blockIdx.x] = {s0.cycles, s1.cycles, s0.ns, s1.ns, rank == 0 ? kK7Bytes / W : 0, cc, cn, 0, read_smid(), rank, ctl->err, 0};
}

// ---------------------------------------------------------------- host
struct Ctx {
    hb::Run& run;
    int sm = 0;
    bool quick = false;
    Parts P;
    Discards12 disc;
    long long li = 0;
    cudaEvent_t ev0{}, ev1{};
    std::unique_ptr<hb::CsvWriter> tput, vis, width;
    Ctx(hb::Run& r, bool q) : run(r), sm(r.dev().sm_count), quick(q), disc(r) {
        CUDA_CHECK(cudaEventCreate(&ev0));
        CUDA_CHECK(cudaEventCreate(&ev1));
    }
    hb::CsvWriter& tputcsv() {
        if (!tput)
            tput = std::make_unique<hb::CsvWriter>(hb::Run::results("12__dsmem_atomics__throughput.csv"),
                hb::with_mandatory({"kernel", "op", "pattern", "threads", "cs", "n_sources", "dest_busy", "role", "rank", "smid", "cluster_id", "ops",
                                    "cycles", "ops_per_cycle", "bpc", "completion_cycles", "completion_ns", "ms_event", "check_ok"}));
        return *tput;
    }
};
constexpr size_t kSmem = kOneCtaPerSmSmem;

// generic recorded cluster launch: fn(prepared) launches; returns placement and clock sample
template <class F>
static std::pair<hb::ClockSample, std::vector<uint32_t>> rlaunch(Ctx& x, const std::string& label, int cs, int ncl, uint32_t* d_place, F&& go,
                                                                 float* ms = nullptr) {
    for (int t = 0; t < 20; ++t) {
        CUDA_CHECK(cudaEventRecord(x.ev0));
        go();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(x.ev1));
        CUDA_CHECK(cudaDeviceSynchronize());
        if (ms) CUDA_CHECK(cudaEventElapsedTime(ms, x.ev0, x.ev1));
        const auto s = x.run.clock_note(label, "launch", x.li);
        std::vector<uint32_t> pl(4 * (size_t)cs * ncl);
        CUDA_CHECK(cudaMemcpy(pl.data(), d_place, pl.size() * 4, cudaMemcpyDeviceToHost));
        const Placement P(pl, cs, ncl);
        if (!P.ok()) { x.disc.add(label, "placement", x.li, s); continue; }
        if (s.bad()) { x.disc.add(label, "clock", x.li, s); continue; }
        return {s, P.sm};
    }
    hb::die(label + ": no clean launch");
}

static hb::ClusterLaunch cl_launch(int cs, int ncl, int threads) {
    hb::ClusterLaunch L;
    L.grid = dim3(cs * ncl); L.block = dim3(threads); L.smem = kSmem; L.cluster = dim3(cs, 1, 1); L.explicit_cluster = true;
    return L;
}

// ---------------------------------------------------------------- K4 host
using K4Fn = void (*)(K4P);
static K4Fn k4fn(int op) {
    switch (op) {
        case O_RED: return k4_issue<O_RED>;
        case O_ATOM: return k4_issue<O_ATOM>;
        case O_REDASYNC: return k4_issue<O_REDASYNC>;
        case O_STASYNC: return k4_issue<O_STASYNC>;
        default: return k4_issue<O_STASYNC4>;
    }
}
static void run_k4(Ctx& x, bool ring) {
    const int rec = x.quick ? 2 : 10;
    const std::vector<int> css = ring ? std::vector<int>{2, 4, 8, 16} : std::vector<int>{2};
    x.run.lock_clocks_note(ring ? "k4ring" : "k4", "before");
    for (int cs : css)
        for (int op : {O_RED, O_ATOM, O_REDASYNC, O_STASYNC, O_STASYNC4})
            for (int pat : {P_DISTINCT, P_PER32, P_ONE}) {
                if (op >= O_STASYNC && pat != P_DISTINCT) continue;
                if (ring && pat != P_DISTINCT) continue;
                hb::ClusterLaunch L = cl_launch(cs, 1, 1024);
                const int ncl = ring ? L.max_active_clusters(k4fn(op)) : 1;
                L = cl_launch(cs, ncl, 1024);
                DevBuf<uint32_t> d_place(4 * (size_t)cs * ncl);
                DevBuf<K4Rec> d_rec((size_t)cs * ncl);
                K4P p{};
                p.pattern = (uint32_t)pat; p.steps = x.quick ? 256 : 1024; p.ring = ring; p.issuers_mask = ring ? 0xFFFFu : 1u; p.zero = 0;
                p.place = d_place.d; p.rec = d_rec.d;
                hb::ClusterLaunch::Prepared prep;
                K4Fn fn = k4fn(op);
                L.prepare(fn, prep);
                const std::string label = fmt("%s/cs%d/%s/%s", ring ? "k4ring" : "k4", cs, kOp4Name[op], kPat4Name[pat]);
                for (int l = 0; l < 2 + rec; ++l) {
                    float ms = 0;
                    auto [s, sm] = rlaunch(x, label, cs, ncl, d_place.d, [&] { CUDA_CHECK(cudaLaunchKernelEx(&prep.cfg, fn, p)); }, &ms);
                    if (l < 2) continue;
                    const auto r = d_rec.get();
                    const std::string pre = x.run.prefix(x.li, s);
                    for (int b = 0; b < cs * ncl; ++b) {
                        const K4Rec& q = r[b];
                        const int src = ring ? (b / cs) * cs + (int)((q.rank + cs - 1) % cs) : (q.rank == 1 ? b - 1 : b);
                        const bool issuer = q.ops > 0, receiver = ring || q.rank == 1;
                        // check: add forms Σ = received ops; st.async: every written word holds 1 (Σ = words written ∩ region)
                        int ok = -1;
                        if (receiver) {
                            const uint64_t got = r[src].ops * (op == O_STASYNC4 ? 4 : 1);
                            const uint64_t expect = op >= O_STASYNC ? std::min<uint64_t>(got, kRegionWords) : got;
                            ok = q.sum == expect && !q.err;
                        }
                        if (issuer) {
                            const uint64_t cyc = q.c1 - q.c0;
                            x.tputcsv().add(pre + fmt(",%s,%s,%s,1024,%d,1,0,src,%u,%u,%d,%llu,%llu,%.5f,%.4f,,,%.6f,", ring ? "k4ring" : "k4", kOp4Name[op],
                                                      kPat4Name[pat], cs, q.rank, q.smid, b / cs, (unsigned long long)q.ops, (unsigned long long)cyc,
                                                      double(q.ops) / cyc, double(q.ops) * op4_bytes(op) / cyc, ms));
                        }
                        if (receiver) {
                            const uint64_t ops = r[src].ops;
                            const std::string comp = op >= O_REDASYNC ? fmt("%llu,%llu", (unsigned long long)q.comp_c, (unsigned long long)q.comp_n) : ",";
                            x.tputcsv().add(pre + fmt(",%s,%s,%s,1024,%d,1,0,dst,%u,%u,%d,%llu,,,,%s,%.6f,%d", ring ? "k4ring" : "k4", kOp4Name[op],
                                                      kPat4Name[pat], cs, q.rank, q.smid, b / cs, (unsigned long long)ops, comp.c_str(), ms, ok));
                            if (ok == 0) std::printf("  WARNING %s: destination sum %llu (err %u)\n", label.c_str(), (unsigned long long)q.sum, q.err);
                        }
                    }
                    x.tputcsv().flush();
                    ++x.li;
                }
                x.tputcsv().sync();
                std::printf("%s done\n", label.c_str());
            }
    x.run.lock_clocks_note(ring ? "k4ring" : "k4", "after");
}

// ---------------------------------------------------------------- K5 host
using K5Fn = void (*)(K5P);
static K5Fn k5fn(int op) { return op == O_RED ? k5_contend<O_RED> : op == O_ATOM ? k5_contend<O_ATOM> : k5_contend<O_REDASYNC>; }
static void run_k5(Ctx& x) {
    const int rec = x.quick ? 2 : 10, cs = 16;
    DevBuf<uint32_t> d_place(4 * cs);
    DevBuf<K4Rec> d_rec(cs);
    DevBuf<uint64_t> d_local(2);
    x.run.lock_clocks_note("k5", "before");
    for (int op : {O_RED, O_ATOM, O_REDASYNC})
        for (int pat : {P_ONE, P_BANK, P_DISTINCT})
            for (uint32_t n : {1u, 2u, 4u, 8u, 15u})
                for (int busy : {0, 1}) {
                    if (x.quick && n != 1 && n != 15) continue;
                    hb::ClusterLaunch L = cl_launch(cs, 1, 1024);
                    K5P p{};
                    p.pattern = (uint32_t)pat; p.n_src = n; p.ops_per_thread = pat == P_ONE ? 64 : 1024; p.dest_busy = busy; p.zero = 0;
                    p.place = d_place.d; p.rec = d_rec.d; p.local = d_local.d;
                    if (x.quick) p.ops_per_thread = std::min<uint32_t>(p.ops_per_thread, 256);
                    hb::ClusterLaunch::Prepared prep;
                    K5Fn fn = k5fn(op);
                    L.prepare(fn, prep);
                    const std::string label = fmt("k5/%s/%s/n%u/busy%d", kOp4Name[op], kPat4Name[pat], n, busy);
                    for (int l = 0; l < 2 + rec; ++l) {
                        d_local.zero();
                        float ms = 0;
                        auto [s, sm] = rlaunch(x, label, cs, 1, d_place.d, [&] { CUDA_CHECK(cudaLaunchKernelEx(&prep.cfg, fn, p)); }, &ms);
                        if (l < 2) continue;
                        const auto r = d_rec.get();
                        const auto loc = d_local.get();
                        const std::string pre = x.run.prefix(x.li, s);
                        uint64_t total = 0;
                        for (int b = 1; b <= (int)n; ++b) {
                            const K4Rec& q = r[b];
                            total += q.ops;
                            const uint64_t cyc = q.c1 - q.c0;
                            x.tputcsv().add(pre + fmt(",k5,%s,%s,1024,16,%u,%d,src,%u,%u,0,%llu,%llu,%.5f,%.4f,,,%.6f,", kOp4Name[op], kPat4Name[pat], n, busy,
                                                      q.rank, q.smid, (unsigned long long)q.ops, (unsigned long long)cyc, double(q.ops) / cyc,
                                                      double(q.ops) * 4 / cyc, ms));
                        }
                        const K4Rec& d = r[0];
                        // rank 0's region holds Σ received + its own local adds
                        const int ok = !d.err && d.sum == total + loc[0];
                        x.tputcsv().add(pre + fmt(",k5,%s,%s,1024,16,%u,%d,dst,0,%u,0,%llu,%llu,%.5f,%.4f,%llu,%llu,%.6f,%d", kOp4Name[op], kPat4Name[pat], n, busy,
                                                  d.smid, (unsigned long long)total, (unsigned long long)d.comp_c, double(total) / d.comp_c,
                                                  double(total) * 4 / d.comp_c, (unsigned long long)d.comp_c, (unsigned long long)d.comp_n, ms, ok));
                        if (busy)
                            x.tputcsv().add(pre + fmt(",k5,atom.shared::cta.add.u32,%s,992,16,%u,1,dst_local,0,%u,0,%llu,%llu,%.5f,%.4f,,,%.6f,", kPat4Name[pat], n,
                                                      d.smid, (unsigned long long)loc[0], (unsigned long long)loc[1], loc[1] ? double(loc[0]) / loc[1] : 0.0,
                                                      loc[1] ? double(loc[0]) * 4 / loc[1] : 0.0, ms));
                        if (!ok) std::printf("  WARNING %s: destination sum %llu, expected %llu (err %u)\n", label.c_str(), (unsigned long long)d.sum,
                                             (unsigned long long)(total + loc[0]), d.err);
                        x.tputcsv().flush();
                        ++x.li;
                    }
                    x.tputcsv().sync();
                    std::printf("%s done\n", label.c_str());
                }
    x.run.lock_clocks_note("k5", "after");
}

// ---------------------------------------------------------------- K6 host
using K6Fn = void (*)(uint32_t, uint64_t, uint32_t*, uint32_t*, uint32_t*, uint32_t*);
static K6Fn k6fn(int m, int gt, int poll1) {
#define K6(M) if (m == M) return poll1 ? (gt ? k6_pingpong<M, 1, 1> : k6_pingpong<M, 0, 1>) : (gt ? k6_pingpong<M, 1, 0> : k6_pingpong<M, 0, 0>);
    K6(M_ST) K6(M_REL) K6(M_STASYNC) K6(M_REDASYNC) K6(M_RED) K6(M_ATOM)
#undef K6
    hb::die("k6 kernel");
}
static void run_k6(Ctx& x, int poll1) {
    const int cs = 16, rec = x.quick ? 2 : 10, npairs = cs * (cs - 1) / 2;
    DevBuf<uint32_t> d_place(4 * cs), d_rt((size_t)npairs * kTrials), d_g((size_t)npairs * 4 * kTrials), d_info(4 * cs);
    if (!x.vis)
        x.vis = std::make_unique<hb::CsvWriter>(hb::Run::results("12__dsmem_atomics__visibility.csv"),
            hb::with_mandatory({"primitive", "scope", "variant", "smid_src", "smid_dst", "trial", "one_way_ns", "one_way_back_ns", "round_trip_cycles"}));
    x.run.lock_clocks_note("k6", "before");
    for (int m = 0; m < 6; ++m)
        for (int gt = 0; gt < 2; ++gt) {
            const uint32_t trials = x.quick ? 32 : gt ? kTrialsGt : kTrials;
            hb::ClusterLaunch L;
            L.grid = dim3(cs); L.block = dim3(32); L.smem = kSmem; L.cluster = dim3(cs, 1, 1); L.explicit_cluster = true;
            K6Fn fn = k6fn(m, gt, poll1);
            hb::ClusterLaunch::Prepared prep;
            L.prepare(fn, prep);
            const std::string label = fmt("k6/%s/gt%d/poll1_%d", kM6Name[m], gt, poll1);
            for (int l = 0; l < 2 + rec; ++l) {
                auto [s, sm] = rlaunch(x, label, cs, 1, d_place.d, [&] {
                    CUDA_CHECK(cudaLaunchKernelEx(&prep.cfg, fn, trials, (uint64_t)0, d_place.d, d_rt.d, d_g.d, d_info.d)); });
                const auto info = d_info.get();
                for (int b = 0; b < cs; ++b)
                    if (info[4 * b + 1] || info[4 * b + 2]) hb::die(label + fmt(": rank %d timeouts %u mismatches %u", b, info[4 * b + 1], info[4 * b + 2]));
                if (l < 2) continue;
                const auto rt = d_rt.get();
                const auto g = d_g.get();
                const std::string pre = x.run.prefix(x.li, s);
                int pair = 0;
                for (int a = 0; a < cs; ++a)
                    for (int b = a + 1; b < cs; ++b, ++pair)
                        for (uint32_t i = 0; i < trials; ++i) {
                            auto G = [&](int side, int k) { return g[((size_t)(pair * 2 + side) * 2 + k) * trials + i]; };
                            std::string ow, owb, r;
                            if (gt) { ow = std::to_string((int32_t)(G(1, 0) - G(0, 0))); owb = std::to_string((int32_t)(G(0, 1) - G(1, 1))); }
                            else r = std::to_string(rt[(size_t)pair * trials + i]);
                            x.vis->add(pre + fmt(",%s,cluster,%s%s,%u,%u,%u,%s,%s,%s", kM6Name[m], gt ? "gt" : "nogt", poll1 ? "_poll1" : "", sm[a], sm[b], i,
                                                 ow.c_str(), owb.c_str(),
                                                 r.c_str()));
                        }
                x.vis->flush();
                ++x.li;
            }
            x.vis->sync();
            std::printf("%s done\n", label.c_str());
        }
    x.run.lock_clocks_note("k6", "after");
}

// ---------------------------------------------------------------- K7 host
using K7Fn = void (*)(uint64_t, uint32_t*, K4Rec*);
static K7Fn k7fn(int p) {
    switch (p) {
        case W_U8: return k7_width<W_U8>;
        case W_U16: return k7_width<W_U16>;
        case W_U32: return k7_width<W_U32>;
        case W_U64: return k7_width<W_U64>;
        case W_V2: return k7_width<W_V2>;
        case W_V4: return k7_width<W_V4>;
        case A_B32: return k7_width<A_B32>;
        case A_B64: return k7_width<A_B64>;
        case A_V2: return k7_width<A_V2>;
        default: return k7_width<A_V4>;
    }
}
static void run_k7(Ctx& x) {
    const int rec = x.quick ? 2 : 10;
    DevBuf<uint32_t> d_place(8);
    DevBuf<K4Rec> d_rec(2);
    if (!x.width)
        x.width = std::make_unique<hb::CsvWriter>(hb::Run::results("12__dsmem_atomics__width.csv"),
            hb::with_mandatory({"primitive", "width_bytes", "bytes", "cycles", "bpc", "side", "smid_src", "smid_dst", "check_ok"}));
    x.run.lock_clocks_note("k7", "before");
    for (int pw = 0; pw <= A_V4; ++pw) {
        hb::ClusterLaunch L;
        L.grid = dim3(2); L.block = dim3(32); L.smem = kSmem; L.cluster = dim3(2, 1, 1); L.explicit_cluster = true;
        K7Fn fn = k7fn(pw);
        hb::ClusterLaunch::Prepared prep;
        L.prepare(fn, prep);
        const std::string label = fmt("k7/%s", kP7Name[pw]);
        for (int l = 0; l < 2 + rec; ++l) {
            auto [s, sm] = rlaunch(x, label, 2, 1, d_place.d, [&] { CUDA_CHECK(cudaLaunchKernelEx(&prep.cfg, fn, (uint64_t)0, d_place.d, d_rec.d)); });
            if (l < 2) continue;
            const auto r = d_rec.get();
            const std::string pre = x.run.prefix(x.li, s);
            const uint64_t cyc = r[0].c1 - r[0].c0;
            x.width->add(pre + fmt(",%s,%u,%u,%llu,%.4f,issuer,%u,%u,%d", kP7Name[pw], p7_bytes(pw), kK7Bytes, (unsigned long long)cyc, double(kK7Bytes) / cyc,
                                   sm[0], sm[1], r[1].err ? 0 : 1));
            if (pw >= A_B32)
                x.width->add(pre + fmt(",%s,%u,%u,%llu,%.4f,dest,%u,%u,%d", kP7Name[pw], p7_bytes(pw), kK7Bytes, (unsigned long long)r[1].comp_c,
                                       double(kK7Bytes) / r[1].comp_c, sm[0], sm[1], r[1].err ? 0 : 1));
            x.width->flush();
            ++x.li;
        }
        x.width->sync();
        std::printf("%s done\n", label.c_str());
    }
    x.run.lock_clocks_note("k7", "after");
}

// ---------------------------------------------------------------- ncu1 (one profiled K4/K5 configuration, kernel replay)
static void run_ncu1(Ctx& x, int idx) {
    struct NC { int kernel, op, pat, n; };
    const std::vector<NC> v = {{4, O_RED, P_DISTINCT, 1}, {4, O_ATOM, P_DISTINCT, 1}, {4, O_REDASYNC, P_DISTINCT, 1}, {4, O_STASYNC, P_DISTINCT, 1},
                               {4, O_STASYNC4, P_DISTINCT, 1}, {4, O_RED, P_ONE, 1}, {4, O_ATOM, P_ONE, 1}, {4, O_REDASYNC, P_ONE, 1},
                               {4, O_RED, P_PER32, 1}, {5, O_RED, P_BANK, 15}, {5, O_ATOM, P_BANK, 15}, {5, O_RED, P_ONE, 15}};
    if (idx < 0) { for (size_t i = 0; i < v.size(); ++i) std::printf("%zu k%d %s %s n%d\n", i, v[i].kernel, kOp4Name[v[i].op], kPat4Name[v[i].pat], v[i].n); return; }
    const NC c = v.at(idx);
    hb::CsvWriter man("bin/ncu_manifest.csv",
                      hb::with_mandatory({"cfg", "kernel", "op", "pattern", "n_sources", "ops", "role"}));
    const auto s = x.run.dev().sample();
    if (c.kernel == 4) {
        hb::ClusterLaunch L = cl_launch(2, 1, 1024);
        DevBuf<uint32_t> d_place(8);
        DevBuf<K4Rec> d_rec(2);
        K4P p{};
        p.pattern = (uint32_t)c.pat; p.steps = 1024; p.ring = 0; p.zero = 0; p.place = d_place.d; p.rec = d_rec.d;
        for (uint32_t mask : {0u, 1u}) {  // all-exit-like baseline (no issuer), then the profiled launch
            p.issuers_mask = mask;
            CUDA_CHECK(L.launch(k4fn(c.op), p));
            CUDA_CHECK(cudaDeviceSynchronize());
            man.add(x.run.prefix(x.li++, s) + fmt(",%d,k4,%s,%s,1,%llu,%s", idx, kOp4Name[c.op], kPat4Name[c.pat], mask ? 1024ull * 4096 : 0ull,
                                                  mask ? "profiled" : "baseline"));
        }
    } else {
        hb::ClusterLaunch L = cl_launch(16, 1, 1024);
        DevBuf<uint32_t> d_place(64);
        DevBuf<K4Rec> d_rec(16);
        DevBuf<uint64_t> d_local(2);
        K5P p{};
        p.pattern = (uint32_t)c.pat; p.ops_per_thread = c.pat == P_ONE ? 64 : 1024; p.zero = 0; p.place = d_place.d; p.rec = d_rec.d; p.local = d_local.d;
        for (uint32_t n : {0u, (uint32_t)c.n}) {
            p.n_src = n;
            CUDA_CHECK(L.launch(k5fn(c.op), p));
            CUDA_CHECK(cudaDeviceSynchronize());
            man.add(x.run.prefix(x.li++, s) + fmt(",%d,k5,%s,%s,%u,%llu,%s", idx, kOp4Name[c.op], kPat4Name[c.pat], n,
                                                  (unsigned long long)n * 1024 * p.ops_per_thread, n ? "profiled" : "baseline"));
        }
    }
    man.sync();
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s {k4|k4ring|k5|k6|k7|ncu1|clockcheck} [--quick] [--cfg N]\n", argv[0]); return 1; }
    const std::string mode = argv[1];
    hb::Run run(EXP, argc, argv);
    Ctx x(run, hb::has_flag(argc, argv, "--quick"));
    x.P = load_parts(x.sm, false);
    if (mode == "clockcheck") run.lock_clocks_note(hb::arg_str(argc, argv, "--label", "check"), "check");
    else if (mode == "k4") run_k4(x, false);
    else if (mode == "k4ring") run_k4(x, true);
    else if (mode == "k5") run_k5(x);
    else if (mode == "k6") run_k6(x, 0);
    else if (mode == "k6poll1") run_k6(x, 1);
    else if (mode == "k7") run_k7(x);
    else if (mode == "ncu1") run_ncu1(x, (int)hb::arg_int(argc, argv, "--cfg", -1));
    else hb::die("unknown mode " + mode);
    std::printf("discarded launches: %d\n", x.disc.n);
    run.complete();
    return 0;
}
