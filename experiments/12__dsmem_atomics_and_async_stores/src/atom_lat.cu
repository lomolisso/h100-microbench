/*
 * atom_lat.cu — experiment 12, K0, K1 and the global part of K6: the L2 partition of every SM on this card, the home
 * partition of every 4 KiB page of the global buffers, the dependent latency of atomics on every path (local smem, own
 * smem through the DSMEM window, a peer CTA's smem, L2 near / far, HBM, host memory), and the round trip of a hand-off
 * through global flags signalled by atomics.
 *
 * Usage: bin/atom_lat {k0a|k0b|calib|smem|dsmem|pairs|l2|l2fine|hbm|sys|flags|ncu_l2|clockcheck} [--quick] [--label L].
 *   One invocation = one run_id (results/12__dsmem_atomics_and_async_stores__runs.csv). Every launch is followed by an
 *   NVML sample; a launch with a throttle reason, an SM clock below the lock, or a placement other than intended is
 *   discarded (results/12__atomics__discards.csv) and repeated.
 *
 * Ops (atomics.cuh): add.u32, add.u64, inc.u32, min.s32, max.u32, and.b32, exch.b32, cas.b32, cas.b64, add.f32, add.f64,
 *   add.noftz.f16x2, add.noftz.bf16x2; global targets also add.v2/v4.f32 and add.noftz.v2/v4.f16x2. Semantics: the default
 *   (relaxed, .gpu scope) for every op; .relaxed / .acq_rel (atom) or .release (red) × .cta/.cluster/.gpu/.sys for
 *   add.u32 atom, add.u32 red+ld and cas.b32.
 * Kinds (one dependent hop; the value it returns gives the next element):
 *   atom    old = atom.op [elem], identity operand (common12.hpp: memory unchanged; exch writes the previous element, cas
 *           compares with 0 and never matches = "cas_fail"; variant plus1: add.u32 with operand 1)
 *   red+ld  red.op [elem], identity; then a plain load of the same word returns the next index (red has no return value:
 *           this is the time until the issuing thread can read its own reduction)
 *   ld      plain load of the chain (ld.shared::cta / ld.shared::cluster / ld.global.cg): same-binary reference
 *   st+ld   (global) st.global of word 1 of the element's line, then ld.global.cg of word 0: same-binary reference
 *   same    one fixed word, the address depends on the previous result (+ (old & zero)): cas.b32/b64 succeeding
 *           (compare = old, new = old + 16), inc.u32, add.u32
 * Timing (K1, one thread = lane 0 of a one-warp CTA): 64 warm hops (smem/DSMEM only), t0 = read_clock64_after(index),
 *   h1 = 4096 hops, t1, h2 = 16384 hops, t2 (#pragma unroll 1 around 16 inlined hops); latency = (c(h2) − c(h1)) /
 *   (h2 − h1) in the analysis (the fixed timer/loop cost cancels). Global targets: one count of hops per launch (pass
 *   protocol below), h ∈ {4096, 16384} in separate launches. Per-access ("fine") timing: a = %clock64; hop;
 *   b = read_clock64_after(index); b − a buffered in smem (batches of 49 152), copied out between batches (outside any
 *   timed hop). The host checks the final index against the chain position it predicts.
 *
 * k0a — per-SM partition. Every line of the first 8 MiB of a 512 MiB buffer is labelled with SMs A = 0 and B = 2
 *   (label_buffer, common12.hpp; partition 0 := the partition of SM 0). Then for every SM s (two rounds, different
 *   lines): L2 flush → 64 lines of home 0 written by SM 0 and 64 lines of home 1 by SM 2 (each line lives only at its
 *   home) → SM s reads all 128 once (timed, dependent). partition(s) = the home whose lines it reads faster.
 *   → results/12__atomics__sm_partition.csv: smid, gpc, partition, med_home0_cycles, med_home1_cycles, near_cycles,
 *     far_cycles, n_lines (timer pair included).
 * k0b — the labels of this invocation's 512 MiB buffer (every binary run labels its own buffer: homes depend on the
 *   physical pages): every 4 KiB page from its first line, every line of the first 16 MiB.
 *   → results/12__atomics__homes_pages.csv (page_offset, cycles_b, cycles_a, home), 12__atomics__homes_lines.csv
 *     (line_offset, cycles_b, cycles_a, home); every mode → one row in 12__atomics__homes_runs.csv.
 * calib — per-access timer cost: lane 0 chases ld.shared::cta (4096 hops), each hop timed alone; median − 23 cycles
 *   (the dependent ld.shared latency from 01/04) is the cost the fine samples carry.
 * smem (T1, T2) — kernel k1_smem<OP, SEM, KIND, SP>, 32 threads, 200 KB dynamic smem (1 CTA per SM). Every CTA builds
 *   the chain in its own smem: 128 KiB region, elements of op_bytes (4 or 8 B), next = one random cycle (Sattolo,
 *   04's pchase_chain.hpp, kChainSeed) over 32 768 / 16 384 elements; then lane 0 chases it. T1: plain launch of 132
 *   CTAs, .shared::cta addresses (ATOMS / LDS). T2: 66 clusters of 2 (one full wave), the chase goes through
 *   .shared::cluster addresses of the CTA's own rank (mapa; ATOM.E / LD.E through the DSMEM window). All 132 SMs chase
 *   at the same time (no shared resource: each SM uses only its own smem). 2 warm-up + 10 recorded launches.
 * dsmem (T3, lone cluster of CS ∈ {2, 4, 16}; hardware places it on SMs 124/125, in GPC 0 from SM 0, in GPC 1) —
 *   k1_smem in mode 1: rank 0 chases the chain of every other rank in turn (mapa(region, dst)); full op list.
 * pairs (T3 per SM pair) — mode 2: full waves of CS 16 / 8 / 4 / 2 (max(20, max active clusters) per launch); for
 *   src = 0 … CS−1, separated by cluster.sync(), rank src chases every other rank (h1 = 4096 hops only: 64 + 4096 hops,
 *   fixed cost ≤ 0.05 cycles/hop); add.u32 atom, cas.b32 (cas_fail), add.u32 red+ld, ld. 2 warm-up + 5 recorded launches.
 * l2 (T4, T5) — k1_glob<OP, SEM, KIND, 0>: one pinned warp (claim_target_sm) on the first SM of every GPC. Elements are
 *   128-B lines of the first 16 MiB of the buffer: "near" = the lines homed in the issuer's partition, "far" = the lines
 *   homed in the other partition (K0b page labels; 8 MiB each, minus the last line of every 2 MiB page, which is kept
 *   for the TLB warm-up). Chain = one random cycle over these lines. Protocol per launch: CleanL2 (flush + clean read)
 *   → the chain is written (whole lines, every word = the encoding) by one pinned warp on an SM of the lines' home
 *   partition (another GPC than the issuer's for near lines): the lines now live only at their home → pass 1 (h hops,
 *   timed) → pass 2 (the same h hops again, no flush) → pass 2ld (ld.global.cg chain over the same lines: does the
 *   atomic leave a near copy, as loads and stores do?). exch: pass 2 walks the reversed chain back (and restores it);
 *   2ld follows the restored chain. Before t0 lane 0 loads the TLB lines of the region (not in the chain).
 *   h ∈ {4096, 16384}; 2 warm-up + 10 recorded launches per (SM, near/far, op, kind, sem). Semantics sweep only on SMs
 *   0 and 2 (one per partition).
 * l2fine — k1_glob<…, 1> per-access timing of the same protocol (4096 hops, passes 1/2/2ld), add.u32 atom and cas.b32
 *   (cas_fail), ld, every first SM of a GPC, near/far, 3 recorded launches.
 * hbm (T6) — per-access timing (k1_glob<…, 1>), 131 072 hops per launch over a random cycle of the lines of the second
 *   256 MiB of the buffer (TLB lines excluded): chain written by a full-grid kernel, then CleanL2, so every hop misses
 *   L2; each launch starts where the previous one ended (no line is visited twice in a run). Near/far by the line's page
 *   label. Every op × {atom, red+ld} + ld + st+ld, on the first SM of every GPC; 2 warm-up + 3 recorded launches.
 *   add.u32 atom → raw rows (fine.csv); every other configuration → exact (cycles, count) histograms per home.
 * sys (T7) — chain over 4096 lines of cudaHostAlloc'd mapped memory (written by the host), h ∈ {256, 1024}, SMs 0 and 2;
 *   add.u32 / cas.b32 atom and add.u32 red+ld at every semantics, ld; cudaDevAttrHostNativeAtomicSupported is printed
 *   and recorded in the variant column.
 * flags (K6, global) — k6_flags<SIG, GT>: two pinned one-warp blocks (claims for SMs A and B) meet through a global
 *   counter, then 1000 ping-pong trials: A: t0, signal(flag_B, e), wait until flag_A ≥ e, t1 = read_clock64_after;
 *   B: wait flag_B ≥ e, signal(flag_A, e). Signals: st.release.gpu (reference), red.release.gpu.add 1,
 *   atom.acq_rel.gpu.exch (returned value consumed before polling). Waits: ld.acquire.gpu. Flags in separate lines of
 *   the labelled buffer, homed near the consumer, near the producer, or (same-partition pairs) both near / both far.
 *   GT variant: %globaltimer stamps at send and observation on both sides (one-way = observer − sender, 32 ns ticks).
 *   Pairs: SMs (0, 2) different partitions; (0, first SM of another GPC of SM 0's partition) same partition.
 *   → results/12__dsmem_atomics__visibility.csv (shared with dsmem_async K6; scope = gpu): primitive, scope,
 *     smid_src, smid_dst, trial, one_way_ns, round_trip_cycles, + variant (flag homes), one_way_back_ns.
 *
 * → results/12__atomics__latency.csv (one row per measurement and launch): op, kind, variant, sem, scope, target
 *   (smem | smem_dsmem_if | dsmem | l2_near | l2_far | hbm | sysmem | calib), pass (1 | 2 | 2ld; 0 = not applicable), cs,
 *   cluster_id, src_rank, dst_rank, smid_src, smid_dst, gpc_src, part_src, home_partition, hops, total_cycles,
 *   cycles_per_op (= total / hops, raw), check_ok. Per-access modes (l2fine, hbm): total_cycles = Σ of the samples (each
 *   includes the timer); hbm writes one row per home partition (hops = samples of that home).
 * ncu_l2 — one k1_glob launch of 16 hops (baseline), then SM 0 × {near, far} × {add.u32 atom, add.u32 red+ld, cas.b32, ld}:
 *   one 4096-hop protocol each; every k1_glob launch is profiled (src/run.sh) and listed in bin/ncu_manifest.csv.
 * → results/12__atomics__fine.csv (per access): op, kind, target, smid, line_offset, home_partition, pass, hop_idx, cycles.
 * → results/12__atomics__fine_hist.csv: op, kind, target, smid, home_partition, pass, cycles, count.
 */
#include <cstring>
#include <memory>

#include "common12.hpp"

enum Kind { K_ATOM = 0, K_REDLD = 1, K_LD = 2, K_STLD = 3, K_SAME = 4 };
static const char* kKindName[] = {"atom", "red+ld", "ld", "st+ld", "same"};
static const char* sem_part(int s) { return s == 0 ? "default" : s <= 4 ? "relaxed" : "acq_rel"; }
static const char* scope_part(int s) { static const char* n[] = {"default", "cta", "cluster", "gpu", "sys"}; return n[s == 0 ? 0 : (s - 1) % 4 + 1]; }
static const char* sem_name(int sem, int kind) {
    if (sem >= 5 && kind == K_REDLD) return "release";  // red has no acquire side
    return sem_part(sem);
}

// ---------------------------------------------------------------- one dependent hop
// idx = the value read by the previous hop (common12.hpp chain encoding); address = base + idx·U, U = 4 (smem) / 8 (global)
template <int OP, int SP, int SEM, int KIND>
__device__ __forceinline__ uint32_t hop1(uint64_t base, uint32_t idx, uint32_t& prev, uint64_t opnd, uint64_t zero) {
    constexpr uint64_t U = SP == AS_GLOBAL ? 8 : 4;
    const uint64_t a = base + (uint64_t)idx * U;
    uint64_t w;
    if constexpr (KIND == K_ATOM) {
        uint64_t v = opnd;
        if constexpr (OP == AO_EXCH_B32) v = prev;  // the previous element's word: reverses the chain
        w = atomx<OP, SP, SEM>(a, v, 0);
    } else if constexpr (KIND == K_REDLD) {
        redx<OP, SP, SEM>(a, opnd);
        if constexpr (op_is64(OP)) w = ld64x<SP>(a); else w = ld32x<SP>(a);
    } else if constexpr (KIND == K_LD) {
        if constexpr (op_is64(OP)) w = ld64x<SP>(a); else w = ld32x<SP>(a);
    } else {
        static_assert(KIND == K_STLD, "hop1: kind");
        st32x<SP>(a + 4, idx);
        w = ld32x<SP>(a);
    }
    prev = idx;
    return dec12(OP, w) + (uint32_t)((w >> 32) & zero);
}
// one hop on a fixed word (kind "same"): returns the new "old" value; the address depends on the previous one
template <int OP, int SP, int SEM>
__device__ __forceinline__ uint64_t same1(uint64_t a, uint64_t old, uint64_t opnd, uint64_t zero) {
    const uint64_t aa = a + (old & zero);
    if constexpr (OP == AO_CAS_B32 || OP == AO_CAS_B64) {
        const uint64_t r = atomx<OP, SP, SEM>(aa, old + 16, old);  // succeeds: memory holds `old`
        return r + 16;                                           // the value now in memory
    } else {
        return atomx<OP, SP, SEM>(aa, opnd + (old & zero), 0);
    }
}

template <int OP, int SP, int SEM, int KIND>
__device__ __forceinline__ void run_hops(uint64_t base, uint32_t& idx, uint32_t& prev, uint64_t& old, uint32_t n, uint64_t opnd,
                                         uint64_t zero) {
#pragma unroll 1
    for (uint32_t k = 0; k < n; k += 16) {
#pragma unroll
        for (int u = 0; u < 16; ++u) {
            if constexpr (KIND == K_SAME) old = same1<OP, SP, SEM>(base, old, opnd, zero);
            else idx = hop1<OP, SP, SEM, KIND>(base, idx, prev, opnd, zero);
        }
    }
    if constexpr (KIND == K_SAME) idx = (uint32_t)old;
}

// ---------------------------------------------------------------- K1 smem / DSMEM kernel
struct K1Rec { uint64_t c1, c2; uint32_t final_idx, src, dst, cluster; };
struct K1S {
    const uint32_t* next;  // successor table (n elements)
    uint32_t n, stride, mode, h1, h2, warm;  // n elements of `stride` bytes
    uint64_t opnd, zero;
    uint32_t* place;
    K1Rec* rec;
};
constexpr uint32_t kRegionBytes = 128 * 1024;

template <int OP, int SEM, int KIND, int SP>
__global__ void __launch_bounds__(32) k1_smem(K1S a) {
    extern __shared__ __align__(16) unsigned char dyn[];
    record_place(a.place);
    const uint32_t region = smem_u32(dyn);
    constexpr uint32_t wb = op_is64(OP) ? 8 : 4;  // word bytes; every word of an element holds the encoding
    for (uint32_t q = threadIdx.x; q < a.n * (a.stride / wb); q += 32) {
        const uint32_t i = q / (a.stride / wb);
        const uint64_t w = enc12(OP, a.next[i] * (a.stride / 4));
        if constexpr (wb == 8) *reinterpret_cast<uint64_t*>(dyn + (size_t)q * 8) = w;
        else *reinterpret_cast<uint32_t*>(dyn + (size_t)q * 4) = (uint32_t)w;
    }
    __syncwarp();
    uint32_t rank = 0, cs = 1, cid = 0;
    if constexpr (SP == AS_CLUSTER) {
        const ClusterRegs c = read_cluster_regs();
        rank = c.ctarank; cs = c.nctarank; cid = c.clusterid_x;
        fence_acq_rel_cluster();
        cluster_sync();
    } else {
        __syncthreads();
    }
    auto chase = [&](uint32_t dst, K1Rec& r) {
        const uint64_t base = SP == AS_CLUSTER ? (uint64_t)mapa_u32(region, dst) : (uint64_t)region;
        uint32_t idx = dec12(OP, enc12(OP, 0)), prev = idx;  // element 0
        uint64_t old = enc12(OP, a.next[0] * (a.stride / 4));  // kind same: the word of element 0
        run_hops<OP, SP, SEM, KIND>(base, idx, prev, old, a.warm, a.opnd, a.zero);
        const uint64_t t0 = read_clock64_after(idx, a.zero);
        run_hops<OP, SP, SEM, KIND>(base, idx, prev, old, a.h1, a.opnd, a.zero);
        const uint64_t t1 = read_clock64_after(idx, a.zero);
        run_hops<OP, SP, SEM, KIND>(base, idx, prev, old, a.h2, a.opnd, a.zero);
        const uint64_t t2 = read_clock64_after(idx, a.zero);
        r.c1 = t1 - t0; r.c2 = t2 - t1; r.final_idx = idx; r.dst = dst; r.cluster = cid;
    };
    if (a.mode == 0) {  // every CTA chases its own region (T1 .shared::cta, T2 through the window)
        if (threadIdx.x == 0) { K1Rec r; r.src = rank; chase(rank, r); a.rec[blockIdx.x] = r; }
    } else if (a.mode == 1) {  // rank 0 → every peer in turn
        if (rank == 0 && threadIdx.x == 0)
            for (uint32_t d = 1; d < cs; ++d) { K1Rec r; r.src = 0; chase(d, r); a.rec[cid * (cs - 1) + d - 1] = r; }
    } else {  // every ordered pair, sources in turn
        for (uint32_t s = 0; s < cs; ++s) {
            if (rank == s && threadIdx.x == 0)
                for (uint32_t d = 0; d < cs; ++d) {
                    if (d == s) continue;
                    K1Rec r; r.src = s; chase(d, r);
                    a.rec[(cid * cs + s) * (cs - 1) + (d < s ? d : d - 1)] = r;
                }
            __syncwarp();
            if constexpr (SP == AS_CLUSTER) cluster_sync();
        }
    }
    if constexpr (SP == AS_CLUSTER) cluster_sync();
}

// ---------------------------------------------------------------- K1 global kernel (one pinned warp)
struct K1GRec { uint64_t cycles; uint32_t final_idx, last_idx, smid, n_batches; uint64_t sink; };
struct K1G {
    char* buf;
    uint32_t start, prev, hops;
    const uint64_t* tlb;
    uint32_t n_tlb;
    uint64_t opnd, zero, same_old;
    uint32_t target;
    uint32_t* claim;
    K1GRec* rec;
    uint32_t* samples;  // FINE: hops entries
};
constexpr uint32_t kFineBatch = 49152;  // samples per smem batch (192 KiB)

template <int OP, int SEM, int KIND, int FINE>
__global__ void __launch_bounds__(32) k1_glob(K1G a) {
    extern __shared__ __align__(16) unsigned char dyn[];
    if (!claim_target_sm(a.target, a.claim)) return;
    uint32_t* sbuf = reinterpret_cast<uint32_t*>(dyn);
    const uint64_t base = chain_base_adj(OP, (uint64_t)a.buf);
    uint32_t idx = a.start, prev = a.prev;  // values as read from the chain (tagged)
    uint64_t old = a.same_old, tl = 0;
    const uint64_t same_addr = base + (uint64_t)a.start * 8;
    if (threadIdx.x == 0)
        for (uint32_t i = 0; i < a.n_tlb; ++i) {  // TLB warm-up: one line per 2 MiB page, not in the chain
            uint32_t v;
            asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(a.tlb[i] + (tl & a.zero)) : "memory");
            tl += v;
        }
    uint64_t cycles = 0;
    uint32_t nb = 0;
    if constexpr (!FINE) {
        if (threadIdx.x == 0) {
            const uint64_t t0 = read_clock64_after(tl, a.zero);
            if constexpr (KIND == K_SAME) run_hops<OP, AS_GLOBAL, SEM, KIND>(same_addr, idx, prev, old, a.hops, a.opnd, a.zero);
            else run_hops<OP, AS_GLOBAL, SEM, KIND>(base, idx, prev, old, a.hops, a.opnd, a.zero);
            cycles = read_clock64_after(idx, a.zero) - t0;
        }
    } else {
        uint64_t dep = tl;
        for (uint32_t k0 = 0; k0 < a.hops; k0 += kFineBatch) {
            const uint32_t m = min(kFineBatch, a.hops - k0);
            if (threadIdx.x == 0) {
#pragma unroll 1
                for (uint32_t k = 0; k < m; ++k) {
                    const uint64_t t0 = read_clock64_after(dep, a.zero);
                    if constexpr (KIND == K_SAME) { old = same1<OP, AS_GLOBAL, SEM>(same_addr, old, a.opnd, a.zero); idx = (uint32_t)old; }
                    else idx = hop1<OP, AS_GLOBAL, SEM, KIND>(base, idx, prev, a.opnd, a.zero);
                    const uint64_t t1 = read_clock64_after(idx, a.zero);
                    sbuf[k] = (uint32_t)(t1 - t0);
                    dep = idx;
                }
            }
            __syncwarp();
            for (uint32_t k = threadIdx.x; k < m; k += 32) a.samples[k0 + k] = sbuf[k];
            __syncwarp();
            ++nb;
        }
    }
    if (threadIdx.x == 0) *a.rec = {cycles, idx, prev, read_smid(), nb, tl & a.zero};
}

// Chain writer: for i < n, line list[i] gets the encoding of line list[next[i]] in every word (whole-line write).
// Pinned one-warp version (home touch) and grid-wide version (T6 chain before CleanL2).
template <bool PINNED>
__global__ void __launch_bounds__(PINNED ? 32 : 1024) k_write_chain(char* buf, const uint32_t* list, const uint32_t* next, uint32_t n, int op,
                                                                  uint32_t target, uint32_t* claim) {
    if constexpr (PINNED) { if (!claim_target_sm(target, claim)) return; }
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t w0 = PINNED ? 0 : (blockIdx.x * blockDim.x + threadIdx.x) / 32, nw = PINNED ? 1 : gridDim.x * blockDim.x / 32;
    for (uint32_t i = w0; i < n; i += nw) {
        const uint64_t e = enc12(op, list[next[i]] * (kLine / 8));
        const uint64_t line = (uint64_t)buf + (uint64_t)list[i] * kLine;
        if (op_is64(op)) { if (lane < 16) *reinterpret_cast<volatile uint64_t*>(line + lane * 8) = e; }
        else *reinterpret_cast<volatile uint32_t*>(line + lane * 4) = (uint32_t)e;
    }
}

// ---------------------------------------------------------------- calibration: per-access timer cost on ld.shared
__global__ void __launch_bounds__(32) k_calib(const uint32_t* next, uint32_t n, uint32_t hops, uint64_t zero, uint32_t* samples) {
    extern __shared__ __align__(16) unsigned char dyn[];
    uint32_t* w = reinterpret_cast<uint32_t*>(dyn);
    for (uint32_t i = threadIdx.x; i < n; i += 32) w[i] = next[i] * 4;
    __syncwarp();
    if (threadIdx.x) return;
    const uint32_t base = smem_u32(dyn);
    uint32_t v = 0;
    for (uint32_t k = 0; k < hops; ++k) {
        const uint64_t t0 = read_clock64_after(v, zero);
        asm volatile("ld.shared::cta.u32 %0, [%1];" : "=r"(v) : "r"(base + v) : "memory");
        samples[k] = (uint32_t)(read_clock64_after(v, zero) - t0);
    }
}

// ---------------------------------------------------------------- K6 global flags
enum Sig { SIG_ST_REL = 0, SIG_RED_REL = 1, SIG_ATOM_EXCH = 2 };
static const char* kSigName[] = {"st.release.gpu", "red.release.gpu.add", "atom.acq_rel.gpu.exch"};
struct K6G {
    uint32_t *flag_a, *flag_b;  // flag_a: polled by A, written by B; flag_b: polled by B
    uint32_t sm_a, sm_b, trials;
    uint32_t *claim, *ready, *err;
    uint64_t zero;
    uint32_t* rt;     // trials: A's round trip cycles
    uint64_t* gts;    // 4 × trials: A send, B seen, B send, A seen (%globaltimer, GT variant)
};
template <int SIG>
__device__ __forceinline__ uint32_t signal_g(uint32_t* f, uint32_t e) {
    if constexpr (SIG == SIG_ST_REL) { st_release_gpu_u32(f, e); return 0; }
    else if constexpr (SIG == SIG_RED_REL) { red_release_gpu_add_u32(f, 1); return 0; }
    else return atom_exch_acq_rel_gpu_u32(f, e);
}
__device__ __forceinline__ uint32_t wait_g(uint32_t* f, uint32_t e, uint32_t* err) {
    uint32_t v;
    uint64_t n = 0;
    while ((v = ld_acquire_gpu_u32(f)) < e)
        if (++n > (1ull << 26)) { atomicOr(err, 1u); return e; }
    return v;
}
template <int SIG, int GT>
__global__ void __launch_bounds__(32) k6_flags(K6G a) {
    __shared__ uint32_t s_rt[1024];
    __shared__ uint64_t s_g[4 * 1024];
    const uint32_t sm = read_smid();
    int role = -1;
    if (threadIdx.x == 0) {
        if (sm == a.sm_a && atomicCAS(&a.claim[0], 0u, blockIdx.x + 1) == 0u) role = 0;
        else if (sm == a.sm_b && atomicCAS(&a.claim[1], 0u, blockIdx.x + 1) == 0u) role = 1;
    }
    role = __shfl_sync(0xffffffffu, role, 0);
    if (role < 0 || threadIdx.x) return;
    atomicAdd(a.ready, 1u);
    for (uint64_t n = 0; atomicAdd(a.ready, 0u) < 2u;)
        if (++n > (1ull << 24)) { atomicOr(a.err, 2u); return; }
    uint64_t dep = 0;
    for (uint32_t i = 0; i < a.trials; ++i) {
        const uint32_t e = i + 1;
        if (role == 0) {
            const uint64_t t0 = read_clock64_after(dep, a.zero);
            if constexpr (GT) s_g[4 * i] = read_globaltimer();
            const uint32_t r = signal_g<SIG>(a.flag_b + (dep & a.zero), e);
            const uint32_t v = wait_g(a.flag_a + (r & a.zero), e, a.err);
            const uint64_t t1 = read_clock64_after(v, a.zero);
            if constexpr (GT) s_g[4 * i + 3] = read_globaltimer() + (v & a.zero);
            s_rt[i] = (uint32_t)(t1 - t0);
            dep = v;
        } else {
            const uint32_t v = wait_g(a.flag_b + (dep & a.zero), e, a.err);
            if constexpr (GT) { s_g[4 * i + 1] = read_globaltimer() + (v & a.zero); s_g[4 * i + 2] = read_globaltimer(); }
            const uint32_t r = signal_g<SIG>(a.flag_a + (v & a.zero), e);
            dep = v + (r & a.zero);
        }
    }
    for (uint32_t i = 0; i < a.trials; ++i) {
        if (role == 0) { a.rt[i] = s_rt[i]; if (GT) { a.gts[4 * i] = s_g[4 * i]; a.gts[4 * i + 3] = s_g[4 * i + 3]; } }
        else if (GT) { a.gts[4 * i + 1] = s_g[4 * i + 1]; a.gts[4 * i + 2] = s_g[4 * i + 2]; }
    }
}

// ---------------------------------------------------------------- configuration lists → kernels
// X(OP, SEM, KIND): smem/DSMEM list; the global list adds the vector ops and st+ld.
#define K1_COMMON(X)                                                                                                     \
    X(AO_ADD_U32, 0, K_ATOM) X(AO_ADD_U64, 0, K_ATOM) X(AO_INC_U32, 0, K_ATOM) X(AO_MIN_S32, 0, K_ATOM)                    \
    X(AO_MAX_U32, 0, K_ATOM) X(AO_AND_B32, 0, K_ATOM) X(AO_EXCH_B32, 0, K_ATOM) X(AO_CAS_B32, 0, K_ATOM)                   \
    X(AO_CAS_B64, 0, K_ATOM) X(AO_ADD_F32, 0, K_ATOM) X(AO_ADD_F64, 0, K_ATOM) X(AO_ADD_F16X2, 0, K_ATOM)                  \
    X(AO_ADD_BF16X2, 0, K_ATOM)                                                                                          \
    X(AO_ADD_U32, 0, K_REDLD) X(AO_ADD_U64, 0, K_REDLD) X(AO_INC_U32, 0, K_REDLD) X(AO_MIN_S32, 0, K_REDLD)                \
    X(AO_MAX_U32, 0, K_REDLD) X(AO_AND_B32, 0, K_REDLD) X(AO_ADD_F32, 0, K_REDLD) X(AO_ADD_F64, 0, K_REDLD)                \
    X(AO_ADD_F16X2, 0, K_REDLD) X(AO_ADD_BF16X2, 0, K_REDLD)                                                             \
    X(AO_ADD_U32, 0, K_LD) X(AO_ADD_F16X2, 0, K_LD) X(AO_CAS_B32, 0, K_SAME) X(AO_CAS_B64, 0, K_SAME) X(AO_INC_U32, 0, K_SAME) X(AO_ADD_U32, 0, K_SAME) \
    K1_SEMS(X, 1) K1_SEMS(X, 2) K1_SEMS(X, 3) K1_SEMS(X, 4) K1_SEMS(X, 5) K1_SEMS(X, 6) K1_SEMS(X, 7) K1_SEMS(X, 8)
#define K1_SEMS(X, S) X(AO_ADD_U32, S, K_ATOM) X(AO_ADD_U32, S, K_REDLD) X(AO_CAS_B32, S, K_ATOM)
#define K1_GLOBAL_EXTRA(X)                                                                                               \
    X(AO_ADD_V2F32, 0, K_ATOM) X(AO_ADD_V4F32, 0, K_ATOM) X(AO_ADD_V2F16X2, 0, K_ATOM) X(AO_ADD_V4F16X2, 0, K_ATOM)        \
    X(AO_ADD_V2F32, 0, K_REDLD) X(AO_ADD_V4F32, 0, K_REDLD) X(AO_ADD_V2F16X2, 0, K_REDLD) X(AO_ADD_V4F16X2, 0, K_REDLD)    \
    X(AO_ADD_U32, 0, K_STLD) X(AO_ADD_U64, 0, K_LD)

struct Cfg { int op, sem, kind; };
using K1SFn = void (*)(K1S);
using K1GFn = void (*)(K1G);
template <int SP>
static K1SFn pick_smem(const Cfg& c) {
#define X(O, S, K) if (c.op == O && c.sem == S && c.kind == K) return k1_smem<O, S, K, SP>;
    K1_COMMON(X)
#undef X
    hb::die("no smem kernel for this configuration");
}
template <int FINE>
static K1GFn pick_glob(const Cfg& c) {
#define X(O, S, K) if (c.op == O && c.sem == S && c.kind == K) return k1_glob<O, S, K, FINE>;
    K1_COMMON(X)
    K1_GLOBAL_EXTRA(X)
#undef X
    hb::die("no global kernel for this configuration");
}
static std::vector<Cfg> cfgs_common() {
    std::vector<Cfg> v;
#define X(O, S, K) v.push_back({O, S, K});
    K1_COMMON(X)
#undef X
    return v;
}
static std::vector<Cfg> cfgs_global() {
    std::vector<Cfg> v = cfgs_common();
#define X(O, S, K) v.push_back({O, S, K});
    K1_GLOBAL_EXTRA(X)
#undef X
    return v;
}

// ---------------------------------------------------------------- host context
struct Ctx {
    hb::Run& run;
    int sm = 0;
    bool quick = false;
    Parts P;
    hb::SmClaim claim;
    hb::L2Flusher flush;
    CleanL2 clean;
    Discards12 disc;
    char* gbuf = nullptr;
    uint64_t gbytes = 0;
    Homes H;
    long long li = 0;
    std::unique_ptr<hb::CsvWriter> lat, fine, hist;
    Ctx(hb::Run& r, bool q) : run(r), sm(r.dev().sm_count), quick(q), disc(r) {}
    hb::CsvWriter& latcsv() {
        if (!lat)
            lat = std::make_unique<hb::CsvWriter>(hb::Run::results("12__atomics__latency.csv"),
                hb::with_mandatory({"op", "kind", "variant", "sem", "scope", "target", "pass", "cs", "cluster_id", "src_rank", "dst_rank",
                                    "smid_src", "smid_dst", "gpc_src", "part_src", "home_partition", "hops", "total_cycles", "cycles_per_op",
                                    "check_ok"}));
        return *lat;
    }
    hb::CsvWriter& finecsv() {
        if (!fine)
            fine = std::make_unique<hb::CsvWriter>(hb::Run::results("12__atomics__fine.csv"),
                hb::with_mandatory({"op", "kind", "target", "smid", "line_offset", "home_partition", "pass", "hop_idx", "cycles"}));
        return *fine;
    }
    hb::CsvWriter& histcsv() {
        if (!hist)
            hist = std::make_unique<hb::CsvWriter>(hb::Run::results("12__atomics__fine_hist.csv"),
                hb::with_mandatory({"op", "kind", "target", "smid", "home_partition", "pass", "cycles", "count"}));
        return *hist;
    }
};

static const uint32_t kBufBytesMiB = 512, kL2RegionMiB = 16, kHbmFirstMiB = 256;

static void setup_buffer(Ctx& x, const char* name, uint32_t full_lines) {
    x.gbytes = (uint64_t)kBufBytesMiB * kMiB;
    x.gbuf = alloc_aligned(x.gbytes);
    CUDA_CHECK(cudaMemset(x.gbuf, 0, x.gbytes));
    x.run.lock_clocks_note("label", "before");
    x.H = label_buffer(x.run, x.claim, x.flush, x.P, x.gbuf, (uint32_t)(x.gbytes / kPage), full_lines);
    x.run.lock_clocks_note("label", "after");
    write_homes_summary(x.run, x.H, name, x.gbytes);
}

static std::string cfg_cols(const Cfg& c, const char* variant) {
    return fmt("%s,%s,%s,%s,%s", kOpName[c.op], kKindName[c.kind], variant, sem_name(c.sem, c.kind), scope_part(c.sem));
}
static const char* variant_of(const Cfg& c, bool plus1) {
    if (plus1) return "plus1";
    if (c.kind == K_SAME) return c.op == AO_CAS_B32 || c.op == AO_CAS_B64 ? "same_word_success" : "same_word";
    if (c.kind == K_ATOM && (c.op == AO_CAS_B32 || c.op == AO_CAS_B64)) return "cas_fail";
    if (c.kind == K_ATOM && c.op == AO_EXCH_B32) return "reverse_chain";
    return "identity";
}

// ---------------------------------------------------------------- k0a
static void run_k0a(Ctx& x) {
    x.P.part.assign(x.sm, -1);
    x.P.part[0] = 0; x.P.part[2] = 1;  // provisional: SM 0 defines partition 0; label_buffer dies on ambiguous labels below
    const uint32_t full = x.quick ? 16384 : 65536;
    setup_buffer(x, "k0a", full);
    // candidate lines: the first line of every labelled 4 KiB page of the fully labelled prefix, by home
    std::vector<uint32_t> l0, l1;
    for (uint32_t i = 0; i < full; i += kLinesPerPage) {
        if (x.H.line[i] == 0) l0.push_back(i);
        if (x.H.line[i] == 1) l1.push_back(i);
    }
    constexpr uint32_t kN = 64;
    const int rounds = 2;
    if (l0.size() < rounds * kN || l1.size() < rounds * kN) hb::die("k0a: too few labelled lines (SMs 0 and 2 in the same partition?)");
    uint32_t* d_list = nullptr;
    uint16_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_list, 2 * kN * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_out, 2 * kN * sizeof(uint16_t)));
    hb::CsvWriter csv(hb::Run::results("12__atomics__sm_partition.csv"),
                      hb::with_mandatory({"smid", "gpc", "partition", "med_home0_cycles", "med_home1_cycles", "near_cycles", "far_cycles", "n_lines"}));
    x.run.lock_clocks_note("k0a", "before");
    const auto s0 = x.run.dev().sample();
    for (int s = 0; s < x.sm; ++s) {
        std::vector<uint16_t> c0, c1;
        for (int r = 0; r < rounds; ++r) {
            // round r: lines r·kN … (r+1)·kN − 1 of each home list; home h written by SM 0 (h = 0) or SM 2 (h = 1)
            std::vector<uint32_t> L(l0.begin() + r * kN, l0.begin() + (r + 1) * kN);
            L.insert(L.end(), l1.begin() + r * kN, l1.begin() + (r + 1) * kN);
            CUDA_CHECK(cudaMemcpy(d_list, L.data(), L.size() * 4, cudaMemcpyHostToDevice));
            x.flush();
            pinned(x.claim, x.sm, 0, "k0a touch", [&](int nb, uint32_t* cl) { touch_list12<<<nb, 32>>>(x.gbuf, d_list, kN, 0, cl); });
            pinned(x.claim, x.sm, 2, "k0a touch", [&](int nb, uint32_t* cl) { touch_list12<<<nb, 32>>>(x.gbuf, d_list + kN, kN, 2, cl); });
            pinned(x.claim, x.sm, (uint32_t)s, "k0a probe",
                   [&](int nb, uint32_t* cl) { probe_list12<<<nb, 32>>>(x.gbuf, d_list, 2 * kN, (uint32_t)s, cl, 0, d_out); });
            std::vector<uint16_t> v(2 * kN);
            CUDA_CHECK(cudaMemcpy(v.data(), d_out, v.size() * 2, cudaMemcpyDeviceToHost));
            c0.insert(c0.end(), v.begin(), v.begin() + kN);
            c1.insert(c1.end(), v.begin() + kN, v.end());
        }
        auto med = [](std::vector<uint16_t> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0.0 : (double)v[v.size() / 2]; };
        const double m0 = med(c0), m1 = med(c1);
        const int p = m0 < m1 ? 0 : 1;
        csv.add(x.run.prefix(-1, s0) + fmt(",%d,%d,%d,%.1f,%.1f,%.1f,%.1f,%zu", s, x.P.gpc[s], p, m0, m1, std::min(m0, m1), std::max(m0, m1), c0.size()));
        if (s % 16 == 15) csv.flush();
        std::printf("  SM %3d GPC %d: home0 %.0f home1 %.0f → partition %d\n", s, x.P.gpc[s], m0, m1, p);
    }
    csv.sync();
    x.run.lock_clocks_note("k0a", "after");
    CUDA_CHECK(cudaFree(d_list));
    CUDA_CHECK(cudaFree(d_out));
}

// ---------------------------------------------------------------- k0b
static void run_k0b(Ctx& x) {
    setup_buffer(x, "k0b", (uint32_t)(kL2RegionMiB * kMiB / kLine));
    const auto s = x.run.dev().sample();
    const std::string pre = x.run.prefix(-1, s);
    hb::CsvWriter pc(hb::Run::results("12__atomics__homes_pages.csv"), hb::with_mandatory({"page_offset", "cycles_b", "cycles_a", "home"}));
    for (size_t i = 0; i < x.H.page.size(); ++i) {
        pc.add(pre + fmt(",%llu,%u,%u,%d", (unsigned long long)i * kPage, x.H.pb[i], x.H.pa[i], x.H.page[i]));
        if (i % 65536 == 65535) pc.flush();
    }
    pc.sync();
    hb::CsvWriter lc(hb::Run::results("12__atomics__homes_lines.csv"), hb::with_mandatory({"line_offset", "cycles_b", "cycles_a", "home"}));
    for (size_t i = 0; i < x.H.line.size(); ++i) {
        lc.add(pre + fmt(",%llu,%u,%u,%d", (unsigned long long)i * kLine, x.H.lb[i], x.H.la[i], x.H.line[i]));
        if (i % 65536 == 65535) lc.flush();
    }
    lc.sync();
}

// ---------------------------------------------------------------- calib
static void run_calib(Ctx& x) {
    const uint32_t n = 4096, hops = 4096;
    const auto next = chain::sattolo_cycle(n);
    DevBuf<uint32_t> d_next(n), d_s(hops);
    CUDA_CHECK(cudaMemcpy(d_next.d, next.data(), n * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaFuncSetAttribute(k_calib, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kOneCtaPerSmSmem));
    for (int l = 0; l < 2 + (x.quick ? 3 : 10); ++l) {
        k_calib<<<1, 32, kOneCtaPerSmSmem>>>(d_next.d, n, hops, 0, d_s.d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto s = x.run.clock_note("calib", "launch", x.li);
        if (l < 2) continue;
        auto v = d_s.get();
        std::sort(v.begin(), v.end());
        x.latcsv().add(x.run.prefix(x.li++, s) + fmt(",ld.u32,ld_fine,median_per_access,default,default,calib,0,1,0,0,0,,,,,,%u,%u,%u,1",
                                                     hops, v[hops / 2], v[hops / 2]));
        x.latcsv().flush();
        std::printf("calib: per-access ld.shared median %u cycles (p5 %u, p95 %u)\n", v[hops / 2], v[hops / 20], v[hops * 19 / 20]);
    }
    x.latcsv().sync();
}

// ---------------------------------------------------------------- smem / dsmem / pairs
struct SmemRun { int cs; int ncl; int mode; bool cluster; };
// One random cycle per element count (elements of 4, 8 or 64 B in the 128 KiB region), its visiting order, device copy.
struct SChain {
    std::vector<uint32_t> next, order;
    DevBuf<uint32_t> d;
    explicit SChain(uint32_t n) : next(chain::sattolo_cycle(n)), order(chain::chase_order(next, n)), d(n) {
        CUDA_CHECK(cudaMemcpy(d.d, next.data(), n * 4, cudaMemcpyHostToDevice));
    }
};
static uint32_t smem_stride(const Cfg& c, bool plus1) {
    if (c.op == AO_INC_U32 || plus1) return 64;  // the +1 moves the next access one word further inside the element
    return op_is64(c.op) ? 8 : 4;
}

static void run_smem_cfg(Ctx& x, const Cfg& c, bool plus1, const SmemRun& R, const char* target, uint32_t h1, uint32_t h2, int launches,
                         std::map<uint32_t, std::unique_ptr<SChain>>& chains) {
    const uint32_t stride = smem_stride(c, plus1), n = kRegionBytes / stride;
    auto& ch = chains[n];
    if (!ch) ch = std::make_unique<SChain>(n);
    K1S a{};
    a.next = ch->d.d;
    a.n = n; a.stride = stride; a.mode = (uint32_t)R.mode; a.h1 = h1; a.h2 = h2; a.warm = 64;
    a.opnd = c.kind == K_LD ? 0 : identity12(c.op, plus1);
    a.zero = 0;
    const int nblk = R.cs * R.ncl;
    const int nrec = R.mode == 0 ? nblk : R.mode == 1 ? R.ncl * (R.cs - 1) : R.ncl * R.cs * (R.cs - 1);
    DevBuf<uint32_t> d_place(4 * (size_t)nblk);
    DevBuf<K1Rec> d_rec((size_t)nrec);
    a.place = d_place.d; a.rec = d_rec.d;
    hb::ClusterLaunch L;
    L.grid = dim3(nblk); L.block = dim3(32); L.smem = kOneCtaPerSmSmem;
    if (R.cluster) { L.cluster = dim3(R.cs, 1, 1); L.explicit_cluster = true; }
    K1SFn k = R.cluster ? pick_smem<AS_CLUSTER>(c) : pick_smem<AS_CTA>(c);
    // expected element after 64 + h1 + h2 hops (exch walks a fresh chain: n ≥ the hop count); kind same: not checked
    const uint32_t expect = ch->order[(64 + h1 + h2) % n];
    const uint32_t tag = tag12(c.op), units = stride / 4;
    const std::string label = fmt("%s/%s/%s/%s/cs%d", target, kOpName[c.op], kKindName[c.kind], kSemName[c.sem], R.cs);
    for (int l = 0; l < 2 + launches; ++l) {
        for (int t = 0;; ++t) {
            if (t >= 20) hb::die(label + ": no clean launch");
            CUDA_CHECK(L.launch(k, a));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            const auto s = x.run.clock_note(label, "launch", x.li);
            const auto place = d_place.get();
            std::vector<uint32_t> smid(nblk);
            for (int b = 0; b < nblk; ++b) smid[b] = place[4 * b];
            const bool ok = R.cluster ? Placement(place, R.cs, R.ncl).ok() : hb::all_distinct(smid);
            if (!ok) { x.disc.add(label, "placement", x.li, s); continue; }
            if (s.bad()) { x.disc.add(label, "clock", x.li, s); continue; }
            if (l < 2) break;
            const auto rec = d_rec.get();
            const std::string pre = x.run.prefix(x.li, s);
            int bad = 0;
            for (int i = 0; i < nrec; ++i) {
                const K1Rec& r = rec[i];
                const int cl = R.mode == 0 ? i / R.cs : (int)r.cluster;
                const int src = R.mode == 0 ? i % R.cs : (int)r.src;
                const uint32_t ssrc = smid[cl * R.cs + src], sdst = smid[cl * R.cs + r.dst];
                const bool ok_idx = c.kind == K_SAME || (dec12(c.op, r.final_idx) - tag) / units == expect;
                bad += !ok_idx;
                for (int q = 0; q < 2; ++q) {
                    const uint32_t h = q ? h2 : h1;
                    if (!h) continue;
                    const uint64_t cyc = q ? r.c2 : r.c1;
                    x.latcsv().add(pre + "," + cfg_cols(c, variant_of(c, plus1)) +
                                   fmt(",%s,0,%d,%d,%d,%u,%u,%u,%d,%d,,%u,%llu,%.4f,%d", target, R.cluster ? R.cs : 1, cl, src, r.dst, ssrc, sdst,
                                       x.P.gpc[ssrc], x.P.part.empty() ? -1 : x.P.part[ssrc], h, (unsigned long long)cyc, (double)cyc / h, (int)ok_idx));
                }
            }
            if (bad) std::printf("  WARNING %s: %d chains ended at an unexpected element\n", label.c_str(), bad);
            x.latcsv().flush();
            ++x.li;
            break;
        }
    }
    x.latcsv().sync();
}

static void run_smem_family(Ctx& x, const char* mode) {
    const bool pairs = !std::strcmp(mode, "pairs");
    std::map<uint32_t, std::unique_ptr<SChain>> chains;
    const int launches = x.quick ? 2 : (pairs ? 5 : 10);
    auto cfgs = cfgs_common();
    if (x.quick) cfgs = {{AO_ADD_U32, 0, K_ATOM}, {AO_ADD_U32, 0, K_REDLD}, {AO_ADD_U32, 0, K_LD}, {AO_CAS_B32, 0, K_ATOM}, {AO_EXCH_B32, 0, K_ATOM},
                         {AO_ADD_F32, 0, K_ATOM}, {AO_ADD_U64, 0, K_ATOM}, {AO_INC_U32, 0, K_ATOM}, {AO_ADD_F16X2, 0, K_ATOM},
                         {AO_CAS_B32, 0, K_SAME}, {AO_ADD_U32, 7, K_ATOM}};
    x.run.lock_clocks_note(mode, "before");
    if (!std::strcmp(mode, "smem")) {
        for (const auto& c : cfgs) {
            run_smem_cfg(x, c, false, {1, x.sm, 0, false}, "smem", 4096, 16384, launches, chains);
            run_smem_cfg(x, c, false, {2, x.sm / 2, 0, true}, "smem_dsmem_if", 4096, 16384, launches, chains);
        }
        const Cfg p1{AO_ADD_U32, 0, K_ATOM};
        run_smem_cfg(x, p1, true, {1, x.sm, 0, false}, "smem", 4096, 16384, launches, chains);
        run_smem_cfg(x, p1, true, {2, x.sm / 2, 0, true}, "smem_dsmem_if", 4096, 16384, launches, chains);
    } else if (!std::strcmp(mode, "dsmem")) {
        for (int cs : {2, 4, 16})
            for (const auto& c : cfgs) {
                if (c.kind == K_SAME && cs != 2) continue;
                run_smem_cfg(x, c, false, {cs, 1, 1, true}, "dsmem", 4096, 16384, launches, chains);
            }
        for (int cs : {2, 16}) run_smem_cfg(x, {AO_ADD_U32, 0, K_ATOM}, true, {cs, 1, 1, true}, "dsmem", 4096, 16384, launches, chains);
    } else {  // pairs: full waves
        const std::vector<Cfg> pc = {{AO_ADD_U32, 0, K_ATOM}, {AO_CAS_B32, 0, K_ATOM}, {AO_ADD_U32, 0, K_REDLD}, {AO_ADD_U32, 0, K_LD}};
        for (int cs : {16, 8, 4, 2}) {
            hb::ClusterLaunch L;
            L.grid = dim3(cs); L.block = dim3(32); L.smem = kOneCtaPerSmSmem; L.cluster = dim3(cs, 1, 1); L.explicit_cluster = true;
            const int ncl = std::max(20, L.max_active_clusters(k1_smem<AO_ADD_U32, 0, K_ATOM, AS_CLUSTER>));
            for (const auto& c : pc) run_smem_cfg(x, c, false, {cs, ncl, 2, true}, "dsmem", 4096, 0, launches, chains);
        }
    }
    x.run.lock_clocks_note(mode, "after");
}

// ---------------------------------------------------------------- global targets
struct GChain {
    std::vector<uint32_t> list, next, order;  // lines (relative to gbuf), successor (index into list), visiting order (list indices)
    std::vector<uint64_t> tlb;                // TLB-warm addresses
    uint32_t *d_list = nullptr, *d_next = nullptr;
    uint64_t* d_tlb = nullptr;
    ~GChain() { cudaFree(d_list); cudaFree(d_next); cudaFree(d_tlb); }
    void upload() {
        CUDA_CHECK(cudaMalloc(&d_list, list.size() * 4));
        CUDA_CHECK(cudaMalloc(&d_next, next.size() * 4));
        CUDA_CHECK(cudaMalloc(&d_tlb, std::max<size_t>(1, tlb.size()) * 8));
        CUDA_CHECK(cudaMemcpy(d_list, list.data(), list.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_next, next.data(), next.size() * 4, cudaMemcpyHostToDevice));
        if (!tlb.empty()) CUDA_CHECK(cudaMemcpy(d_tlb, tlb.data(), tlb.size() * 8, cudaMemcpyHostToDevice));
    }
};
// Lines of pages [p0, p0 + np) homed in `home` (−1: any labelled home), without the last line of every 2 MiB page.
static void build_gchain(Ctx& x, GChain& g, uint32_t p0, uint32_t np, int home, uint64_t order_hops, uint64_t seed) {
    const uint32_t lines_2m = (uint32_t)(2 * kMiB / kLine);
    for (uint32_t pg = p0; pg < p0 + np; ++pg) {
        if (x.H.page[pg] < 0 || (home >= 0 && x.H.page[pg] != home)) continue;
        for (uint32_t i = 0; i < kLinesPerPage; ++i) {
            const uint32_t ln = pg * kLinesPerPage + i;
            if (ln % lines_2m != lines_2m - 1) g.list.push_back(ln);
        }
    }
    for (uint64_t b = (uint64_t)p0 * kPage; b < (uint64_t)(p0 + np) * kPage; b += 2 * kMiB) g.tlb.push_back((uint64_t)x.gbuf + b + 2 * kMiB - kLine);
    g.next = chain::sattolo_cycle((uint32_t)g.list.size(), seed);
    g.order = chain::chase_order(g.next, std::min<uint64_t>(order_hops, g.list.size()) + 1);
    g.upload();
}

struct GLaunch { uint64_t cycles; uint32_t final_idx, last_idx; std::vector<uint32_t> samples; };
// One pinned k1_glob launch; FINE samples copied back.
static GLaunch glob_launch(Ctx& x, const Cfg& c, bool fine, const GChain& g, uint32_t start, uint32_t prev, uint32_t hops, uint32_t sm,
                           uint64_t same_old, char* base, bool plus1 = false) {
    static DevBuf<K1GRec> d_rec(1);
    static uint32_t* d_samples = nullptr;
    static uint32_t cap = 0;
    if (fine && hops > cap) { cudaFree(d_samples); CUDA_CHECK(cudaMalloc(&d_samples, (size_t)hops * 4)); cap = hops; }
    K1G a{};
    a.buf = base; a.start = start; a.prev = prev; a.hops = hops; a.tlb = g.d_tlb; a.n_tlb = (uint32_t)g.tlb.size();
    a.opnd = c.kind == K_LD || c.kind == K_STLD ? 0 : identity12(c.op, plus1);
    a.zero = 0; a.same_old = same_old; a.target = sm; a.rec = d_rec.d; a.samples = d_samples;
    K1GFn k = fine ? pick_glob<1>(c) : pick_glob<0>(c);
    const size_t smem = fine ? kOneCtaPerSmSmem : 0;
    if (fine) CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    pinned(x.claim, x.sm, sm, "k1_glob", [&](int nb, uint32_t* cl) { a.claim = cl; k<<<nb, 32, smem>>>(a); });
    GLaunch r;
    const auto rec = d_rec.get();
    r.cycles = rec[0].cycles; r.final_idx = rec[0].final_idx; r.last_idx = rec[0].last_idx;
    if (fine) {  // total = Σ per-access samples (each includes the timer pair)
        r.samples.resize(hops);
        CUDA_CHECK(cudaMemcpy(r.samples.data(), d_samples, (size_t)hops * 4, cudaMemcpyDeviceToHost));
        r.cycles = 0;
        for (uint32_t v : r.samples) r.cycles += v;
    }
    return r;
}

// T4/T5: one protocol (CleanL2 → home write → pass 1 → pass 2 → 2ld) for SM `sm`, chain g homed in `home`.
// chain value (as read by a hop) of `line` for op's encoding, and back
static uint32_t xval(int op, uint32_t line) { return dec12(op, enc12(op, line * (kLine / 8))); }
static uint32_t line_of(int op, uint32_t v) { return (v - tag12(op)) / (kLine / 8); }

static void l2_protocol(Ctx& x, const Cfg& c, const GChain& g, int home, uint32_t sm, uint32_t toucher, uint32_t hops, bool fine, bool record,
                        const char* target) {
    const std::string label = fmt("%s/%s/%s/%s/sm%u/h%u", target, kOpName[c.op], kKindName[c.kind], kSemName[c.sem], sm, hops);
    for (int t = 0;; ++t) {
        if (t >= 20) hb::die(label + ": no clean launch");
        x.clean();
        pinned(x.claim, x.sm, toucher, "write_chain", [&](int nb, uint32_t* cl) {
            k_write_chain<true><<<nb, 32>>>(x.gbuf, g.d_list, g.d_next, (uint32_t)g.list.size(), c.op, toucher, cl); });
        const uint32_t start = xval(c.op, g.list[0]);
        const uint64_t same_old = enc12(c.op, g.list[g.next[0]] * (kLine / 8));  // kind same: the word of the first line
        const uint32_t prev0 = start;  // exch: the value pass 1 writes into the first element
        GLaunch p1 = glob_launch(x, c, fine, g, start, prev0, hops, sm, same_old, x.gbuf);
        // pass 2: the same hops again (exch: walk the reversed chain back from its last element, which restores it)
        const bool rev = c.kind == K_ATOM && c.op == AO_EXCH_B32;
        const uint64_t same2 = c.kind == K_SAME ? (uint64_t)p1.final_idx : same_old;
        GLaunch p2 = rev ? glob_launch(x, c, fine, g, p1.last_idx, p1.final_idx, hops, sm, same2, x.gbuf)
                         : glob_launch(x, c, fine, g, start, prev0, hops, sm, same2, x.gbuf);
        // pass 2ld: plain ld.global.cg chain over the same lines, with the op's word format
        const Cfg ldx{op_is16x2(c.op) ? AO_ADD_F16X2 : op_is64(c.op) ? AO_ADD_U64 : AO_ADD_U32, 0, K_LD};
        // (kind same: the fixed word was the chain's first element; its pointer is gone, so no 2ld pass)
        const bool has2ld = c.kind != K_SAME;
        GLaunch p3 = has2ld ? glob_launch(x, ldx, fine, g, xval(ldx.op, g.list[0]), 0, hops, sm, 0, x.gbuf) : GLaunch{};
        const auto s = x.run.clock_note(label, "launch", x.li);
        if (s.bad()) { x.disc.add(label, "clock", x.li, s); continue; }
        if (!record) return;
        const std::string pre = x.run.prefix(x.li, s);
        const uint32_t exp_end = g.list[g.order[hops % g.list.size()]];
        const GLaunch* P[3] = {&p1, &p2, &p3};
        static const char* pname[3] = {"1", "2", "2ld"};
        for (int q = 0; q < (has2ld ? 3 : 2); ++q) {
            const int opq = q == 2 ? ldx.op : c.op;
            bool ok = true;
            if (c.kind != K_SAME || q == 2) {
                if (q == 1 && rev) ok = line_of(opq, P[q]->final_idx) == g.list[0];
                else ok = line_of(opq, P[q]->final_idx) == exp_end;
            }
            const Cfg cq = q == 2 ? Cfg{c.op, 0, K_LD} : c;  // the 2ld row carries the op of the atomic pass before it
            x.latcsv().add(pre + "," + cfg_cols(cq, q == 2 ? "after_atomic" : variant_of(c, false)) +
                           fmt(",%s,%s,1,0,0,0,%u,,%d,%d,%d,%u,%llu,%.4f,%d", target, pname[q], sm, x.P.gpc[sm], x.P.part[sm], home, hops,
                               (unsigned long long)P[q]->cycles, (double)P[q]->cycles / hops, (int)ok));
            if (fine) {
                for (uint32_t k = 0; k < hops; ++k) {
                    const uint32_t li_ = rev && q == 1 ? g.order[(hops - 1 - k) % g.list.size()] : g.order[k % g.list.size()];
                    x.finecsv().add(pre + fmt(",%s,%s,%s,%u,%llu,%d,%s,%u,%u", kOpName[cq.op], kKindName[cq.kind], target, sm,
                                              (unsigned long long)g.list[li_] * kLine, home, pname[q], k, P[q]->samples[k]));
                }
                x.finecsv().flush();
            }
        }
        x.latcsv().flush();
        ++x.li;
        return;
    }
}

static void run_l2(Ctx& x, bool fine) {
    x.P = load_parts(x.sm);
    setup_buffer(x, fine ? "l2fine" : "l2", 0);
    const uint32_t np = (uint32_t)(kL2RegionMiB * kMiB / kPage);
    GChain g[2];
    for (int h = 0; h < 2; ++h) build_gchain(x, g[h], 0, np, h, 16384, chain::kChainSeed + h);
    std::printf("l2: chains of %zu / %zu lines (homes 0 / 1)\n", g[0].list.size(), g[1].list.size());
    std::vector<uint32_t> sms;
    for (int gp = 0; gp < 8; ++gp) sms.push_back((uint32_t)first_sm_of_gpc(x.P.gpc, gp));
    if (x.quick) sms = {0, 2};
    std::vector<Cfg> cfgs;
    if (fine) cfgs = {{AO_ADD_U32, 0, K_ATOM}, {AO_CAS_B32, 0, K_ATOM}, {AO_ADD_U32, 0, K_LD}};
    else cfgs = cfgs_global();
    if (x.quick && !fine) cfgs = {{AO_ADD_U32, 0, K_ATOM}, {AO_ADD_U32, 0, K_REDLD}, {AO_ADD_U32, 0, K_LD}, {AO_ADD_U32, 0, K_STLD},
                                  {AO_EXCH_B32, 0, K_ATOM}, {AO_CAS_B32, 0, K_SAME}, {AO_ADD_V4F32, 0, K_ATOM}, {AO_ADD_U32, 7, K_ATOM}};
    const int launches = x.quick ? 2 : fine ? 3 : 10;
    const std::vector<uint32_t> hs = fine ? std::vector<uint32_t>{4096} : std::vector<uint32_t>{4096, 16384};
    x.run.lock_clocks_note(fine ? "l2fine" : "l2", "before");
    for (uint32_t sm : sms)
        for (int near = 1; near >= 0; --near) {
            const int home = near ? x.P.part[sm] : 1 - x.P.part[sm];
            uint32_t toucher = (uint32_t)x.P.first_sm(home, x.P.gpc[sm]);
            const char* target = near ? "l2_near" : "l2_far";
            for (const auto& c : cfgs) {
                if (c.sem != 0 && sm != 0 && sm != 2) continue;  // semantics sweep on SMs 0 and 2 only
                if (c.kind == K_LD && c.op != AO_ADD_U32) continue;  // (u64 / f16x2 loads exist for the 2ld pass only)
                for (uint32_t h : hs)
                    for (int l = 0; l < 2 + launches; ++l) l2_protocol(x, c, g[home], home, sm, toucher, h, fine, l >= 2, target);
                x.latcsv().sync();
            }
        }
    x.run.lock_clocks_note(fine ? "l2fine" : "l2", "after");
}

// T6: per-access, lines of the second 256 MiB, full-grid chain write + CleanL2 per launch.
static void run_hbm(Ctx& x) {
    x.P = load_parts(x.sm);
    setup_buffer(x, "hbm", 0);
    const uint32_t p0 = (uint32_t)(kHbmFirstMiB * kMiB / kPage), np = (uint32_t)((kBufBytesMiB - kHbmFirstMiB) * kMiB / kPage);
    const uint32_t hops = x.quick ? 32768 : 131072;
    const int launches = x.quick ? 2 : 3;
    GChain g;
    build_gchain(x, g, p0, np, -1, (uint64_t)hops * (2 + launches) + 1, chain::kChainSeed + 7);
    std::vector<int8_t> home_of_line(x.gbytes / kLine, -1);
    for (uint32_t ln : g.list) home_of_line[ln] = x.H.page[ln / kLinesPerPage];
    std::vector<uint32_t> sms;
    for (int gp = 0; gp < 8; ++gp) sms.push_back((uint32_t)first_sm_of_gpc(x.P.gpc, gp));
    if (x.quick) sms = {0, 2};
    std::vector<Cfg> cfgs;
    for (const auto& c : cfgs_global()) if (c.sem == 0) cfgs.push_back(c);
    if (x.quick) cfgs = {{AO_ADD_U32, 0, K_ATOM}, {AO_ADD_U32, 0, K_LD}, {AO_CAS_B32, 0, K_ATOM}, {AO_EXCH_B32, 0, K_ATOM}};
    x.run.lock_clocks_note("hbm", "before");
    for (const auto& c : cfgs) {
        if (c.kind == K_SAME) continue;  // one fixed word cannot miss L2 on every hop
        if (c.kind == K_LD && c.op != AO_ADD_U32) continue;
        for (uint32_t sm : sms) {
            const std::string label = fmt("hbm/%s/%s/sm%u", kOpName[c.op], kKindName[c.kind], sm);
            k_write_chain<false><<<4 * x.sm, 1024>>>(x.gbuf, g.d_list, g.d_next, (uint32_t)g.list.size(), c.op, 0, nullptr);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            const bool raw = c.op == AO_ADD_U32 && c.kind == K_ATOM;
            for (int l = 0; l < 2 + launches; ++l) {
                const uint64_t k0 = (uint64_t)l * hops;  // each launch continues the cycle (fresh lines)
                const uint32_t start = xval(c.op, g.list[g.order[k0]]), prev = xval(c.op, g.list[g.order[k0 ? k0 - 1 : 0]]);
                for (int t = 0;; ++t) {
                    if (t >= 20) hb::die(label + ": no clean launch");
                    x.clean();
                    GLaunch r = glob_launch(x, c, true, g, start, prev, hops, sm, 0, x.gbuf);
                    const auto s = x.run.clock_note(label, "launch", x.li);
                    if (s.bad()) { x.disc.add(label, "clock", x.li, s); continue; }
                    if (l < 2) break;
                    const std::string pre = x.run.prefix(x.li, s);
                    const bool ok = line_of(c.op, r.final_idx) == g.list[g.order[k0 + hops]];
                    uint64_t sum[2] = {0, 0}, cnt[2] = {0, 0};
                    std::map<std::pair<int, uint32_t>, uint64_t> hist;
                    for (uint32_t k = 0; k < hops; ++k) {
                        const uint32_t ln = g.list[g.order[k0 + k]];
                        const int hm = home_of_line[ln];
                        if (raw) x.finecsv().add(pre + fmt(",%s,%s,hbm,%u,%llu,%d,1,%u,%u", kOpName[c.op], kKindName[c.kind], sm,
                                                           (unsigned long long)ln * kLine, hm, k, r.samples[k]));
                        else ++hist[{hm, r.samples[k]}];
                        if (hm >= 0) { sum[hm] += r.samples[k]; ++cnt[hm]; }
                        if (raw && k % 65536 == 65535) x.finecsv().flush();
                    }
                    if (raw) x.finecsv().flush();
                    for (const auto& kv : hist)
                        x.histcsv().add(pre + fmt(",%s,%s,hbm,%u,%d,1,%u,%llu", kOpName[c.op], kKindName[c.kind], sm, kv.first.first, kv.first.second,
                                                  (unsigned long long)kv.second));
                    x.histcsv().flush();
                    for (int hm = 0; hm < 2; ++hm)
                        x.latcsv().add(pre + "," + cfg_cols(c, variant_of(c, false)) +
                                       fmt(",hbm,1,1,0,0,0,%u,,%d,%d,%d,%llu,%llu,%.4f,%d", sm, x.P.gpc[sm], x.P.part[sm], hm,
                                           (unsigned long long)cnt[hm], (unsigned long long)sum[hm], cnt[hm] ? (double)sum[hm] / cnt[hm] : 0.0, (int)ok));
                    x.latcsv().flush();
                    if (!ok) std::printf("  WARNING %s: chain ended at an unexpected line\n", label.c_str());
                    ++x.li;
                    break;
                }
            }
        }
        x.latcsv().sync(); x.finecsv().sync(); x.histcsv().sync();
    }
    x.run.lock_clocks_note("hbm", "after");
}

// T7: host-pinned mapped memory.
static void run_sys(Ctx& x) {
    x.P = load_parts(x.sm);
    int native = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&native, cudaDevAttrHostNativeAtomicSupported, 0));
    std::printf("sys: cudaDevAttrHostNativeAtomicSupported = %d\n", native);
    const uint32_t n = 4096;
    char* h = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h, (size_t)n * kLine, cudaHostAllocMapped));
    char* d = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&d, h, 0));
    GChain g;
    for (uint32_t i = 0; i < n; ++i) g.list.push_back(i);
    g.next = chain::sattolo_cycle(n, chain::kChainSeed + 11);
    g.order = chain::chase_order(g.next, n + 1);
    g.upload();
    std::vector<Cfg> cfgs = {{AO_ADD_U32, 0, K_LD}, {AO_ADD_U64, 0, K_ATOM}, {AO_ADD_F32, 0, K_ATOM}, {AO_EXCH_B32, 0, K_ATOM}};
    for (int sem = 0; sem <= 8; ++sem) { cfgs.push_back({AO_ADD_U32, sem, K_ATOM}); cfgs.push_back({AO_ADD_U32, sem, K_REDLD}); cfgs.push_back({AO_CAS_B32, sem, K_ATOM}); }
    if (x.quick) cfgs = {{AO_ADD_U32, 0, K_LD}, {AO_ADD_U32, 0, K_ATOM}, {AO_ADD_U32, 4, K_ATOM}, {AO_ADD_U32, 8, K_ATOM}};
    const int launches = x.quick ? 2 : 10;
    const std::string variant = fmt("host_native_atomic_%d", native);
    x.run.lock_clocks_note("sys", "before");
    for (uint32_t sm : {0u, 2u})
        for (const auto& c : cfgs)
            for (uint32_t hops : {256u, 1024u})
                for (int l = 0; l < 2 + launches; ++l) {
                    for (uint32_t i = 0; i < n; ++i) {  // host writes the chain (every word of the line)
                        const uint64_t e = enc12(c.op, g.list[g.next[i]] * (kLine / 8));
                        for (uint32_t w = 0; w < kLine / 8; ++w)
                            reinterpret_cast<uint64_t*>(h + (size_t)i * kLine)[w] = op_is64(c.op) ? e : (e | e << 32);
                    }
                    const std::string label = fmt("sys/%s/%s/%s/sm%u/h%u", kOpName[c.op], kKindName[c.kind], kSemName[c.sem], sm, hops);
                    GLaunch r = glob_launch(x, c, false, g, xval(c.op, 0), xval(c.op, 0), hops, sm, enc12(c.op, g.next[0] * (kLine / 8)), d);
                    const auto s = x.run.clock_note(label, "launch", x.li);
                    if (s.bad()) { x.disc.add(label, "clock", x.li, s); --l; continue; }
                    if (l < 2) continue;
                    const bool ok = line_of(c.op, r.final_idx) == g.order[hops % n];
                    x.latcsv().add(x.run.prefix(x.li, s) + "," + cfg_cols(c, variant.c_str()) +
                                   fmt(",sysmem,1,1,0,0,0,%u,,%d,%d,,%u,%llu,%.4f,%d", sm, x.P.gpc[sm], x.P.part[sm], hops,
                                       (unsigned long long)r.cycles, (double)r.cycles / hops, (int)ok));
                    x.latcsv().flush();
                    ++x.li;
                }
    x.latcsv().sync();
    x.run.lock_clocks_note("sys", "after");
    CUDA_CHECK(cudaFreeHost(h));
}

// ---------------------------------------------------------------- ncu_l2: counters of single L2 protocols (run under ncu)
// One k1_glob launch of 16 hops first (baseline), then for SM 0 × {near, far} × {add.u32 atom, add.u32 red+ld, cas.b32, ld}
// one protocol of 4096 hops (pass 1, pass 2, 2ld). Every k1_glob launch gets a manifest row, in launch order.
static void run_ncu_l2(Ctx& x) {
    x.P = load_parts(x.sm);
    setup_buffer(x, "ncu_l2", 0);
    const uint32_t np = (uint32_t)(kL2RegionMiB * kMiB / kPage);
    GChain g[2];
    for (int h = 0; h < 2; ++h) build_gchain(x, g[h], 0, np, h, 16384, chain::kChainSeed + h);
    hb::CsvWriter man("bin/ncu_manifest.csv",
                      hb::with_mandatory({"order", "target", "op", "kind", "pass", "hops", "smid", "home_partition"}));
    const auto s = x.run.dev().sample();
    int order = 0;
    const uint32_t sm = 0;
    {
        const Cfg c{AO_ADD_U32, 0, K_LD};
        x.clean();
        pinned(x.claim, x.sm, sm, "write_chain", [&](int nb, uint32_t* cl) {
            k_write_chain<true><<<nb, 32>>>(x.gbuf, g[0].d_list, g[0].d_next, (uint32_t)g[0].list.size(), c.op, sm, cl); });
        glob_launch(x, c, false, g[0], xval(c.op, g[0].list[0]), 0, 16, sm, 0, x.gbuf);
        man.add(x.run.prefix(-1, s) + fmt(",%d,baseline,add.u32,ld,0,16,%u,%d", order++, sm, x.P.part[sm]));
    }
    for (int near = 1; near >= 0; --near) {
        const int home = near ? x.P.part[sm] : 1 - x.P.part[sm];
        const uint32_t toucher = (uint32_t)x.P.first_sm(home, x.P.gpc[sm]);
        for (const Cfg& c : {Cfg{AO_ADD_U32, 0, K_ATOM}, Cfg{AO_ADD_U32, 0, K_REDLD}, Cfg{AO_CAS_B32, 0, K_ATOM}, Cfg{AO_ADD_U32, 0, K_LD}}) {
            l2_protocol(x, c, g[home], home, sm, toucher, 4096, false, true, near ? "l2_near" : "l2_far");
            for (const char* pass : {"1", "2", "2ld"})
                man.add(x.run.prefix(-1, s) + fmt(",%d,%s,%s,%s,%s,4096,%u,%d", order++, near ? "l2_near" : "l2_far", kOpName[c.op],
                                                  !std::strcmp(pass, "2ld") ? "ld" : kKindName[c.kind], pass, sm, home));
        }
    }
    man.sync();
}

// ---------------------------------------------------------------- K6 global flags
template <int SIG, int GT>
static void flags_launch(Ctx& x, K6G a) {
    for (int t = 0; t < kMaxClaimRetries; ++t) {
        CUDA_CHECK(cudaMemset(a.claim, 0, 2 * 4));
        CUDA_CHECK(cudaMemset(a.ready, 0, 4));
        CUDA_CHECK(cudaMemset(a.flag_a, 0, 4));
        CUDA_CHECK(cudaMemset(a.flag_b, 0, 4));
        k6_flags<SIG, GT><<<4 * x.sm, 32>>>(a);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        uint32_t cl[2];
        CUDA_CHECK(cudaMemcpy(cl, a.claim, 8, cudaMemcpyDeviceToHost));
        if (cl[0] && cl[1]) return;
    }
    hb::die("flags: could not pin both SMs");
}
static void run_flags(Ctx& x) {
    x.P = load_parts(x.sm);
    setup_buffer(x, "flags", 0);
    const uint32_t trials = 1000;
    DevBuf<uint32_t> d_claim(2), d_ready(1), d_err(1), d_rt(trials);
    DevBuf<uint64_t> d_g(4 * trials);
    hb::CsvWriter vis(hb::Run::results("12__dsmem_atomics__visibility.csv"),
                      hb::with_mandatory({"primitive", "scope", "variant", "smid_src", "smid_dst", "trial", "one_way_ns", "one_way_back_ns",
                                          "round_trip_cycles"}));
    // a line homed in partition p (first line of a labelled page of the L2 region; pages taken in order, one per use)
    uint32_t next_page[2] = {0, 0};
    auto line_homed = [&](int p) {
        for (uint32_t& i = next_page[p]; i < x.H.page.size(); ++i)
            if (x.H.page[i] == p) return (uint32_t*)(x.gbuf + (uint64_t)(i++) * kPage);
        hb::die("no page homed in partition " + std::to_string(p));
    };
    const uint32_t A = 0, B_diff = 2, B_same = (uint32_t)x.P.first_sm(x.P.part[0], x.P.gpc[0]);
    struct Pair { uint32_t a, b; };
    const std::vector<Pair> pairs = {{A, B_diff}, {A, B_same}};
    const int launches = x.quick ? 2 : 10;
    x.run.lock_clocks_note("flags", "before");
    for (const auto& pr : pairs)
        for (int hv = 0; hv < 2; ++hv) {
            // different partitions: hv 0 = each flag homed near its consumer, 1 = near its producer;
            // same partition: hv 0 = both flags homed in the pair's partition, 1 = both in the other partition
            const int pa = x.P.part[pr.a], pb = x.P.part[pr.b];
            const int home_fa = hv == 0 ? pa : (pa != pb ? pb : 1 - pa), home_fb = hv == 0 ? pb : (pa != pb ? pa : 1 - pb);
            const char* variant = pa != pb ? (hv == 0 ? "home_near_consumer" : "home_near_producer") : (hv == 0 ? "home_near_both" : "home_far_both");
            K6G a{};
            a.flag_a = line_homed(home_fa); a.flag_b = line_homed(home_fb);
            a.sm_a = pr.a; a.sm_b = pr.b; a.trials = trials; a.claim = d_claim.d; a.ready = d_ready.d; a.err = d_err.d; a.zero = 0;
            a.rt = d_rt.d; a.gts = d_g.d;
            for (int sig = 0; sig < 3; ++sig)
                for (int gt = 0; gt < 2; ++gt) {
                    const std::string label = fmt("flags/%s/%s/%u-%u/gt%d", kSigName[sig], variant, pr.a, pr.b, gt);
                    for (int l = 0; l < 2 + launches; ++l) {
                        d_err.zero();
                        if (sig == 0) { if (gt) flags_launch<0, 1>(x, a); else flags_launch<0, 0>(x, a); }
                        else if (sig == 1) { if (gt) flags_launch<1, 1>(x, a); else flags_launch<1, 0>(x, a); }
                        else { if (gt) flags_launch<2, 1>(x, a); else flags_launch<2, 0>(x, a); }
                        const auto s = x.run.clock_note(label, "launch", x.li);
                        if (d_err.get()[0]) hb::die(label + ": timeout");
                        if (s.bad()) { x.disc.add(label, "clock", x.li, s); --l; continue; }
                        if (l < 2) continue;
                        const auto rt = d_rt.get();
                        const auto gg = d_g.get();
                        const std::string pre = x.run.prefix(x.li, s);
                        for (uint32_t i = 0; i < trials; ++i) {
                            std::string ow = "", owb = "";
                            if (gt) { ow = fmt("%lld", (long long)(gg[4 * i + 1] - gg[4 * i])); owb = fmt("%lld", (long long)(gg[4 * i + 3] - gg[4 * i + 2])); }
                            vis.add(pre + fmt(",%s,gpu,%s%s,%u,%u,%u,%s,%s,%s", kSigName[sig], variant, gt ? "_gt" : "", pr.a, pr.b, i, ow.c_str(), owb.c_str(),
                                              gt ? "" : std::to_string(rt[i]).c_str()));
                        }
                        vis.flush();
                        ++x.li;
                    }
                    vis.sync();
                }
        }
    x.run.lock_clocks_note("flags", "after");
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s {k0a|k0b|calib|smem|dsmem|pairs|l2|l2fine|hbm|sys|flags|clockcheck} [--quick]\n", argv[0]); return 1; }
    const std::string mode = argv[1];
    hb::Run run(EXP, argc, argv);
    Ctx x(run, hb::has_flag(argc, argv, "--quick"));
    if (mode == "clockcheck") run.lock_clocks_note(hb::arg_str(argc, argv, "--label", "check"), "check");
    else if (mode == "k0a") { x.P.gpc = load_gpc_map(x.sm); run_k0a(x); }
    else if (mode == "k0b") { x.P = load_parts(x.sm); run_k0b(x); }
    else if (mode == "calib") run_calib(x);
    else if (mode == "smem" || mode == "dsmem" || mode == "pairs") { x.P = load_parts(x.sm, false); run_smem_family(x, mode.c_str()); }
    else if (mode == "l2") run_l2(x, false);
    else if (mode == "l2fine") run_l2(x, true);
    else if (mode == "hbm") run_hbm(x);
    else if (mode == "sys") run_sys(x);
    else if (mode == "flags") run_flags(x);
    else if (mode == "ncu_l2") run_ncu_l2(x);
    else hb::die("unknown mode " + mode);
    if (x.lat) x.lat->sync();
    if (x.fine) x.fine->sync();
    if (x.hist) x.hist->sync();
    std::printf("discarded launches: %d\n", x.disc.n);
    run.complete();
    return 0;
}
