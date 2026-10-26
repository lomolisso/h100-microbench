/*
 * atom_tput.cu — experiment 12, K2 and K3: throughput of atomics (red without return, atom with return) per SM and
 * GPU-wide, on local smem, own smem through the DSMEM window, a peer CTA's smem, L2 lines homed near / far, and HBM;
 * without contention and with same-bank / same-word / hot-line contention; plain ld / st of the same addresses as
 * same-binary references.
 *
 * Usage: bin/atom_tput {k2|k2all|k2x|k3a|k3b|k3c|k3d|ncu1|clockcheck} [--quick] [--cfg N]. One invocation = one run_id.
 *
 * Kernel k_tput<OP, KIND, ILP, SP>(TP): every thread issues, per step, ILP independent operations (KIND red: red.op;
 *   atom: atom.op whose returned values are XORed into an accumulator by volatile PTX after the step's last issue, so
 *   consumption never stalls an issue; ld: ld.global.cg / ld.shared, consumed the same way; st: st.global / st.shared);
 *   OP ∈ {add.u32, add.f32, add.u64, cas.b32 (atom only; compare 0xFFFFFFFF never matches: memory unchanged),
 *   add.noftz.f16x2}; operand 1 (1.0 for f32; f16x2 halves 1.0), so that Σ memory − Σ before = issued ops for the
 *   integer adds (checked on the last recorded launch of every configuration: check_ok).
 *   Address unit u (op size) of a "virtual array" of W units:
 *     smem (SP = AS_CTA, AS_CLUSTER)  128 KiB region of the target CTA (own; own through mapa(own rank); peer = rank ^ 1
 *                                     of a cluster of 2): address = region + u·size;
 *     global (SP = AS_GLOBAL)         the pages of a list (4 KiB pages; mixed = every page of the footprint in order,
 *                                     local / remote = the pages homed in the block's own / the other L2 partition, from
 *                                     this invocation's K0b labels): address = buf + list[u / units_per_page]·4096 +
 *                                     (u mod units_per_page)·size; the list lookup is ld.global.nc, warp-uniform for every
 *                                     pattern (a warp instruction never leaves its page).
 *   Patterns (s = step, j < ILP, g = thread index in its group, G = threads of the group, w/NW = warp index/count of the
 *   group, b = block index in the group; u wraps modulo W, kept incrementally: no division in the loop):
 *     coal     u = (s·ILP + j)·G + g                         (a: 32 consecutive words per warp instruction)
 *     sector   u = ((s·ILP + j)·G + g)·32/size               (b: one 32-B sector per lane)
 *     line     u = ((s·ILP + j)·G + g)·128/size              (b: one 128-B line per lane)
 *     bank     u = 32·lane + 1024·(((s·ILP + j)·NWb + wb) mod 32)   (c, smem: 32 lanes in bank 0, distinct words)
 *     warp1    u = ((s·ILP + j)·NW + w)·128/size             (d: the 32 lanes of a warp on one word; one word per line)
 *     block1   u = b·128/size                                (e: every thread of a block on one word)
 *     hotN     u = ((w + s·ILP + j) mod N)·128/size + lane   (K3 c: a warp instruction covers one of N hot lines)
 *     hot1     u = 0                                         (K3 c: every thread of the GPU on one word)
 *   (lane & zero) is added to every u so that ptxas cannot prove a warp-uniform address (it would insert a REDUX warp
 *   reduction for some red ops).
 *   Groups and tickets (global): mixed — one group of every active block; local/remote — one group per L2 partition
 *   (Parts from K0a). Blocks on SMs outside reader_gpc_mask exit at once. Each block takes a ticket in its group; G is
 *   computed by the host from the placement (exactly 2 blocks per SM, or ≤ 1 per SM), checked after the launch.
 *   Timed region per block: bar.sync → thread 0 %clock64 + %globaltimer → steps → every thread fence.acq_rel.gpu (its
 *   red / st are performed) → bar.sync → thread 0 timers after the accumulator. GPU-wide: cudaEvent pair per launch.
 *
 * K2 (k2: one CTA; k2all: one CTA on every SM; 1024-thread CTAs, 2 warm-up + 10 recorded launches): threads ∈ {128, 256,
 *   512, 1024} × ILP ∈ {1, 2, 4, 8}, steps 4096 (T6: as many as fit one pass over 256 MiB), for targets
 *     smem (T1, plain launch, 200 KB dynamic smem), smem_dsmem_if (T2: lone cluster of 2, rank 0 → own smem through the
 *     window; k2all: every CTA), dsmem (T3: lone cluster of 2, rank 0 → rank 1; k2all: 66 clusters, every CTA → its
 *     peer), l2_near / l2_far (T4/T5: local / remote list, 8 MiB per partition, first 16 MiB of the buffer, L2-resident
 *     after the warm-ups), hbm (T6: mixed list over the second 256 MiB, CleanL2 before every launch);
 *   patterns coal, warp1, block1 everywhere, bank (smem targets), sector, line (global targets); full grid for add.u32
 *   red / atom; add.f32, add.u64, cas.b32, add.f16x2 and the ld / st references at 1024 threads, coal, every ILP.
 *   k2all: 1024 threads × ILP sweep, coal/warp1/block1(/bank), add.u32 red/atom, ld/st.
 *   k2x (kind red_rt): local smem, red.add.u32 whose operand is 1 + a run-time zero, coal/bank/warp1/block1, 1 CTA and 132
 *   CTAs, ILP sweep. Reason: ptxas compiles red.shared.add.u32 with the constant 1 into ATOMS.POPC.INC.32 (adds the number of
 *   lanes on the same word in one operation); a run-time operand keeps ATOMS.ADD, so the two kinds separate the effect.
 *   → results/12__atomics__throughput_sm.csv (one row per CTA and launch): op, kind, target, pattern, threads, ilp,
 *     placement (1sm | 132sm), block, smid, ops, cycles, ops_per_cycle, bpc, steps, footprint_bytes, part, check_ok.
 * K3 (2 × 132 CTAs × 1024 threads, ILP 4, 2 warm-up + 10 recorded launches; global lists of the 512 MiB buffer):
 *   k3a  footprint ∈ {1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 256} MiB × coal/sector × red/atom × add.u32/add.f32/add.u64 ×
 *        mixed/local/remote, + ld and st (same patterns and variants);
 *   k3b  remote, 16 MiB, coal, red/atom add.u32 + st + ld, CTAs of 1…4 GPCs of partition 0, then 1…4 of partition 1,
 *        then all GPCs (link: remote atomics vs the partition link, with remote stores as the reference);
 *   k3c  hot spots, mixed, red/atom × add.u32/add.f32: hot1 and hotN, N ∈ {1, 2, 4, …, 1024}; hot1 / hot lines N ∈ {1, 32}
 *        vs active SMs ∈ {1, 2, 4, 8, 16, 33, 66, 132} (one CTA of 1024 threads per SM, 120 KiB smem);
 *   k3d  each GPC alone, local, 16 MiB, coal, red/atom add.u32 + ld + st.
 *   → results/12__atomics__throughput_gpu.csv (one row per launch): op, kind, pattern, variant, footprint_bytes, n_hot_lines,
 *     reader_gpc_mask, blocks, threads, ilp, ops, ms_event, gops, tbps_payload, sm_mhz_blocks, discard, steps,
 *     active_sms, check_ok, span_ns, step.
 *   → results/12__atomics__throughput_gpu_blocks.csv (one row per CTA and launch): step, op, kind, pattern, variant,
 *     footprint_bytes, n_hot_lines, reader_gpc_mask, block, smid, gpc, partition, ticket, ops, cycles, ns, start_ns,
 *     ops_per_cycle, discard.
 * ncu1 — one configuration (--cfg N of the list printed by `ncu1 --list`), 2 warm-up launches + one launch to profile,
 *   preceded by an all-exit launch of the same grid (baseline); run under ncu by src/run.sh.
 */
#include <cstring>
#include <memory>
#include <type_traits>

#include "common12.hpp"

enum TKind { TK_RED = 0, TK_ATOM = 1, TK_LD = 2, TK_ST = 3, TK_REDRT = 4 };
static const char* kTKindName[] = {"red", "atom", "ld", "st", "red_rt"};
enum Pat { PA_COAL = 0, PA_SECTOR, PA_LINE, PA_BANK, PA_WARP1, PA_BLOCK1, PA_HOTN, PA_HOT1 };
static const char* kPatName[] = {"coal", "sector", "line", "bank", "warp1", "block1", "hotN", "hot1"};
enum Var { V_MIXED = 0, V_LOCAL = 1, V_REMOTE = 2 };
static const char* kVarName[] = {"mixed", "local", "remote"};
constexpr uint32_t kRegion = 128 * 1024;
constexpr size_t kForceOneSmem = 120 * 1024;

__constant__ uint8_t c_part12[256], c_gpc12[256];

struct TRec { uint64_t c0, c1, n0, n1, ops, sum; uint32_t smid, part, ticket, flags; };
struct TP {
    char* buf;
    const uint32_t* list[2];  // page lists (global); [0] used by mixed
    uint32_t W[2];            // units in list g (power of two not required)
    uint32_t G[2];            // threads of group g
    uint32_t variant, pattern, steps, nhot, gpc_mask, smem_mode;  // smem_mode: 0 own (cta), 1 own via window, 2 peer
    uint32_t active_rank0_only;
    uint32_t* tickets;
    TRec* rec;
    uint32_t* place;
    uint64_t zero;
};

template <int OP>
__device__ __forceinline__ uint64_t operand_one() {
    if constexpr (OP == AO_ADD_F32) return 0x3F800000u;
    else if constexpr (OP == AO_ADD_F16X2) return 0x3C003C00u;
    else return 1;
}
template <int OP, int KIND, int SP>
__device__ __forceinline__ uint64_t do_op(uint64_t a, uint64_t zero) {
    if constexpr (KIND == TK_RED) { redx<OP, SP, 0>(a, operand_one<OP>()); return 0; }
    else if constexpr (KIND == TK_REDRT) { redx<OP, SP, 0>(a, operand_one<OP>() + zero); return 0; }  // operand not a constant
    else if constexpr (KIND == TK_ATOM) {
        if constexpr (OP == AO_CAS_B32) return atomx<OP, SP, 0>(a, 1, 0xFFFFFFFFu);
        else return atomx<OP, SP, 0>(a, operand_one<OP>(), 0);
    } else if constexpr (KIND == TK_LD) {
        if constexpr (op_is64(OP)) return ld64x<SP>(a); else return ld32x<SP>(a);
    } else {
        if constexpr (op_is64(OP)) {
            if constexpr (SP == AS_GLOBAL) asm volatile("st.global.u64 [%0], %1;" :: "l"(a), "l"(1ull) : "memory");
            else asm volatile("st.shared::cluster.u64 [%0], %1;" :: "r"((uint32_t)a), "l"(1ull) : "memory");
        } else st32x<SP>(a, 1u);
        return 0;
    }
}

// global kernels with ILP ≤ 4 run 2 CTAs of 1024 threads per SM (K3): ≤ 32 registers per thread
template <int OP, int KIND, int ILP, int SP>
__global__ void __launch_bounds__(1024, (SP == AS_GLOBAL && ILP <= 4) ? 2 : 1) k_tput(TP p) {
    extern __shared__ __align__(16) unsigned char dyn[];
    __shared__ uint32_t s_ticket;
    __shared__ unsigned long long s_sum;
    constexpr uint32_t size = op_bytes(OP), upp = kPage / size;  // units per page
    const uint32_t smid = read_smid(), lane = threadIdx.x & 31, T = blockDim.x;
    uint32_t rank = 0;
    if constexpr (SP != AS_GLOBAL) {
        record_place(p.place);
        if constexpr (SP == AS_CLUSTER) rank = read_cluster_regs().ctarank;
        for (uint32_t i = threadIdx.x; i < kRegion / 4; i += T) reinterpret_cast<uint32_t*>(dyn)[i] = 0;  // checked sums
        if (threadIdx.x == 0) s_sum = 0;
        if constexpr (SP == AS_CLUSTER) { fence_acq_rel_cluster(); cluster_sync(); } else __syncthreads();
    }
    const bool active = SP == AS_GLOBAL ? ((p.gpc_mask >> c_gpc12[smid]) & 1u) != 0 : !(p.active_rank0_only && rank != 0);
    const uint32_t grp = SP == AS_GLOBAL && p.variant != V_MIXED ? c_part12[smid] : 0;
    const uint32_t lst = p.variant == V_REMOTE ? 1 - grp : grp;
    if (threadIdx.x == 0) s_ticket = SP == AS_GLOBAL ? (active ? atomicAdd(&p.tickets[grp], 1u) : ~0u) : 0;
    __syncthreads();
    uint64_t acc = 0;
    TimerPair t0{}, t1{};
    if (active) {
        // (selects, not p.G[grp]: a run-time index into the parameter struct would copy it to local memory)
        const uint32_t G = SP == AS_GLOBAL ? (grp ? p.G[1] : p.G[0]) : T, W = SP == AS_GLOBAL ? (lst ? p.W[1] : p.W[0]) : kRegion / size;
        const uint32_t g = s_ticket * T + threadIdx.x, w = g / 32, NW = G / 32, b = s_ticket, wb = threadIdx.x / 32, NWb = T / 32;
        // per-slot unit u[j] and per-step increment (both < W); hotN keeps the line rotation r[j] instead
        uint32_t u[ILP], inc = 0;
        const uint32_t str = p.pattern == PA_SECTOR ? 32 / size : (p.pattern == PA_LINE || p.pattern == PA_WARP1 || p.pattern == PA_BLOCK1 ||
                                                                   p.pattern == PA_HOTN) ? 128 / size : 1;
        auto modw = [&](uint64_t v) { return (uint32_t)(v % W); };
#pragma unroll
        for (int j = 0; j < ILP; ++j) {
            switch (p.pattern) {
                case PA_COAL: case PA_SECTOR: case PA_LINE: u[j] = modw(((uint64_t)j * G + g) * str); break;
                case PA_WARP1: u[j] = modw(((uint64_t)j * NW + w) * str); break;
                case PA_BLOCK1: u[j] = modw((uint64_t)b * str); break;
                case PA_BANK: u[j] = (j * NWb + wb) & 31; break;                 // rotation index
                case PA_HOTN: u[j] = (w + j) % p.nhot; break;                    // hot line index
                default: u[j] = 0;
            }
        }
        switch (p.pattern) {
            case PA_COAL: case PA_SECTOR: case PA_LINE: inc = modw((uint64_t)ILP * G * str); break;
            case PA_WARP1: inc = modw((uint64_t)ILP * NW * str); break;
            case PA_BANK: inc = (ILP * NWb) & 31; break;
            case PA_HOTN: inc = ILP % p.nhot; break;
            default: inc = 0;
        }
        const uint32_t lim = p.pattern == PA_BANK ? 32 : p.pattern == PA_HOTN ? p.nhot : W;
        // unit = u[j]·m1 + lane·m2 (+ lane & zero): bank m1 = 1024, m2 = 32; hotN m1 = units per line, m2 = 1; else m1 = 1, m2 = 0
        const uint32_t m1 = p.pattern == PA_BANK ? 1024u : p.pattern == PA_HOTN ? 128u / size : 1u;
        const uint32_t lanem = lane * (p.pattern == PA_BANK ? 32u : p.pattern == PA_HOTN ? 1u : 0u) + (lane & (uint32_t)p.zero);
        uint64_t sbase = 0;
        if constexpr (SP == AS_CTA) sbase = smem_u32(dyn);
        else if constexpr (SP == AS_CLUSTER) sbase = mapa_u32(smem_u32(dyn), p.smem_mode == 2 ? rank ^ 1u : rank);
        const uint32_t* list = lst ? p.list[1] : p.list[0];
        using R = std::conditional_t<op_is64(OP), uint64_t, uint32_t>;
        R acc_r = 0;
        __syncthreads();
        if (threadIdx.x == 0) t0 = read_timers_pair();
#pragma unroll 1
        for (uint32_t s = 0; s < p.steps; ++s) {
            R r[ILP];
#pragma unroll
            for (int j = 0; j < ILP; ++j) {
                const uint32_t un = u[j] * m1 + lanem;
                uint64_t a;
                if constexpr (SP == AS_GLOBAL) {
                    uint32_t pg;
                    asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(pg) : "l"(list + un / upp));
                    a = (uint64_t)p.buf + (uint64_t)pg * kPage + (uint64_t)(un % upp) * size;
                } else {
                    a = sbase + (uint64_t)un * size;
                }
                r[j] = (R)do_op<OP, KIND, SP>(a, p.zero);
            }
            if constexpr (KIND == TK_ATOM || KIND == TK_LD) {
#pragma unroll
                for (int j = 0; j < ILP; ++j) {
                    if constexpr (op_is64(OP)) asm volatile("xor.b64 %0, %0, %1;" : "+l"(acc_r) : "l"(r[j]));
                    else asm volatile("xor.b32 %0, %0, %1;" : "+r"(acc_r) : "r"(r[j]));
                }
            }
#pragma unroll
            for (int j = 0; j < ILP; ++j) { u[j] += inc; if (u[j] >= lim) u[j] -= lim; }
        }
        acc = acc_r;
        fence_acq_rel_gpu();  // this thread's red / st have been performed
        __syncthreads();
        if (threadIdx.x == 0) t1 = read_timers_after(acc, p.zero);
    } else {
        __syncthreads();
        __syncthreads();
    }
    if constexpr (SP != AS_GLOBAL) {  // owner sums its region after every writer has finished (integer adds: checked)
        if constexpr (SP == AS_CLUSTER) cluster_sync(); else __syncthreads();
        unsigned long long part = 0;
        if constexpr (op_is64(OP)) { for (uint32_t i = threadIdx.x; i < kRegion / 8; i += T) part += reinterpret_cast<uint64_t*>(dyn)[i]; }
        else { for (uint32_t i = threadIdx.x; i < kRegion / 4; i += T) part += reinterpret_cast<uint32_t*>(dyn)[i]; }
        for (int o = 16; o; o >>= 1) part += __shfl_xor_sync(0xffffffffu, part, o);
        if (lane == 0) atomicAdd(&s_sum, part);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const uint64_t ops = active ? (uint64_t)T * p.steps * ILP : 0;
        p.rec[blockIdx.x] = {t0.cycles, t1.cycles, t0.ns, t1.ns, ops, SP == AS_GLOBAL ? (uint64_t)(acc & p.zero) : (uint64_t)s_sum, smid,
                             SP == AS_GLOBAL ? (uint32_t)c_part12[smid] : 0u, active ? s_ticket : ~0u, 0};
    }
}

// Σ of the u32 (or u64) words of a page list's first `units` units → *out (integer-add checks)
__global__ void k_sum(const char* buf, const uint32_t* list, uint32_t units, uint32_t size, unsigned long long* out) {
    unsigned long long s = 0;
    const uint32_t upp = kPage / size;
    for (uint32_t u = blockIdx.x * blockDim.x + threadIdx.x; u < units; u += gridDim.x * blockDim.x) {
        const char* a = buf + (uint64_t)list[u / upp] * kPage + (uint64_t)(u % upp) * size;
        s += size == 8 ? *reinterpret_cast<const unsigned long long*>(a) : *reinterpret_cast<const uint32_t*>(a);
    }
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(out, s);
}

// ---------------------------------------------------------------- kernel selection
using TFn = void (*)(TP);
template <int OP, int KIND, int SP>
static TFn pick_ilp(int ilp) {
    switch (ilp) {
        case 1: return k_tput<OP, KIND, 1, SP>;
        case 2: return k_tput<OP, KIND, 2, SP>;
        case 4: return k_tput<OP, KIND, 4, SP>;
        default: return k_tput<OP, KIND, 8, SP>;
    }
}
template <int SP>
static TFn pick_k(int op, int kind, int ilp) {
#define PK(O, K) if (op == O && kind == K) return pick_ilp<O, K, SP>(ilp);
    PK(AO_ADD_U32, TK_RED) PK(AO_ADD_U32, TK_ATOM) PK(AO_ADD_F32, TK_RED) PK(AO_ADD_F32, TK_ATOM) PK(AO_ADD_U64, TK_RED) PK(AO_ADD_U64, TK_ATOM)
    PK(AO_CAS_B32, TK_ATOM) PK(AO_ADD_F16X2, TK_RED) PK(AO_ADD_F16X2, TK_ATOM) PK(AO_ADD_U32, TK_LD) PK(AO_ADD_U32, TK_ST) PK(AO_ADD_U32, TK_REDRT)
#undef PK
    hb::die("no k_tput instantiation");
}
static TFn pick(int sp, int op, int kind, int ilp) {
    return sp == AS_CTA ? pick_k<AS_CTA>(op, kind, ilp) : sp == AS_CLUSTER ? pick_k<AS_CLUSTER>(op, kind, ilp) : pick_k<AS_GLOBAL>(op, kind, ilp);
}

// ---------------------------------------------------------------- host
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
    uint64_t gbytes = 512 * kMiB;
    Homes H;
    long long li = 0;
    uint32_t* d_tickets = nullptr;
    TRec* d_rec = nullptr;
    uint32_t* d_place = nullptr;
    unsigned long long* d_sum = nullptr;
    cudaEvent_t ev0{}, ev1{};
    std::unique_ptr<hb::CsvWriter> sm_csv, gpu_csv, blk_csv;
    Ctx(hb::Run& r, bool q) : run(r), sm(r.dev().sm_count), quick(q), disc(r) {}
};

// page list: pages of [p0, p0 + np) with home h (−1 = every page, labelled or not), as uint32 page indices
struct PList {
    std::vector<uint32_t> pages;
    uint32_t* d = nullptr;
    ~PList() { cudaFree(d); }
    void upload() {
        CUDA_CHECK(cudaMalloc(&d, std::max<size_t>(1, pages.size()) * 4));
        CUDA_CHECK(cudaMemcpy(d, pages.data(), pages.size() * 4, cudaMemcpyHostToDevice));
    }
};
static void make_list(Ctx& x, PList& L, uint32_t p0, uint32_t np, int home) {
    for (uint32_t i = p0; i < p0 + np; ++i)
        if (home < 0 || x.H.page[i] == home) L.pages.push_back(i);
    L.upload();
}

struct TCfg {
    int sp = AS_GLOBAL, op = AO_ADD_U32, kind = TK_RED, ilp = 4, threads = 1024, blocks = 264, pattern = PA_COAL, variant = V_MIXED;
    uint32_t steps = 256, nhot = 1, gpc_mask = 0xFF, smem_mode = 0;
    int cs = 1, ncl = 0;          // smem targets: cluster size (1 = plain launch), clusters
    bool rank0_only = false, one_per_sm = false, clean_each = false;
    bool split = false;              // global local/remote K3: the footprint is split over the two partitions' lists
    uint64_t footprint = 16 * kMiB;  // global: bytes touched (per list: footprint, or footprint / 2 when split)
    uint32_t first_page = 0;         // global: first page of the region
    const char* target = "";
    const char* step = "";
};

static std::string tlabel(const TCfg& c) {
    return fmt("%s/%s/%s/%s/%s/%s/t%d/i%d/F%lluM/N%u/m%02x", c.step, c.target, kOpName[c.op], kTKindName[c.kind], kPatName[c.pattern],
               kVarName[c.variant], c.threads, c.ilp, (unsigned long long)(c.footprint / kMiB), c.nhot, c.gpc_mask);
}

struct Lists { PList l[3][2]; };  // [variant][group/partition]; mixed uses [0][0]

// run one configuration: 2 warm-ups + `rec` recorded launches; k2 rows (per CTA) or k3 rows (per launch + per CTA)
static void run_cfg(Ctx& x, const TCfg& c, int recorded, bool k3, PList* lists[2]) {
    const bool glob = c.sp == AS_GLOBAL;
    TP p{};
    p.buf = x.gbuf;
    p.variant = (uint32_t)c.variant; p.pattern = (uint32_t)c.pattern; p.steps = c.steps; p.nhot = c.nhot; p.gpc_mask = c.gpc_mask;
    p.smem_mode = c.smem_mode; p.active_rank0_only = c.rank0_only; p.tickets = x.d_tickets; p.rec = x.d_rec; p.place = x.d_place; p.zero = 0;
    const uint32_t size = op_bytes(c.op);
    if (glob) {
        for (int i = 0; i < 2; ++i) {
            PList* L = lists[i] ? lists[i] : lists[0];
            p.list[i] = L->d;
            p.W[i] = (uint32_t)std::min<uint64_t>((uint64_t)L->pages.size() * (kPage / size), (c.split ? c.footprint / 2 : c.footprint) / size);
        }
        // threads per group: 2 CTAs per SM (or 1 with one_per_sm) on the active GPCs, split by partition
        int bps = c.one_per_sm ? 1 : 2;
        uint32_t act[2] = {0, 0}, all = 0;
        for (int s = 0; s < x.sm; ++s)
            if ((c.gpc_mask >> x.P.gpc[s]) & 1u) { act[x.P.part[s]] += bps; all += bps; }
        if (c.blocks < 2 * x.sm && !c.one_per_sm) hb::die("global K2/K3 configs use 2 CTAs per SM or one_per_sm");
        if (c.one_per_sm && c.blocks < x.sm) {  // fewer CTAs than SMs: the group sizes follow the actual placement (set below)
            act[0] = act[1] = all = (uint32_t)c.blocks;
        }
        p.G[0] = (c.variant == V_MIXED ? all : act[0]) * c.threads;
        p.G[1] = act[1] * c.threads;
    }
    TFn fn = pick(c.sp, c.op, c.kind, c.ilp);
    hb::ClusterLaunch L;
    L.grid = dim3(c.blocks); L.block = dim3(c.threads);
    L.smem = glob ? (c.one_per_sm ? kForceOneSmem : 0) : kOneCtaPerSmSmem;
    if (!glob && c.cs > 1) { L.cluster = dim3(c.cs, 1, 1); L.explicit_cluster = true; }
    hb::ClusterLaunch::Prepared prep;
    L.prepare(fn, prep);
    const std::string label = tlabel(c);
    auto one = [&](bool timed) {
        if (c.clean_each) x.clean();
        CUDA_CHECK(cudaMemsetAsync(x.d_tickets, 0, 2 * sizeof(uint32_t)));
        if (timed) CUDA_CHECK(cudaEventRecord(x.ev0));
        CUDA_CHECK(cudaLaunchKernelEx(&prep.cfg, fn, p));
        CUDA_CHECK(cudaGetLastError());
        if (timed) CUDA_CHECK(cudaEventRecord(x.ev1));
        CUDA_CHECK(cudaDeviceSynchronize());
    };
    // integer-add check: Σ before / after the last recorded launch
    const bool checkable = (c.op == AO_ADD_U32 || c.op == AO_ADD_U64) && (c.kind == TK_RED || c.kind == TK_ATOM || c.kind == TK_REDRT);
    auto gsum = [&]() -> unsigned long long {
        unsigned long long tot = 0;
        for (int i = 0; i < (c.variant == V_MIXED ? 1 : 2); ++i) {
            CUDA_CHECK(cudaMemset(x.d_sum, 0, 8));
            k_sum<<<264, 1024>>>(x.gbuf, p.list[i], p.W[i], size, x.d_sum);
            CUDA_CHECK(cudaGetLastError());
            unsigned long long v;
            CUDA_CHECK(cudaMemcpy(&v, x.d_sum, 8, cudaMemcpyDeviceToHost));
            tot += v;
        }
        return tot;
    };
    for (int i = 0; i < 2; ++i) one(false);
    int valid = 0, discards = 0;
    std::vector<double> rate;
    while (valid < recorded) {
        const bool last = valid == recorded - 1;
        unsigned long long s0 = 0;
        if (glob && checkable && last) s0 = gsum();
        one(true);
        const auto cs = x.run.clock_note(label, "launch", x.li);
        std::vector<TRec> r(c.blocks);
        CUDA_CHECK(cudaMemcpy(r.data(), x.d_rec, c.blocks * sizeof(TRec), cudaMemcpyDeviceToHost));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, x.ev0, x.ev1));
        // placement
        std::vector<int> per(x.sm, 0);
        for (const auto& b : r) if (b.smid < (uint32_t)x.sm) ++per[b.smid];
        bool place_ok = true;
        if (glob) {
            for (int s = 0; s < x.sm; ++s) place_ok &= c.one_per_sm ? per[s] <= 1 : per[s] == 2;
        } else {
            std::vector<uint32_t> pl(4 * (size_t)c.blocks);
            CUDA_CHECK(cudaMemcpy(pl.data(), x.d_place, pl.size() * 4, cudaMemcpyDeviceToHost));
            std::vector<uint32_t> sm(c.blocks);
            for (int b = 0; b < c.blocks; ++b) sm[b] = pl[4 * b];
            place_ok = hb::all_distinct(sm);
        }
        const std::string discard = !place_ok ? "placement" : cs.bad() ? "clock" : "";
        uint64_t ops = 0, lo = ~0ull, hi = 0;
        std::vector<double> mhz;
        for (const auto& b : r) {
            if (!b.ops) continue;
            ops += b.ops;
            lo = std::min(lo, b.n0); hi = std::max(hi, b.n1);
            if (b.n1 > b.n0) mhz.push_back(1e3 * double(b.c1 - b.c0) / double(b.n1 - b.n0));
        }
        std::sort(mhz.begin(), mhz.end());
        const double sm_mhz = mhz.empty() ? 0 : mhz[mhz.size() / 2];
        int check = -1;
        if (checkable && last) {
            if (glob) {
                const unsigned long long s1 = gsum();
                const unsigned long long d = s1 - s0;
                check = (c.op == AO_ADD_U32 ? (uint32_t)d == (uint32_t)ops : d == ops) ? 1 : 0;
            } else {  // each owner's sum = ops it received (own region: own ops; peer: the peer's ops)
                check = 1;
                for (int b = 0; b < c.blocks; ++b) {
                    const int src = c.smem_mode == 2 ? (b ^ 1) : b;
                    const uint64_t expect = r[src].ops;
                    if (c.op == AO_ADD_U32 ? (uint32_t)r[b].sum != (uint32_t)expect : r[b].sum != expect) check = 0;
                }
            }
            if (check == 0) std::printf("  WARNING %s: memory sum does not match the issued ops\n", label.c_str());
        }
        const std::string pre = x.run.prefix(x.li, cs);
        if (!k3) {
            if (!x.sm_csv)
                x.sm_csv = std::make_unique<hb::CsvWriter>(hb::Run::results("12__atomics__throughput_sm.csv"),
                    hb::with_mandatory({"op", "kind", "target", "pattern", "threads", "ilp", "placement", "block", "smid", "ops", "cycles",
                                        "ops_per_cycle", "bpc", "steps", "footprint_bytes", "part", "check_ok", "discard"}));
            for (int b = 0; b < c.blocks; ++b) {
                if (!r[b].ops) continue;
                const uint64_t cyc = r[b].c1 - r[b].c0;
                x.sm_csv->add(pre + fmt(",%s,%s,%s,%s,%d,%d,%s,%d,%u,%llu,%llu,%.5f,%.4f,%u,%llu,%d,%d,%s", kOpName[c.op], kTKindName[c.kind], c.target,
                                        kPatName[c.pattern], c.threads, c.ilp, c.blocks > 2 ? "132sm" : "1sm", b, r[b].smid, (unsigned long long)r[b].ops,
                                        (unsigned long long)cyc, double(r[b].ops) / cyc, double(r[b].ops) * size / cyc, c.steps,
                                        (unsigned long long)(glob ? c.footprint : kRegion), glob ? (int)r[b].part : (x.P.part.empty() ? -1 : x.P.part[r[b].smid]),
                                        check, discard.c_str()));
            }
            x.sm_csv->flush();
        } else {
            if (!x.gpu_csv) {
                x.gpu_csv = std::make_unique<hb::CsvWriter>(hb::Run::results("12__atomics__throughput_gpu.csv"),
                    hb::with_mandatory({"op", "kind", "pattern", "variant", "footprint_bytes", "n_hot_lines", "reader_gpc_mask", "blocks", "threads", "ilp",
                                        "ops", "ms_event", "gops", "tbps_payload", "sm_mhz_blocks", "discard", "steps", "active_sms", "check_ok",
                                        "span_ns", "step"}));
                x.blk_csv = std::make_unique<hb::CsvWriter>(hb::Run::results("12__atomics__throughput_gpu_blocks.csv"),
                    hb::with_mandatory({"step", "op", "kind", "pattern", "variant", "footprint_bytes", "n_hot_lines", "reader_gpc_mask", "block", "smid", "gpc",
                                        "partition", "ticket", "ops", "cycles", "ns", "start_ns", "ops_per_cycle", "discard"}));
            }
            std::set<uint32_t> act;
            for (const auto& b : r) if (b.ops) act.insert(b.smid);
            x.gpu_csv->add(pre + fmt(",%s,%s,%s,%s,%llu,%u,%u,%d,%d,%d,%llu,%.6f,%.4f,%.6f,%.2f,%s,%u,%zu,%d,%llu,%s", kOpName[c.op], kTKindName[c.kind],
                                     kPatName[c.pattern], kVarName[c.variant], (unsigned long long)c.footprint, c.nhot, c.gpc_mask, c.blocks, c.threads, c.ilp,
                                     (unsigned long long)ops, ms, ops / (ms * 1e-3) / 1e9, ops * (double)size / (ms * 1e-3) / 1e12, sm_mhz,
                                     discard.c_str(), c.steps, act.size(), check, (unsigned long long)(hi > lo ? hi - lo : 0), c.step));
            for (int b = 0; b < c.blocks; ++b) {
                const auto& q = r[b];
                const uint64_t cyc = q.c1 - q.c0;
                x.blk_csv->add(pre + fmt(",%s,%s,%s,%s,%s,%llu,%u,%u,%d,%u,%d,%u,%d,%llu,%llu,%llu,%lld,%.5f,%s", c.step, kOpName[c.op], kTKindName[c.kind],
                                         kPatName[c.pattern], kVarName[c.variant], (unsigned long long)c.footprint, c.nhot, c.gpc_mask, b, q.smid,
                                         q.smid < (uint32_t)x.sm ? x.P.gpc[q.smid] : -1, q.part, q.ticket == ~0u ? -1 : (int)q.ticket,
                                         (unsigned long long)q.ops, (unsigned long long)(q.ops ? cyc : 0), (unsigned long long)(q.ops ? q.n1 - q.n0 : 0),
                                         q.ops ? (long long)(q.n0 - lo) : -1ll, q.ops && cyc ? double(q.ops) / cyc : 0.0, discard.c_str()));
            }
            x.gpu_csv->flush();
            x.blk_csv->flush();
        }
        ++x.li;
        if (!discard.empty()) {
            x.disc.add(label, discard.c_str(), x.li - 1, cs);
            if (++discards > 20) hb::die(label + ": too many discards");
            continue;
        }
        rate.push_back(ops / (ms * 1e-3) / 1e9);
        ++valid;
    }
    if (x.sm_csv) x.sm_csv->sync();
    if (x.gpu_csv) { x.gpu_csv->sync(); x.blk_csv->sync(); }
    std::sort(rate.begin(), rate.end());
    std::printf("%-78s %9.3f Gops/s\n", label.c_str(), rate[rate.size() / 2]);
}

static void setup(Ctx& x) {
    x.P = load_parts(x.sm);
    std::vector<uint8_t> part(256, 0), gpc(256, 0);
    for (int s = 0; s < x.sm; ++s) { part[s] = (uint8_t)x.P.part[s]; gpc[s] = (uint8_t)x.P.gpc[s]; }
    CUDA_CHECK(cudaMemcpyToSymbol(c_part12, part.data(), 256));
    CUDA_CHECK(cudaMemcpyToSymbol(c_gpc12, gpc.data(), 256));
    x.gbuf = alloc_aligned(x.gbytes);
    CUDA_CHECK(cudaMemset(x.gbuf, 0, x.gbytes));
    x.run.lock_clocks_note("label", "before");
    x.H = label_buffer(x.run, x.claim, x.flush, x.P, x.gbuf, (uint32_t)(x.gbytes / kPage), 0);
    x.run.lock_clocks_note("label", "after");
    write_homes_summary(x.run, x.H, "atom_tput", x.gbytes);
    CUDA_CHECK(cudaMalloc(&x.d_tickets, 2 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&x.d_rec, 2 * 132 * sizeof(TRec)));
    CUDA_CHECK(cudaMalloc(&x.d_place, 4 * 2 * 132 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&x.d_sum, 8));
    CUDA_CHECK(cudaEventCreate(&x.ev0));
    CUDA_CHECK(cudaEventCreate(&x.ev1));
}

// ---------------------------------------------------------------- K2
static void run_k2(Ctx& x, bool all) {
    const uint32_t l2pages = (uint32_t)(16 * kMiB / kPage), hbm0 = (uint32_t)(256 * kMiB / kPage), hbmn = (uint32_t)(256 * kMiB / kPage);
    PList near_far[2], hbm;
    for (int h = 0; h < 2; ++h) make_list(x, near_far[h], 0, l2pages, h);
    make_list(x, hbm, hbm0, hbmn, -1);
    const int rec = x.quick ? 2 : 10;
    struct Tgt { const char* name; int sp, cs; uint32_t smode; };
    const std::vector<Tgt> tg = {{"smem", AS_CTA, 1, 0}, {"smem_dsmem_if", AS_CLUSTER, 2, 1}, {"dsmem", AS_CLUSTER, 2, 2},
                                 {"l2_near", AS_GLOBAL, 1, 0}, {"l2_far", AS_GLOBAL, 1, 0}, {"hbm", AS_GLOBAL, 1, 0}};
    const std::vector<int> threads = x.quick ? std::vector<int>{1024} : all ? std::vector<int>{1024} : std::vector<int>{128, 256, 512, 1024};
    const std::vector<int> ilps = x.quick ? std::vector<int>{1, 4} : std::vector<int>{1, 2, 4, 8};
    x.run.lock_clocks_note(all ? "k2all" : "k2", "before");
    for (const auto& t : tg) {
        const bool glob = t.sp == AS_GLOBAL;
        std::vector<int> pats = {PA_COAL, PA_WARP1, PA_BLOCK1};
        if (!glob) pats.push_back(PA_BANK); else if (!all) { pats.push_back(PA_SECTOR); pats.push_back(PA_LINE); }
        struct OK { int op, kind; bool full; };
        std::vector<OK> oks = {{AO_ADD_U32, TK_RED, true}, {AO_ADD_U32, TK_ATOM, true}, {AO_ADD_U32, TK_LD, false}, {AO_ADD_U32, TK_ST, false}};
        if (!all) for (int op : {AO_ADD_F32, AO_ADD_U64, AO_ADD_F16X2}) { oks.push_back({op, TK_RED, false}); oks.push_back({op, TK_ATOM, false}); }
        if (!all) oks.push_back({AO_CAS_B32, TK_ATOM, false});
        for (const auto& ok : oks)
            for (int pat : pats) {
                if (!ok.full && pat != PA_COAL) continue;
                for (int T : threads) {
                    if (!ok.full && T != 1024) continue;
                    for (int ilp : ilps) {
                        TCfg c;
                        c.step = all ? "k2all" : "k2"; c.target = t.name; c.sp = t.sp; c.op = ok.op; c.kind = ok.kind; c.ilp = ilp; c.threads = T;
                        c.pattern = pat; c.smem_mode = t.smode;
                        PList* lists[2] = {nullptr, nullptr};
                        if (!glob) {
                            c.cs = t.cs;
                            c.blocks = all ? x.sm : t.cs;
                            c.rank0_only = !all;
                            c.steps = 4096;
                        } else {
                            c.one_per_sm = true;
                            c.blocks = all ? x.sm : 1;
                            if (!std::strcmp(t.name, "hbm")) {
                                c.variant = V_MIXED; c.footprint = 256 * kMiB; c.clean_each = true;
                                lists[0] = &hbm;
                                // one pass over 256 MiB at most (every access misses L2)
                                const uint64_t per_step = (uint64_t)T * ilp * c.blocks * (pat == PA_LINE ? 128 : pat == PA_SECTOR ? 32 : op_bytes(c.op));
                                c.steps = (uint32_t)std::max<uint64_t>(16, std::min<uint64_t>(4096, (256 * kMiB) / per_step));
                            } else {
                                c.variant = !std::strcmp(t.name, "l2_near") ? V_LOCAL : V_REMOTE;
                                c.footprint = 8 * kMiB;
                                lists[0] = &near_far[0]; lists[1] = &near_far[1];
                                c.steps = 4096;
                            }
                        }
                        if (pat == PA_BLOCK1 || pat == PA_WARP1) c.steps = std::min<uint32_t>(c.steps, glob ? 1024 : 4096);
                        if (x.quick) c.steps = std::min<uint32_t>(c.steps, 512);
                        run_cfg(x, c, rec, false, lists);
                    }
                }
            }
    }
    x.run.lock_clocks_note(all ? "k2all" : "k2", "after");
}

// k2x: local-smem red.add.u32 with a run-time operand (1 + zero): ptxas turns red.shared.add with the constant 1 into
// ATOMS.POPC.INC (one increment by the number of lanes on the same word); a run-time operand keeps a plain ATOMS.ADD.
static void run_k2x(Ctx& x) {
    const int rec = x.quick ? 2 : 10;
    x.run.lock_clocks_note("k2x", "before");
    for (int all : {0, 1})
        for (int pat : {PA_COAL, PA_BANK, PA_WARP1, PA_BLOCK1})
            for (int ilp : {1, 2, 4, 8}) {
                TCfg c;
                c.step = "k2x"; c.target = "smem"; c.sp = AS_CTA; c.op = AO_ADD_U32; c.kind = TK_REDRT; c.ilp = ilp; c.threads = 1024; c.pattern = pat;
                c.cs = 1; c.blocks = all ? x.sm : 1; c.steps = x.quick ? 512 : 4096;
                PList* lists[2] = {nullptr, nullptr};
                run_cfg(x, c, rec, false, lists);
            }
    x.run.lock_clocks_note("k2x", "after");
}

// ---------------------------------------------------------------- K3
struct GL { PList mixed, home[2]; };
static void make_gl(Ctx& x, GL& g, uint64_t footprint) {
    const uint32_t np = (uint32_t)(x.gbytes / kPage);
    make_list(x, g.mixed, 0, np, -1);
    for (int h = 0; h < 2; ++h) make_list(x, g.home[h], 0, np, h);
    (void)footprint;
}
static void k3_cfg(Ctx& x, GL& g, TCfg c, int rec) {
    c.sp = AS_GLOBAL;
    if (!c.one_per_sm) c.blocks = 2 * x.sm;
    PList* lists[2] = {&g.mixed, nullptr};
    if (c.variant != V_MIXED) { lists[0] = &g.home[0]; lists[1] = &g.home[1]; }
    c.split = c.variant != V_MIXED;  // local/remote: the footprint is split over the two partitions' lists
    run_cfg(x, c, rec, true, lists);
}

static void run_k3(Ctx& x, char which) {
    GL g;
    make_gl(x, g, 0);
    const int rec = x.quick ? 2 : 10;
    x.run.lock_clocks_note(fmt("k3%c", which), "before");
    if (which == 'a') {
        std::vector<uint64_t> fps = {1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 256};
        if (x.quick) fps = {4, 16, 128};
        struct OK { int op, kind; };
        std::vector<OK> oks = {{AO_ADD_U32, TK_RED}, {AO_ADD_U32, TK_ATOM}, {AO_ADD_F32, TK_RED}, {AO_ADD_F32, TK_ATOM}, {AO_ADD_U64, TK_RED},
                               {AO_ADD_U64, TK_ATOM}, {AO_ADD_U32, TK_LD}, {AO_ADD_U32, TK_ST}};
        if (x.quick) oks = {{AO_ADD_U32, TK_RED}, {AO_ADD_U32, TK_ATOM}, {AO_ADD_U32, TK_LD}, {AO_ADD_U32, TK_ST}};
        for (const auto& ok : oks)
            for (int pat : {PA_COAL, PA_SECTOR})
                for (int v : {V_MIXED, V_LOCAL, V_REMOTE})
                    for (uint64_t f : fps) {
                        TCfg c;
                        c.step = "k3a"; c.target = "global"; c.op = ok.op; c.kind = ok.kind; c.pattern = pat; c.variant = v; c.footprint = f * kMiB;
                        c.steps = pat == PA_SECTOR ? 64 : 256;
                        k3_cfg(x, g, c, rec);
                    }
    } else if (which == 'b') {
        std::vector<uint32_t> masks;
        for (int p = 0; p < 2; ++p) {
            uint32_t m = 0;
            for (int gp = 0; gp < 8; ++gp)
                if (x.P.gpc_part[gp] == p) { m |= 1u << gp; masks.push_back(m); }
        }
        masks.push_back(0xFF);
        for (uint32_t m : masks)
            for (int kind : {TK_RED, TK_ATOM, TK_ST, TK_LD}) {
                TCfg c;
                c.step = "k3b"; c.target = "global"; c.op = AO_ADD_U32; c.kind = kind; c.variant = V_REMOTE; c.footprint = 16 * kMiB; c.gpc_mask = m;
                c.steps = 256;
                k3_cfg(x, g, c, rec);
            }
    } else if (which == 'c') {
        for (int op : {AO_ADD_U32, AO_ADD_F32})
            for (int kind : {TK_RED, TK_ATOM}) {
                {
                    TCfg c;
                    c.step = "k3c"; c.target = "global"; c.op = op; c.kind = kind; c.pattern = PA_HOT1; c.nhot = 0; c.footprint = 4 * kMiB; c.steps = 2;
                    k3_cfg(x, g, c, rec);
                }
                for (uint32_t n = 1; n <= 1024; n *= 2) {
                    if (x.quick && n != 1 && n != 32 && n != 1024) continue;
                    TCfg c;
                    c.step = "k3c"; c.target = "global"; c.op = op; c.kind = kind; c.pattern = PA_HOTN; c.nhot = n; c.footprint = 4 * kMiB;
                    c.steps = std::min<uint32_t>(256, 4 * n);
                    k3_cfg(x, g, c, rec);
                }
                for (int nsm : {1, 2, 4, 8, 16, 33, 66, 132}) {
                    if (x.quick && nsm != 1 && nsm != 132) continue;
                    for (int hot : {0, 1, 32}) {  // 0 = hot1 (one word), else hotN with N lines
                        TCfg c;
                        c.step = "k3c_sms"; c.target = "global"; c.op = op; c.kind = kind; c.pattern = hot ? PA_HOTN : PA_HOT1; c.nhot = (uint32_t)hot;
                        c.footprint = 4 * kMiB; c.one_per_sm = true; c.blocks = nsm; c.steps = hot ? std::min<uint32_t>(256, 8 * hot) : 4;
                        k3_cfg(x, g, c, rec);
                    }
                }
            }
    } else {  // 'd': one GPC alone
        for (int gp = 0; gp < 8; ++gp)
            for (int kind : {TK_RED, TK_ATOM, TK_ST, TK_LD}) {
                if (x.quick && gp > 1) continue;
                TCfg c;
                c.step = "k3d"; c.target = "global"; c.op = AO_ADD_U32; c.kind = kind; c.variant = V_LOCAL; c.footprint = 16 * kMiB; c.gpc_mask = 1u << gp;
                c.steps = 256;
                k3_cfg(x, g, c, rec);
            }
    }
    x.run.lock_clocks_note(fmt("k3%c", which), "after");
}

// ---------------------------------------------------------------- ncu1: one profiled configuration
static std::vector<TCfg> ncu_cfgs(Ctx& x) {
    std::vector<TCfg> v;
    auto add = [&](const char* step, int sp, int op, int kind, int pat, int var, uint64_t fp, uint32_t n, uint32_t steps, int blocks, bool one,
                   const char* tgt, uint32_t smode = 0, int cs = 1) {
        TCfg c;
        c.step = step; c.sp = sp; c.op = op; c.kind = kind; c.pattern = pat; c.variant = var; c.footprint = fp; c.nhot = n; c.steps = steps;
        c.blocks = blocks; c.one_per_sm = one; c.target = tgt; c.smem_mode = smode; c.cs = cs;
        v.push_back(c);
    };
    const int B = 2 * x.sm;
    for (int kind : {TK_RED, TK_ATOM, TK_ST, TK_LD})
        for (int var : {V_LOCAL, V_REMOTE}) add("ncu_k3a", AS_GLOBAL, AO_ADD_U32, kind, PA_COAL, var, 16 * kMiB, 1, 64, B, false, "global");
    for (int kind : {TK_RED, TK_ATOM}) {
        add("ncu_k3a", AS_GLOBAL, AO_ADD_U32, kind, PA_COAL, V_MIXED, 256 * kMiB, 1, 64, B, false, "global");
        add("ncu_k3c", AS_GLOBAL, AO_ADD_U32, kind, PA_HOT1, V_MIXED, 4 * kMiB, 0, 2, B, false, "global");
        add("ncu_k3c", AS_GLOBAL, AO_ADD_U32, kind, PA_HOTN, V_MIXED, 4 * kMiB, 1, 4, B, false, "global");
        add("ncu_k3c", AS_GLOBAL, AO_ADD_U32, kind, PA_HOTN, V_MIXED, 4 * kMiB, 32, 128, B, false, "global");
        add("ncu_k3c", AS_GLOBAL, AO_ADD_F32, kind, PA_HOT1, V_MIXED, 4 * kMiB, 0, 2, B, false, "global");
        // smem targets (kernel replay): local, own-window, peer, coal and same word
        for (int pat : {PA_COAL, PA_BANK, PA_BLOCK1}) {
            add("ncu_k2", AS_CTA, AO_ADD_U32, kind, pat, V_MIXED, kRegion, 1, 1024, 1, false, "smem");
            add("ncu_k2", AS_CLUSTER, AO_ADD_U32, kind, pat, V_MIXED, kRegion, 1, 1024, 2, false, "dsmem", 2, 2);
        }
        add("ncu_k2", AS_CLUSTER, AO_ADD_U32, kind, PA_COAL, V_MIXED, kRegion, 1, 1024, 2, false, "smem_dsmem_if", 1, 2);
    }
    return v;
}
static void run_ncu1(Ctx& x, int idx, bool list) {
    GL g;
    if (!list) make_gl(x, g, 0);
    const auto v = ncu_cfgs(x);
    if (list) { for (size_t i = 0; i < v.size(); ++i) std::printf("%zu %s\n", i, tlabel(v[i]).c_str()); return; }
    TCfg c = v.at(idx);
    // baseline: the same grid, every block exits at once (gpc_mask 0 / no active rank)
    TCfg b = c;
    b.step = "ncu_baseline";
    if (c.sp == AS_GLOBAL) { b.gpc_mask = 0; } else { b.steps = 0; }
    hb::CsvWriter man("bin/ncu_manifest.csv", hb::with_mandatory({"cfg", "label", "role", "op", "kind", "pattern", "variant",
                                                                                              "target", "ops", "steps", "blocks", "threads", "ilp"}));
    const auto s = x.run.dev().sample();
    auto row = [&](const TCfg& q, const char* role, uint64_t ops) {
        man.add(x.run.prefix(x.li, s) + fmt(",%d,%s,%s,%s,%s,%s,%s,%s,%llu,%u,%d,%d,%d", idx, tlabel(q).c_str(), role, kOpName[q.op], kTKindName[q.kind],
                                            kPatName[q.pattern], kVarName[q.variant], q.target, (unsigned long long)ops, q.steps, q.blocks, q.threads, q.ilp));
    };
    // record=1 launches (the run_cfg loop: 2 warm-ups + 1 recorded); the profiled kernel is the last k_tput launch of each run_cfg
    if (c.sp == AS_GLOBAL) k3_cfg(x, g, b, 1); else { PList* l[2] = {nullptr, nullptr}; run_cfg(x, b, 1, false, l); }
    row(b, "baseline", 0);
    if (c.sp == AS_GLOBAL) k3_cfg(x, g, c, 1); else { c.rank0_only = c.cs == 2; PList* l[2] = {nullptr, nullptr}; run_cfg(x, c, 1, false, l); }
    uint64_t act = c.sp == AS_GLOBAL ? (uint64_t)(c.one_per_sm ? c.blocks : 2 * x.sm) : (c.cs == 2 ? 1 : (uint64_t)c.blocks);
    row(c, "profiled", act * c.threads * c.steps * c.ilp);
    man.sync();
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s {k2|k2all|k3a|k3b|k3c|k3d|ncu1|clockcheck} [--quick] [--cfg N|--list]\n", argv[0]); return 1; }
    const std::string mode = argv[1];
    const bool list = hb::has_flag(argc, argv, "--list");
    hb::Run run(EXP, argc, argv);
    Ctx x(run, hb::has_flag(argc, argv, "--quick"));
    if (mode == "clockcheck") { run.lock_clocks_note(hb::arg_str(argc, argv, "--label", "check"), "check"); run.complete(); return 0; }
    if (!list) setup(x); else x.P = load_parts(x.sm);
    if (mode == "k2") run_k2(x, false);
    else if (mode == "k2all") run_k2(x, true);
    else if (mode == "k2x") run_k2x(x);
    else if (mode == "k3a" || mode == "k3b" || mode == "k3c" || mode == "k3d") run_k3(x, mode[2]);
    else if (mode == "ncu1") run_ncu1(x, (int)hb::arg_int(argc, argv, "--cfg", 0), list);
    else hb::die("unknown mode " + mode);
    std::printf("discarded launches: %d\n", x.disc.n);
    run.complete();
    return 0;
}
