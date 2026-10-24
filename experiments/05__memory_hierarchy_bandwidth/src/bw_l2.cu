/*
 * bw_l2.cu — experiment 05, K3 + K5 (L2 part): L2 read/store bandwidth GPU-wide, per partition and per SM, for data
 * homed in the reader's own L2 partition ("local"), in the other partition ("remote"), or both ("mixed").
 *
 * Usage: bin/bw_l2 {k3|link|ncu|clockcheck} [--quick] [--label L]. One invocation = one run_id.
 *
 * Partitions. Each SM belongs to one of two L2 partitions through its GPC (bw_common.hpp). Each 128 B line has a home
 * partition fixed by its address; an access from the other partition leaves a copy of the line in the accessor's
 * partition, so every repeated access is a near hit (measured in experiment 04).
 *
 * Step 1 — home labelling (every invocation; buffer of kBuf = 96 MiB, 32 MiB with --quick, from cudaMalloc).
 *   Per 8 MiB chunk (65 536 lines): L2 flush (hb::L2Flusher) → touch_lines on SM A (smid 0, partition 0) writes every
 *   line completely → probe_lines on SM B (smid 2, partition 1) reads every line once with one dependent
 *   ld.global.cg.u32 and times it (t0 = %clock64; load; t1 = read_clock64_after(value); cycles = t1 − t0, timer pair
 *   included). Then the same with A and B swapped. One-warp blocks pinned by claim_target_sm (hb::SmClaim, retried).
 *   2-means over all samples gives the near/far threshold. A line is far for B and near for A ⇒ home 0; near for B
 *   and far for A ⇒ home 1; anything else ⇒ ambiguous (excluded from the lists). Chunk 0 is also probed from the first
 *   SM of every GPC (toucher: A or B, whichever is in the other partition) to check the GPC → partition grouping.
 *   → results/05__bandwidth__l2_home.csv (k3 mode only): line_offset, cycles_b (touch A, probe B), cycles_a (touch B,
 *     probe A), home (0 | 1 | -1 ambiguous), cycles_g0 … cycles_g7 (chunk 0 only; probe = first SM of GPC g).
 *
 * Step 2 — streaming kernel k3_stream<OP, ILP, LIST> (k3_ncu: same body, profiled under ncu). Lines of a footprint F
 *   are the buffer's first F bytes. Groups: "mixed" — one group of all blocks over all lines; "local" — the blocks of
 *   partition p read the lines homed in p; "remote" — the blocks of partition p read the lines homed in 1 − p. On entry
 *   each block reads %smid → partition, takes a ticket within its group (atomic), and each warp w of the group gets a
 *   fixed range of its group's line list: start = w·L/W, count = (w+1)·L/W − start (L lines, W warps; when L < 4·ILP·W:
 *   start = 4·ILP·w mod L, count = 4·ILP, wrapping). LIST variants copy the warp's list entries into shared memory
 *   once (no list traffic in the timed loop); "mixed" computes line = start + j. Per step a warp issues ILP accesses;
 *   each covers 4 lines (8 lanes × 16 B per line): ld.global.cg.v4.f32 (OP ld), st.global.v4.f32 (OP st), or
 *   alternately ld/st (OP mix). Loads of a step are issued before their data is XORed into an accumulator.
 *   Timed region per block: bar.sync → thread 0 timers → `reps` passes over the warp's range → bar.sync → thread 0
 *   timers after its last loaded value. Each rep adds (rep & zero) = 0 to the base address: ptxas cannot prove it is
 *   0, so it cannot hoist the rep-invariant loads out of the rep loop. bytes per block = Σ warps count·128·reps.
 *   GPU-wide time = cudaEvent pair around the launch. Blocks whose partition is not in reader_mask exit at once
 *   (one-direction first pass).
 *   k3_direct<ILP>: the plain grid-stride stream (no lists): element e = base + i·T_total for i < ILP, ld.global.cg.v4.
 *
 * Configurations (2 × 132 blocks × 1024 threads, ILP 4, unless stated; 2 warm-up + 10 recorded launches; reps chosen
 * so that a launch moves ≈ 8 GiB):
 *   steady       variant ∈ {mixed, local, remote} × F ∈ kFootprints (1 … 96 MiB), op ld; op st and mix for mixed.
 *   list check   mixed_list (identity list through smem) and mixed_direct (k3_direct, 2048 reps) at 16 MiB.
 *   ilp          mixed 16 MiB, ILP ∈ {1, 2, 4} at 2 blocks/SM and ILP ∈ {1, 2, 4, 8} at 1 block/SM (1024 threads).
 *   active_sms   mixed 16 MiB, n ∈ kActive blocks of 1024 threads, 1 block/SM (dynamic smem 120 KiB).
 *   first pass   flush + clean (CleanL2: no dirty lines left) → home touch (every line of the footprint read once by a block of its home partition: k3_stream
 *                local, 1 rep; lines are clean and live only at home) → pass 1 (timed, 1 rep) → pass 2 (timed, 1 rep,
 *                no flush: copies now exist). Variants: both directions {local, remote, mixed} × F ∈ {8, 16, 24} MiB;
 *                one direction (reader partition r ∈ {0, 1} only; the other partition's blocks exit) {local, remote}
 *                × F = 48 MiB (≈ 24 MiB homed in each partition). Block configs: 1 block/SM (smem-forced) of 256 or
 *                512 threads (≈ 12 / 6 steps per warp), and 2 × 1024 threads/SM (1–2 steps per warp). 20 launches each.
 *   steady st/mix local and remote at 16 MiB (K5).
 *   cache ops    load cache operator ∈ {cs, lu, cv} (instead of cg) × {mixed, local, remote} at 16 MiB.
 *   reader_mask (k3 CSV): partitions whose blocks work (bit p = partition p; 3 = all). The kernel receives it as a GPC
 *   bitmask (reader_gpcs): blocks on SMs of other GPCs exit at once.
 * Placement: blocks must fill every SM exactly as intended (2 per SM, or ≤ 1 per SM); otherwise the launch is discarded
 * and repeated, as is a launch whose NVML sample shows a throttle reason or a clock below the lock.
 *   → results/05__bandwidth__l2.csv (one row per launch): variant, op, footprint_bytes, blocks, threads, ilp, reps,
 *     pass (0 = steady state, 1/2 = first-pass test), reader_mask, bytes, ms_event, tbps, sm_mhz_blocks (median
 *     cycles/ns of the blocks), bpc_chip (= tbps / sm clock), span_ns (min block start → max block end, %globaltimer),
 *     tbps_span, discard.
 *   → results/05__bandwidth__l2_blocks.csv (one row per block and launch): variant, op, footprint_bytes, pass, block,
 *     smid, gpc, partition, ticket, bytes, cycles, ns, start_ns (relative to the earliest block of the launch),
 *     bpc_sm, discard.
 * link mode (partition link): the same streaming kernel with only the blocks on SMs of a chosen set of GPCs working
 *   (bitmask; the others exit at once): {G1}, {G1,G2}, {G1,G2,G3}, partition 1, {G4}, {G4,G5}, {G4,G5,G6}, partition 0,
 *   both partitions. Per set: steady state at 16 MiB of st local, st remote, ld local (2 × 132 blocks × 1024 threads,
 *   ILP 4; stores never leave a copy in the writer's partition, so remote stores cross the link on every pass), and the
 *   first-pass test, ld local and remote, 48 MiB prefix, 1 block × 512 threads per SM.
 *   → results/05__bandwidth__l2_link.csv and 05__bandwidth__l2_link_blocks.csv: the columns of the k3 files, with
 *     reader_gpc_mask (GPC bitmask) in place of reader_mask. Labels → 05__bandwidth__l2_home.csv as in k3 mode.
 * ncu mode (application replay, --cache-control none; see src/run.sh): labelling, then one k3_ncu launch in which every
 *   block exits at once (the counters the profiler itself adds), then for each profiled configuration the warm-up
 *   launches (k3_stream) and one k3_ncu launch; first-pass configurations profile pass 1 and pass 2.
 *   Writes bin/ncu_manifest_l2.csv.
 * --quick: 32 MiB buffer, F ∈ {4, 16, 28} MiB, short lists of other configurations, 3 recorded launches.
 */
#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include "bw_common.hpp"

static const char* EXP = "05__memory_hierarchy_bandwidth";
static constexpr uint32_t kLine = 128, kChunkLines = 65536;  // 8 MiB chunks
static constexpr int kWarmups = 2, kMaxDiscards = 20, kMaxClaimRetries = 50;
static constexpr uint64_t kMiB = 1ull << 20, kTargetBytes = 8ull << 30;
static constexpr size_t kForceOneSmem = 120 * 1024;  // two blocks no longer fit in 228 KiB → 1 block/SM

enum Op { OP_LD = 0, OP_ST = 1, OP_MIX = 2 };
static const char* kOpName[] = {"ld", "st", "mix"};

__constant__ uint8_t c_part[256], c_gpc[256];

// ---------------------------------------------------------------- labelling kernels (one pinned warp)
__global__ void __launch_bounds__(32) touch_lines(char* buf, uint32_t first, uint32_t n, uint32_t target, uint32_t* claim) {
    if (!claim_target_sm(target, claim)) return;
    const uint32_t lane = threadIdx.x;
    for (uint32_t q = lane; q < n * 8; q += 32) {  // whole lines: 8 lanes × 16 B
        const uint64_t a = (uint64_t)buf + (uint64_t)(first + q / 8) * kLine + (q % 8) * 16;
        st_v4<C_WB>(a, 0u);
    }
}

__global__ void __launch_bounds__(32) probe_lines(const char* buf, uint32_t first, uint32_t n, uint32_t target, uint32_t* claim,
                                                  uint64_t zero, uint16_t* out) {
    if (!claim_target_sm(target, claim)) return;
    if (threadIdx.x != 0) return;
    uint64_t dep = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const uint64_t a = (uint64_t)buf + (uint64_t)(first + i) * kLine + (dep & zero);
        const uint64_t t0 = read_clock64();
        uint32_t v;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory");
        const uint64_t t1 = read_clock64_after(v, zero);
        out[i] = (uint16_t)(t1 - t0 > 65535 ? 65535 : t1 - t0);
        dep = v;
    }
}

// ---------------------------------------------------------------- streaming kernel
struct StreamArgs {
    char* buf;
    const uint32_t* list[2];  // LIST: line indices of group g
    uint32_t len[2];          // lines in group g's list
    uint32_t warps[2];        // warps of group g (expected)
    uint32_t reps;
    uint32_t by_partition;    // 0: one group (g = 0) of all blocks; 1: g = partition of the block's SM
    uint32_t reader_gpcs;     // bit g clear: blocks on SMs of GPC g exit at once
    uint32_t max_count;       // shared-memory list slots per warp (LIST)
    uint32_t* tickets;        // [2], zeroed before every launch
    BlockRec* rec;
    uint64_t zero;
};

__host__ __device__ inline void warp_range(uint32_t w, uint32_t W, uint32_t L, uint32_t step, uint32_t& start, uint32_t& count) {
    if (L == 0) {
        start = count = 0;
    } else if ((uint64_t)L >= (uint64_t)step * W) {
        start = (uint32_t)((uint64_t)w * L / W);
        count = (uint32_t)((uint64_t)(w + 1) * L / W) - start;
    } else {
        start = (uint32_t)((uint64_t)w * step % L);
        count = step;
    }
}

// One step of a warp: ILP accesses (lanes 8k…8k+7 cover line[i] in 16-B pieces), all issued before any loaded data is
// consumed; okm bit i = slot i is inside the warp's range.
template <int OP, int ILP, int C>
__device__ __forceinline__ void access_step(uint64_t base, const uint32_t (&line)[ILP], uint32_t okm, uint32_t r, uint32_t& acc) {
    V4 v[ILP];
#pragma unroll
    for (int i = 0; i < ILP; ++i) {
        if (!((okm >> i) & 1)) continue;
        const uint64_t ad = base + (uint64_t)line[i] * kLine;
        if (OP == OP_LD || (OP == OP_MIX && i % 2 == 0)) v[i] = ld_v4<C>(ad);  // folded after unrolling
        else st_v4<C_WB>(ad, r);
    }
#pragma unroll
    for (int i = 0; i < ILP; ++i)
        if (OP == OP_LD || (OP == OP_MIX && i % 2 == 0))
            if ((okm >> i) & 1) xor_into(acc, v[i]);
}

template <int OP, int ILP, bool LIST, int C>
__device__ __forceinline__ void stream_body(const StreamArgs& a) {
    extern __shared__ uint32_t s_idx[];
    __shared__ uint32_t s_ticket;
    __shared__ uint32_t s_lines;  // line accesses of the block (one 32-bit shared atomic per warp)
    constexpr uint32_t kStep = 4 * ILP;  // lines per warp step
    const uint32_t smid = read_smid(), part = c_part[smid];
    const uint32_t g = a.by_partition ? part : 0;
    const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32, nw = blockDim.x / 32;
    const bool active = (a.reader_gpcs >> c_gpc[smid]) & 1u;
    if (threadIdx.x == 0) {
        s_ticket = active ? atomicAdd(&a.tickets[g], 1u) : ~0u;
        s_lines = 0;
    }
    __syncthreads();
    if (!active) {
        if (threadIdx.x == 0) a.rec[blockIdx.x] = {0, 0, 0, 0, 0, smid, part, ~0u, 0};
        return;
    }
    const uint32_t W = a.warps[g], L = a.len[g];
    uint32_t wg = s_ticket * nw + warp;
    const uint32_t overflow = wg >= W;  // more blocks in this group than expected: placement check discards the launch
    wg %= W;
    uint32_t start, count;
    warp_range(wg, W, L, kStep, start, count);
    uint32_t* my = s_idx + warp * a.max_count;
    if constexpr (LIST) {
        const uint32_t padded = (count + kStep - 1) / kStep * kStep;
        for (uint32_t j = lane; j < padded; j += 32) {
            uint32_t pos = start + j;
            if (pos >= L) pos -= L;
            my[j] = j < count ? a.list[g][pos] : 0u;
        }
    }
    if (lane == 0) atomicAdd(&s_lines, count * a.reps);
    const uint32_t sub = (lane % 8) * 16, li = lane / 8;
    const uint64_t base = (uint64_t)a.buf + sub;
    uint32_t acc = 0;
    __syncthreads();
    TimerPair t0{};
    if (threadIdx.x == 0) t0 = read_timers_pair();
    if (count <= kStep) {
        // the warp's whole range is one step (every footprint up to 16 MiB mixed / 32 MiB local or remote with this
        // grid): its line indices stay in registers for all reps, and the timed loop contains only the accesses.
        // base + (r & zero) = base, but ptxas cannot prove it, so the rep-invariant loads stay inside the rep loop.
        uint32_t line[ILP], okm = 0;
#pragma unroll
        for (int i = 0; i < ILP; ++i) {
            const uint32_t j = 4 * i + li;
            okm |= (j < count) << i;
            if constexpr (LIST) line[i] = my[j];
            else { line[i] = start + j; if (line[i] >= L) line[i] -= L; }
        }
#pragma unroll 1
        for (uint32_t r = 0; r < a.reps; ++r) access_step<OP, ILP, C>(base + (r & a.zero), line, okm, r, acc);
    } else {
#pragma unroll 1
        for (uint32_t r = 0; r < a.reps; ++r) {
#pragma unroll 1
            for (uint32_t j0 = 0; j0 < count; j0 += kStep) {
                uint32_t line[ILP], okm = 0;  // 32-bit line indices first (the list slice is padded to whole steps)
#pragma unroll
                for (int i = 0; i < ILP; ++i) {
                    const uint32_t j = j0 + 4 * i + li;
                    okm |= (j < count) << i;
                    if constexpr (LIST) line[i] = my[j];
                    else { line[i] = start + j; if (line[i] >= L) line[i] -= L; }
                }
                access_step<OP, ILP, C>(base + (r & a.zero), line, okm, r, acc);
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const TimerPair t1 = read_timers_after(acc, a.zero);
        a.rec[blockIdx.x] = {t0.cycles, t1.cycles, t0.ns, t1.ns, (uint64_t)s_lines * kLine, smid, part, s_ticket, overflow};
    }
}

template <int OP, int ILP, bool LIST, int C>
__global__ void __launch_bounds__(1024, ILP >= 8 ? 1 : 2) k3_stream(StreamArgs a) { stream_body<OP, ILP, LIST, C>(a); }
template <int OP, int ILP, bool LIST, int C>
__global__ void __launch_bounds__(1024, ILP >= 8 ? 1 : 2) k3_ncu(StreamArgs a) { stream_body<OP, ILP, LIST, C>(a); }

// Plain grid-stride stream (no lists): thread gt reads elements gt + i·T + k·ILP·T (i < ILP) of n4 16-B elements.
template <int ILP>
__global__ void __launch_bounds__(1024, 2) k3_direct(const char* buf, uint64_t n4, uint32_t reps, BlockRec* rec, uint64_t zero) {
    __shared__ uint32_t s_units;  // 16-byte loads of the block over all reps
    const uint64_t T = (uint64_t)gridDim.x * blockDim.x, gt = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x == 0) s_units = 0;
    __syncthreads();
    uint32_t mine = 0;
    for (uint64_t b = gt; b < n4; b += ILP * T)
        for (int i = 0; i < ILP; ++i) mine += (b + i * T < n4);
    mine = __reduce_add_sync(0xffffffffu, mine * reps);
    if (threadIdx.x % 32 == 0) atomicAdd(&s_units, mine);
    uint32_t acc = 0;
    __syncthreads();
    TimerPair t0{};
    if (threadIdx.x == 0) t0 = read_timers_pair();
#pragma unroll 1
    for (uint32_t r = 0; r < reps; ++r) {
#pragma unroll 1
        for (uint64_t b = gt; b < n4; b += ILP * T) {
            V4 v[ILP];
#pragma unroll
            for (int i = 0; i < ILP; ++i)
                if (b + i * T < n4) v[i] = ld_v4<C_CG>((uint64_t)buf + (r & zero) + (b + i * T) * 16);
#pragma unroll
            for (int i = 0; i < ILP; ++i)
                if (b + i * T < n4) acc ^= xor4(v[i]);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const TimerPair t1 = read_timers_after(acc, zero);
        rec[blockIdx.x] = {t0.cycles, t1.cycles, t0.ns, t1.ns, (uint64_t)s_units * 16, read_smid(), c_part[read_smid()], blockIdx.x, 0};
    }
}

// ---------------------------------------------------------------- host
struct Labels {
    std::vector<int8_t> home;          // per line: 0, 1, −1 ambiguous
    std::vector<uint16_t> cyc_b, cyc_a;  // touch A probe B / touch B probe A
    std::vector<std::vector<uint16_t>> cyc_g;  // chunk 0, probe = first SM of GPC g
    double thr = 0, c_near = 0, c_far = 0;
};

struct Ctx {
    hb::Run& run;
    int sm_count = 0;
    std::vector<int> gpc, part;
    uint64_t buf_bytes = 0;
    uint32_t n_lines = 0;
    char* d_buf = nullptr;
    uint32_t* d_tickets = nullptr;
    BlockRec* d_rec = nullptr;
    uint32_t* d_list[2][2] = {};  // [list kind: 0 = home lists, 1 = identity][partition]
    uint32_t* d_ident = nullptr;
    uint32_t* d_home[2] = {};
    std::vector<uint32_t> home_lines[2];  // line indices homed in partition p, ascending
    hb::L2Flusher flush;
    CleanL2 clean;  // flush + read of a clean buffer (first-pass test)
    hb::SmClaim claim;
    cudaEvent_t ev0{}, ev1{};
    long long launch_idx = 0;
    explicit Ctx(hb::Run& r) : run(r) {}
};

static void touch(Ctx& x, uint32_t first, uint32_t n, uint32_t sm) {
    for (int t = 0; t < kMaxClaimRetries; ++t) {
        x.claim.reset();
        touch_lines<<<hb::SmClaim::blocks(x.sm_count), 32>>>(x.d_buf, first, n, sm, x.claim.ptr());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        if (x.claim.ok(sm)) return;
    }
    hb::die("touch: could not pin SM " + std::to_string(sm));
}
static std::vector<uint16_t> probe(Ctx& x, uint32_t first, uint32_t n, uint32_t sm, uint16_t* d_out) {
    for (int t = 0; t < kMaxClaimRetries; ++t) {
        x.claim.reset();
        probe_lines<<<hb::SmClaim::blocks(x.sm_count), 32>>>(x.d_buf, first, n, sm, x.claim.ptr(), 0, d_out);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        if (x.claim.ok(sm)) {
            std::vector<uint16_t> h(n);
            CUDA_CHECK(cudaMemcpy(h.data(), d_out, n * sizeof(uint16_t), cudaMemcpyDeviceToHost));
            return h;
        }
    }
    hb::die("probe: could not pin SM " + std::to_string(sm));
}

// Flush, touch [first, first+n) from SM `toucher`, time every line from SM `prober`.
static std::vector<uint16_t> touch_probe(Ctx& x, uint32_t first, uint32_t n, uint32_t toucher, uint32_t prober, uint16_t* d_out) {
    x.flush();
    touch(x, first, n, toucher);
    return probe(x, first, n, prober, d_out);
}

// 1-D 2-means on samples ≤ cap; returns the threshold (midpoint) and the centres.
static void two_means(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, double cap, double& lo, double& hi) {
    std::vector<uint64_t> hist(65536, 0);
    for (auto v : a) ++hist[v];
    for (auto v : b) ++hist[v];
    lo = 200; hi = 500;
    for (int it = 0; it < 100; ++it) {
        const double thr = (lo + hi) / 2;
        double s0 = 0, n0 = 0, s1 = 0, n1 = 0;
        for (int v = 0; v <= cap; ++v) {
            if (!hist[v]) continue;
            if (v <= thr) { s0 += double(v) * hist[v]; n0 += hist[v]; } else { s1 += double(v) * hist[v]; n1 += hist[v]; }
        }
        const double nlo = n0 ? s0 / n0 : lo, nhi = n1 ? s1 / n1 : hi;
        if (std::fabs(nlo - lo) < 1e-6 && std::fabs(nhi - hi) < 1e-6) break;
        lo = nlo; hi = nhi;
    }
}

static Labels label_homes(Ctx& x, bool write_csv) {
    Labels lb;
    const uint32_t A = (uint32_t)first_sm_of_gpc(x.gpc, 0), B = (uint32_t)first_sm_of_gpc(x.gpc, 1);
    if (x.part[A] != 0 || x.part[B] != 1) hb::die("labelling SMs are not in partitions 0/1");
    uint16_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, kChunkLines * sizeof(uint16_t)));
    lb.cyc_b.resize(x.n_lines);
    lb.cyc_a.resize(x.n_lines);
    x.run.lock_clocks_note("label", "before");
    for (uint32_t c0 = 0; c0 < x.n_lines; c0 += kChunkLines) {
        const auto b = touch_probe(x, c0, kChunkLines, A, B, d_out);
        const auto a = touch_probe(x, c0, kChunkLines, B, A, d_out);
        std::copy(b.begin(), b.end(), lb.cyc_b.begin() + c0);
        std::copy(a.begin(), a.end(), lb.cyc_a.begin() + c0);
    }
    lb.cyc_g.resize(8);
    for (int g = 0; g < 8; ++g) {
        const uint32_t p = (uint32_t)first_sm_of_gpc(x.gpc, g);
        lb.cyc_g[g] = touch_probe(x, 0, kChunkLines, kPartitionOfGpc[g] == 0 ? B : A, p, d_out);
    }
    x.run.lock_clocks_note("label", "after");
    CUDA_CHECK(cudaFree(d_out));
    two_means(lb.cyc_b, lb.cyc_a, 1500, lb.c_near, lb.c_far);
    lb.thr = (lb.c_near + lb.c_far) / 2;
    lb.home.resize(x.n_lines);
    uint64_t n0 = 0, n1 = 0, amb = 0;
    for (uint32_t i = 0; i < x.n_lines; ++i) {
        const bool far_b = lb.cyc_b[i] > lb.thr, far_a = lb.cyc_a[i] > lb.thr;
        lb.home[i] = (far_b && !far_a) ? 0 : (!far_b && far_a) ? 1 : -1;
        n0 += lb.home[i] == 0; n1 += lb.home[i] == 1; amb += lb.home[i] < 0;
    }
    std::printf("labels: near %.1f far %.1f thr %.1f cycles; home0 %llu home1 %llu ambiguous %llu of %u lines\n", lb.c_near, lb.c_far,
                lb.thr, (unsigned long long)n0, (unsigned long long)n1, (unsigned long long)amb, x.n_lines);
    for (int g = 0; g < 8; ++g) {  // a probe in partition q sees far ⇔ home ≠ q
        uint64_t agree = 0, n = 0;
        for (uint32_t i = 0; i < kChunkLines; ++i) {
            if (lb.home[i] < 0) continue;
            const bool far = lb.cyc_g[g][i] > lb.thr;
            agree += far == (lb.home[i] != kPartitionOfGpc[g]);
            ++n;
        }
        std::printf("  GPC %d (partition %d): %.4f %% of chunk-0 lines agree with the labels\n", g, kPartitionOfGpc[g], 100.0 * agree / n);
    }
    for (uint32_t unit : {4u, 32u}) {  // label uniformity within 512 B and 4 KiB units
        uint64_t mixed = 0, units = 0;
        for (uint32_t u = 0; u + unit <= x.n_lines; u += unit) {
            bool h0 = false, h1 = false;
            for (uint32_t i = u; i < u + unit; ++i) { h0 |= lb.home[i] == 0; h1 |= lb.home[i] == 1; }
            mixed += h0 && h1;
            ++units;
        }
        std::printf("  %u-B units with lines of both homes: %llu of %llu\n", unit * kLine, (unsigned long long)mixed, (unsigned long long)units);
    }
    if (write_csv) {
        hb::CsvWriter csv(hb::Run::results("05__bandwidth__l2_home.csv"),
                          hb::with_mandatory({"line_offset", "cycles_b", "cycles_a", "home", "cycles_g0", "cycles_g1", "cycles_g2", "cycles_g3",
                                              "cycles_g4", "cycles_g5", "cycles_g6", "cycles_g7", "toucher_a", "toucher_b", "thr_cycles"}));
        const hb::ClockSample cs = x.run.dev().sample();
        const std::string pre = x.run.prefix(-1, cs);
        for (uint32_t i = 0; i < x.n_lines; ++i) {
            char r[512];
            int k = std::snprintf(r, sizeof r, "%s,%llu,%u,%u,%d", pre.c_str(), (unsigned long long)i * kLine, lb.cyc_b[i], lb.cyc_a[i], lb.home[i]);
            for (int g = 0; g < 8; ++g)
                k += i < kChunkLines ? std::snprintf(r + k, sizeof r - k, ",%u", lb.cyc_g[g][i]) : std::snprintf(r + k, sizeof r - k, ",");
            std::snprintf(r + k, sizeof r - k, ",%u,%u,%.1f", A, B, lb.thr);
            csv.add(r);
            if (i % 65536 == 65535) csv.flush();
        }
        csv.sync();
    }
    return lb;
}

// ---------------------------------------------------------------- one streaming configuration
enum Variant { V_MIXED = 0, V_LOCAL = 1, V_REMOTE = 2, V_MIXED_LIST = 3, V_DIRECT = 4 };
static const char* kVariantName[] = {"mixed", "local", "remote", "mixed_list", "mixed_direct"};

struct Cfg {
    int variant, op, ilp;
    uint64_t footprint;
    int blocks, threads;
    bool one_per_sm;
    int reader_mask = 3;
    uint32_t reps = 0;  // 0: choose for kTargetBytes
    int pass = 0;       // 0 steady; 1/2 first-pass test
    int cache = C_CG;   // load cache operator (stores: always st.global)
    int gpc_mask = -1;  // ≥ 0: GPCs whose blocks work (link mode); −1: the GPCs of the partitions in reader_mask
};

static uint32_t gpcs_of(const Cfg& c) {
    if (c.gpc_mask >= 0) return (uint32_t)c.gpc_mask;
    uint32_t m = 0;
    for (int g = 0; g < 8; ++g) m |= ((c.reader_mask >> kPartitionOfGpc[g]) & 1u) << g;
    return m;
}

using StreamFn = void (*)(StreamArgs);
template <int OP, bool LIST, bool NCU>
static StreamFn pick_ilp(int ilp) {
    switch (ilp) {
        case 1: return NCU ? k3_ncu<OP, 1, LIST, C_CG> : k3_stream<OP, 1, LIST, C_CG>;
        case 2: return NCU ? k3_ncu<OP, 2, LIST, C_CG> : k3_stream<OP, 2, LIST, C_CG>;
        case 4: return NCU ? k3_ncu<OP, 4, LIST, C_CG> : k3_stream<OP, 4, LIST, C_CG>;
        case 8: return NCU ? k3_ncu<OP, 8, LIST, C_CG> : k3_stream<OP, 8, LIST, C_CG>;
    }
    hb::die("bad ilp");
}
// Load cache operators other than .cg: loads only, ILP 4.
template <bool LIST, bool NCU>
static StreamFn pick_cache(int cache) {
    switch (cache) {
        case C_CS: return NCU ? k3_ncu<OP_LD, 4, LIST, C_CS> : k3_stream<OP_LD, 4, LIST, C_CS>;
        case C_LU: return NCU ? k3_ncu<OP_LD, 4, LIST, C_LU> : k3_stream<OP_LD, 4, LIST, C_LU>;
        case C_CV: return NCU ? k3_ncu<OP_LD, 4, LIST, C_CV> : k3_stream<OP_LD, 4, LIST, C_CV>;
    }
    hb::die("bad cache operator");
}
template <bool NCU>
static StreamFn pick(int op, bool list, int ilp, int cache) {
    if (cache != C_CG) {
        if (op != OP_LD || ilp != 4) hb::die("cache operators other than cg: loads with ILP 4 only");
        return list ? pick_cache<true, NCU>(cache) : pick_cache<false, NCU>(cache);
    }
    if (op == OP_LD) return list ? pick_ilp<OP_LD, true, NCU>(ilp) : pick_ilp<OP_LD, false, NCU>(ilp);
    if (op == OP_ST) return list ? pick_ilp<OP_ST, true, NCU>(ilp) : pick_ilp<OP_ST, false, NCU>(ilp);
    return list ? pick_ilp<OP_MIX, true, NCU>(ilp) : pick_ilp<OP_MIX, false, NCU>(ilp);
}

struct Prepared {
    StreamArgs args{};
    StreamFn fn = nullptr;
    size_t smem = 0;
    uint64_t bytes_per_rep = 0;
    bool direct = false;
    uint64_t n4 = 0;
};

static uint32_t lines_in_prefix(const std::vector<uint32_t>& v, uint32_t n) {
    return (uint32_t)(std::lower_bound(v.begin(), v.end(), n) - v.begin());
}

template <bool NCU = false>
static Prepared prepare(Ctx& x, Cfg& c) {
    Prepared p;
    const uint32_t nF = (uint32_t)(c.footprint / kLine);
    if (nF > x.n_lines) hb::die("footprint beyond buffer");
    const uint32_t blocks_per_sm = c.one_per_sm ? 1 : 2, wpb = c.threads / 32;
    if (c.variant == V_DIRECT) {
        p.direct = true;
        p.n4 = c.footprint / 16;
        if (!c.reps) c.reps = 2048;
        p.bytes_per_rep = c.footprint;
        return p;
    }
    const bool list = c.variant != V_MIXED;
    StreamArgs& a = p.args;
    a.buf = x.d_buf;
    a.tickets = x.d_tickets;
    a.rec = x.d_rec;
    a.zero = 0;
    const uint32_t gm = gpcs_of(c);
    a.reader_gpcs = gm;
    std::vector<int> active_sms(2, 0);  // SMs of working GPCs, per partition
    for (int s = 0; s < x.sm_count; ++s) active_sms[x.part[s]] += (gm >> x.gpc[s]) & 1;
    const uint32_t step = 4 * c.ilp;
    if (c.variant == V_MIXED || c.variant == V_MIXED_LIST) {
        a.by_partition = 0;
        a.len[0] = a.len[1] = nF;
        int active_blocks = 0;  // blocks of working GPCs
        if (gm == 0xFF) active_blocks = c.blocks;
        else active_blocks = (active_sms[0] + active_sms[1]) * blocks_per_sm;
        a.warps[0] = a.warps[1] = std::max<uint32_t>(1, active_blocks * wpb);  // ≥ 1: an all-exit baseline launch has no active warps
        a.list[0] = a.list[1] = x.d_ident;
    } else {
        a.by_partition = 1;
        for (int q = 0; q < 2; ++q) {
            const int src = c.variant == V_LOCAL ? q : 1 - q;
            a.list[q] = x.d_home[src];
            a.len[q] = lines_in_prefix(x.home_lines[src], nF);
            a.warps[q] = std::max<uint32_t>(1, active_sms[q] * blocks_per_sm * wpb);
        }
    }
    uint32_t maxc = 0;
    for (int q = 0; q < 2; ++q) {
        if (a.by_partition && active_sms[q] == 0) continue;
        const int g = a.by_partition ? q : 0;
        for (uint32_t w = 0; w < a.warps[g]; ++w) {
            uint32_t s, n;
            warp_range(w, a.warps[g], a.len[g], step, s, n);
            maxc = std::max(maxc, n);
            p.bytes_per_rep += (uint64_t)n * kLine;
        }
        if (!a.by_partition) break;
    }
    a.max_count = (maxc + step - 1) / step * step;
    p.smem = list ? (size_t)a.max_count * wpb * sizeof(uint32_t) : 0;
    if (p.smem > 100 * 1024) hb::die("list slice too large for shared memory");
    if (c.one_per_sm) p.smem = std::max(p.smem, kForceOneSmem);
    if (!c.reps) c.reps = (uint32_t)std::max<uint64_t>(1, (uint64_t)std::llround(double(kTargetBytes) / p.bytes_per_rep));
    a.reps = c.reps;
    p.fn = pick<NCU>(c.op, list, c.ilp, c.cache);
    if (p.smem > 48 * 1024) CUDA_CHECK(cudaFuncSetAttribute(p.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)p.smem));
    return p;
}

static void launch(Ctx& x, const Cfg& c, const Prepared& p, bool timed) {
    CUDA_CHECK(cudaMemsetAsync(x.d_tickets, 0, 2 * sizeof(uint32_t)));
    if (timed) CUDA_CHECK(cudaEventRecord(x.ev0));
    if (p.direct) k3_direct<4><<<c.blocks, c.threads>>>(x.d_buf, p.n4, c.reps, x.d_rec, 0);
    else p.fn<<<c.blocks, c.threads, p.smem>>>(p.args);
    CUDA_CHECK(cudaGetLastError());
    if (timed) CUDA_CHECK(cudaEventRecord(x.ev1));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// k3 mode: 05__bandwidth__l2{,_blocks}.csv with reader_mask = partition bitmask (1, 2, 3);
// link mode: 05__bandwidth__l2_link{,_blocks}.csv with reader_gpc_mask = GPC bitmask (bit g = GPC g works).
struct Writers {
    bool link;
    hb::CsvWriter launches, blocks;
    explicit Writers(bool link_mode = false)
        : link(link_mode),
          launches(hb::Run::results(link_mode ? "05__bandwidth__l2_link.csv" : "05__bandwidth__l2.csv"),
                   hb::with_mandatory({"variant", "op", "cache_op", "footprint_bytes", "blocks", "threads", "ilp", "reps", "pass",
                                       link_mode ? "reader_gpc_mask" : "reader_mask", "bytes",
                                       "ms_event", "tbps", "sm_mhz_blocks", "bpc_chip", "span_ns", "tbps_span", "discard"})),
          blocks(hb::Run::results(link_mode ? "05__bandwidth__l2_link_blocks.csv" : "05__bandwidth__l2_blocks.csv"),
                 hb::with_mandatory({"variant", "op", "cache_op", "footprint_bytes", "pass", "block", "smid", "gpc", "partition", "ticket", "bytes", "cycles",
                                     "ns", "start_ns", "bpc_sm", "discard"})) {}
};

static std::string cfg_label(const Cfg& c) {
    char b[200];
    std::snprintf(b, sizeof b, "l2_%s_%s_%s_F%lluM_b%d_t%d_ilp%d_m%d_g%02x_p%d", kVariantName[c.variant], kOpName[c.op], cache_name(c.cache),
                  (unsigned long long)(c.footprint / kMiB), c.blocks, c.threads, c.ilp, c.reader_mask, gpcs_of(c), c.pass);
    return b;
}

// After a timed launch: check placement, write the launch and block rows. Returns "" if valid, else the discard reason.
static std::string record(Ctx& x, Writers& w, const Cfg& c, const std::string& label) {
    const hb::ClockSample cs = x.run.clock_note(label, "launch", x.launch_idx);
    std::vector<BlockRec> r(c.blocks);
    CUDA_CHECK(cudaMemcpy(r.data(), x.d_rec, c.blocks * sizeof(BlockRec), cudaMemcpyDeviceToHost));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, x.ev0, x.ev1));
    const PlacementCheck pc(r, x.sm_count);
    bool overflow = false;
    for (const auto& b : r) overflow |= b.flags != 0;
    const bool place_ok = !overflow && (c.one_per_sm ? pc.at_most_one() : pc.exactly(2));
    const std::string discard = !place_ok ? "placement" : cs.bad() ? "clock" : "";
    uint64_t bytes = 0, t_lo = ~0ull, t_hi = 0;
    std::vector<double> mhz;
    for (const auto& b : r) {
        if (!b.bytes) continue;
        bytes += b.bytes;
        t_lo = std::min(t_lo, b.n0);
        t_hi = std::max(t_hi, b.n1);
        if (b.n1 > b.n0) mhz.push_back(1e3 * double(b.c1 - b.c0) / double(b.n1 - b.n0));
    }
    std::sort(mhz.begin(), mhz.end());
    const double sm_mhz = mhz.empty() ? 0 : mhz[mhz.size() / 2];
    const double tbps = bytes / (ms * 1e-3) / 1e12, span = double(t_hi - t_lo);
    const std::string pre = x.run.prefix(x.launch_idx, cs);
    char row[768];
    std::snprintf(row, sizeof row, "%s,%s,%s,%s,%llu,%d,%d,%d,%u,%d,%d,%llu,%.6f,%.6f,%.2f,%.4f,%.0f,%.6f,%s", pre.c_str(), kVariantName[c.variant],
                  kOpName[c.op], cache_name(c.cache), (unsigned long long)c.footprint, c.blocks, c.threads, c.ilp, c.reps, c.pass,
                  w.link ? (int)gpcs_of(c) : c.reader_mask, (unsigned long long)bytes, ms, tbps, sm_mhz, sm_mhz > 0 ? tbps * 1e12 / (sm_mhz * 1e6) : 0.0, span,
                  span > 0 ? bytes / span / 1e3 : 0.0, discard.c_str());
    w.launches.add(row);
    for (int i = 0; i < c.blocks; ++i) {
        const auto& b = r[i];
        const uint64_t cyc = b.c1 - b.c0;
        std::snprintf(row, sizeof row, "%s,%s,%s,%s,%llu,%d,%d,%u,%d,%u,%d,%llu,%llu,%llu,%lld,%.4f,%s", pre.c_str(), kVariantName[c.variant], kOpName[c.op],
                      cache_name(c.cache), (unsigned long long)c.footprint, c.pass, i, b.smid, b.smid < (uint32_t)x.sm_count ? x.gpc[b.smid] : -1, b.part,
                      b.ticket == ~0u ? -1 : (int)b.ticket, (unsigned long long)b.bytes, (unsigned long long)cyc, (unsigned long long)(b.n1 - b.n0),
                      b.bytes ? (long long)(b.n0 - t_lo) : -1ll, (b.bytes && cyc) ? double(b.bytes) / cyc : 0.0, discard.c_str());
        w.blocks.add(row);
    }
    w.launches.flush();
    w.blocks.flush();
    ++x.launch_idx;
    return discard;
}

static double steady(Ctx& x, Writers& w, Cfg c, int recorded) {
    Prepared p = prepare(x, c);
    const std::string label = cfg_label(c);
    for (int i = 0; i < kWarmups; ++i) launch(x, c, p, false);
    int valid = 0, discards = 0;
    std::vector<double> tb;
    while (valid < recorded) {
        launch(x, c, p, true);
        const std::string d = record(x, w, c, label);
        if (!d.empty()) {
            if (++discards > kMaxDiscards) hb::die(label + ": too many discards");
            continue;
        }
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, x.ev0, x.ev1));
        tb.push_back(double(p.bytes_per_rep) * c.reps / (ms * 1e-3) / 1e12);
        ++valid;
    }
    w.launches.sync();
    w.blocks.sync();
    std::sort(tb.begin(), tb.end());
    std::printf("%-52s reps %5u  %7.3f TB/s%s\n", label.c_str(), c.reps, tb[tb.size() / 2], discards ? "  (discards)" : "");
    return tb[tb.size() / 2];
}

// First-pass test: flush, home touch, pass 1 and pass 2, both recorded.
static void first_pass(Ctx& x, Writers& w, Cfg c, int recorded) {
    Cfg touch_cfg{V_LOCAL, OP_LD, 4, c.footprint, 2 * x.sm_count, 1024, false, 3, 1, 0};
    Prepared pt = prepare(x, touch_cfg);
    c.reps = 1;
    Prepared p = prepare(x, c);
    int valid = 0, discards = 0;
    std::vector<double> t1, t2;
    for (int i = 0; i < kWarmups; ++i) { x.clean(); launch(x, touch_cfg, pt, false); launch(x, c, p, false); }
    while (valid < recorded) {
        x.clean();
        launch(x, touch_cfg, pt, false);
        Cfg c1 = c, c2 = c;
        c1.pass = 1;
        c2.pass = 2;
        launch(x, c1, p, true);
        const std::string d1 = record(x, w, c1, cfg_label(c1));
        launch(x, c2, p, true);
        const std::string d2 = record(x, w, c2, cfg_label(c2));
        if (!d1.empty() || !d2.empty()) {
            if (++discards > kMaxDiscards) hb::die(cfg_label(c) + ": too many discards");
            continue;
        }
        ++valid;
    }
    w.launches.sync();
    w.blocks.sync();
    std::printf("%-52s first-pass test: %d launches%s\n", cfg_label(c).c_str(), recorded, discards ? " (discards)" : "");
}

static void setup(Ctx& x, bool quick) {
    x.sm_count = x.run.dev().sm_count;
    x.gpc = load_gpc_map(x.sm_count);
    x.part = partition_of_sm(x.gpc);
    std::vector<uint8_t> cp(256, 0);
    for (int s = 0; s < x.sm_count; ++s) cp[s] = (uint8_t)x.part[s];
    CUDA_CHECK(cudaMemcpyToSymbol(c_part, cp.data(), 256));
    for (int s = 0; s < x.sm_count; ++s) cp[s] = (uint8_t)x.gpc[s];
    CUDA_CHECK(cudaMemcpyToSymbol(c_gpc, cp.data(), 256));
    x.buf_bytes = (quick ? 32 : 96) * kMiB;
    x.n_lines = (uint32_t)(x.buf_bytes / kLine);
    CUDA_CHECK(cudaMalloc(&x.d_buf, x.buf_bytes));
    CUDA_CHECK(cudaMemset(x.d_buf, 0, x.buf_bytes));
    CUDA_CHECK(cudaMalloc(&x.d_tickets, 2 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&x.d_rec, 4 * x.sm_count * sizeof(BlockRec)));
    CUDA_CHECK(cudaEventCreate(&x.ev0));
    CUDA_CHECK(cudaEventCreate(&x.ev1));
    std::vector<uint32_t> ident(x.n_lines);
    std::iota(ident.begin(), ident.end(), 0u);
    CUDA_CHECK(cudaMalloc(&x.d_ident, x.n_lines * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(x.d_ident, ident.data(), x.n_lines * sizeof(uint32_t), cudaMemcpyHostToDevice));
}

static void upload_lists(Ctx& x, const Labels& lb) {
    for (int q = 0; q < 2; ++q) {
        x.home_lines[q].clear();
        for (uint32_t i = 0; i < x.n_lines; ++i)
            if (lb.home[i] == q) x.home_lines[q].push_back(i);
        CUDA_CHECK(cudaMalloc(&x.d_home[q], std::max<size_t>(1, x.home_lines[q].size()) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(x.d_home[q], x.home_lines[q].data(), x.home_lines[q].size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
}

static void run_k3(Ctx& x, bool quick) {
    const Labels lb = label_homes(x, true);
    upload_lists(x, lb);
    Writers w;
    const int R = quick ? 3 : 10, RF = quick ? 3 : 20;
    const int B2 = 2 * x.sm_count;
    const std::vector<uint64_t> F = quick ? std::vector<uint64_t>{4, 16, 28}
                                          : std::vector<uint64_t>{1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 96};
    x.run.lock_clocks_note("k3", "before");
    // list check and ILP
    steady(x, w, Cfg{V_DIRECT, OP_LD, 4, 16 * kMiB, B2, 1024, false}, R);
    steady(x, w, Cfg{V_MIXED_LIST, OP_LD, 4, 16 * kMiB, B2, 1024, false}, R);
    for (int ilp : {1, 2, 4}) steady(x, w, Cfg{V_MIXED, OP_LD, ilp, 16 * kMiB, B2, 1024, false}, R);
    for (int ilp : {1, 2, 4, 8}) steady(x, w, Cfg{V_MIXED, OP_LD, ilp, 16 * kMiB, x.sm_count, 1024, true}, R);
    // footprint sweep
    for (uint64_t f : F) {
        for (int v : {V_MIXED, V_LOCAL, V_REMOTE}) steady(x, w, Cfg{v, OP_LD, 4, f * kMiB, B2, 1024, false}, R);
        steady(x, w, Cfg{V_MIXED, OP_ST, 4, f * kMiB, B2, 1024, false}, R);
        steady(x, w, Cfg{V_MIXED, OP_MIX, 4, f * kMiB, B2, 1024, false}, R);
    }
    for (int op : {OP_ST, OP_MIX})
        for (int v : {V_LOCAL, V_REMOTE}) steady(x, w, Cfg{v, op, 4, 16 * kMiB, B2, 1024, false}, R);
    // load cache operators other than .cg (do any of them avoid the copy in the reader's partition?)
    for (int cache : {C_CS, C_LU, C_CV})
        for (int v : {V_MIXED, V_LOCAL, V_REMOTE}) {
            Cfg c{v, OP_LD, 4, 16 * kMiB, B2, 1024, false};
            c.cache = cache;
            steady(x, w, c, R);
        }
    // active SMs (1 block of 1024 threads per SM)
    const std::vector<int> act = quick ? std::vector<int>{1, 8, 66, 132}
                                       : std::vector<int>{1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 80, 96, 112, 128, 132};
    for (int n : act) {
        Cfg c{V_MIXED, OP_LD, 4, 16 * kMiB, n, 1024, true};
        c.reps = (uint32_t)std::max<uint64_t>(1, std::min<uint64_t>(kTargetBytes, (uint64_t)n * (256ull << 20)) / (16 * kMiB));
        steady(x, w, c, R);
    }
    // first-pass test: 1 block/SM of 256 or 512 threads (each warp makes ≈ 12 or 6 steps through its range, so the pass
    // reaches a steady flow) and, for reference, 2 blocks/SM of 1024 threads (1–2 steps per warp)
    for (int thr : {256, 512, 1024}) {
        const bool one = thr != 1024;
        const int blocks = one ? x.sm_count : B2;
        for (uint64_t f : quick ? std::vector<uint64_t>{16} : std::vector<uint64_t>{8, 16, 24})
            for (int v : {V_LOCAL, V_REMOTE, V_MIXED}) first_pass(x, w, Cfg{v, OP_LD, 4, f * kMiB, blocks, thr, one}, RF);
        const uint64_t f1 = quick ? 24 : 48;
        for (int r : {1, 2})
            for (int v : {V_LOCAL, V_REMOTE}) first_pass(x, w, Cfg{v, OP_LD, 4, f1 * kMiB, blocks, thr, one, r}, RF);
    }
    x.run.lock_clocks_note("k3", "after");
}

// Partition link: the same SMs, reading or writing data homed in their own partition (local) or in the other one
// (remote), for growing sets of GPCs of one partition, then for both partitions.
static void run_link(Ctx& x, bool quick) {
    const Labels lb = label_homes(x, true);
    upload_lists(x, lb);
    Writers w(true);
    const int R = quick ? 3 : 10, RF = quick ? 3 : 20, B2 = 2 * x.sm_count;
    // GPC sets (bit g = GPC g): partition 1 = G1, G2, G3, G7; partition 0 = G4, G5, G6, G0; both partitions
    const std::vector<int> masks = quick ? std::vector<int>{0x02, 0x8E, 0xFF}
                                         : std::vector<int>{0x02, 0x06, 0x0E, 0x8E, 0x10, 0x30, 0x70, 0x71, 0xFF};
    x.run.lock_clocks_note("link", "before");
    for (int m : masks) {
        for (int op : {OP_ST, OP_LD})
            for (int v : {V_LOCAL, V_REMOTE}) {
                if (op == OP_LD && v == V_REMOTE) continue;  // steady remote loads are served by near copies (k3 mode)
                Cfg c{v, op, 4, 16 * kMiB, B2, 1024, false};
                c.gpc_mask = m;
                steady(x, w, c, R);
            }
        for (int v : {V_LOCAL, V_REMOTE}) {
            Cfg c{v, OP_LD, 4, (quick ? 24 : 48) * kMiB, x.sm_count, 512, true};
            c.gpc_mask = m;
            first_pass(x, w, c, RF);
        }
    }
    x.run.lock_clocks_note("link", "after");
}

static void run_ncu(Ctx& x, bool quick) {
    const Labels lb = label_homes(x, false);
    upload_lists(x, lb);
    FILE* man = std::fopen("bin/ncu_manifest_l2.csv", "w");
    if (!man) hb::die("cannot write bin/ncu_manifest_l2.csv");
    std::fprintf(man, "%s,variant,op,cache_op,footprint_bytes,blocks,threads,ilp,reps,pass,reader_mask,bytes,ms_event,tbps\n", hb::join(hb::mandatory_cols()).c_str());
    const int B2 = 2 * x.sm_count;
    auto profiled = [&](Cfg c, const Prepared& pn, uint64_t bytes) {
        launch(x, c, pn, true);
        const hb::ClockSample cs = x.run.clock_note("ncu_" + cfg_label(c), "launch", x.launch_idx);
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, x.ev0, x.ev1));
        std::fprintf(man, "%s,%s,%s,%s,%llu,%d,%d,%d,%u,%d,%d,%llu,%.6f,%.6f\n", x.run.prefix(x.launch_idx, cs).c_str(), kVariantName[c.variant],
                     kOpName[c.op], cache_name(c.cache), (unsigned long long)c.footprint, c.blocks, c.threads, c.ilp, c.reps, c.pass, c.reader_mask,
                     (unsigned long long)bytes, ms, bytes / (ms * 1e-3) / 1e12);
        ++x.launch_idx;
    };
    x.run.lock_clocks_note("ncu", "before");
    const std::vector<uint64_t> F = quick ? std::vector<uint64_t>{16} : std::vector<uint64_t>{1, 4, 16, 48};
    std::vector<Cfg> st;
    for (uint64_t f : F)
        for (int v : {V_MIXED, V_LOCAL, V_REMOTE}) st.push_back(Cfg{v, OP_LD, 4, f * kMiB, B2, 1024, false});
    st.push_back(Cfg{V_MIXED, OP_ST, 4, 16 * kMiB, B2, 1024, false});
    st.push_back(Cfg{V_MIXED, OP_MIX, 4, 16 * kMiB, B2, 1024, false});
    st.push_back(Cfg{V_REMOTE, OP_ST, 4, 16 * kMiB, B2, 1024, false});
    for (int cache : {C_CS, C_LU, C_CV})
        for (int v : {V_MIXED, V_LOCAL, V_REMOTE}) {
            Cfg c{v, OP_LD, 4, 16 * kMiB, B2, 1024, false};
            c.cache = cache;
            st.push_back(c);
        }
    {  // profiler baseline: every block exits at once (reader_mask 0), so the counters show only what ncu itself adds
        Cfg c{V_MIXED, OP_LD, 4, 16 * kMiB, B2, 1024, false, 0, 1};
        Prepared pn = prepare<true>(x, c);
        profiled(c, pn, 0);
    }
    for (Cfg c : st) {
        Prepared p = prepare<false>(x, c);
        Prepared pn = prepare<true>(x, c);
        for (int i = 0; i < kWarmups; ++i) launch(x, c, p, false);
        profiled(c, pn, p.bytes_per_rep * c.reps);
    }
    std::vector<Cfg> fp;
    for (int v : {V_LOCAL, V_REMOTE, V_MIXED}) fp.push_back(Cfg{v, OP_LD, 4, 24 * kMiB, B2, 1024, false});
    for (int v : {V_LOCAL, V_REMOTE}) fp.push_back(Cfg{v, OP_LD, 4, (quick ? 24 : 48) * kMiB, B2, 1024, false, 1});
    for (int v : {V_LOCAL, V_REMOTE}) fp.push_back(Cfg{v, OP_LD, 4, 24 * kMiB, x.sm_count, 256, true});
    for (Cfg c : fp) {
        Cfg touch_cfg{V_LOCAL, OP_LD, 4, c.footprint, B2, 1024, false, 3, 1, 0};
        Prepared pt = prepare(x, touch_cfg);
        c.reps = 1;
        Prepared pn = prepare<true>(x, c);
        x.clean();
        launch(x, touch_cfg, pt, false);
        Cfg c1 = c, c2 = c;
        c1.pass = 1;
        c2.pass = 2;
        profiled(c1, pn, pn.bytes_per_rep);
        profiled(c2, pn, pn.bytes_per_rep);
    }
    std::fclose(man);
    x.run.lock_clocks_note("ncu", "after");
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "k3";
    const bool quick = hb::has_flag(argc, argv, "--quick");
    hb::Run run(EXP, argc, argv);
    const std::string label = hb::arg_str(argc, argv, "--label", mode.c_str());
    Ctx x(run);
    if (mode == "clockcheck") {
        run.lock_clocks_note(label, "check");
    } else {
        setup(x, quick);
        if (mode == "k3") run_k3(x, quick);
        else if (mode == "link") run_link(x, quick);
        else if (mode == "ncu") run_ncu(x, quick);
        else hb::die("unknown mode " + mode);
    }
    run.complete();
    return 0;
}
