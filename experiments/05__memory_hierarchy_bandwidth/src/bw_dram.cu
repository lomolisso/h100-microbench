/*
 * bw_dram.cu — experiment 05, K4 + K5 (HBM part): HBM read, write and read/write-mix bandwidth GPU-wide, its
 * dependence on the cache operator and on the bytes in flight, and the bandwidth versus the number of active SMs.
 *
 * Usage: bin/bw_dram {k4|ncu|clockcheck} [--quick] [--label L]. One invocation = one run_id.
 *
 * Buffers: in and out, 2 GiB each (n4 = 2^27 16-byte elements; 512 MiB each with --quick), far larger than the 60 MiB L2.
 * Kernel k4<OP, C, ILP>(in, out, n4, rec, zero), grid-stride over 16-byte elements, T_total threads, gt = global thread:
 *   read   elements gt + i·T_total + k·ILP·T_total (i < ILP) of `in`: ld.global.{cg|cs}.v4.f32, all ILP loads of a step
 *          issued before their data is XORed (volatile xor, see bw_common.hpp).
 *   write  the same elements of `out`: st.global.{cg|cs}.v4.f32.
 *   mix    5 reads + 1 write per thread and step: groups g < G = n4/5: reads in[g + k·G] (k < 5), writes out[g].
 *   mix11  1 read + 1 write: g < n4/2, ILP elements per step: reads in[g…], then writes out[g…] (50/50 by bytes); the
 *          stored value is the element index, so the stores issue right behind the loads (two independent streams).
 *   Timed region per block: bar.sync → thread 0 %clock64/%globaltimer → loop → bar.sync → timers after the block's
 *   last loaded value. GPU-wide time: cudaEvent pair around the launch. bytes = elements read + written × 16 B.
 *   Before every launch (warm-ups included) the L2 is flushed and left holding only clean lines (CleanL2,
 *   bw_common.hpp), so no write-back of earlier data overlaps the timed kernel.
 * Configurations (2 warm-up + 10 recorded launches):
 *   full grid  4 × 132 blocks × 1024 threads (2 resident per SM, 2 waves): op ∈ {read, write, mix, mix11} × cache
 *              operator ∈ {cg, cs}, ILP 4; read.cg also with ILP ∈ {1, 2, 8} (launch bounds 1024 × 1 for ILP 8).
 *   active SMs n ∈ 1…132 blocks × 1024 threads, ILP 4 (64 KiB in flight per block), 120 KiB dynamic smem so that at
 *              most one block fits per SM (host check; ld/st.cg bypass L1, so the carveout does not matter);
 *              read.cg and write.cg, 5 recorded launches each.
 *   A launch is discarded and repeated if its NVML sample shows a throttle reason or a clock below the lock, or (active
 *   SMs) if two blocks shared an SM.
 *   → results/05__bandwidth__dram.csv (one row per launch): op, cache_op, ilp, blocks, threads, active_sms (distinct %smid),
 *     bytes, ms_event, tbps, gbps_per_sm (= tbps / active_sms), sm_mhz_blocks, span_ns, discard.
 *   → results/05__bandwidth__dram_blocks.csv (one row per block and launch): op, cache_op, ilp, blocks, block, smid, gpc,
 *     partition, bytes, cycles, ns, start_ns, bpc_sm, discard.
 * ncu mode: one launch of each full-grid op × cache operator (ILP 4), profiled by src/run.sh with kernel replay and
 *   --cache-control all (every pass starts with a flushed L2, like the timed runs). Writes bin/ncu_manifest_dram.csv.
 * diag mode (diagnostics, console only): occupancy and registers of every kernel, and read.cg ILP 4 TB/s for
 *   132…1056 blocks with and without the L2 clean step.
 */
#include <algorithm>
#include <string>
#include <vector>

#include "bw_common.hpp"

static const char* EXP = "05__memory_hierarchy_bandwidth";
static constexpr int kWarmups = 2, kMaxDiscards = 20;
static constexpr size_t kForceOneSmem = 120 * 1024;

enum Op { OP_READ = 0, OP_WRITE = 1, OP_MIX = 2, OP_MIX11 = 3 };
static const char* kOpName[] = {"read", "write", "mix", "mix11"};
static const char* kCacheName[] = {"ca", "cg", "cs", "wb"};

__constant__ uint8_t c_part[256];

// Element indices are 32-bit (n4 ≤ 2^27 here) and full steps run unguarded, with one guarded tail step: this keeps the
// ILP loads of a step back to back in SASS under the 32-register limit of 2 × 1024 threads per SM (checked in the SASS).
template <int C, int ILP>
__device__ __forceinline__ void read_step(uint64_t ib, uint32_t b, uint32_t T, uint32_t n, bool guard, uint32_t& acc) {
    V4 v[ILP];
#pragma unroll
    for (int i = 0; i < ILP; ++i)
        if (!guard || b + i * T < n) v[i] = ld_v4<C>(ib + (uint64_t)(b + i * T) * 16);
#pragma unroll
    for (int i = 0; i < ILP; ++i)
        if (!guard || b + i * T < n) xor_into(acc, v[i]);
}
template <int C, int ILP>
__device__ __forceinline__ void write_step(uint64_t ob, uint32_t b, uint32_t T, uint32_t n, bool guard) {
#pragma unroll
    for (int i = 0; i < ILP; ++i)
        if (!guard || b + i * T < n) st_v4<C>(ob + (uint64_t)(b + i * T) * 16, b);
}

template <int OP, int C, int ILP>
__device__ __forceinline__ void k4_body(const char* in, char* out, uint32_t n4, BlockRec* rec, uint64_t zero) {
    __shared__ uint32_t s_units;  // 16-byte elements read + written by the block
    const uint32_t T = gridDim.x * blockDim.x, gt = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t ib = (uint64_t)in, ob = (uint64_t)out;
    // elements of the strided stream: n for read/write, n4/2 for mix11 (read stream and write stream of equal size)
    const uint32_t n = OP == OP_MIX11 ? n4 / 2 : n4, G = n4 / 5;
    if (threadIdx.x == 0) s_units = 0;
    __syncthreads();
    // elements this thread moves, counted with the bounds of the timed loop; summed per warp, then one 32-bit shared
    // atomic per warp (a 64-bit shared atomicAdd from every thread is a CAS loop that costs tens of µs per block)
    uint32_t mine = 0;
    if constexpr (OP == OP_MIX) {
        for (uint32_t g = gt; g < G; g += T) mine += 6;
    } else {
        for (uint32_t b = gt; b < n; b += ILP * T)
            for (int i = 0; i < ILP; ++i) mine += (b + i * T < n) * (OP == OP_MIX11 ? 2 : 1);
    }
    mine = __reduce_add_sync(0xffffffffu, mine);
    if (threadIdx.x % 32 == 0) atomicAdd(&s_units, mine);
    uint32_t acc = 0;
    __syncthreads();
    TimerPair t0{};
    if (threadIdx.x == 0) t0 = read_timers_pair();
    if constexpr (OP == OP_MIX) {
#pragma unroll 1
        for (uint32_t g = gt; g < G; g += T) {
            V4 v[5];
#pragma unroll
            for (int k = 0; k < 5; ++k) v[k] = ld_v4<C>(ib + (uint64_t)(g + k * G) * 16);
            uint32_t s = 0;
#pragma unroll
            for (int k = 0; k < 5; ++k) xor_into(s, v[k]);
            st_v4<C>(ob + (uint64_t)g * 16, s);
            acc ^= s;
        }
    } else {
        uint32_t b = gt;
#pragma unroll 1
        for (; b + (ILP - 1) * T < n; b += ILP * T) {
            if constexpr (OP == OP_READ || OP == OP_MIX11) read_step<C, ILP>(ib, b, T, n, false, acc);
            if constexpr (OP == OP_WRITE || OP == OP_MIX11) write_step<C, ILP>(ob, b, T, n, false);  // mix11: data = index
        }
        if (b < n) {
            if constexpr (OP == OP_READ || OP == OP_MIX11) read_step<C, ILP>(ib, b, T, n, true, acc);
            if constexpr (OP == OP_WRITE || OP == OP_MIX11) write_step<C, ILP>(ob, b, T, n, true);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const TimerPair t1 = read_timers_after(acc, zero);
        const uint32_t sm = read_smid();
        rec[blockIdx.x] = {t0.cycles, t1.cycles, t0.ns, t1.ns, (uint64_t)s_units * 16, sm, c_part[sm], blockIdx.x, 0};
    }
}

template <int OP, int C, int ILP>
__global__ void __launch_bounds__(1024, ILP >= 8 ? 1 : 2) k4(const char* in, char* out, uint32_t n4, BlockRec* rec, uint64_t zero) {
    k4_body<OP, C, ILP>(in, out, n4, rec, zero);
}

using K4Fn = void (*)(const char*, char*, uint32_t, BlockRec*, uint64_t);

template <int OP, int C>
static K4Fn pick_ilp(int ilp) {
    switch (ilp) {
        case 1: return k4<OP, C, 1>;
        case 2: return k4<OP, C, 2>;
        case 4: return k4<OP, C, 4>;
        case 8: return k4<OP, C, 8>;
    }
    hb::die("bad ilp");
}
template <int C>
static K4Fn pick_op(int op, int ilp) {
    switch (op) {
        case OP_READ: return pick_ilp<OP_READ, C>(ilp);
        case OP_WRITE: return pick_ilp<OP_WRITE, C>(ilp);
        case OP_MIX: return k4<OP_MIX, C, 4>;
        case OP_MIX11: return pick_ilp<OP_MIX11, C>(ilp);
    }
    hb::die("bad op");
}
static K4Fn pick(int op, int cache, int ilp) { return cache == C_CS ? pick_op<C_CS>(op, ilp) : pick_op<C_CG>(op, ilp); }

struct Cfg {
    int op, cache, ilp, blocks, threads;
    bool one_per_sm;
};

struct Ctx {
    hb::Run& run;
    int sm_count = 0;
    std::vector<int> gpc, part;
    uint32_t n4 = 0;
    char *d_in = nullptr, *d_out = nullptr;
    BlockRec* d_rec = nullptr;
    CleanL2 flush;
    cudaEvent_t ev0{}, ev1{};
    long long launch_idx = 0;
    explicit Ctx(hb::Run& r) : run(r) {}
};

static std::string cfg_label(const Cfg& c) {
    return std::string("dram_") + kOpName[c.op] + "_" + kCacheName[c.cache] + "_ilp" + std::to_string(c.ilp) + "_b" + std::to_string(c.blocks);
}

static void launch(Ctx& x, const Cfg& c, bool timed) {
    const K4Fn fn = pick(c.op, c.cache, c.ilp);
    const size_t smem = c.one_per_sm ? kForceOneSmem : 0;
    if (smem > 48 * 1024) CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    x.flush();
    if (timed) CUDA_CHECK(cudaEventRecord(x.ev0));
    fn<<<c.blocks, c.threads, smem>>>(x.d_in, x.d_out, x.n4, x.d_rec, 0);
    CUDA_CHECK(cudaGetLastError());
    if (timed) CUDA_CHECK(cudaEventRecord(x.ev1));
    CUDA_CHECK(cudaDeviceSynchronize());
}

struct Writers {
    hb::CsvWriter launches, blocks;
    Writers()
        : launches(hb::Run::results("05__bandwidth__dram.csv"),
                   hb::with_mandatory({"op", "cache_op", "ilp", "blocks", "threads", "active_sms", "bytes", "ms_event", "tbps", "gbps_per_sm",
                                       "sm_mhz_blocks", "span_ns", "discard"})),
          blocks(hb::Run::results("05__bandwidth__dram_blocks.csv"),
                 hb::with_mandatory({"op", "cache_op", "ilp", "blocks", "block", "smid", "gpc", "partition", "bytes", "cycles", "ns", "start_ns",
                                     "bpc_sm", "discard"})) {}
};

static double run_cfg(Ctx& x, Writers& w, const Cfg& c, int recorded) {
    const std::string label = cfg_label(c);
    for (int i = 0; i < kWarmups; ++i) launch(x, c, false);
    std::vector<BlockRec> r(c.blocks);
    std::vector<double> tb;
    int valid = 0, discards = 0;
    while (valid < recorded) {
        launch(x, c, true);
        const hb::ClockSample cs = x.run.clock_note(label, "launch", x.launch_idx);
        CUDA_CHECK(cudaMemcpy(r.data(), x.d_rec, c.blocks * sizeof(BlockRec), cudaMemcpyDeviceToHost));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, x.ev0, x.ev1));
        const PlacementCheck pc(r, x.sm_count);
        const std::string discard = (c.one_per_sm && !pc.at_most_one()) ? "placement" : cs.bad() ? "clock" : "";
        uint64_t bytes = 0, t_lo = ~0ull, t_hi = 0;
        std::vector<double> mhz;
        int active = 0;
        for (int n : pc.per_sm) active += n > 0;
        for (const auto& b : r) {
            bytes += b.bytes;
            t_lo = std::min(t_lo, b.n0);
            t_hi = std::max(t_hi, b.n1);
            if (b.n1 > b.n0) mhz.push_back(1e3 * double(b.c1 - b.c0) / double(b.n1 - b.n0));
        }
        std::sort(mhz.begin(), mhz.end());
        const double tbps = bytes / (ms * 1e-3) / 1e12;
        const std::string pre = x.run.prefix(x.launch_idx, cs);
        char row[768];
        std::snprintf(row, sizeof row, "%s,%s,%s,%d,%d,%d,%d,%llu,%.6f,%.6f,%.4f,%.2f,%llu,%s", pre.c_str(), kOpName[c.op], kCacheName[c.cache], c.ilp,
                      c.blocks, c.threads, active, (unsigned long long)bytes, ms, tbps, tbps * 1e3 / active, mhz.empty() ? 0.0 : mhz[mhz.size() / 2],
                      (unsigned long long)(t_hi - t_lo), discard.c_str());
        w.launches.add(row);
        for (int i = 0; i < c.blocks; ++i) {
            const auto& b = r[i];
            const uint64_t cyc = b.c1 - b.c0;
            std::snprintf(row, sizeof row, "%s,%s,%s,%d,%d,%d,%u,%d,%u,%llu,%llu,%llu,%llu,%.4f,%s", pre.c_str(), kOpName[c.op], kCacheName[c.cache], c.ilp,
                          c.blocks, i, b.smid, b.smid < (uint32_t)x.sm_count ? x.gpc[b.smid] : -1, b.part, (unsigned long long)b.bytes,
                          (unsigned long long)cyc, (unsigned long long)(b.n1 - b.n0), (unsigned long long)(b.n0 - t_lo), cyc ? double(b.bytes) / cyc : 0.0,
                          discard.c_str());
            w.blocks.add(row);
        }
        w.launches.flush();
        w.blocks.flush();
        ++x.launch_idx;
        if (!discard.empty()) {
            if (++discards > kMaxDiscards) hb::die(label + ": too many discards");
            continue;
        }
        tb.push_back(tbps);
        ++valid;
    }
    w.launches.sync();
    w.blocks.sync();
    std::sort(tb.begin(), tb.end());
    std::printf("%-36s %7.3f TB/s%s\n", label.c_str(), tb[tb.size() / 2], discards ? "  (discards)" : "");
    return tb[tb.size() / 2];
}

static void setup(Ctx& x, bool quick) {
    x.sm_count = x.run.dev().sm_count;
    x.gpc = load_gpc_map(x.sm_count);
    x.part = partition_of_sm(x.gpc);
    std::vector<uint8_t> cp(256, 0);
    for (int s = 0; s < x.sm_count; ++s) cp[s] = (uint8_t)x.part[s];
    CUDA_CHECK(cudaMemcpyToSymbol(c_part, cp.data(), 256));
    const size_t bytes = quick ? (512ull << 20) : (2ull << 30);
    x.n4 = (uint32_t)(bytes / 16);
    CUDA_CHECK(cudaMalloc(&x.d_in, bytes));
    CUDA_CHECK(cudaMalloc(&x.d_out, bytes));
    CUDA_CHECK(cudaMemset(x.d_in, 0, bytes));
    CUDA_CHECK(cudaMemset(x.d_out, 0, bytes));
    CUDA_CHECK(cudaMalloc(&x.d_rec, 4 * x.sm_count * sizeof(BlockRec)));
    CUDA_CHECK(cudaEventCreate(&x.ev0));
    CUDA_CHECK(cudaEventCreate(&x.ev1));
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "k4";
    const bool quick = hb::has_flag(argc, argv, "--quick");
    hb::Run run(EXP, argc, argv);
    const std::string label = hb::arg_str(argc, argv, "--label", mode.c_str());
    Ctx x(run);
    if (mode == "clockcheck") {
        run.lock_clocks_note(label, "check");
        run.complete();
        return 0;
    }
    setup(x, quick);
    const int B4 = 4 * x.sm_count;
    if (mode == "k4") {
        Writers w;
        const int R = quick ? 3 : 10, RS = quick ? 2 : 5;
        run.lock_clocks_note("k4", "before");
        for (int cache : {C_CG, C_CS})
            for (int op : {OP_READ, OP_WRITE, OP_MIX, OP_MIX11}) run_cfg(x, w, Cfg{op, cache, 4, B4, 1024, false}, R);
        for (int ilp : {1, 2, 8}) run_cfg(x, w, Cfg{OP_READ, C_CG, ilp, B4, 1024, false}, R);
        std::vector<int> act;
        if (quick) act = {1, 2, 8, 33, 66, 132};
        else for (int n = 1; n <= x.sm_count; ++n) act.push_back(n);
        for (int op : {OP_READ, OP_WRITE})
            for (int n : act) run_cfg(x, w, Cfg{op, C_CG, 4, n, 1024, true}, RS);
        run.lock_clocks_note("k4", "after");
    } else if (mode == "diag") {  // occupancy of every full-grid kernel (no CSV output)
        for (int cache : {C_CG, C_CS})
            for (int op : {OP_READ, OP_WRITE, OP_MIX, OP_MIX11})
                for (int ilp : {1, 2, 4, 8}) {
                    const K4Fn fn = pick(op, cache, ilp);
                    int nb = 0;
                    cudaFuncAttributes fa{};
                    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, fn, 1024, 0));
                    CUDA_CHECK(cudaFuncGetAttributes(&fa, fn));
                    std::printf("%s %s ilp %d: %d blocks/SM, %d regs, %zu B static smem, %zu B local\n", kOpName[op], kCacheName[cache], ilp, nb,
                                fa.numRegs, fa.sharedSizeBytes, fa.localSizeBytes);
                }
        for (int fl : {1, 0})
            for (int blocks : {132, 264, 528, 1056}) {
                const K4Fn fn = pick(OP_READ, C_CG, 4);
                double tb = 0;
                for (int r = 0; r < 4; ++r) {
                    if (fl) x.flush();
                    CUDA_CHECK(cudaEventRecord(x.ev0));
                    fn<<<blocks, 1024>>>(x.d_in, x.d_out, x.n4, x.d_rec, 0);
                    CUDA_CHECK(cudaEventRecord(x.ev1));
                    CUDA_CHECK(cudaDeviceSynchronize());
                    float ms = 0;
                    CUDA_CHECK(cudaEventElapsedTime(&ms, x.ev0, x.ev1));
                    if (r) tb += x.n4 * 16.0 / (ms * 1e-3) / 1e12 / 3;
                }
                std::printf("read cg ilp 4, %u MiB, flush %d, %d blocks: %.3f TB/s\n", (unsigned)(x.n4 >> 16), fl, blocks, tb);
            }
    } else if (mode == "ncu") {
        FILE* man = std::fopen("bin/ncu_manifest_dram.csv", "w");
        if (!man) hb::die("cannot write bin/ncu_manifest_dram.csv");
        std::fprintf(man, "%s,op,cache_op,ilp,blocks,threads,ms_event\n", hb::join(hb::mandatory_cols()).c_str());
        run.lock_clocks_note("ncu", "before");
        for (int cache : {C_CG, C_CS})
            for (int op : {OP_READ, OP_WRITE, OP_MIX, OP_MIX11}) {
                const Cfg c{op, cache, 4, B4, 1024, false};
                launch(x, c, true);
                const hb::ClockSample cs = run.clock_note("ncu_" + cfg_label(c), "launch", x.launch_idx);
                float ms = 0;
                CUDA_CHECK(cudaEventElapsedTime(&ms, x.ev0, x.ev1));
                std::fprintf(man, "%s,%s,%s,%d,%d,%d,%.6f\n", run.prefix(x.launch_idx, cs).c_str(), kOpName[op], kCacheName[cache], 4, B4, 1024, ms);
                ++x.launch_idx;
            }
        std::fclose(man);
        run.lock_clocks_note("ncu", "after");
    } else {
        hb::die("unknown mode " + mode);
    }
    run.complete();
    return 0;
}
