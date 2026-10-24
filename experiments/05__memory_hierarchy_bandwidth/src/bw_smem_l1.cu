/*
 * bw_smem_l1.cu — experiment 05, K1 + K2: load/store bandwidth of one SM's shared memory (K1) and L1 cache (K2),
 * in bytes per SM cycle, for one block alone and for 132 blocks on 132 distinct SMs at the same time.
 *
 * Usage: bin/bw_smem_l1 {k12|ncu|clockcheck} [--quick] [--label L]. One invocation = one run_id.
 *
 * Kernel k12<LEVEL, OP, W, ILP, PAT, T>(gbuf, iters, zero, out): T threads (128…1024), one 64 KiB region per block.
 *   Addresses: thread t, slot i ∈ [0, ILP): byte offset W·t + W·T·(i mod S), S = 64 KiB / (W·T) (pattern "stride");
 *   a warp touches W·32 consecutive bytes per instruction, so shared-memory accesses are bank-conflict-free and global
 *   accesses fully coalesced. Pattern "accelsim" (L1, W = 16, ILP 4): offsets 16·t + {0, 128, 256, 384} B.
 *   Timed region: bar.sync → thread 0 reads %clock64/%globaltimer → `iters` steps of ILP independent accesses to the
 *   same ILP addresses (plus (iteration & zero) = 0, which ptxas cannot see is 0, so the global loads cannot be hoisted
 *   out of the loop) → bar.sync → thread 0 reads the timers after its own last loaded value (read_timers_after).
 *   Loads: each loop iteration issues the step's ILP loads, then XORs their data into ILP accumulators (volatile xor,
 *   bw_common.hpp); in-order issue makes the next step wait for the XORs, so a thread has at most ILP loads outstanding,
 *   and every thread's loads complete before it reaches the closing bar.sync. Stores write the loop counter; their
 *   steps are unrolled to ≥ 8 stores per loop iteration, copy u using slots u·ILP + i (mod S), so that consecutive
 *   stores of a thread never write the same address. bytes = T · iters · ILP · W; bpc_per_sm = bytes / cycles.
 *     LEVEL smem: ld.shared.v4.f32 / ld.shared.f32 / st.shared.v4.f32 / st.shared.f32 on 64 KiB dynamic smem, written with
 *                 the .volatile qualifier (same SASS LDS/STS): the addresses repeat every iteration and nothing in the
 *                 loop writes shared memory, so ptxas would otherwise hoist the plain loads out of the timed loop.
 *     LEVEL l1:   ld.global.ca.v4.f32 / ld.global.ca.f32 on the block's 64 KiB slice of a global buffer, warmed by
 *                 one untimed pass of the same loads (L1 hits afterwards; checked with ncu); st.global.v4.f32 /
 *                 st.global.f32 (L1 is write-through: these measure the store path from the SM to L2).
 *   Sweep: level × op × W ∈ {4, 16} × T ∈ {128, 256, 512, 1024} × ILP ∈ {1, 2, 4, 8} × iters ∈ {4096, 8192}
 *   (+ accelsim: T × iters, ILP 4) × placement, except L1 16-B loads at ILP 8 with 1024 threads: 8 slots of 16 KiB do not
 *   fit in 64 KiB, the repeated slots have identical addresses, and ptxas merges repeated global loads. Placement:
 *     single   1 block (on whichever SM the scheduler picks; %smid recorded);
 *     all_sms  132 blocks; the host requires 132 distinct %smid values, otherwise the launch is discarded and repeated.
 *   2 warm-up + 10 recorded (valid) launches per configuration. The slope between iters 4096 and 8192 removes the
 *   fixed cost of the timed region (barriers, timer reads, loop exit).
 *   → results/05__bandwidth__smem_l1.csv: level, op, width_bytes, threads, ilp, iters, pattern, placement, block, bytes,
 *     cycles, bpc_per_sm, ns, smid, discard (one row per block and launch; discard = "" | placement | clock; a
 *     discarded launch is kept and repeated). Mandatory columns first; clocks = NVML sample after the launch.
 * ncu mode: one launch each of smem ld/st v4 and f32 and l1 ld v4/f32 (single block, T = 1024, ILP 4, iters 4096),
 *   profiled by src/run.sh (kernel replay; the kernel warms its own L1). Writes bin/ncu_manifest_smem_l1.csv.
 * --quick: T ∈ {256, 1024}, ILP ∈ {1, 4}, iters 4096 and 8192, 3 recorded launches (smoke test).
 */
#include <string>
#include <vector>

#include "bw_common.hpp"

static const char* EXP = "05__memory_hierarchy_bandwidth";
static constexpr uint32_t kRegion = 64 * 1024;
static constexpr int kWarmups = 2, kMaxDiscards = 20;

enum Level { L_SMEM = 0, L_L1 = 1 };
enum Op { OP_LD = 0, OP_ST = 1 };
enum Pattern { P_STRIDE = 0, P_ACCELSIM = 1 };
static const char* kLevelName[] = {"smem", "l1"};
static const char* kOpName[] = {"ld", "st"};
static const char* kPatName[] = {"stride", "accelsim"};

struct BlockOut { uint64_t cycles, ns; uint32_t smid, sink; };

// One access of width W (4 or 16 B) at a shared (32-bit) or global (64-bit) address. Loads return the raw words
// (y = z = w = 0 for W = 4); the caller XORs them only after all ILP loads of a step are issued.
struct Words { uint32_t x, y, z, w; };
// acc ^= data as volatile PTX, so that it stays behind all loads of the step (bw_common.hpp, xor_into)
template <int W>
__device__ __forceinline__ void consume(uint32_t& acc, const Words& v) {
    if constexpr (W == 16) xor_into(acc, V4{v.x, v.y, v.z, v.w});
    else asm volatile("xor.b32 %0, %0, %1;" : "+r"(acc) : "r"(v.x));
}
template <int LEVEL, int W>
__device__ __forceinline__ Words ld_one(uint64_t a) {
    Words v{0, 0, 0, 0};
    if constexpr (W == 16) {
        if constexpr (LEVEL == L_SMEM)
            asm volatile("ld.volatile.shared.v4.f32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"((uint32_t)a) : "memory");
        else
            asm volatile("ld.global.ca.v4.f32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(a) : "memory");
    } else {
        if constexpr (LEVEL == L_SMEM) asm volatile("ld.volatile.shared.f32 %0, [%1];" : "=r"(v.x) : "r"((uint32_t)a) : "memory");
        else asm volatile("ld.global.ca.f32 %0, [%1];" : "=r"(v.x) : "l"(a) : "memory");
    }
    return v;
}
template <int LEVEL, int W>
__device__ __forceinline__ void st_one(uint64_t a, uint32_t v) {
    const float f = __uint_as_float(v);
    if constexpr (W == 16) {
        if constexpr (LEVEL == L_SMEM) asm volatile("st.volatile.shared.v4.f32 [%0], {%1, %1, %1, %1};" :: "r"((uint32_t)a), "f"(f) : "memory");
        else asm volatile("st.global.v4.f32 [%0], {%1, %1, %1, %1};" :: "l"(a), "f"(f) : "memory");
    } else {
        if constexpr (LEVEL == L_SMEM) asm volatile("st.volatile.shared.f32 [%0], %1;" :: "r"((uint32_t)a), "f"(f) : "memory");
        else asm volatile("st.global.f32 [%0], %1;" :: "l"(a), "f"(f) : "memory");
    }
}

// Byte offset of slot k of thread t: W·t + W·T·(k mod S) (stride) or 16·t + 128·k (accelsim) — T is a template
// parameter, so the slot part is a compile-time immediate in the SASS address ([R + imm]).
template <int W, int T, int PAT>
__device__ __forceinline__ constexpr uint32_t slot_off(int k) {
    return PAT == P_ACCELSIM ? 128u * k : (uint32_t)W * T * (k % (int)(kRegion / (W * T)));
}

template <int LEVEL, int OP, int W, int ILP, int PAT, int T>
__global__ void __launch_bounds__(T) k12(char* gbuf, int iters, uint64_t zero, BlockOut* out) {
    extern __shared__ __align__(16) unsigned char smem[];
    // Loads: one step of ILP loads per loop iteration, consumed before the next step → at most ILP loads outstanding per
    // thread. Stores never wait, so their steps are unrolled to ≥ 8 accesses per iteration to amortise the loop overhead;
    // copy u of a store step uses slots u·ILP … u·ILP + ILP − 1 (mod S), so consecutive stores never hit the same address.
    constexpr int kUnroll = (OP == OP_LD || ILP >= 8) ? 1 : 8 / ILP;
    const uint32_t t = threadIdx.x;
    const uint64_t base = ((LEVEL == L_SMEM) ? (uint64_t)smem_u32(smem) : (uint64_t)(gbuf + (size_t)blockIdx.x * kRegion)) +
                          (PAT == P_ACCELSIM ? 16u : (uint32_t)W) * t;
    uint32_t acc[ILP];
#pragma unroll
    for (int i = 0; i < ILP; ++i) acc[i] = 0;
    Words v[ILP];
    if constexpr (LEVEL == L_L1 && OP == OP_LD) {  // warm L1 with the exact lines of the timed loop
#pragma unroll
        for (int i = 0; i < ILP; ++i) v[i] = ld_one<LEVEL, W>(base + slot_off<W, T, PAT>(i));
#pragma unroll
        for (int i = 0; i < ILP; ++i) acc[i] ^= v[i].x ^ v[i].y ^ v[i].z ^ v[i].w;
    }
    __syncthreads();
    TimerPair t0{};
    if (t == 0) t0 = read_timers_pair();
#pragma unroll 1
    for (int r = 0; r < iters; r += kUnroll) {
        // + (r & zero) = + 0, but ptxas cannot prove the addresses loop-invariant: it would otherwise hoist the global
        // loads (LDG.STRONG.SM) out of the loop. Shared accesses are .volatile for the same reason (ld_one/st_one).
        const uint64_t p = base + ((uint64_t)r & zero);
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            if constexpr (OP == OP_LD) {
#pragma unroll
                for (int i = 0; i < ILP; ++i) v[i] = ld_one<LEVEL, W>(p + slot_off<W, T, PAT>(i));
#pragma unroll
                for (int i = 0; i < ILP; ++i) consume<W>(acc[i], v[i]);
            } else {
#pragma unroll
                for (int i = 0; i < ILP; ++i) st_one<LEVEL, W>(p + slot_off<W, T, PAT>(u * ILP + i), (uint32_t)(r + u));
            }
        }
    }
    uint32_t fold = 0;
#pragma unroll
    for (int i = 0; i < ILP; ++i) fold ^= acc[i];
    __syncthreads();
    if (t == 0) {
        const TimerPair t1 = read_timers_after(fold, zero);
        out[blockIdx.x] = {t1.cycles - t0.cycles, t1.ns - t0.ns, read_smid(), fold};
    }
}

using KernelFn = void (*)(char*, int, uint64_t, BlockOut*);

template <int LEVEL, int OP, int W, int ILP, int PAT>
static KernelFn pick_t(int threads) {
    switch (threads) {
        case 128: return k12<LEVEL, OP, W, ILP, PAT, 128>;
        case 256: return k12<LEVEL, OP, W, ILP, PAT, 256>;
        case 512: return k12<LEVEL, OP, W, ILP, PAT, 512>;
        case 1024: return k12<LEVEL, OP, W, ILP, PAT, 1024>;
    }
    hb::die("bad thread count " + std::to_string(threads));
}
template <int LEVEL, int OP, int W, int PAT>
static KernelFn pick(int ilp, int threads) {
    switch (ilp) {
        case 1: return pick_t<LEVEL, OP, W, 1, PAT>(threads);
        case 2: return pick_t<LEVEL, OP, W, 2, PAT>(threads);
        case 4: return pick_t<LEVEL, OP, W, 4, PAT>(threads);
        case 8: return pick_t<LEVEL, OP, W, 8, PAT>(threads);
    }
    hb::die("bad ilp " + std::to_string(ilp));
}

struct Cfg {
    int level, op, width, threads, ilp, iters, pattern;
    bool all_sms;
    KernelFn fn;
};

struct Ctx {
    hb::Run& run;
    char* d_buf = nullptr;
    BlockOut* d_out = nullptr;
    int sm_count = 0;
    long long launch_idx = 0;
};

static void launch(const Cfg& c, Ctx& x) {
    const size_t smem = c.level == L_SMEM ? kRegion : 0;
    if (smem > 48 * 1024) CUDA_CHECK(cudaFuncSetAttribute(c.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    c.fn<<<c.all_sms ? x.sm_count : 1, c.threads, smem>>>(x.d_buf, c.iters, 0, x.d_out);
    CUDA_CHECK(cudaGetLastError());
}

static std::string cfg_label(const Cfg& c) {
    return std::string("k12_") + kLevelName[c.level] + "_" + kOpName[c.op] + "_w" + std::to_string(c.width) + "_t" +
           std::to_string(c.threads) + "_ilp" + std::to_string(c.ilp) + "_it" + std::to_string(c.iters) + "_" +
           kPatName[c.pattern] + (c.all_sms ? "_all" : "_single");
}

static void run_cfg(const Cfg& c, Ctx& x, hb::CsvWriter& csv, int recorded) {
    const std::string label = cfg_label(c);
    hb::warmup([&] { launch(c, x); }, kWarmups);
    const int nb = c.all_sms ? x.sm_count : 1;
    std::vector<BlockOut> h(nb);
    const double bytes = double(c.threads) * c.iters * c.ilp * c.width;
    int valid = 0, discards = 0;
    double last = 0;
    while (valid < recorded) {
        launch(c, x);
        CUDA_CHECK(cudaDeviceSynchronize());
        const hb::ClockSample cs = x.run.clock_note(label, "launch", x.launch_idx);
        CUDA_CHECK(cudaMemcpy(h.data(), x.d_out, nb * sizeof(BlockOut), cudaMemcpyDeviceToHost));
        std::vector<uint32_t> sm(nb);
        for (int b = 0; b < nb; ++b) sm[b] = h[b].smid;
        const char* discard = !hb::all_distinct(sm) ? "placement" : cs.bad() ? "clock" : "";
        for (int b = 0; b < nb; ++b) {
            char r[512];
            std::snprintf(r, sizeof r, "%s,%s,%s,%d,%d,%d,%d,%s,%s,%d,%.0f,%llu,%.4f,%llu,%u,%s", x.run.prefix(x.launch_idx, cs).c_str(),
                          kLevelName[c.level], kOpName[c.op], c.width, c.threads, c.ilp, c.iters, kPatName[c.pattern],
                          c.all_sms ? "all_sms" : "single", b, bytes, (unsigned long long)h[b].cycles, bytes / h[b].cycles,
                          (unsigned long long)h[b].ns, h[b].smid, discard);
            csv.add(r);
        }
        csv.flush();
        ++x.launch_idx;
        if (*discard) {
            if (++discards > kMaxDiscards) hb::die(label + ": too many discarded launches");
            continue;
        }
        last = bytes / h[0].cycles;
        ++valid;
    }
    csv.sync();
    std::printf("%-48s %8.2f B/cycle/SM (block 0)%s\n", label.c_str(), last,
                discards ? (" discards " + std::to_string(discards)).c_str() : "");
}

template <int LEVEL, int OP, int W, int PAT>
static Cfg make(int threads, int ilp, int iters, bool all) {
    return {LEVEL, OP, W, threads, ilp, iters, PAT, all, pick<LEVEL, OP, W, PAT>(ilp, threads)};
}

static std::vector<Cfg> sweep(bool quick) {
    const std::vector<int> T = quick ? std::vector<int>{256, 1024} : std::vector<int>{128, 256, 512, 1024};
    const std::vector<int> I = quick ? std::vector<int>{1, 4} : std::vector<int>{1, 2, 4, 8};
    std::vector<Cfg> v;
    for (bool all : {false, true})
        for (int it : {4096, 8192})
            for (int t : T) {
                for (int i : I) {
                    // 16-B loads at ILP 8 and 1024 threads would need 128 KiB: slots 4–7 repeat slots 0–3, and ptxas merges
                    // the repeated global loads (4 LDG for 8 counted). Not run for L1; shared loads are .volatile (8 LDS).
                    const bool l1_v4_fits = 16 * t * i <= (int)kRegion;
                    v.push_back(make<L_SMEM, OP_LD, 16, P_STRIDE>(t, i, it, all));
                    v.push_back(make<L_SMEM, OP_LD, 4, P_STRIDE>(t, i, it, all));
                    v.push_back(make<L_SMEM, OP_ST, 16, P_STRIDE>(t, i, it, all));
                    v.push_back(make<L_SMEM, OP_ST, 4, P_STRIDE>(t, i, it, all));
                    if (l1_v4_fits) v.push_back(make<L_L1, OP_LD, 16, P_STRIDE>(t, i, it, all));
                    v.push_back(make<L_L1, OP_LD, 4, P_STRIDE>(t, i, it, all));
                    v.push_back(make<L_L1, OP_ST, 16, P_STRIDE>(t, i, it, all));
                    v.push_back(make<L_L1, OP_ST, 4, P_STRIDE>(t, i, it, all));
                }
                v.push_back(make<L_L1, OP_LD, 16, P_ACCELSIM>(t, 4, it, all));
            }
    return v;
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "k12";
    const bool quick = hb::has_flag(argc, argv, "--quick");
    hb::Run run(EXP, argc, argv);
    const std::string label = hb::arg_str(argc, argv, "--label", mode.c_str());
    Ctx x{run};
    x.sm_count = run.dev().sm_count;
    CUDA_CHECK(cudaMalloc(&x.d_buf, (size_t)x.sm_count * kRegion));
    CUDA_CHECK(cudaMemset(x.d_buf, 0, (size_t)x.sm_count * kRegion));
    CUDA_CHECK(cudaMalloc(&x.d_out, x.sm_count * sizeof(BlockOut)));
    if (mode == "clockcheck") {
        run.lock_clocks_note(label, "check");
    } else if (mode == "k12") {
        hb::CsvWriter csv(hb::Run::results("05__bandwidth__smem_l1.csv"),
                          hb::with_mandatory({"level", "op", "width_bytes", "threads", "ilp", "iters", "pattern", "placement", "block",
                                              "bytes", "cycles", "bpc_per_sm", "ns", "smid", "discard"}));
        run.lock_clocks_note("k12", "before");
        for (const Cfg& c : sweep(quick)) run_cfg(c, x, csv, quick ? 3 : 10);
        run.lock_clocks_note("k12", "after");
    } else if (mode == "ncu") {
        FILE* man = std::fopen("bin/ncu_manifest_smem_l1.csv", "w");
        if (!man) hb::die("cannot write bin/ncu_manifest_smem_l1.csv");
        std::fprintf(man, "%s,level,op,width_bytes,threads,ilp,iters,pattern,placement,bytes,cycles,bpc_per_sm,smid\n",
                     hb::join(hb::mandatory_cols()).c_str());
        const std::vector<Cfg> cfgs = {make<L_SMEM, OP_LD, 16, P_STRIDE>(1024, 4, 4096, false), make<L_SMEM, OP_LD, 4, P_STRIDE>(1024, 4, 4096, false),
                                       make<L_SMEM, OP_ST, 16, P_STRIDE>(1024, 4, 4096, false), make<L_SMEM, OP_ST, 4, P_STRIDE>(1024, 4, 4096, false),
                                       make<L_L1, OP_LD, 16, P_STRIDE>(1024, 4, 4096, false), make<L_L1, OP_LD, 4, P_STRIDE>(1024, 4, 4096, false)};
        run.lock_clocks_note("ncu", "before");
        for (const Cfg& c : cfgs) {
            launch(c, x);
            CUDA_CHECK(cudaDeviceSynchronize());
            const hb::ClockSample cs = run.clock_note("ncu_" + cfg_label(c), "launch", x.launch_idx);
            BlockOut o;
            CUDA_CHECK(cudaMemcpy(&o, x.d_out, sizeof o, cudaMemcpyDeviceToHost));
            const double bytes = double(c.threads) * c.iters * c.ilp * c.width;
            std::fprintf(man, "%s,%s,%s,%d,%d,%d,%d,%s,single,%.0f,%llu,%.4f,%u\n", run.prefix(x.launch_idx, cs).c_str(), kLevelName[c.level],
                         kOpName[c.op], c.width, c.threads, c.ilp, c.iters, kPatName[c.pattern], bytes, (unsigned long long)o.cycles,
                         bytes / o.cycles, o.smid);
            ++x.launch_idx;
        }
        std::fclose(man);
        run.lock_clocks_note("ncu", "after");
    } else {
        hb::die("unknown mode " + mode);
    }
    CUDA_CHECK(cudaFree(x.d_buf));
    CUDA_CHECK(cudaFree(x.d_out));
    run.complete();
    return 0;
}
