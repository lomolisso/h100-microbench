/*
 * sync_grid.cu — experiment 07, K5: cost of the grid-wide barrier cooperative_groups::this_grid().sync() versus the
 * number of blocks (the GPU-scope reference for the cluster and flag primitives).
 *
 * Usage: bin/sync_grid {k5|ncu|clockcheck} [--quick] [--label L]. One invocation = one run_id.
 *
 * Kernel k5_grid_sync: cooperative launch (cudaLaunchKernelEx + cudaLaunchAttributeCooperative, no clusters) of
 * B ∈ {1, 8, 66, 132} blocks × 1024 threads (quick: B ∈ {1, 132}), 200 KB dynamic smem → 1 block/SM. A launch whose
 * blocks do not sit on distinct SMs is discarded and repeated. One untimed grid.sync(), then per rep every warp of
 * every block: untimed fence.acq_rel.gpu (no store pending since the last fence: the grid barrier's MEMBAR costs ~256
 * cycles more otherwise), bar.sync + volatile shared load (t0 waits for it: bar.sync is deferred, see sync_common.hpp);
 * t0 = %clock64 (lane 0); grid.sync(); t1 = %clock64; block_record (two bar.sync, outside t0..t1). Warp 0 executes the
 * grid barrier's global atomic and spin, so its t1 follows the barrier's completion; other warps can read t1 before
 * their CTA's final (deferred) BAR.SYNC completed, so cycles = warp 0 is the grid-barrier cost.
 * The grid barrier itself (CUDA 12.9 cooperative_groups, header-only, no -rdc) in SASS: BAR.SYNC; thread 0 of
 * each block: MEMBAR.ALL.GPU, ATOM.E.ADD.STRONG.GPU on one counter in the driver-provided grid workspace, spin on
 * LD.E.STRONG.GPU + CCTL.IVALL until the counter's top bit flips; BAR.SYNC.
 * Variant b<B>_after_st_global (B ∈ {1, 132}): every thread stores one word (st.global) right before t0.
 * Per B: 2 unrecorded warm-up launches, then 5 recorded launches × 1024 reps (quick: 2 × 64).
 *   → results/07__sync__primitives__k5.csv: one row per (block, rep): kernel k5, primitive grid_sync, cs 0,
 *     block_threads 1024, variant b<B>, cycles = warp 0's t1 − t0, rank = blockIdx.x, smid, n_ops 1,
 *     cycles_warp_max = the slowest warp of that block. Timer-pair floor rows (kernel k0) from k_timer_floor.
 * ncu mode: one launch per B (1024 reps); manifest bin/ncu_manifest_k5.csv (one row per profiled launch, in order).
 */
#include <algorithm>
#include <cooperative_groups.h>
#include <string>
#include <vector>

#include "sync_common.hpp"

namespace cg = cooperative_groups;
static const char* EXP = "07__synchronization_primitives";

// rec[blockIdx.x * reps + r].
template <bool ST>
__global__ void __launch_bounds__(1024) k5_grid_sync(uint32_t reps, Rec* rec, uint32_t* gbuf) {
    extern __shared__ __align__(16) uint8_t dyn[];  // 200 KB, only reserved (1 block/SM)
    __shared__ uint32_t s_w[64], s_probe;
    cg::grid_group grid = cg::this_grid();
    const uint32_t smid = read_smid(), zero = smid >> 31, probe = smem_u32(&s_probe);  // zero: 0 at run time
    if (threadIdx.x == 0) { dyn[0] = 0; s_probe = 0; }
    grid.sync();
#pragma unroll 1
    for (uint32_t r = 0; r < reps; ++r) {
        fence_acq_rel_gpu();  // untimed drain: a .gpu fence costs ~256 cycles more if any store was issued since the last fence
        // t0 after the CTA is aligned: bar.sync is deferred, an early warp would read t0 before it completed
        const uint32_t dep = bar_sync0_then_lds(probe);
        if constexpr (ST) asm volatile("st.global.u32 [%0], %1;" :: "l"(gbuf + blockIdx.x * 1024u + threadIdx.x), "r"(r) : "memory");
        const uint64_t t0 = read_clock64_after(dep, zero);
        grid.sync();
        const uint64_t t1 = read_clock64();
        block_record(s_w, (uint32_t)(t1 - t0), 0u, smid, rec, blockIdx.x * reps + r, false);
    }
}

static bool k5_launch(int B, bool st, uint32_t reps, DevBuf<Rec>& rec, std::vector<Rec>& h) {
    static bool attr = false;
    static DevBuf<uint32_t> gbuf(132u * 1024u);  // one word per thread (after_st_global)
    if (!attr) {
        CUDA_CHECK(cudaFuncSetAttribute(k5_grid_sync<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kOneCtaPerSmSmem));
        CUDA_CHECK(cudaFuncSetAttribute(k5_grid_sync<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kOneCtaPerSmSmem));
        attr = true;
    }
    cudaLaunchConfig_t cfg{};
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeCooperative;
    at[0].val.cooperative = 1;
    cfg.gridDim = dim3(B);
    cfg.blockDim = dim3(1024);
    cfg.dynamicSmemBytes = kOneCtaPerSmSmem;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    if (st) CUDA_CHECK(cudaLaunchKernelEx(&cfg, k5_grid_sync<true>, reps, rec.d, gbuf.d));
    else CUDA_CHECK(cudaLaunchKernelEx(&cfg, k5_grid_sync<false>, reps, rec.d, gbuf.d));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    h = rec.get((size_t)B * reps);
    std::vector<uint32_t> sm(B);
    for (int b = 0; b < B; ++b) sm[b] = h[(size_t)b * reps].smid;
    return hb::all_distinct(sm);
}

static void k5_point(Out07& o, int B, bool st, uint32_t reps, int launches) {
    DevBuf<Rec> rec((size_t)B * reps);
    std::vector<Rec> h;
    for (int w = 0; w < kWarmups; ++w) k5_launch(B, st, reps, rec, h);
    const std::string variant = "b" + std::to_string(B) + (st ? "_after_st_global" : ""), label = "k5_grid_sync_" + variant;
    std::vector<uint32_t> c0, cm;
    for (int l = 0, tries = 0; l < launches;) {
        if (!k5_launch(B, st, reps, rec, h)) {
            ++o.discarded;
            if (++tries > 20) hb::die(label + ": blocks never on distinct SMs");
            continue;
        }
        const hb::ClockSample s = o.run.clock_note(label, "launch", o.li);
        o.prim_rows(s, "k5", "grid_sync", 0, 1024, variant, h, reps, B, 1);
        o.flush();
        ++o.li;
        ++l;
        for (const Rec& x : h) { c0.push_back(x.c0); cm.push_back(x.cmax); }
    }
    o.sync();
    std::nth_element(c0.begin(), c0.begin() + c0.size() / 2, c0.end());
    std::nth_element(cm.begin(), cm.begin() + cm.size() / 2, cm.end());
    std::printf("k5 grid_sync B %3d%s: %8u cycles warp 0, %8u slowest warp (medians over blocks x reps)\n", B, st ? " after st" : "",
                c0[c0.size() / 2], cm[cm.size() / 2]);
}

static void run_k5(hb::Run& run, bool quick) {
    Out07 o(run, "k5");
    const uint32_t reps = quick ? kQuickReps : kReps;
    const int launches = quick ? kQuickLaunches : kLaunches;
    timer_floor(o, reps, launches);
    const std::vector<int> bs = quick ? std::vector<int>{1, 132} : std::vector<int>{1, 8, 66, 132};
    for (int B : bs) k5_point(o, B, false, reps, launches);
    for (int B : {1, 132}) k5_point(o, B, true, reps, launches);  // grid.sync behind one pending store per thread
    std::printf("k5: %d launches discarded (blocks not on distinct SMs)\n", o.discarded);
}

static void run_ncu(hb::Run& run) {
    hb::CsvWriter man("bin/ncu_manifest_k5.csv", hb::with_mandatory({"kernel", "primitive", "cs", "block_threads", "variant"}));
    long long li = 0;
    for (int B : {1, 8, 66, 132}) {
        DevBuf<Rec> rec((size_t)B * kReps);
        std::vector<Rec> h;
        k5_launch(B, false, kReps, rec, h);
        const hb::ClockSample s = run.clock_note("ncu_k5_grid_sync_b" + std::to_string(B), "launch", li);
        man.add(run.prefix(li++, s) + ",k5,grid_sync,0,1024,b" + std::to_string(B));
        man.flush();
    }
    man.sync();
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "k5";
    hb::Run run(EXP, argc, argv);
    if (mode == "clockcheck") run.lock_clocks_note(hb::arg_str(argc, argv, "--label", "check"), "check");
    else if (mode == "k5") run_k5(run, hb::has_flag(argc, argv, "--quick"));
    else if (mode == "ncu") run_ncu(run);
    else hb::die("unknown mode " + mode);
    run.complete();
    return 0;
}
