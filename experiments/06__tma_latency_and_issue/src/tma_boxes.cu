/*
 * tma_boxes.cu — experiment 06, K4: latency of one 16-KiB tensor-TMA load versus box shape, swizzle, source (L2 warm /
 * HBM cold) and the state of the tensor-map descriptor (cold, prefetched with prefetch.tensormap, warm).
 *
 * Usage: bin/tma_boxes {k4|clockcheck} [--quick] [--label L]. One invocation = one run_id.
 *
 * Tensor: 3-D float32, dims {x 1024, y 256, z 256} (256 MiB, row stride 4 KiB, plane stride 1 MiB), L2 promotion none,
 *   OOB fill none. Boxes (x×y×z floats, all 16 KiB): 256×16×1, 64×64×1, 16×256×1, 16×16×16, 4×4×256; swizzle NONE for
 *   every box and 128B for the boxes whose inner extent is ≤ 128 B (16×256×1, 16×16×16, 4×4×256); an encode that the
 *   driver rejects is recorded with encode_ok = 0 and no timing.
 * Kernel k4_box(A, B, C, D, pos, hbm, reps, out): 1 block of 1 warp, thread 0; destination: 160 KiB of dynamic smem
 *   aligned to 1024 B (a swizzled box may occupy more smem than its 16 KiB, see smem_extent); A, B, C, D = four
 *   identical tensor maps passed as separate __grid_constant__ parameters (different 128-B lines of the parameter
 *   space), so that each can be used for the first time at a chosen point of the launch.
 *   Footprint probe (untimed, first thing in the launch): the 160 KiB are filled with 0xFF by generic stores,
 *   fence.proxy.async.shared::cta, then the first (untimed) load with A; the thread then scans the 160 KiB:
 *   smem_extent_bytes = end of the last 16-B chunk that changed, smem_written_bytes = changed chunks × 16 (the
 *   tensor holds zeros).
 *   One timed load = arrive.expect_tx(bar, 16384) (untimed); t0 = clock64; cp.async.bulk.tensor.3d.tile [dst],
 *   [map, {x, y, z}], [bar]; try_wait.parity loop; t1 = clock64 → cycles = t1 − t0.
 *   Sequence per launch (positions from `pos`, chosen by the host; box-aligned coordinates):
 *     L2 source:  untimed load with A at P (data → L2, A's descriptor used once); timed load with B at P
 *                 (descriptor cold, data warm) → cold; prefetch.tensormap [C], spin 5 µs on %globaltimer, timed load
 *                 with C at P → prefetched; `reps` timed loads with A at P → warm; finally a timed load with D at P
 *                 (descriptor first used at the end of the launch) → cold_late, which separates the descriptor state
 *                 from the position in the sequence.
 *     HBM source: L2 cleaned before the launch (05's CleanL2); same sequence, but every load uses its own, never
 *                 touched position (P0 untimed A, P1 B cold, P2 C prefetched, P3.. A warm, last D cold_late): data
 *                 always cold.
 *   reps = 16. Per (box, swizzle, source): 2 unrecorded warm-up launches, 20 recorded launches (new positions each).
 *   Odd launches first complete one empty phase (arrive + wait), so that every load of the sequence waits on the other
 *   mbarrier parity: each descriptor state is measured at both parities (the HBM-source completion depends on it).
 *   --quick: 3 recorded launches.
 *   → results/06__tma__boxes.csv: box, box_x, box_y, box_z, swizzle, source, descriptor_state, rep, parity, cycles, smid,
 *     encode_ok, smem_extent_bytes, smem_written_bytes (one row per timed load).
 */
#include <random>
#include <string>
#include <vector>

#include "../../05__memory_hierarchy_bandwidth/src/bw_common.hpp"
#include "tma_common.hpp"

static const char* EXP = "06__tma_latency_and_issue";
static constexpr uint32_t kBoxBytes = 16384, kWarmReps = 16, kNPos = 4 + kWarmReps, kDstBytes = 160u << 10;
static constexpr int kOut = 6 + kWarmReps;  // cold, prefetched, warm[reps], cold_late, smid, extent, written
static constexpr uint32_t kDimX = 1024, kDimY = 256, kDimZ = 256;

struct Pos { int32_t x, y, z, pad; };

// t0 is read after the coordinates have arrived in registers (they come from a global array: a load still in flight
// would otherwise be timed as part of the TMA load).
__device__ __forceinline__ uint64_t timed_load(uint32_t d, const CUtensorMap* m, const Pos& p, uint32_t b, uint32_t parity) {
    mbar_arrive_expect_tx(b, kBoxBytes);
    const uint64_t t0 = read_clock64_after((uint64_t)(uint32_t)(p.x ^ p.y ^ p.z), (uint64_t)p.pad);  // p.pad = 0: runtime zero
    tma_load_3d(d, m, p.x, p.y, p.z, b);
    wait_loop<W_TRY>(b, parity, 0);
    return read_clock64() - t0;
}

// out[0] cold, out[1] prefetched, out[2 + i] warm rep i, out[2 + reps] cold_late, then smid, smem extent, smem written
__global__ void __launch_bounds__(32) k4_box(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B,
                                              const __grid_constant__ CUtensorMap C, const __grid_constant__ CUtensorMap D,
                                              const Pos* pos, int hbm, uint32_t reps, uint32_t parity0, uint64_t* out) {
    extern __shared__ uint8_t dyn_raw[];
    __shared__ alignas(8) uint64_t bar;
    if (threadIdx.x != 0) return;
    const uint32_t d = (smem_u32(dyn_raw) + 1023u) & ~1023u;  // 1024-B alignment (128B swizzle)
    const uint32_t b = smem_u32(&bar);
    mbar_init(b, 1);
    fence_mbarrier_init();
    uint32_t parity = 0;
    if (parity0) {  // one extra phase (an empty arrive) so that every timed load of this launch waits on the other parity
        mbar_arrive(b);
        wait_loop<W_TRY>(b, parity, 0);
        parity ^= 1;
    }
#pragma unroll 1
    for (uint32_t o = 0; o < kDstBytes; o += 16)
        asm volatile("st.shared.v4.u32 [%0], {%1, %1, %1, %1};" :: "r"(d + o), "r"(0xffffffffu) : "memory");
    fence_proxy_async_shared_cta();
    timed_load(d, &A, pos[0], b, parity);  // untimed use: warms the data (L2 source) and A's descriptor
    parity ^= 1;
    uint32_t extent = 0, written = 0;
#pragma unroll 1
    for (uint32_t o = 0; o < kDstBytes; o += 16) {
        uint32_t x, y, z, w;
        asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(d + o) : "memory");
        if ((x & y & z & w) != 0xffffffffu) { extent = o + 16; written += 16; }
    }
    out[0] = timed_load(d, &B, hbm ? pos[1] : pos[0], b, parity);
    parity ^= 1;
    prefetch_tensormap(&C);
    const uint64_t until = read_globaltimer() + 5000;
    while (read_globaltimer() < until) { }
    out[1] = timed_load(d, &C, hbm ? pos[2] : pos[0], b, parity);
    parity ^= 1;
#pragma unroll 1
    for (uint32_t r = 0; r < reps; ++r) {
        out[2 + r] = timed_load(d, &A, hbm ? pos[3 + r] : pos[0], b, parity);
        parity ^= 1;
    }
    out[2 + reps] = timed_load(d, &D, hbm ? pos[3 + reps] : pos[0], b, parity);
    out[3 + reps] = read_smid();
    out[4 + reps] = extent;
    out[5 + reps] = written;
}

struct Box { uint32_t x, y, z; };

static void run_k4(hb::Run& run, bool quick) {
    hb::CsvWriter csv(hb::Run::results("06__tma__boxes.csv"),
                      hb::with_mandatory({"box", "box_x", "box_y", "box_z", "swizzle", "source", "descriptor_state", "rep",
                                          "parity", "cycles", "smid", "encode_ok", "smem_extent_bytes", "smem_written_bytes"}));
    const size_t bytes = (size_t)kDimX * kDimY * kDimZ * sizeof(float);
    float* d_t;
    CUDA_CHECK(cudaMalloc(&d_t, bytes));
    CUDA_CHECK(cudaMemset(d_t, 0, bytes));
    Pos* d_pos;
    uint64_t* d_out;
    CUDA_CHECK(cudaMalloc(&d_pos, kNPos * sizeof(Pos)));
    CUDA_CHECK(cudaMalloc(&d_out, kOut * sizeof(uint64_t)));
    CleanL2 clean;
    const int smem = kDstBytes + 1024;
    CUDA_CHECK(cudaFuncSetAttribute(k4_box, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    const std::vector<Box> boxes = {{256, 16, 1}, {64, 64, 1}, {16, 256, 1}, {16, 16, 16}, {4, 4, 256}};
    const int launches = quick ? 3 : 20;
    std::mt19937_64 rng(0x06ull << 56 | 20261024ull);
    long long li = 0;
    run.lock_clocks_note("k4", "before");
    for (const Box& bx : boxes) {
        for (int sw = 0; sw <= 1; ++sw) {
            if (sw && bx.x * sizeof(float) > 128) continue;
            const CUtensorMapSwizzle swz = sw ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE;
            const char* swname = sw ? "128B" : "none";
            const std::string bname = std::to_string(bx.x) + "x" + std::to_string(bx.y) + "x" + std::to_string(bx.z);
            const cuuint64_t gdim[3] = {kDimX, kDimY, kDimZ};
            const cuuint64_t gstr[2] = {(cuuint64_t)kDimX * 4, (cuuint64_t)kDimX * kDimY * 4};
            const cuuint32_t box[3] = {bx.x, bx.y, bx.z};
            alignas(64) CUtensorMap A{}, B{}, C{}, D{};
            const CUresult e = encode_tiled(&A, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 3, d_t, gdim, gstr, box, swz);
            if (e != CUDA_SUCCESS) {
                const hb::ClockSample s = run.clock_note("k4_" + bname + "_" + swname + "_encode", "check", li);
                char b[512];
                std::snprintf(b, sizeof b, "%s,%s,%u,%u,%u,%s,,,,,,,0,,", run.prefix(li, s).c_str(), bname.c_str(), bx.x, bx.y, bx.z, swname);
                csv.add(b);
                csv.sync();
                std::printf("k4 %s swizzle %s: encode rejected (CUresult %d)\n", bname.c_str(), swname, (int)e);
                ++li;
                continue;
            }
            B = A;
            C = A;
            D = A;
            const uint32_t nx = kDimX / bx.x, ny = kDimY / bx.y, nz = kDimZ / bx.z;
            std::uniform_int_distribution<uint64_t> pick(0, (uint64_t)nx * ny * nz - 1);
            for (int src = 0; src <= 1; ++src) {
                const char* sname = src ? "hbm" : "l2";
                double sums[4] = {0, 0, 0, 0};
                for (int l = 0; l < launches + 2; ++l) {
                    std::vector<Pos> P(kNPos);
                    std::vector<uint64_t> used;
                    for (auto& p : P) {  // distinct box positions
                        uint64_t k;
                        do { k = pick(rng); } while (std::find(used.begin(), used.end(), k) != used.end());
                        used.push_back(k);
                        p = Pos{(int32_t)((k % nx) * bx.x), (int32_t)((k / nx % ny) * bx.y), (int32_t)((k / nx / ny) * bx.z), 0};
                    }
                    CUDA_CHECK(cudaMemcpy(d_pos, P.data(), kNPos * sizeof(Pos), cudaMemcpyHostToDevice));
                    if (src) clean();
                    const uint32_t parity0 = (uint32_t)(l & 1);  // alternate the parity sequence between launches
                    k4_box<<<1, 32, smem>>>(A, B, C, D, d_pos, src, kWarmReps, parity0, d_out);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaDeviceSynchronize());
                    if (l < 2) continue;  // warm-up launches
                    const hb::ClockSample s = run.clock_note("k4_" + bname + "_" + swname + "_" + sname, "launch", li);
                    std::vector<uint64_t> h(kOut);
                    CUDA_CHECK(cudaMemcpy(h.data(), d_out, h.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
                    const uint32_t smid = (uint32_t)h[3 + kWarmReps];
                    const uint64_t extent = h[4 + kWarmReps], written = h[5 + kWarmReps];
                    // timed load k (0 cold, 1 prefetched, 2.. warm, last cold_late) waits on parity (parity0 + 1 + k) mod 2:
                    // the untimed first load used parity0
                    auto add = [&](const char* state, int rep, int k, uint64_t cyc) {
                        char b[512];
                        std::snprintf(b, sizeof b, "%s,%s,%u,%u,%u,%s,%s,%s,%d,%u,%llu,%u,1,%llu,%llu", run.prefix(li, s).c_str(),
                                      bname.c_str(), bx.x, bx.y, bx.z, swname, sname, state, rep, (parity0 + 1 + k) & 1,
                                      (unsigned long long)cyc, smid, (unsigned long long)extent, (unsigned long long)written);
                        csv.add(b);
                    };
                    add("cold", 0, 0, h[0]);
                    add("prefetched", 0, 1, h[1]);
                    for (uint32_t r = 0; r < kWarmReps; ++r) add("warm", (int)r, 2 + (int)r, h[2 + r]);
                    add("cold_late", 0, 2 + (int)kWarmReps, h[2 + kWarmReps]);
                    sums[0] += h[0];
                    sums[1] += h[1];
                    for (uint32_t r = 0; r < kWarmReps; ++r) sums[2] += double(h[2 + r]) / kWarmReps;
                    sums[3] += h[2 + kWarmReps];
                    csv.flush();
                    ++li;
                }
                csv.sync();
                std::printf("k4 %-10s swizzle %-4s %-3s: cold %7.1f  prefetched %7.1f  warm %7.1f  cold_late %7.1f cycles (mean)\n",
                            bname.c_str(), swname, sname, sums[0] / launches, sums[1] / launches, sums[2] / launches,
                            sums[3] / launches);
                std::fflush(stdout);
            }
        }
    }
    run.lock_clocks_note("k4", "after");
    cudaFree(d_t);
    cudaFree(d_pos);
    cudaFree(d_out);
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "k4";
    hb::Run run(EXP, argc, argv);
    if (mode == "clockcheck") run.lock_clocks_note(hb::arg_str(argc, argv, "--label", "check"), "check");
    else if (mode == "k4") run_k4(run, hb::has_flag(argc, argv, "--quick"));
    else hb::die("unknown mode " + mode);
    run.complete();
    return 0;
}
