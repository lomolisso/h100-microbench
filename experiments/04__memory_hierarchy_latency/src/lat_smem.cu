/*
 * lat_smem.cu — experiment 04, K1: dependent-load latency of the SM's own shared memory, through the local
 * shared-memory path and through the DSMEM interface addressed at the own CTA.
 *
 * Usage: bin/lat_smem {k1|clockcheck} [--label L]. One invocation = one run_id (results/04__memory_hierarchy_latency__runs.csv).
 *
 * Kernel k1_chase<V>(perm, n_words, stride, hops, zero, out): grid 1 block × 32 threads, dynamic smem = n_words × 4 B.
 *   Chain: word i holds the address of word next(i), next(i) = (i + stride) mod n_words for stride ∈ {1, 32} words
 *   (32 words = every hop hits the same bank), or a random single-cycle permutation of the n_words words
 *   (pchase_chain.hpp, Sattolo, seed kChainSeed) for stride "random". All 32 threads write the chain, then thread 0 times.
 *   Timed region: t0 = %clock64; `hops` dependent loads p = ld [p] (#pragma unroll 1 outer loop around 16 inlined
 *   loads); t1 = read_clock64_after(p) (cannot issue before the last load has returned). Variants:
 *     lds           ld.shared.u32            → SASS LDS
 *     lds_volatile  ld.volatile.shared.u32   → SASS LDS (volatile: no reordering/caching by the compiler)
 *     dsmem_if      ld.shared::cluster.u32 on addresses mapa'd to the CTA's own rank, launched as an explicit
 *                   cluster of 1 (cudaLaunchKernelEx + ClusterDimension {1,1,1}) → SASS generic LD.E through the
 *                   DSMEM interface ("loopback")
 *   Sweep: 3 variants × stride {1, 32, random} × n_words {1024, 8192, 49152} × hops {4096, 16384};
 *   2 warm-up launches + 10 recorded launches per configuration. Latency = slope between the two hop counts
 *   (removes the fixed timer/loop region, see 01); cycles_per_hop = total_cycles / hops is the raw ratio.
 *   → results/04__latency__smem.csv: variant, stride, n_words, hops, cycles_per_hop, total_cycles, smid
 *     (one row per launch; mandatory columns first, clocks from the NVML sample taken after the launch).
 */
#include <string>
#include <vector>

#include "harness.cuh"
#include "host.hpp"
#include "pchase_chain.hpp"

static const char* EXP = "04__memory_hierarchy_latency";
static constexpr int kWarmups = 2, kRecorded = 10, kUnroll = 16;

enum Variant { V_LDS = 0, V_LDS_VOL = 1, V_DSMEM = 2 };
static const char* kVariantName[] = {"lds", "lds_volatile", "dsmem_if"};

struct K1Out { uint64_t cycles; uint32_t sink, smid; };

template <int V>
__device__ __forceinline__ uint32_t load_step(uint32_t p) {
    uint32_t v;
    if constexpr (V == V_LDS) asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(p) : "memory");
    else if constexpr (V == V_LDS_VOL) asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(v) : "r"(p) : "memory");
    else asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(v) : "r"(p) : "memory");
    return v;
}

template <int V>
__global__ void __launch_bounds__(32) k1_chase(const uint32_t* perm, int n_words, int stride, int hops, uint64_t zero, K1Out* out) {
    extern __shared__ uint32_t s[];
    const uint32_t rank = (V == V_DSMEM) ? read_cluster_regs().ctarank : 0;
    for (int i = threadIdx.x; i < n_words; i += blockDim.x) {
        const int nx = perm ? (int)perm[i] : (i + stride) % n_words;
        uint32_t a = smem_u32(&s[nx]);
        if constexpr (V == V_DSMEM) a = mapa_u32(a, rank);
        s[i] = a;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    uint32_t p = smem_u32(&s[0]);
    if constexpr (V == V_DSMEM) p = mapa_u32(p, rank);
    const uint64_t t0 = read_clock64();
#pragma unroll 1
    for (int i = 0; i < hops / kUnroll; ++i) {
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) p = load_step<V>(p);
    }
    const uint64_t t1 = read_clock64_after(p, zero);
    out->cycles = t1 - t0;
    out->sink = p;
    out->smid = read_smid();
}

template <int V>
static void run_variant(hb::Run& run, hb::CsvWriter& csv, K1Out* d_out, long long& launch_idx) {
    hb::ClusterLaunch cl;
    cl.grid = dim3(1);
    cl.block = dim3(32);
    cl.explicit_cluster = (V == V_DSMEM);  // a cluster of 1: needed for the shared::cluster window
    for (int n : {1024, 8192, 49152}) {
        cl.smem = (size_t)n * sizeof(uint32_t);
        // random: single-cycle permutation of the n words
        const std::vector<uint32_t> perm = chain::sattolo_cycle((uint32_t)n);
        uint32_t* d_perm = nullptr;
        CUDA_CHECK(cudaMalloc(&d_perm, n * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_perm, perm.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));
        for (int stride : {1, 32, 0}) {  // 0 = random
            const std::string sname = stride ? std::to_string(stride) : "random";
            for (int hops : {4096, 16384}) {
                const std::string label = std::string("k1_") + kVariantName[V] + "_s" + sname + "_n" + std::to_string(n) + "_h" + std::to_string(hops);
                auto launch = [&] {
                    CUDA_CHECK(cl.launch(k1_chase<V>, stride ? nullptr : (const uint32_t*)d_perm, n, stride, hops, (uint64_t)0, d_out));
                };
                hb::warmup(launch, kWarmups);
                double last = 0;
                for (int l = 0; l < kRecorded; ++l) {
                    launch();
                    CUDA_CHECK(cudaDeviceSynchronize());
                    const hb::ClockSample cs = run.clock_note(label, "launch", launch_idx);
                    K1Out o;
                    CUDA_CHECK(cudaMemcpy(&o, d_out, sizeof o, cudaMemcpyDeviceToHost));
                    last = double(o.cycles) / hops;
                    char b[512];
                    std::snprintf(b, sizeof b, "%s,%s,%s,%d,%d,%.4f,%llu,%u", run.prefix(launch_idx, cs).c_str(), kVariantName[V],
                                  sname.c_str(), n, hops, last, (unsigned long long)o.cycles, o.smid);
                    csv.add(b);
                    csv.flush();
                    ++launch_idx;
                }
                std::printf("k1 %-12s stride %-6s n %5d hops %5d: %.3f cycles/hop\n", kVariantName[V], sname.c_str(), n, hops, last);
            }
            csv.sync();
        }
        CUDA_CHECK(cudaFree(d_perm));
    }
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "k1";
    hb::Run run(EXP, argc, argv);
    const std::string label = hb::arg_str(argc, argv, "--label", mode.c_str());
    if (mode == "clockcheck") {
        run.lock_clocks_note(label, "check");
    } else if (mode == "k1") {
        hb::CsvWriter csv(hb::Run::results("04__latency__smem.csv"),
                          hb::with_mandatory({"variant", "stride", "n_words", "hops", "cycles_per_hop", "total_cycles", "smid"}));
        K1Out* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(K1Out)));
        long long launch_idx = 0;
        run.lock_clocks_note("k1", "before");
        run_variant<V_LDS>(run, csv, d_out, launch_idx);
        run_variant<V_LDS_VOL>(run, csv, d_out, launch_idx);
        run_variant<V_DSMEM>(run, csv, d_out, launch_idx);
        run.lock_clocks_note("k1", "after");
        CUDA_CHECK(cudaFree(d_out));
    } else {
        hb::die("unknown mode " + mode);
    }
    run.complete();
    return 0;
}
