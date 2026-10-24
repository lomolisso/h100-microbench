/*
 * lat_footprint.cu — experiment 04, K2: latency of a dependent global-memory pointer chase versus working-set size
 * (L1 → L2 → HBM), for three load cache operators, from 4 SMs in different GPCs.
 *
 * Usage: bin/lat_footprint {k2|ncu|clockcheck} [--sms a,b,c,d] [--quick] [--label L]. One invocation = one run_id.
 *
 * Chain (pchase_chain.hpp): one 2 GiB cudaMalloc buffer; footprint F = n lines of 128 B; line i holds at its first
 *   qword the address of line next[i], next = one random cycle over the n lines (Sattolo, kChainSeed). The host
 *   builds next[] for each F, writes it into a zeroed pinned host buffer and copies F bytes with one cudaMemcpy
 *   ("fill from host"); the chase starts at line 0.
 * Placement: claim_target_sm — 2 × 132 one-warp blocks, no dynamic smem; the first block on the probe SM runs the
 *   chase, all others return at once (only one thread on the GPU works). The host checks the claim and re-launches
 *   if no block ran on the probe SM (counted in `retries`). PreferredSharedMemoryCarveout = MaxL1, so L1 gets the
 *   largest carveout the hardware allows for a block without dynamic smem (ncu reports the configuration).
 * Kernel k2_chase<V>(start, warm_hops, hops, target, claim, zero, out), thread 0 of the winning block:
 *   warm_hops untimed dependent loads, then t0 = read_clock64_after(p) (after the warm pass has returned), `hops`
 *   dependent loads p = ld [p] (#pragma unroll 1 around 16 inlined loads), t1 = read_clock64_after(p).
 *   V: ca = ld.global.ca.u64 (L1 allowed), cg = ld.global.cg.u64 (L1 bypass), cv = ld.global.cv.u64 (volatile,
 *   "don't cache"; whether it bypasses L2 is checked with the ncu run and the curve).
 * Sweep: F = 1 KiB · 1.25^k rounded to 128 B, up to 2 GiB, plus 128 KiB, 25 MiB, 30 MiB, 50 MiB, 60 MiB, 2 GiB.
 *   hops = max(n, 2^20) rounded up to 16. For each F, for each V ∈ {ca, cg, cv}, for each probe SM (--sms: 2 SMs in
 *   each of the two L2-partition groups found by K3's matrix, chosen by src/pick_toucher.py --k2-sms; default the
 *   first SM of GPCs 1, 3, 5, 7 of the 03 map):
 *     n ≤ 2^20: every launch first chases all n lines untimed (warm_hops = n, inside the same launch, so that L1
 *               contents survive), then the timed hops; 5 recorded launches.
 *     n > 2^20: one untimed warm launch of n hops, then 3 recorded launches of n hops (warm_hops = 0).
 *   With ≥ 2^20 hops per timed region the fixed timer region (≤ 192 cycles, 01) is < 2e-4 cycles/hop: not subtracted.
 *   --quick: F ∈ {16 KiB, 1 MiB, 8 MiB, 256 MiB}, one probe SM, 2 recorded launches (smoke test).
 *   → results/04__latency__footprint.csv: load_variant, footprint_bytes, hops, warm_hops, rep, cycles_per_hop,
 *     total_cycles, smid, gpc, retries (one row per recorded launch).
 * ncu mode (profiled with application replay, see src/run.sh): for V × F ∈ {64 KiB, 8 MiB, 256 MiB} on the first
 *   probe SM: fill, one warm launch (k2_chase, n hops), then one launch of k2_ncu<V> (same body; the kernel
 *   src/run.sh profiles) with warm_hops = 0 and hops = max(n, 2^20). Writes bin/ncu_manifest.csv (one row per k2_ncu launch) for src/post_ncu.py.
 */
#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "gpc_map.hpp"
#include "harness.cuh"
#include "host.hpp"
#include "pchase_chain.hpp"

static const char* EXP = "04__memory_hierarchy_latency";
static constexpr int kUnroll = 16, kMaxRetries = 20;
static constexpr uint64_t kMinHops = 1ull << 20, kBufBytes = 2ull << 30;

enum Variant { V_CA = 0, V_CG = 1, V_CV = 2 };
static const char* kVariantName[] = {"ca", "cg", "cv"};

struct K2Out { uint64_t cycles, sink; uint32_t smid; };

template <int V>
__device__ __forceinline__ uint64_t ld_step(uint64_t p) {
    uint64_t v;
    if constexpr (V == V_CA) asm volatile("ld.global.ca.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    else if constexpr (V == V_CG) asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    else asm volatile("ld.global.cv.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

template <int V>
__device__ __forceinline__ void chase_body(const uint64_t* start, uint64_t warm_hops, uint64_t hops, uint32_t target,
                                           uint32_t* claim, uint64_t zero, K2Out* out) {
    if (!claim_target_sm(target, claim)) return;
    if (threadIdx.x != 0) return;
    uint64_t p = (uint64_t)start;
#pragma unroll 1
    for (uint64_t i = 0; i < warm_hops / kUnroll; ++i) {
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) p = ld_step<V>(p);
    }
    const uint64_t t0 = read_clock64_after(p, zero);
#pragma unroll 1
    for (uint64_t i = 0; i < hops / kUnroll; ++i) {
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) p = ld_step<V>(p);
    }
    const uint64_t t1 = read_clock64_after(p, zero);
    out->cycles = t1 - t0;
    out->sink = p;
    out->smid = read_smid();
}

template <int V>
__global__ void __launch_bounds__(32) k2_chase(const uint64_t* start, uint64_t warm_hops, uint64_t hops, uint32_t target,
                                                uint32_t* claim, uint64_t zero, K2Out* out) {
    chase_body<V>(start, warm_hops, hops, target, claim, zero, out);
}
// Identical body under another name: the only kernel that src/run.sh profiles.
template <int V>
__global__ void __launch_bounds__(32) k2_ncu(const uint64_t* start, uint64_t warm_hops, uint64_t hops, uint32_t target,
                                              uint32_t* claim, uint64_t zero, K2Out* out) {
    chase_body<V>(start, warm_hops, hops, target, claim, zero, out);
}

static uint64_t round16(uint64_t h) { return (h + kUnroll - 1) / kUnroll * kUnroll; }

static std::vector<uint64_t> footprints(bool quick) {
    if (quick) return {16ull << 10, 1ull << 20, 8ull << 20, 256ull << 20};
    std::vector<uint64_t> f;
    for (double x = 1024; x < (double)kBufBytes; x *= 1.25) f.push_back((uint64_t)std::llround(x / 128.0) * 128);
    for (uint64_t e : {128ull << 10, 25ull << 20, 30ull << 20, 50ull << 20, 60ull << 20, (unsigned long long)kBufBytes}) f.push_back(e);
    std::sort(f.begin(), f.end());
    f.erase(std::unique(f.begin(), f.end()), f.end());
    return f;
}

struct Ctx {
    hb::Run& run;
    hb::SmClaim claim;
    uint64_t* d_buf = nullptr;
    uint64_t* h_buf = nullptr;
    K2Out* d_out = nullptr;
    int blocks = 0;
    explicit Ctx(hb::Run& r) : run(r), blocks(hb::SmClaim::blocks(r.dev().sm_count)) {
        CUDA_CHECK(cudaMalloc(&d_buf, kBufBytes));
        CUDA_CHECK(cudaMallocHost(&h_buf, kBufBytes));
        std::memset(h_buf, 0, kBufBytes);
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(K2Out)));
        for (auto k : {k2_chase<V_CA>, k2_chase<V_CG>, k2_chase<V_CV>, k2_ncu<V_CA>, k2_ncu<V_CG>, k2_ncu<V_CV>})
            CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, (int)cudaSharedmemCarveoutMaxL1));
    }
    ~Ctx() { cudaFree(d_buf); cudaFreeHost(h_buf); cudaFree(d_out); }

    // Writes the chain of footprint F into the device buffer (one cudaMemcpy of F bytes); returns n lines.
    uint64_t fill(uint64_t F) {
        const uint32_t n = (uint32_t)(F / chain::kLineBytes);
        const std::vector<uint32_t> next = chain::sattolo_cycle(n);
        const uint64_t base = (uint64_t)d_buf;
        for (uint32_t i = 0; i < n; ++i) h_buf[(uint64_t)i * chain::kQwordsPerLine] = base + (uint64_t)next[i] * chain::kLineBytes;
        CUDA_CHECK(cudaMemcpy(d_buf, h_buf, F, cudaMemcpyHostToDevice));
        return n;
    }

    // One claimed launch; returns the number of retries needed (dies after kMaxRetries).
    template <class K>
    int launch(K kernel, uint64_t warm_hops, uint64_t hops, uint32_t target) {
        for (int r = 0; r <= kMaxRetries; ++r) {
            claim.reset();
            kernel<<<blocks, 32>>>(d_buf, warm_hops, hops, target, claim.ptr(), 0ull, d_out);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            if (claim.ok(target)) return r;
        }
        hb::die("no block ran on SM " + std::to_string(target) + " after retries");
    }
    K2Out out() const {
        K2Out o;
        CUDA_CHECK(cudaMemcpy(&o, d_out, sizeof o, cudaMemcpyDeviceToHost));
        return o;
    }
};

template <int V>
static void run_point(Ctx& c, hb::CsvWriter& csv, uint64_t F, uint64_t n, uint32_t sm, int gpc, int reps, long long& launch_idx) {
    const std::string label = std::string("k2_") + kVariantName[V] + "_F" + std::to_string(F) + "_sm" + std::to_string(sm);
    const bool inkernel_warm = n <= kMinHops;
    const uint64_t hops = round16(std::max<uint64_t>(n, kMinHops));
    const uint64_t warm = inkernel_warm ? round16(n) : 0;
    int retries = 0;
    if (!inkernel_warm) retries += c.launch(k2_chase<V>, round16(n), 0, sm);  // untimed warm launch
    double sum = 0;
    for (int r = 0; r < reps; ++r) {
        retries += c.launch(k2_chase<V>, warm, hops, sm);
        const hb::ClockSample s = c.run.clock_note(label, "launch", launch_idx);
        const K2Out o = c.out();
        if (o.smid != sm) hb::die("claimed block reported smid " + std::to_string(o.smid));
        const double cph = double(o.cycles) / hops;
        sum += cph;
        char b[640];
        std::snprintf(b, sizeof b, "%s,%s,%llu,%llu,%llu,%d,%.4f,%llu,%u,%d,%d", c.run.prefix(launch_idx, s).c_str(), kVariantName[V],
                      (unsigned long long)F, (unsigned long long)hops, (unsigned long long)warm, r, cph,
                      (unsigned long long)o.cycles, o.smid, gpc, retries);
        csv.add(b);
        csv.flush();
        ++launch_idx;
    }
    csv.sync();
    std::printf("k2 %s F %11llu sm %3u: %8.2f cycles/hop (mean of %d), retries %d\n", kVariantName[V], (unsigned long long)F, sm,
                sum / reps, reps, retries);
    std::fflush(stdout);
}

static void run_k2(hb::Run& run, bool quick, const char* sms_arg) {
    const std::vector<int> gpc = load_gpc_map(run.dev().sm_count);
    std::vector<uint32_t> probes;
    if (const char* l = sms_arg) {  // --sms a,b,c,d (src/pick_toucher.py --k2-sms)
        std::stringstream ss(l);
        for (std::string t; std::getline(ss, t, ',');) probes.push_back((uint32_t)std::stoul(t));
    } else {
        for (int g : {1, 3, 5, 7}) probes.push_back((uint32_t)first_sm_of_gpc(gpc, g));
    }
    if (quick) probes.resize(1);
    hb::CsvWriter csv(hb::Run::results("04__latency__footprint.csv"),
                      hb::with_mandatory({"load_variant", "footprint_bytes", "hops", "warm_hops", "rep", "cycles_per_hop",
                                          "total_cycles", "smid", "gpc", "retries"}));
    Ctx c(run);
    long long launch_idx = 0;
    run.lock_clocks_note("k2", "before");
    for (uint64_t F : footprints(quick)) {
        const uint64_t n = c.fill(F);
        const int reps = quick ? 2 : (n <= kMinHops ? 5 : 3);
        for (uint32_t sm : probes) {
            run_point<V_CA>(c, csv, F, n, sm, gpc[sm], reps, launch_idx);
            run_point<V_CG>(c, csv, F, n, sm, gpc[sm], reps, launch_idx);
            run_point<V_CV>(c, csv, F, n, sm, gpc[sm], reps, launch_idx);
        }
    }
    run.lock_clocks_note("k2", "after");
}

template <int V>
static void ncu_point(Ctx& c, FILE* man, uint64_t F, uint64_t n, uint32_t sm, int gpc, long long& launch_idx) {
    const uint64_t hops = round16(std::max<uint64_t>(n, kMinHops));
    int retries = c.launch(k2_chase<V>, round16(n), 0, sm);  // warm (not profiled)
    retries += c.launch(k2_ncu<V>, 0, hops, sm);
    const hb::ClockSample s = c.run.clock_note(std::string("ncu_") + kVariantName[V] + "_F" + std::to_string(F), "launch", launch_idx);
    const K2Out o = c.out();
    std::fprintf(man, "%s,%s,%llu,%llu,%.4f,%u,%d,%d\n", c.run.prefix(launch_idx, s).c_str(), kVariantName[V], (unsigned long long)F,
                 (unsigned long long)hops, double(o.cycles) / hops, o.smid, gpc, retries);
    std::fflush(man);
    ++launch_idx;
}

static void run_ncu(hb::Run& run) {
    const std::vector<int> gpc = load_gpc_map(run.dev().sm_count);
    const uint32_t sm = (uint32_t)first_sm_of_gpc(gpc, 1);
    FILE* man = std::fopen("bin/ncu_manifest.csv", "w");
    if (!man) hb::die("cannot write bin/ncu_manifest.csv");
    std::fprintf(man, "%s,load_variant,footprint_bytes,hops,cycles_per_hop_under_ncu,smid,gpc,retries\n", hb::join(hb::mandatory_cols()).c_str());
    Ctx c(run);
    long long launch_idx = 0;
    run.lock_clocks_note("ncu", "before");
    for (uint64_t F : {64ull << 10, 8ull << 20, 256ull << 20}) {
        const uint64_t n = c.fill(F);
        ncu_point<V_CA>(c, man, F, n, sm, gpc[sm], launch_idx);
        ncu_point<V_CG>(c, man, F, n, sm, gpc[sm], launch_idx);
        ncu_point<V_CV>(c, man, F, n, sm, gpc[sm], launch_idx);
    }
    run.lock_clocks_note("ncu", "after");
    std::fclose(man);
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "k2";
    hb::Run run(EXP, argc, argv);
    if (mode == "clockcheck") run.lock_clocks_note(hb::arg_str(argc, argv, "--label", "check"), "check");
    else if (mode == "k2") run_k2(run, hb::has_flag(argc, argv, "--quick"), hb::arg_str(argc, argv, "--sms", nullptr));
    else if (mode == "ncu") run_ncu(run);
    else hb::die("unknown mode " + mode);
    run.complete();
    return 0;
}
