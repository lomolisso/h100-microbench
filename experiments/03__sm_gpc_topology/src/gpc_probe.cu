/*
 * gpc_probe.cu — experiment 03: which logical SMs (%smid) form a GPC on this card.
 *
 * Usage: bin/gpc_probe {cosched|clockdom|ncu|clockcheck} [--label L]. Each invocation is one run (run_id)
 * registered in results/03__sm_gpc_topology__runs.csv. Every recorded launch is followed by an NVML clock sample
 * (clock log); its SM/memory clocks fill the mandatory columns of that launch's rows, and a throttle reason or a
 * clock below the lock gives clock_ok = 0 (the launch is kept and counted, never dropped).
 *
 * cosched — cluster co-scheduling probe. All CTAs of a cluster run on SMs of one GPC, so SMs that ever share a
 *   cluster belong to one co-scheduling domain (union-find in src/build_map.py).
 *   Kernel sleep_kernel(rec, cycles): dynamic shared memory = smem_bytes (allocated, unused: it only sets how many
 *   CTAs fit on an SM). Every thread spins on %clock64 for `cycles` = 100 000 SM cycles (≈ 61 µs, so all CTAs of a
 *   wave overlap), the block synchronises, and thread 0 stores {blockIdx.x, %smid, %clusterid.x, %cluster_ctarank,
 *   %cluster_nctarank, %is_explicit_cluster} to rec[blockIdx.x]. No timed region: only the placement is used.
 *   Sweep (policy Default, cudaLaunchKernelEx with ClusterDimension {CS,1,1}; CS 16 sets NonPortableClusterSize):
 *     CS ∈ {16, 8, 4, 3, 2} × waves ∈ {1, 3} × smem ∈ {200 KB, 120 KB, 80 KB} × block ∈ {1024, 512} = 60 configs;
 *     grid = waves · cudaOccupancyMaxActiveClusters(config) · CS. 200/120 KB allow 1 CTA/SM, 80 KB allows 2.
 *     2 warm-ups per config, then 50 rounds; one round launches all 60 configs in order, so smem and block size
 *     change from one launch to the next (different SM subsets are free when a cluster is placed).
 *   The host checks every record: all blocks wrote, clusterid = block / CS, rank = block % CS, nctarank = CS,
 *   is_explicit_cluster = 1. Violations, and clusters whose CTAs share an SM, are counted and printed.
 *     → results/03__sm_gpc_topology__edges.csv (one row per CTA): launch_id (round 0–49 of this config), cs,
 *       policy, smem_bytes, block_threads, cluster_id, rank, smid, grid_blocks, waves, clock_ok.
 *
 * clockdom — per-SM clock rate. The SMs of a GPC share one clock; under the lock the rate of different GPCs differs
 *   by a small, launch-dependent amount, so SMs with identical rates in every launch share a clock domain.
 *   Kernel clockdom_kernel(rec, window_ns), grid = SM count × 32 threads, 200 KB dynamic smem → exactly 1 CTA/SM
 *   (host checks all %smid distinct; otherwise the launch is repeated and counted). Thread 0 only:
 *     waits for a %globaltimer tick edge and reads %clock64 right after it (g0, c0); spins until the first
 *     %globaltimer value ≥ g0 + window_ns (also a tick edge) and reads %clock64 right after it (g1, c1).
 *   Both endpoints sit on tick edges, so g1 − g0 is exact in ns (the 32 ns quantisation cancels) and the error is
 *   the edge-detection loop time (tens of cycles) on each end. window_ns = 2^25 ns (33.6 ms, ≈ 5.5e7 cycles) →
 *   rate resolution ≈ 0.002 MHz. sm_mhz = (c1 − c0) / (g1 − g0) · 1000. 2 warm-ups, then 50 launches.
 *     → results/03__sm_gpc_topology__clock_domains.csv: smid, span_ns, span_cycles, sm_mhz, retries.
 *
 * ncu — the launches that src/run.sh profiles (no warm-ups): sleep_kernel at one wave for CS ∈ {2, 3, 4, 8, 16}
 *   × smem ∈ {200 KB (1 CTA/SM), 80 KB (2 CTAs/SM)}, 1024 threads. Per-GPC cluster counts (gpc__cgas_launched)
 *   show the GPC sizes as the hardware counts them. Writes bin/ncu_manifest.csv for src/post_profile.py.
 * clockcheck — one clock-log row and the nvidia-smi clock report.
 */
#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include "harness.cuh"
#include "host.hpp"

static const char* EXP = "03__sm_gpc_topology";
static constexpr int kWarmups = 2;
static constexpr uint64_t kSleepCycles = 100000;

struct Record {
    uint32_t block, smid, clusterid, rank, nrank, is_explicit;
};

__device__ __forceinline__ uint32_t read_is_explicit_cluster() {
    uint32_t r;
    asm volatile("{\n\t.reg .pred p;\n\tmov.pred p, %%is_explicit_cluster;\n\tselp.u32 %0, 1, 0, p;\n\t}" : "=r"(r));
    return r;
}

__global__ void __launch_bounds__(1024) sleep_kernel(Record* rec, uint64_t cycles) {
    const uint64_t c0 = read_clock64();
    while (read_clock64() - c0 < cycles) {
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const ClusterRegs c = read_cluster_regs();
        Record r;
        r.block = blockIdx.x;
        r.smid = read_smid();
        r.clusterid = c.clusterid_x;
        r.rank = c.ctarank;
        r.nrank = c.nctarank;
        r.is_explicit = read_is_explicit_cluster();
        rec[blockIdx.x] = r;
    }
}

struct DomRecord {
    uint32_t smid, pad;
    uint64_t g0, c0, g1, c1;
};

__global__ void __launch_bounds__(32) clockdom_kernel(DomRecord* rec, uint64_t window_ns) {
    if (threadIdx.x != 0) return;
    uint64_t g0, c0, g1, c1;
    const uint64_t g = read_globaltimer();
    do {  // first read after a tick edge, %clock64 right behind it
        g0 = read_globaltimer();
        c0 = read_clock64();
    } while (g0 == g);
    do {  // first read at or past g0 + window: again the first read after a tick edge
        g1 = read_globaltimer();
        c1 = read_clock64();
    } while (g1 - g0 < window_ns);
    DomRecord r;
    r.smid = read_smid();
    r.pad = 0;
    r.g0 = g0;
    r.c0 = c0;
    r.g1 = g1;
    r.c1 = c1;
    rec[blockIdx.x] = r;
}

// ---------------------------------------------------------------- cluster launches
struct Config {
    int cs, waves, threads;
    size_t smem;
    int api_clusters;  // cudaOccupancyMaxActiveClusters
    int grid() const { return waves * api_clusters * cs; }
};

static hb::ClusterLaunch make_launch(int cs, int grid, int threads, size_t smem) {
    hb::ClusterLaunch cl;
    cl.grid = dim3(grid);
    cl.block = dim3(threads);
    cl.smem = smem;
    cl.cluster = dim3(cs, 1, 1);
    cl.explicit_cluster = true;
    cl.policy = cudaClusterSchedulingPolicyDefault;
    return cl;
}

static Config make_config(int cs, int waves, int threads, size_t smem) {
    Config c{cs, waves, threads, smem, 0};
    c.api_clusters = make_launch(cs, cs, threads, smem).max_active_clusters(sleep_kernel);
    if (c.api_clusters <= 0) hb::die("cudaOccupancyMaxActiveClusters = 0 for CS " + std::to_string(cs));
    return c;
}

struct Launcher {
    Record* d_rec = nullptr;
    std::vector<Record> h;
    explicit Launcher(int max_grid) : h(max_grid) { CUDA_CHECK(cudaMalloc(&d_rec, max_grid * sizeof(Record))); }
    ~Launcher() { cudaFree(d_rec); }
    void run(const Config& c, bool keep_records = true) {
        if (c.grid() > (int)h.size()) hb::die("grid larger than the record buffer");
        hb::ClusterLaunch::Prepared p;
        make_launch(c.cs, c.grid(), c.threads, c.smem).prepare(sleep_kernel, p);  // also sets the smem attribute
        CUDA_CHECK(cudaMemset(d_rec, 0xff, c.grid() * sizeof(Record)));
        CUDA_CHECK(cudaLaunchKernelEx(&p.cfg, sleep_kernel, d_rec, kSleepCycles));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        if (keep_records) CUDA_CHECK(cudaMemcpy(h.data(), d_rec, c.grid() * sizeof(Record), cudaMemcpyDeviceToHost));
    }
};

// Consistency of the records with the launch configuration; returns the number of violations (first 3 printed).
static long check_records(const std::vector<Record>& h, const Config& c) {
    long bad = 0;
    for (int i = 0; i < c.grid(); ++i) {
        const Record& r = h[i];
        const bool ok = r.block == (uint32_t)i && r.clusterid == (uint32_t)(i / c.cs) && r.rank == (uint32_t)(i % c.cs) &&
                        r.nrank == (uint32_t)c.cs && r.is_explicit == 1u;
        if (!ok && bad++ < 3)
            std::fprintf(stderr, "WARNING cs %d grid %d: block %d record {block %u, cluster %u, rank %u/%u, explicit %u}\n", c.cs,
                         c.grid(), i, r.block, r.clusterid, r.rank, r.nrank, r.is_explicit);
    }
    return bad;
}

static int colocated_clusters(const std::vector<Record>& h, const Config& c) {
    int n = 0;
    for (int k = 0; k + c.cs <= c.grid(); k += c.cs) {
        std::set<uint32_t> s;
        for (int i = k; i < k + c.cs; ++i) s.insert(h[i].smid);
        n += (int)s.size() < c.cs;
    }
    return n;
}

static std::vector<Config> cosched_configs() {
    std::vector<Config> v;
    for (int cs : {16, 8, 4, 3, 2})
        for (int waves : {1, 3})
            for (size_t smem : {200 * 1024, 120 * 1024, 80 * 1024})
                for (int threads : {1024, 512}) v.push_back(make_config(cs, waves, threads, smem));
    return v;
}

// ================================================================ cosched
static void run_cosched(hb::Run& run) {
    const int kRounds = 50;
    const std::vector<Config> cfgs = cosched_configs();
    hb::CsvWriter csv(hb::Run::results("03__sm_gpc_topology__edges.csv"),
                      hb::with_mandatory({"launch_id", "cs", "policy", "smem_bytes", "block_threads", "cluster_id", "rank", "smid",
                                          "grid_blocks", "waves", "clock_ok"}));
    int max_grid = 0;
    for (const Config& c : cfgs) {
        max_grid = std::max(max_grid, c.grid());
        std::printf("cs %2d waves %d smem %6zu threads %4d: api %3d clusters, grid %4d\n", c.cs, c.waves, c.smem, c.threads,
                    c.api_clusters, c.grid());
    }
    Launcher L(max_grid);
    for (const Config& c : cfgs)
        for (int w = 0; w < kWarmups; ++w) L.run(c, false);

    long long idx = 0;
    long bad_records = 0, bad_clock = 0, colocated = 0;
    char b[256];
    run.clock_note("cosched", "before");
    for (int round = 0; round < kRounds; ++round) {
        for (const Config& c : cfgs) {
            L.run(c);
            const hb::ClockSample s = run.clock_note("cosched", "launch", idx);
            bad_clock += s.bad();
            bad_records += check_records(L.h, c);
            colocated += colocated_clusters(L.h, c);
            const std::string prefix = run.prefix(idx, s);
            for (int i = 0; i < c.grid(); ++i) {
                const Record& r = L.h[i];
                std::snprintf(b, sizeof b, "%s,%d,%d,default,%zu,%d,%u,%u,%u,%d,%d,%d", prefix.c_str(), round, c.cs, c.smem,
                              c.threads, r.clusterid, r.rank, r.smid, c.grid(), c.waves, (int)!s.bad());
                csv.add(b);
            }
            csv.flush();
            ++idx;
        }
        csv.sync();
        if (round % 10 == 9) std::printf("cosched: round %d/%d done\n", round + 1, kRounds);
    }
    run.clock_note("cosched", "after");
    std::printf("cosched: %lld launches, record violations %ld, clock-flagged launches %ld, clusters with a shared SM %ld\n", idx,
                bad_records, bad_clock, colocated);
}

// ================================================================ clockdom
static void run_clockdom(hb::Run& run) {
    const int sm = run.dev().sm_count, kLaunches = 50;
    const uint64_t window_ns = 1ull << 25;
    hb::CsvWriter csv(hb::Run::results("03__sm_gpc_topology__clock_domains.csv"),
                      hb::with_mandatory({"smid", "span_ns", "span_cycles", "sm_mhz", "retries"}));
    DomRecord* d_rec;
    CUDA_CHECK(cudaMalloc(&d_rec, sm * sizeof(DomRecord)));
    std::vector<DomRecord> h(sm);
    hb::ClusterLaunch cl;
    cl.grid = dim3(sm);
    cl.block = dim3(32);
    cl.smem = 200 * 1024;
    hb::ClusterLaunch::Prepared p;
    cl.prepare(clockdom_kernel, p);
    auto launch = [&]() {
        CUDA_CHECK(cudaLaunchKernelEx(&p.cfg, clockdom_kernel, d_rec, window_ns));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h.data(), d_rec, sm * sizeof(DomRecord), cudaMemcpyDeviceToHost));
        std::vector<uint32_t> ids(sm);
        for (int i = 0; i < sm; ++i) ids[i] = h[i].smid;
        return hb::all_distinct(ids);
    };
    for (int w = 0; w < kWarmups; ++w) launch();
    long long idx = 0;
    long total_retries = 0, bad_clock = 0;
    char b[256];
    run.clock_note("clockdom", "before");
    for (int l = 0; l < kLaunches; ++l) {
        int retries = 0;
        while (!launch()) {
            if (++retries > 10) hb::die("clockdom: %smid not distinct in 10 consecutive launches");
        }
        total_retries += retries;
        const hb::ClockSample s = run.clock_note("clockdom", "launch", idx);
        bad_clock += s.bad();
        const std::string prefix = run.prefix(idx, s);
        for (const DomRecord& r : h) {
            const uint64_t ns = r.g1 - r.g0, cyc = r.c1 - r.c0;
            std::snprintf(b, sizeof b, "%s,%u,%llu,%llu,%.6f,%d", prefix.c_str(), r.smid, (unsigned long long)ns,
                          (unsigned long long)cyc, double(cyc) / double(ns) * 1e3, retries);
            csv.add(b);
        }
        csv.flush();
        ++idx;
    }
    csv.sync();
    run.clock_note("clockdom", "after");
    CUDA_CHECK(cudaFree(d_rec));
    std::printf("clockdom: %lld launches, repeated launches (smid not distinct) %ld, clock-flagged launches %ld\n", idx, total_retries,
                bad_clock);
}

// ================================================================ profiled launches (ncu)
static void run_ncu(hb::Run& run) {
    std::vector<Config> cfgs;
    for (size_t smem : {200 * 1024, 80 * 1024})
        for (int cs : {2, 3, 4, 8, 16}) cfgs.push_back(make_config(cs, 1, 1024, smem));
    int max_grid = 0;
    for (const Config& c : cfgs) max_grid = std::max(max_grid, c.grid());
    Launcher L(max_grid);
    FILE* f = std::fopen("bin/ncu_manifest.csv", "w");
    if (!f) hb::die("cannot write bin/ncu_manifest.csv");
    std::fprintf(f, "%s\n", hb::join(hb::with_mandatory({"cs", "policy", "smem_bytes", "block_threads", "grid_blocks", "waves",
                                                         "api_max_active_clusters", "sleep_cycles"})).c_str());
    long long idx = 0;
    for (const Config& c : cfgs) {
        L.run(c, false);
        const hb::ClockSample s = run.clock_note("ncu", "launch", idx);
        std::fprintf(f, "%s,%d,default,%zu,%d,%d,%d,%d,%llu\n", run.prefix(idx, s).c_str(), c.cs, c.smem, c.threads, c.grid(), c.waves,
                     c.api_clusters, (unsigned long long)kSleepCycles);
        ++idx;
    }
    std::fclose(f);
}

// ================================================================
int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s {cosched|clockdom|ncu|clockcheck} [--label L]\n", argv[0]);
        return 1;
    }
    const std::string cmd = argv[1];
    const std::string label = hb::arg_str(argc, argv, "--label", cmd.c_str());
    hb::Run run(EXP, argc, argv);
    if (cmd == "cosched") run_cosched(run);
    else if (cmd == "clockdom") run_clockdom(run);
    else if (cmd == "ncu") run_ncu(run);
    else if (cmd == "clockcheck") run.lock_clocks_note(label, "check");
    else hb::die("unknown subcommand " + cmd);
    run.complete();
    return 0;
}
