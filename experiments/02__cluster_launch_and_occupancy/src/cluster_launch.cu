/*
 * cluster_launch.cu — experiment 02: cost of launching a thread-block-cluster kernel relative to a non-cluster
 * kernel with the same grid and block, the number of co-resident clusters per cluster size (and the SMs left
 * idle), and the rank -> SM placement under each cudaClusterSchedulingPolicy.
 *
 * Usage: bin/cluster_launch {overhead|occupancy|ncu|nsys|clockcheck} [--label L]. Each invocation is one run
 * (run_id) registered in results/02__cluster_launch_and_occupancy__runs.csv. Every recorded launch is followed by
 * an NVML clock sample (clock log); its SM/memory clock fill the mandatory columns of that launch's rows, and a
 * throttle reason or a clock below the lock gives clock_ok = 0 (the launch is kept and counted as discarded).
 *
 * Kernel sleep_kernel(rec, cycles), 1024 threads per block, dynamic shared memory = smem_bytes (allocated, not
 * used: it only limits how many CTAs fit on an SM). Thread 0 reads %globaltimer on entry (t_start). Every thread
 * then spins on %clock64 until `cycles` SM cycles have elapsed (no nanosleep), the block synchronises, and
 * thread 0 reads %globaltimer again (t_end) and stores {blockIdx.x, %smid, %clusterid.x, %cluster_ctarank,
 * %cluster_nctarank, %is_explicit_cluster, t_start, t_end} to rec[blockIdx.x]. Occupancy at 1024 threads is at
 * most 2 CTAs/SM (2048 threads/SM); 200 KB of dynamic smem allows exactly 1 CTA/SM.
 * Launches go through cudaLaunchKernelEx with a configuration prepared beforehand (hb::ClusterLaunch::prepare):
 *   no cluster  : no launch attributes (cs = 0, policy = none in the CSVs; %clusterid.x = blockIdx.x);
 *   cluster CS  : cudaLaunchAttributeClusterDimension = {CS,1,1} (also for CS = 1: an explicit cluster of one CTA)
 *                 + cudaLaunchAttributeClusterSchedulingPolicyPreference; CS 16 sets NonPortableClusterSizeAllowed.
 * The host checks every record: all blocks wrote, clusterid = block / CS, rank = block % CS, nctarank = CS,
 * is_explicit_cluster = 1 exactly for cluster launches. Violations are printed and counted, never hidden.
 *
 * overhead (K1) — first the event floor: 1000 event pairs with nothing in between (us_event; their median is
 *   ms_floor), and 200 event-timed launches of an empty 1024-thread kernel for no cluster and CS 1/2/4/8/16
 *   (grid = max(CS,1), 0 smem; the bare launch cost of each configuration).
 *     → results/02__cluster_launch__floor.csv: mode (empty_pair | empty_kernel), cs, policy, launches, us_event.
 *   Then sleep_kernel(cycles = 100000) for smem ∈ {0, 64 KB, 200 KB} × grid target g ∈ {1, 8, 16, 32, 64, 128, 132,
 *   264, 528, 1056, 2112}. The launch list of one (smem, g): no-cluster at g and at every padded size
 *   ceil(g/CS)·CS that differs from g, then CS 1, 2, 4, 8, 16 (policy Default) at their padded grid. Every entry
 *   gets 2 warm-ups, then 5 reps; one rep runs the whole list in order, so each cluster launch has a no-cluster
 *   launch with an identical grid in the same rep. Each launch is timed by its own event pair.
 *   overhead_pct = ((ms_event − ms_floor) / (ms_event of the no-cluster launch with the same grid in the same
 *   rep − ms_floor) − 1) · 100 (0 for no-cluster rows).
 *     → results/02__cluster_launch__overhead.csv: cs, policy, grid_blocks, smem_bytes, rep, ms_event, ms_floor,
 *       overhead_pct, grid_target, sleep_cycles, api_max_active_clusters (cudaOccupancyMaxActiveClusters; empty for
 *       no cluster), api_blocks_per_wave (max active clusters · CS, or max active blocks per SM · SM count), clock_ok.
 *
 * occupancy (K2) — sleep_kernel(cycles = 2000000, ≈ 1.2 ms, so all co-resident blocks overlap) for CS ∈ {2,4,8,16}
 *   × policy ∈ {Default, Spread, LoadBalancing} × smem ∈ {0, 64 KB, 200 KB} × grid = m · api_max_active_clusters
 *   · CS, m ∈ {1, 4}; 2 warm-ups, then 5 consecutive reps. Per launch the host computes from the records:
 *   first wave = clusters all of whose blocks started before the earliest block end; sms_used = distinct %smid of
 *   first-wave blocks; sms_idle = SM count − sms_used; max_ctas_per_sm over first-wave blocks; clusters_colocated
 *   = clusters (whole launch) whose CTAs occupy fewer than CS distinct SMs.
 *     → results/02__cluster_launch__occupancy.csv: cs, policy, smem_bytes, api_max_active_clusters,
 *       measured_first_wave_clusters, sms_used, sms_idle, grid_blocks, grid_mult, rep, max_ctas_per_sm,
 *       clusters_colocated, clock_ok.
 *
 * Placement — every block of every recorded K1 and K2 launch (warm-ups are not recorded):
 *     → results/02__cluster_launch__placement.csv: cs, policy, grid_blocks, smem_bytes, rep, block_idx, cluster_id,
 *       rank, smid, t_start_ns, t_end_ns (absolute %globaltimer), probe (k1_overhead | k2_occupancy), sleep_cycles.
 *       Joins with overhead/occupancy rows on (run_id, launch_idx).
 *
 * ncu — the launches that src/run.sh profiles: one per K2 configuration at m = 1 (cycles 2000000), then no
 *   cluster and CS 1/2/4/8/16 at grid 2112 / 200 KB (cycles 100000). No warm-ups. Writes bin/ncu_manifest.csv
 *   (profiled-launch order, configuration, mandatory columns) for src/post_profile.py.
 * nsys — for no cluster and CS 1/2/4/8/16: 2 warm-ups, then 20 back-to-back launches at grid 2112 / 200 KB
 *   (cycles 100000) without synchronisation in between; manifest bin/nsys_manifest.csv for src/post_profile.py.
 * clockcheck — one clock-log row and the nvidia-smi clock report.
 */
#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "harness.cuh"
#include "host.hpp"

static const char* EXP = "02__cluster_launch_and_occupancy";
static constexpr int kThreads = 1024;
static constexpr int kWarmups = 2;
static constexpr int kReps = 5;

struct Record {
    uint32_t block, smid, clusterid, rank, nrank, is_explicit;
    uint64_t t_start, t_end;
};

__device__ __forceinline__ uint32_t read_is_explicit_cluster() {
    uint32_t r;
    asm volatile("{\n\t.reg .pred p;\n\tmov.pred p, %%is_explicit_cluster;\n\tselp.u32 %0, 1, 0, p;\n\t}" : "=r"(r));
    return r;
}

__global__ void __launch_bounds__(kThreads) sleep_kernel(Record* rec, uint64_t cycles) {
    uint64_t t_start = 0;
    if (threadIdx.x == 0) t_start = read_globaltimer();
    const uint64_t c0 = read_clock64();
    while (read_clock64() - c0 < cycles) {
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint64_t t_end = read_globaltimer();
        const ClusterRegs c = read_cluster_regs();
        Record r;
        r.block = blockIdx.x;
        r.smid = read_smid();
        r.clusterid = c.clusterid_x;
        r.rank = c.ctarank;
        r.nrank = c.nctarank;
        r.is_explicit = read_is_explicit_cluster();
        r.t_start = t_start;
        r.t_end = t_end;
        rec[blockIdx.x] = r;
    }
}

__global__ void __launch_bounds__(kThreads) empty_kernel() {}

// ---------------------------------------------------------------- launch configurations
struct Config {
    int cs;  // 0 = no cluster attribute
    cudaClusterSchedulingPolicy policy;
    const char* policy_name;
};
static const Config kNoCluster{0, cudaClusterSchedulingPolicyDefault, "none"};
static Config cluster_cfg(int cs, cudaClusterSchedulingPolicy p = cudaClusterSchedulingPolicyDefault) {
    const char* n = p == cudaClusterSchedulingPolicySpread ? "spread" : p == cudaClusterSchedulingPolicyLoadBalancing ? "load_balancing" : "default";
    return {cs, p, n};
}

static hb::ClusterLaunch make_launch(const Config& c, int grid, size_t smem, int threads = kThreads) {
    hb::ClusterLaunch cl;
    cl.grid = dim3(grid);
    cl.block = dim3(threads);
    cl.smem = smem;
    cl.cluster = dim3(c.cs > 0 ? c.cs : 1, 1, 1);
    cl.explicit_cluster = c.cs > 0;
    cl.policy = c.policy;
    return cl;
}
static int pad(int g, int cs) { return cs > 1 ? (g + cs - 1) / cs * cs : g; }

static int api_max_clusters(const Config& c, int grid, size_t smem) {
    return c.cs > 0 ? make_launch(c, grid, smem).max_active_clusters(sleep_kernel) : -1;
}
static int api_blocks_per_wave(const Config& c, int grid, size_t smem, int sm_count) {
    if (c.cs > 0) return api_max_clusters(c, grid, smem) * c.cs;
    if (smem > 48 * 1024) CUDA_CHECK(cudaFuncSetAttribute(sleep_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    int n = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, sleep_kernel, kThreads, smem));
    return n * sm_count;
}

// ---------------------------------------------------------------- one timed launch with records
struct Launcher {
    Record* d_rec = nullptr;
    std::vector<Record> h;
    cudaEvent_t e0, e1;
    explicit Launcher(int max_grid) : h(max_grid) {
        CUDA_CHECK(cudaMalloc(&d_rec, max_grid * sizeof(Record)));
        CUDA_CHECK(cudaEventCreate(&e0));
        CUDA_CHECK(cudaEventCreate(&e1));
    }
    ~Launcher() { cudaFree(d_rec); cudaEventDestroy(e0); cudaEventDestroy(e1); }

    // Returns the event time in ms; the records of the launch are in h[0 .. grid).
    float run(const Config& c, int grid, size_t smem, uint64_t cycles, bool keep_records = true) {
        hb::ClusterLaunch cl = make_launch(c, grid, smem);
        hb::ClusterLaunch::Prepared p;
        cl.prepare(sleep_kernel, p);
        CUDA_CHECK(cudaMemset(d_rec, 0xff, grid * sizeof(Record)));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(e0));
        CUDA_CHECK(cudaLaunchKernelEx(&p.cfg, sleep_kernel, d_rec, cycles));
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        if (keep_records) CUDA_CHECK(cudaMemcpy(h.data(), d_rec, grid * sizeof(Record), cudaMemcpyDeviceToHost));
        return ms;
    }
};

// Consistency of the records with the launch configuration; returns the number of violations (printed).
static long check_records(const std::vector<Record>& h, int grid, const Config& c, const std::string& what) {
    const int cs = c.cs > 0 ? c.cs : 1;
    long bad = 0;
    for (int i = 0; i < grid; ++i) {
        const Record& r = h[i];
        const bool ok = r.block == (uint32_t)i && r.clusterid == (uint32_t)(i / cs) && r.rank == (uint32_t)(i % cs) &&
                        r.nrank == (uint32_t)cs && r.is_explicit == (c.cs > 0 ? 1u : 0u) && r.t_end >= r.t_start;
        if (!ok && bad++ < 3)
            std::fprintf(stderr, "WARNING %s: block %d record {block %u, cluster %u, rank %u/%u, explicit %u}\n", what.c_str(), i,
                         r.block, r.clusterid, r.rank, r.nrank, r.is_explicit);
    }
    return bad;
}

// Clusters whose CTAs occupy fewer than CS distinct SMs.
static int colocated_clusters(const std::vector<Record>& h, int grid, int cs) {
    int n = 0;
    for (int c = 0; c + cs <= grid; c += cs) {
        std::set<uint32_t> s;
        for (int i = c; i < c + cs; ++i) s.insert(h[i].smid);
        n += (int)s.size() < cs;
    }
    return n;
}

static void add_placement(hb::CsvWriter& csv, const std::string& prefix, const Config& c, int grid, size_t smem, int rep,
                          const std::vector<Record>& h, const char* probe, uint64_t cycles) {
    char b[512];
    for (int i = 0; i < grid; ++i) {
        const Record& r = h[i];
        std::snprintf(b, sizeof b, "%s,%d,%s,%d,%zu,%d,%d,%u,%u,%u,%llu,%llu,%s,%llu", prefix.c_str(), c.cs, c.policy_name, grid,
                      smem, rep, i, r.clusterid, r.rank, r.smid, (unsigned long long)r.t_start, (unsigned long long)r.t_end,
                      probe, (unsigned long long)cycles);
        csv.add(b);
    }
}

static std::vector<std::string> placement_cols() {
    return hb::with_mandatory({"cs", "policy", "grid_blocks", "smem_bytes", "rep", "block_idx", "cluster_id", "rank", "smid",
                               "t_start_ns", "t_end_ns", "probe", "sleep_cycles"});
}

static const size_t kSmem[] = {0, 64 * 1024, 200 * 1024};
static const int kCs[] = {1, 2, 4, 8, 16};

// ================================================================ K1: event floor + overhead sweep
static double run_floor(hb::Run& run) {
    hb::CsvWriter csv(hb::Run::results("02__cluster_launch__floor.csv"),
                      hb::with_mandatory({"mode", "cs", "policy", "launches", "us_event"}));
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    char b[512];
    long long idx = 0;

    run.clock_note("floor_empty_pair", "before");
    std::vector<float> ms(1000);
    for (int i = 0; i < 1000; ++i) {
        CUDA_CHECK(cudaEventRecord(e0));
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaEventElapsedTime(&ms[i], e0, e1));
    }
    hb::ClockSample s = run.clock_note("floor_empty_pair", "after");
    for (int i = 0; i < 1000; ++i) {
        std::snprintf(b, sizeof b, "%s,empty_pair,,,0,%.4f", run.prefix(idx++, s).c_str(), ms[i] * 1e3);
        csv.add(b);
    }
    csv.sync();
    std::vector<float> sorted = ms;
    std::sort(sorted.begin(), sorted.end());
    const double floor_ms = 0.5 * (sorted[499] + sorted[500]);
    std::printf("event floor (empty pair) median %.3f us\n", floor_ms * 1e3);

    std::vector<Config> cfgs{kNoCluster};
    for (int cs : kCs) cfgs.push_back(cluster_cfg(cs));
    for (const Config& c : cfgs) {
        const int grid = std::max(c.cs, 1);
        hb::ClusterLaunch cl = make_launch(c, grid, 0);
        hb::ClusterLaunch::Prepared p;
        cl.prepare(empty_kernel, p);
        for (int i = 0; i < 10; ++i) CUDA_CHECK(cudaLaunchKernelEx(&p.cfg, empty_kernel));
        CUDA_CHECK(cudaDeviceSynchronize());
        const std::string label = "floor_empty_kernel_cs" + std::to_string(c.cs);
        run.clock_note(label, "before");
        std::vector<float> t(200);
        for (int i = 0; i < 200; ++i) {
            CUDA_CHECK(cudaEventRecord(e0));
            CUDA_CHECK(cudaLaunchKernelEx(&p.cfg, empty_kernel));
            CUDA_CHECK(cudaEventRecord(e1));
            CUDA_CHECK(cudaEventSynchronize(e1));
            CUDA_CHECK(cudaEventElapsedTime(&t[i], e0, e1));
        }
        CUDA_CHECK(cudaGetLastError());
        s = run.clock_note(label, "after");
        for (int i = 0; i < 200; ++i) {
            std::snprintf(b, sizeof b, "%s,empty_kernel,%d,%s,1,%.4f", run.prefix(idx++, s).c_str(), c.cs, c.policy_name, t[i] * 1e3);
            csv.add(b);
        }
        csv.sync();
        std::sort(t.begin(), t.end());
        std::printf("empty kernel cs %2d: median %.3f us\n", c.cs, t[100] * 1e3);
    }
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    return floor_ms;
}

static void run_overhead(hb::Run& run) {
    const double floor_ms = run_floor(run);
    const int sm_count = run.dev().sm_count;
    const uint64_t cycles = 100000;
    const int targets[] = {1, 8, 16, 32, 64, 128, 132, 264, 528, 1056, 2112};
    hb::CsvWriter ovh(hb::Run::results("02__cluster_launch__overhead.csv"),
                      hb::with_mandatory({"cs", "policy", "grid_blocks", "smem_bytes", "rep", "ms_event", "ms_floor", "overhead_pct",
                                          "grid_target", "sleep_cycles", "api_max_active_clusters", "api_blocks_per_wave", "clock_ok"}));
    hb::CsvWriter plc(hb::Run::results("02__cluster_launch__placement.csv"), placement_cols());
    Launcher L(4096);
    long long idx = 0;
    long bad_records = 0, bad_clock = 0, colocated = 0;
    char b[512];

    for (size_t smem : kSmem) {
        for (int g : targets) {
            // launch list: no-cluster at every grid that a cluster size pads g to (g first), then CS 1..16
            struct Entry { Config c; int grid; int api_clusters; int api_bpw; };
            std::vector<Entry> list;
            std::set<int> nc_grids{g};
            for (int cs : kCs) nc_grids.insert(pad(g, cs));
            list.push_back({kNoCluster, g, -1, api_blocks_per_wave(kNoCluster, g, smem, sm_count)});
            for (int ng : nc_grids)
                if (ng != g) list.push_back({kNoCluster, ng, -1, api_blocks_per_wave(kNoCluster, ng, smem, sm_count)});
            for (int cs : kCs) {
                const Config c = cluster_cfg(cs);
                const int grid = pad(g, cs);
                list.push_back({c, grid, api_max_clusters(c, grid, smem), api_blocks_per_wave(c, grid, smem, sm_count)});
            }
            for (const Entry& e : list)
                for (int w = 0; w < kWarmups; ++w) L.run(e.c, e.grid, smem, cycles, false);

            const std::string label = "k1_smem" + std::to_string(smem) + "_g" + std::to_string(g);
            run.clock_note(label, "before");
            for (int rep = 0; rep < kReps; ++rep) {
                std::map<int, float> nc_ms;  // grid -> no-cluster event time in this rep
                for (const Entry& e : list) {
                    const float ms = L.run(e.c, e.grid, smem, cycles);
                    const hb::ClockSample s = run.clock_note(label, "launch", idx);
                    bad_clock += s.bad();
                    bad_records += check_records(L.h, e.grid, e.c, label);
                    if (e.c.cs > 1) colocated += colocated_clusters(L.h, e.grid, e.c.cs);
                    double pct = 0;
                    if (e.c.cs == 0) nc_ms[e.grid] = ms;
                    else pct = ((ms - floor_ms) / (nc_ms.at(e.grid) - floor_ms) - 1.0) * 100.0;
                    const std::string prefix = run.prefix(idx, s);
                    std::string api = e.api_clusters >= 0 ? std::to_string(e.api_clusters) : "";
                    std::snprintf(b, sizeof b, "%s,%d,%s,%d,%zu,%d,%.6f,%.6f,%.4f,%d,%llu,%s,%d,%d", prefix.c_str(), e.c.cs, e.c.policy_name,
                                  e.grid, smem, rep, ms, floor_ms, pct, g, (unsigned long long)cycles, api.c_str(), e.api_bpw, (int)!s.bad());
                    ovh.add(b);
                    add_placement(plc, prefix, e.c, e.grid, smem, rep, L.h, "k1_overhead", cycles);
                    ovh.flush();
                    plc.flush();
                    ++idx;
                }
            }
            ovh.sync();
            plc.sync();
            run.clock_note(label, "after");
            std::printf("k1 smem %6zu grid %4d done (%zu launch configs)\n", smem, g, list.size());
        }
    }
    std::printf("k1: %lld launches, record violations %ld, clock-flagged launches %ld, co-located clusters (Default) %ld\n", idx,
                bad_records, bad_clock, colocated);
}

// ================================================================ K2: occupancy probe
static void run_occupancy(hb::Run& run) {
    const int sm_count = run.dev().sm_count;
    const uint64_t cycles = 2000000;
    const cudaClusterSchedulingPolicy policies[] = {cudaClusterSchedulingPolicyDefault, cudaClusterSchedulingPolicySpread,
                                                     cudaClusterSchedulingPolicyLoadBalancing};
    hb::CsvWriter occ(hb::Run::results("02__cluster_launch__occupancy.csv"),
                      hb::with_mandatory({"cs", "policy", "smem_bytes", "api_max_active_clusters", "measured_first_wave_clusters", "sms_used",
                                          "sms_idle", "grid_blocks", "grid_mult", "rep", "max_ctas_per_sm", "clusters_colocated", "clock_ok"}));
    hb::CsvWriter plc(hb::Run::results("02__cluster_launch__placement.csv"), placement_cols());
    Launcher L(8192);
    long long idx = 0;
    long bad_records = 0, bad_clock = 0;
    char b[512];

    for (size_t smem : kSmem) {
        for (int cs : {2, 4, 8, 16}) {
            for (cudaClusterSchedulingPolicy pol : policies) {
                const Config c = cluster_cfg(cs, pol);
                const int api = api_max_clusters(c, cs, smem);
                if (api <= 0) {
                    std::printf("k2 smem %6zu cs %2d %-14s: api reports %d clusters, configuration skipped\n", smem, cs, c.policy_name, api);
                    continue;
                }
                for (int m : {1, 4}) {
                    const int grid = m * api * cs;
                    for (int w = 0; w < kWarmups; ++w) L.run(c, grid, smem, cycles, false);
                    const std::string label = "k2_smem" + std::to_string(smem) + "_cs" + std::to_string(cs) + "_" + c.policy_name + "_m" + std::to_string(m);
                    run.clock_note(label, "before");
                    for (int rep = 0; rep < kReps; ++rep) {
                        L.run(c, grid, smem, cycles);
                        const hb::ClockSample s = run.clock_note(label, "launch", idx);
                        bad_clock += s.bad();
                        bad_records += check_records(L.h, grid, c, label);
                        uint64_t min_end = ~0ull;
                        for (int i = 0; i < grid; ++i) min_end = std::min(min_end, L.h[i].t_end);
                        int first_wave = 0;
                        std::map<uint32_t, int> per_sm;
                        for (int k = 0; k + cs <= grid; k += cs) {
                            bool in = true;
                            for (int i = k; i < k + cs; ++i) in = in && L.h[i].t_start < min_end;
                            if (!in) continue;
                            ++first_wave;
                            for (int i = k; i < k + cs; ++i) ++per_sm[L.h[i].smid];
                        }
                        int max_per_sm = 0;
                        for (auto& kv : per_sm) max_per_sm = std::max(max_per_sm, kv.second);
                        const int used = (int)per_sm.size();
                        const int coloc = colocated_clusters(L.h, grid, cs);
                        const std::string prefix = run.prefix(idx, s);
                        std::snprintf(b, sizeof b, "%s,%d,%s,%zu,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d", prefix.c_str(), cs, c.policy_name, smem, api,
                                      first_wave, used, sm_count - used, grid, m, rep, max_per_sm, coloc, (int)!s.bad());
                        occ.add(b);
                        add_placement(plc, prefix, c, grid, smem, rep, L.h, "k2_occupancy", cycles);
                        occ.flush();
                        plc.flush();
                        if (rep == 0)
                            std::printf("k2 smem %6zu cs %2d %-14s m %d: api %3d clusters, first wave %3d, SMs used %3d idle %3d, max CTAs/SM %d, co-located %d\n",
                                        smem, cs, c.policy_name, m, api, first_wave, used, sm_count - used, max_per_sm, coloc);
                        ++idx;
                    }
                    occ.sync();
                    plc.sync();
                    run.clock_note(label, "after");
                }
            }
        }
    }
    std::printf("k2: %lld launches, record violations %ld, clock-flagged launches %ld\n", idx, bad_records, bad_clock);
}

// ================================================================ profiled launches (ncu / nsys)
static void write_manifest(const char* path, const std::vector<std::string>& rows) {
    FILE* f = std::fopen(path, "w");
    if (!f) hb::die(std::string("cannot write ") + path);
    std::fprintf(f, "%s\n", hb::join(hb::with_mandatory({"probe", "cs", "policy", "grid_blocks", "smem_bytes", "sleep_cycles"})).c_str());
    for (auto& r : rows) std::fprintf(f, "%s\n", r.c_str());
    std::fclose(f);
}

static void run_ncu(hb::Run& run) {
    Launcher L(8192);
    std::vector<std::string> rows;
    char b[512];
    long long idx = 0;
    auto one = [&](const Config& c, int grid, size_t smem, uint64_t cycles, const char* probe) {
        L.run(c, grid, smem, cycles, false);
        const hb::ClockSample s = run.clock_note("ncu", "launch", idx);
        std::snprintf(b, sizeof b, "%s,%s,%d,%s,%d,%zu,%llu", run.prefix(idx, s).c_str(), probe, c.cs, c.policy_name, grid, smem,
                      (unsigned long long)cycles);
        rows.push_back(b);
        ++idx;
    };
    for (size_t smem : kSmem)
        for (int cs : {2, 4, 8, 16})
            for (cudaClusterSchedulingPolicy pol : {cudaClusterSchedulingPolicyDefault, cudaClusterSchedulingPolicySpread,
                                                    cudaClusterSchedulingPolicyLoadBalancing}) {
                const Config c = cluster_cfg(cs, pol);
                one(c, api_max_clusters(c, cs, smem) * cs, smem, 2000000, "k2_occupancy");
            }
    one(kNoCluster, 2112, 200 * 1024, 100000, "k1_overhead");
    for (int cs : kCs) one(cluster_cfg(cs), 2112, 200 * 1024, 100000, "k1_overhead");
    write_manifest("bin/ncu_manifest.csv", rows);
}

static void run_nsys(hb::Run& run) {
    std::vector<std::string> rows;
    char b[512];
    long long idx = 0;
    Record* d_rec;
    CUDA_CHECK(cudaMalloc(&d_rec, 2112 * sizeof(Record)));
    std::vector<Config> cfgs{kNoCluster};
    for (int cs : kCs) cfgs.push_back(cluster_cfg(cs));
    for (const Config& c : cfgs) {
        hb::ClusterLaunch cl = make_launch(c, 2112, 200 * 1024);
        hb::ClusterLaunch::Prepared p;
        cl.prepare(sleep_kernel, p);
        for (int i = 0; i < kWarmups; ++i) CUDA_CHECK(cudaLaunchKernelEx(&p.cfg, sleep_kernel, d_rec, (uint64_t)100000));
        CUDA_CHECK(cudaDeviceSynchronize());
        for (int i = 0; i < 20; ++i) CUDA_CHECK(cudaLaunchKernelEx(&p.cfg, sleep_kernel, d_rec, (uint64_t)100000));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        const hb::ClockSample s = run.clock_note("nsys_cs" + std::to_string(c.cs), "after");
        for (int i = 0; i < kWarmups + 20; ++i) {
            std::snprintf(b, sizeof b, "%s,%s,%d,%s,2112,%d,100000", run.prefix(idx, s).c_str(), i < kWarmups ? "nsys_warmup" : "nsys",
                          c.cs, c.policy_name, 200 * 1024);
            rows.push_back(b);
            ++idx;
        }
    }
    CUDA_CHECK(cudaFree(d_rec));
    write_manifest("bin/nsys_manifest.csv", rows);
}

// ================================================================
int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s {overhead|occupancy|ncu|nsys|clockcheck} [--label L]\n", argv[0]);
        return 1;
    }
    const std::string cmd = argv[1];
    const std::string label = hb::arg_str(argc, argv, "--label", cmd.c_str());
    hb::Run run(EXP, argc, argv);
    if (cmd == "overhead") run_overhead(run);
    else if (cmd == "occupancy") run_occupancy(run);
    else if (cmd == "ncu") run_ncu(run);
    else if (cmd == "nsys") run_nsys(run);
    else if (cmd == "clockcheck") run.lock_clocks_note(label, "check");
    else hb::die("unknown subcommand " + cmd);
    run.complete();
    return 0;
}
