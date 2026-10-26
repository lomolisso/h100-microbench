/*
 * common09.hpp — experiment 09: pieces shared by dsmem_lat and dsmem_vis.
 *
 * Device
 *   record_place(place)    lane 0 of every CTA writes place[4·blockIdx.x + {0: %smid, 1: %clusterid.x,
 *                          2: %cluster_ctarank, 3: %cluster_nctarank}].
 * Host
 *   kPolicies / policy_name  the three cudaClusterSchedulingPolicy values (Default, Spread, LoadBalancing).
 *   DevBuf<T>              device buffer (zeroed), get() copies it back.
 *   Grid09                 a launch of `nclusters` 1-D clusters of `cs` CTAs, `block` threads, 200 KB dynamic smem
 *                          (→ 1 CTA per SM), explicit cluster attribute (also for CS 1) and an explicit scheduling
 *                          policy. nclusters = 0 → max(kMinClusters = 20, cudaOccupancyMaxActiveClusters) (one full
 *                          wave, so that every cluster-capable SM hosts a CTA, but at least 20 clusters).
 *   Placement              host view of the place[] buffer: smid[cluster][rank]; ok() is false if two ranks of a
 *                          cluster share an SM (the launch is discarded and repeated); dies if %clusterid/%rank do not
 *                          match blockIdx (1-D grid: block b = cluster b / cs, rank b % cs).
 *   Discards               one row per discarded launch in results/09__dsmem_latency__discards.csv (label, reason:
 *                          "placement" = ranks not on distinct SMs, "clock" = throttle reason or SM clock below the lock
 *                          in the NVML sample after the launch); the launch is repeated (≤ kMaxTries per launch).
 *   gpc_of                 smid → GPC from the latest run of the 03 map (04's gpc_map.hpp).
 */
#pragma once
#include <cooperative_groups.h>

#include <algorithm>
#include <cstdio>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "harness.cuh"
#include "host.hpp"
#include "../../04__memory_hierarchy_latency/src/gpc_map.hpp"

constexpr size_t kOneCtaPerSmSmem = 200u * 1024u;
constexpr int kMinClusters = 20, kWarmups = 2, kLaunches = 5, kQuickLaunches = 2, kMaxTries = 20;

static const cudaClusterSchedulingPolicy kPolicies[] = {cudaClusterSchedulingPolicyDefault, cudaClusterSchedulingPolicySpread,
                                                        cudaClusterSchedulingPolicyLoadBalancing};
inline const char* policy_name(cudaClusterSchedulingPolicy p) {
    return p == cudaClusterSchedulingPolicyDefault ? "Default" : p == cudaClusterSchedulingPolicySpread ? "Spread" : "LoadBalancing";
}

__device__ __forceinline__ void record_place(uint32_t* place) {
    if ((threadIdx.x & 31) == 0 && threadIdx.x < 32) {
        const ClusterRegs c = read_cluster_regs();
        uint32_t* p = place + 4 * blockIdx.x;
        p[0] = read_smid();
        p[1] = c.clusterid_x;
        p[2] = c.ctarank;
        p[3] = c.nctarank;
    }
}

template <class T>
struct DevBuf {
    T* d = nullptr;
    size_t n = 0;
    explicit DevBuf(size_t n_) : n(n_) { CUDA_CHECK(cudaMalloc(&d, n * sizeof(T))); CUDA_CHECK(cudaMemset(d, 0, n * sizeof(T))); }
    ~DevBuf() { cudaFree(d); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    std::vector<T> get(size_t m = 0) const {
        std::vector<T> h(m ? m : n);
        CUDA_CHECK(cudaMemcpy(h.data(), d, h.size() * sizeof(T), cudaMemcpyDeviceToHost));
        return h;
    }
    void zero(size_t m = 0) { CUDA_CHECK(cudaMemset(d, 0, (m ? m : n) * sizeof(T))); }
};

struct Grid09 {
    hb::ClusterLaunch L;
    int cs = 1, nclusters = 0, max_active = 0;
    template <typename... KArgs>
    Grid09(void (*k)(KArgs...), int cs_, int block, cudaClusterSchedulingPolicy pol, int ncl = 0) : cs(cs_) {
        L.block = dim3(block);
        L.cluster = dim3(cs, 1, 1);
        L.smem = kOneCtaPerSmSmem;
        L.explicit_cluster = true;
        L.policy = pol;
        L.grid = dim3(cs);
        max_active = L.max_active_clusters(k);
        nclusters = ncl > 0 ? ncl : std::max(kMinClusters, max_active);
        L.grid = dim3(nclusters * cs);
    }
    int blocks() const { return nclusters * cs; }
};

struct Placement {
    int cs = 1, nclusters = 0;
    std::vector<uint32_t> sm;  // sm[c * cs + r]
    Placement(const std::vector<uint32_t>& place, int cs_, int ncl) : cs(cs_), nclusters(ncl), sm((size_t)cs_ * ncl) {
        for (int b = 0; b < cs * ncl; ++b) {
            const uint32_t* p = &place[4 * (size_t)b];
            if ((int)p[1] != b / cs || (int)p[2] != b % cs || (int)p[3] != cs)
                hb::die("block " + std::to_string(b) + ": clusterid " + std::to_string(p[1]) + " rank " + std::to_string(p[2]) +
                        " nctarank " + std::to_string(p[3]) + " do not match the 1-D grid");
            sm[b] = p[0];
        }
    }
    uint32_t at(int c, int r) const { return sm[(size_t)c * cs + r]; }
    bool ok() const {
        for (int c = 0; c < nclusters; ++c) {
            const std::set<uint32_t> u(sm.begin() + (size_t)c * cs, sm.begin() + (size_t)(c + 1) * cs);
            if ((int)u.size() != cs) return false;
        }
        return true;
    }
};

struct Discards {
    hb::Run& run;
    hb::CsvWriter csv;
    int n = 0;
    explicit Discards(hb::Run& r)
        : run(r), csv(hb::Run::results("09__dsmem_latency__discards.csv"), hb::with_mandatory({"label", "reason"})) {}
    void add(const std::string& label, const char* reason, long long li, const hb::ClockSample& s) {
        csv.add(run.prefix(li, s) + "," + label + "," + reason);
        csv.sync();
        ++n;
        std::printf("  discarded launch (%s): %s\n", reason, label.c_str());
    }
};

// One recorded launch: launch() runs the kernel (synchronised) and returns its Placement; repeated until the placement
// is clean and the clock sample taken after the launch is good. Returns the clock sample (for the CSV prefix) and the placement.
template <class F>
std::pair<hb::ClockSample, Placement> recorded_launch(hb::Run& run, Discards& disc, const std::string& label, long long li, F&& launch) {
    for (int t = 0; t < kMaxTries; ++t) {
        Placement p = launch();
        const hb::ClockSample s = run.clock_note(label, "launch", li);
        if (!p.ok()) { disc.add(label, "placement", li, s); continue; }
        if (s.bad()) { disc.add(label, "clock", li, s); continue; }
        return {s, std::move(p)};
    }
    hb::die(label + ": no clean launch after " + std::to_string(kMaxTries) + " tries");
}
