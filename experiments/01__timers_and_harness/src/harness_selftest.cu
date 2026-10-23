/*
 * harness_selftest.cu — functional test of the shared harness (experiments/harness). Not a benchmark: it
 * writes no CSV and prints one line per test. The exit status is non-zero if any test fails.
 *
 *   T1 cluster + DSMEM: hb::ClusterLaunch, cluster size 2, one CTA per SM (all SMs, 200 KB dynamic smem).
 *      Each CTA writes blockIdx.x+1 into its smem, cluster_sync(), reads its peer's word through
 *      mapa_u32 + ld_dsmem_u32, then cluster_sync() before exit. Checks: peer values, %cluster_ctarank /
 *      %cluster_nctarank, all %smid distinct (grid-wide and within each cluster).
 *   T2 mbarrier + bulk_g2s: one CTA copies two 16 KB chunks global -> smem through one barrier
 *      (arrive_expect_tx + complete_tx), waiting with parity 0 and then 1. Checks the smem contents.
 *   T3 bulk_s2g: generic smem writes, fence_proxy_async_shared_cta, bar, bulk_s2g + commit + wait<0>.
 *      Checks the global contents.
 *   T4 bulk_s2s_dsmem: cluster of 2. Rank 1 arms its barrier (arrive_expect_tx) before cluster_sync. Rank 0
 *      puts 8 KB into rank 1's smem, signalling rank 1's barrier. Rank 1 waits with acquire.cluster.
 *   T5 remote arrive: cluster of 2. Thread 0 of each CTA arrives on rank 0's barrier (count 2) through
 *      mbar_remote_arrive. Rank 0 waits with acquire.cluster.
 *   T6 bulk_g2s_multicast: cluster of 2. Both CTAs arm their own barrier; rank 0 multicasts 8 KB to both.
 *   T7 L2Flusher, warmup, max_active_clusters: run without error.
 */
#include <vector>

#include "harness.cuh"
#include "host.hpp"

extern __shared__ __align__(1024) unsigned char dyn_smem[];

__global__ void t1_cluster(uint32_t* peer_val, uint32_t* smid, uint32_t* rank, uint32_t* nrank) {
    uint32_t* word = reinterpret_cast<uint32_t*>(dyn_smem);
    if (threadIdx.x == 0) *word = blockIdx.x + 1;
    cluster_sync();
    const ClusterRegs c = read_cluster_regs();
    if (threadIdx.x == 0) {
        peer_val[blockIdx.x] = ld_dsmem_u32(mapa_u32(smem_u32(word), c.ctarank ^ 1u));
        smid[blockIdx.x] = read_smid();
        rank[blockIdx.x] = c.ctarank;
        nrank[blockIdx.x] = c.nctarank;
    }
    cluster_sync();
}

constexpr uint32_t T2_BYTES = 16 * 1024;
__global__ void t2_g2s(const uint32_t* src, uint32_t* bad) {
    __shared__ alignas(8) uint64_t bar;
    uint32_t* buf = reinterpret_cast<uint32_t*>(dyn_smem);
    const uint32_t b = smem_u32(&bar);
    if (threadIdx.x == 0) { mbar_init(b, 1); fence_mbarrier_init(); }
    __syncthreads();
    uint32_t errs = 0;
    for (uint32_t phase = 0; phase < 2; ++phase) {
        if (threadIdx.x == 0) {
            mbar_arrive_expect_tx(b, T2_BYTES);
            bulk_g2s(smem_u32(buf), src + phase * (T2_BYTES / 4), T2_BYTES, b);
        }
        mbar_wait_parity(b, phase);
        for (uint32_t i = threadIdx.x; i < T2_BYTES / 4; i += blockDim.x) errs += buf[i] != phase * (T2_BYTES / 4) + i;
        __syncthreads();  // everyone done reading before the next chunk overwrites buf
    }
    atomicAdd(bad, errs);
}

__global__ void t3_s2g(uint32_t* dst) {
    uint32_t* buf = reinterpret_cast<uint32_t*>(dyn_smem);
    for (uint32_t i = threadIdx.x; i < T2_BYTES / 4; i += blockDim.x) buf[i] = 0xA5000000u + i;
    fence_proxy_async_shared_cta();
    __syncthreads();
    if (threadIdx.x == 0) {
        bulk_s2g(dst, smem_u32(buf), T2_BYTES);
        bulk_commit();
        bulk_wait<0>();
    }
}

constexpr uint32_t T4_BYTES = 8 * 1024;
__global__ void t4_s2s(uint32_t* bad) {
    __shared__ alignas(8) uint64_t bar;
    uint32_t* src = reinterpret_cast<uint32_t*>(dyn_smem);
    uint32_t* dst = reinterpret_cast<uint32_t*>(dyn_smem + T4_BYTES);
    const uint32_t rank = read_cluster_regs().ctarank;
    const uint32_t b = smem_u32(&bar);
    if (threadIdx.x == 0) {
        mbar_init(b, 1);
        fence_mbarrier_init();
        if (rank == 1) mbar_arrive_expect_tx(b, T4_BYTES);
    }
    for (uint32_t i = threadIdx.x; i < T4_BYTES / 4; i += blockDim.x) { src[i] = 0x5A000000u + i; dst[i] = 0; }
    fence_proxy_async_shared_cta();
    cluster_sync();
    if (rank == 0 && threadIdx.x == 0) bulk_s2s_dsmem(mapa_u32(smem_u32(dst), 1), smem_u32(src), T4_BYTES, mapa_u32(b, 1));
    if (rank == 1) {
        mbar_wait_parity_acq_cluster(b, 0);
        uint32_t errs = 0;
        for (uint32_t i = threadIdx.x; i < T4_BYTES / 4; i += blockDim.x) errs += dst[i] != 0x5A000000u + i;
        atomicAdd(bad, errs);
    }
    cluster_sync();
}

__global__ void t5_remote_arrive(uint32_t* done) {
    __shared__ alignas(8) uint64_t bar;
    const uint32_t rank = read_cluster_regs().ctarank;
    const uint32_t b = smem_u32(&bar);
    if (rank == 0 && threadIdx.x == 0) { mbar_init(b, 2); fence_mbarrier_init(); }
    cluster_sync();
    if (threadIdx.x == 0) mbar_remote_arrive(mapa_u32(b, 0));
    if (rank == 0 && threadIdx.x == 0) { mbar_wait_parity_acq_cluster(b, 0); *done = 1; }
    cluster_sync();
}

__global__ void t6_multicast(const uint32_t* src, uint32_t* bad) {
    __shared__ alignas(8) uint64_t bar;
    uint32_t* buf = reinterpret_cast<uint32_t*>(dyn_smem);
    const uint32_t rank = read_cluster_regs().ctarank;
    const uint32_t b = smem_u32(&bar);
    if (threadIdx.x == 0) { mbar_init(b, 1); fence_mbarrier_init(); mbar_arrive_expect_tx(b, T4_BYTES); }
    cluster_sync();
    if (rank == 0 && threadIdx.x == 0) bulk_g2s_multicast(smem_u32(buf), src, T4_BYTES, b, (uint16_t)0x3);
    mbar_wait_parity_acq_cluster(b, 0);
    uint32_t errs = 0;
    for (uint32_t i = threadIdx.x; i < T4_BYTES / 4; i += blockDim.x) errs += buf[i] != i;
    atomicAdd(bad, errs);
    cluster_sync();
}

static int failures = 0;
static void report(const char* name, bool ok, const std::string& detail = "") {
    std::printf("%-28s %s %s\n", name, ok ? "ok" : "FAILED", detail.c_str());
    failures += !ok;
}

int main() {
    const hb::DeviceInfo dev = hb::DeviceInfo::open();
    std::printf("device %s (%s), %d SMs, L2 %d B, driver %s, CUDA %s, power limit %u W, lock %u MHz\n", dev.name.c_str(),
                dev.uuid.c_str(), dev.sm_count, dev.l2_bytes, dev.driver_version.c_str(), dev.cuda_version.c_str(),
                dev.power_limit_mw / 1000, dev.lock_mhz);
    const int G = dev.sm_count - dev.sm_count % 2;
    uint32_t *d_a, *d_b, *d_c, *d_d, *d_src, *d_bad;
    CUDA_CHECK(cudaMalloc(&d_a, G * 4));
    CUDA_CHECK(cudaMalloc(&d_b, G * 4));
    CUDA_CHECK(cudaMalloc(&d_c, G * 4));
    CUDA_CHECK(cudaMalloc(&d_d, G * 4));
    CUDA_CHECK(cudaMalloc(&d_src, 2 * T2_BYTES));
    CUDA_CHECK(cudaMalloc(&d_bad, 4));
    std::vector<uint32_t> h(2 * T2_BYTES / 4);
    for (size_t i = 0; i < h.size(); ++i) h[i] = (uint32_t)i;
    CUDA_CHECK(cudaMemcpy(d_src, h.data(), 2 * T2_BYTES, cudaMemcpyHostToDevice));

    {  // T1
        hb::ClusterLaunch cl;
        cl.grid = dim3(G);
        cl.block = dim3(128);
        cl.cluster = dim3(2, 1, 1);
        cl.smem = 200 * 1024;
        CUDA_CHECK(cl.launch(t1_cluster, d_a, d_b, d_c, d_d));
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<uint32_t> peer(G), smid(G), rank(G), nrank(G);
        CUDA_CHECK(cudaMemcpy(peer.data(), d_a, G * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(smid.data(), d_b, G * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(rank.data(), d_c, G * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(nrank.data(), d_d, G * 4, cudaMemcpyDeviceToHost));
        bool ok = true;
        for (int i = 0; i < G; ++i) ok = ok && peer[i] == (uint32_t)((i ^ 1) + 1) && rank[i] == (uint32_t)(i % 2) && nrank[i] == 2;
        report("T1 cluster DSMEM read", ok);
        report("T1 smid distinct (grid)", hb::all_distinct(smid));
        report("T1 smid distinct (cluster)", hb::distinct_within_clusters(smid, 2));
        report("T1 max active clusters", true, std::to_string(cl.max_active_clusters(t1_cluster)) + " clusters of 2 at 200 KB");
    }
    auto bad_count = [&] { uint32_t v; CUDA_CHECK(cudaMemcpy(&v, d_bad, 4, cudaMemcpyDeviceToHost)); return v; };
    {  // T2
        CUDA_CHECK(cudaMemset(d_bad, 0, 4));
        hb::ClusterLaunch cl;
        cl.block = dim3(128);
        cl.smem = T2_BYTES;
        CUDA_CHECK(cl.launch(t2_g2s, (const uint32_t*)d_src, d_bad));
        CUDA_CHECK(cudaDeviceSynchronize());
        const uint32_t e = bad_count();
        report("T2 mbarrier + bulk_g2s", e == 0, std::to_string(e) + " wrong words");
    }
    {  // T3
        uint32_t* d_dst;
        CUDA_CHECK(cudaMalloc(&d_dst, T2_BYTES));
        CUDA_CHECK(cudaMemset(d_dst, 0, T2_BYTES));
        hb::ClusterLaunch cl;
        cl.block = dim3(128);
        cl.smem = T2_BYTES;
        CUDA_CHECK(cl.launch(t3_s2g, d_dst));
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<uint32_t> out(T2_BYTES / 4);
        CUDA_CHECK(cudaMemcpy(out.data(), d_dst, T2_BYTES, cudaMemcpyDeviceToHost));
        uint32_t e = 0;
        for (size_t i = 0; i < out.size(); ++i) e += out[i] != 0xA5000000u + (uint32_t)i;
        report("T3 bulk_s2g", e == 0, std::to_string(e) + " wrong words");
        CUDA_CHECK(cudaFree(d_dst));
    }
    hb::ClusterLaunch pair;
    pair.grid = dim3(2);
    pair.block = dim3(128);
    pair.cluster = dim3(2, 1, 1);
    pair.smem = 2 * T4_BYTES;
    {  // T4
        CUDA_CHECK(cudaMemset(d_bad, 0, 4));
        CUDA_CHECK(pair.launch(t4_s2s, d_bad));
        CUDA_CHECK(cudaDeviceSynchronize());
        const uint32_t e = bad_count();
        report("T4 bulk_s2s_dsmem", e == 0, std::to_string(e) + " wrong words");
    }
    {  // T5
        CUDA_CHECK(cudaMemset(d_bad, 0, 4));
        CUDA_CHECK(pair.launch(t5_remote_arrive, d_bad));
        CUDA_CHECK(cudaDeviceSynchronize());
        report("T5 mbar_remote_arrive", bad_count() == 1);
    }
    {  // T6
        CUDA_CHECK(cudaMemset(d_bad, 0, 4));
        CUDA_CHECK(pair.launch(t6_multicast, (const uint32_t*)d_src, d_bad));
        CUDA_CHECK(cudaDeviceSynchronize());
        const uint32_t e = bad_count();
        report("T6 bulk_g2s_multicast", e == 0, std::to_string(e) + " wrong words");
    }
    {  // T7
        hb::L2Flusher flush;
        flush();
        hb::warmup([&] { flush(); }, 2);
        report("T7 L2Flusher + warmup", true);
    }
    std::printf("%s: %d failure(s)\n", failures ? "SELFTEST FAILED" : "selftest passed", failures);
    return failures ? 1 : 0;
}
