// Sustained-load kernels for choosing the locked SM clock (tools/clocktest/select_clock.sh).
// Models the heaviest phases of the suite: full-chip HBM streaming (05, 16) and L2-resident streaming
// (05, 11). Runs each phase for --seconds, launching back-to-back kernels; prints achieved bandwidth.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

__global__ void hbm_copy(const float4* __restrict__ src, float4* __restrict__ dst, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        dst[i] = src[i];
}

__global__ void l2_read(const float4* __restrict__ buf, size_t n, int reps, float* out) {
    float acc = 0.f;
    for (int r = 0; r < reps; ++r)
        for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
            float4 v = __ldcg(&buf[i]);
            acc += v.x + v.y + v.z + v.w;
        }
    if (acc == 12345.f) *out = acc;  // keep the loads alive
}

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); std::exit(1); } } while (0)

template <class F> static double run_for(double seconds, size_t bytes_per_launch, F launch) {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now(); size_t launches = 0;
    while (std::chrono::duration<double>(clk::now() - t0).count() < seconds) {
        for (int k = 0; k < 10; ++k) launch();
        CK(cudaDeviceSynchronize()); launches += 10;
    }
    double s = std::chrono::duration<double>(clk::now() - t0).count();
    return launches * (double)bytes_per_launch / s / 1e12;  // TB/s
}

int main(int argc, char** argv) {
    double seconds = 30;
    for (int i = 1; i < argc; ++i) if (!std::strcmp(argv[i], "--seconds") && i + 1 < argc) seconds = std::atof(argv[++i]);
    int sms; CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const int grid = sms * 4, block = 512;

    const size_t hbm_bytes = 4ull << 30, l2_bytes = 32ull << 20;  // 2 x 4 GiB; 32 MiB fits in L2
    float4 *a, *b; float* out;
    CK(cudaMalloc(&a, hbm_bytes)); CK(cudaMalloc(&b, hbm_bytes)); CK(cudaMalloc(&out, sizeof(float)));
    CK(cudaMemset(a, 0, hbm_bytes));
    size_t n_hbm = hbm_bytes / sizeof(float4), n_l2 = l2_bytes / sizeof(float4);

    std::printf("phase=hbm_copy  start\n"); std::fflush(stdout);
    double t1 = run_for(seconds, 2 * hbm_bytes, [&] { hbm_copy<<<grid, block>>>(a, b, n_hbm); });
    std::printf("phase=hbm_copy  tbps=%.3f\n", t1); std::fflush(stdout);

    const int reps = 64;
    std::printf("phase=l2_read   start\n"); std::fflush(stdout);
    double t2 = run_for(seconds, reps * l2_bytes, [&] { l2_read<<<grid, block>>>(a, n_l2, reps, out); });
    std::printf("phase=l2_read   tbps=%.3f\n", t2); std::fflush(stdout);
    CK(cudaGetLastError());
    return 0;
}
