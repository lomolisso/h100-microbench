// Minimal, non-intrusive environment check: one block of 32 threads, a few microseconds of GPU time.
// Prints what the process can see and confirms that sm_90a code runs and %smid/%clock64 are readable.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void probe(unsigned* smid, unsigned long long* cycles) {
    unsigned id; unsigned long long t0, t1;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    if (threadIdx.x == 0) { *smid = id; *cycles = t1 - t0; }
}

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)

int main() {
    int n = 0, drv = 0, rt = 0;
    CK(cudaGetDeviceCount(&n));
    CK(cudaDriverGetVersion(&drv)); CK(cudaRuntimeGetVersion(&rt));
    std::printf("visible_devices=%d driver_api=%d runtime=%d\n", n, drv, rt);
    for (int d = 0; d < n; ++d) {
        cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, d));
        char uuid[64]; char* u = uuid;
        for (int i = 0; i < 16; ++i) u += std::sprintf(u, "%02x", (unsigned char)p.uuid.bytes[i]);
        std::printf("device %d: %s cc=%d.%d sms=%d l2=%d MiB smem/block optin=%zu KiB pci=%02x:%02x uuid=%s\n",
                    d, p.name, p.major, p.minor, p.multiProcessorCount, p.l2CacheSize >> 20,
                    p.sharedMemPerBlockOptin >> 10, p.pciBusID, p.pciDeviceID, uuid);
    }
    unsigned* smid; unsigned long long* cyc;
    CK(cudaMallocManaged(&smid, sizeof *smid)); CK(cudaMallocManaged(&cyc, sizeof *cyc));
    probe<<<1, 32>>>(smid, cyc);
    CK(cudaDeviceSynchronize());
    std::printf("probe kernel ok: smid=%u back-to-back clock64 delta=%llu cycles\n", *smid, *cyc);
    return 0;
}
