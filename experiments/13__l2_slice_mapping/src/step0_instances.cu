/*
 * step0_instances.cu — experiment 13, Step 0 only: does this Nsight Compute export per-unit (per-instance) values of
 * hardware-unit counters? The kernel is the one-line probe the step needs; src/step0.sh profiles it four ways.
 *
 * Kernel k13_step0: one thread (1 block × 1 thread, on whichever SM the scheduler picks; %smid printed) reads one 128 B
 *   line `n` times through a dependent chain of ld.global.cg.u64: qword 0 of the line holds the line's own address, so
 *   every load uses the value of the previous one (no merging, no hoisting). n = 4096: ≈ 4096 L2 read lookups of one
 *   line (one slice), plus a few for the first miss and the list-free setup.
 * Host: cudaMalloc 2 MiB, qword 0 = its own address, one launch inside cudaProfilerStart/Stop (the range for ncu's
 *   range replay modes). No CSV: the deliverable is the ncu output (results/).
 */
#include <cstdio>

#include <cuda_profiler_api.h>

#include "harness.cuh"
#include "host.hpp"

__global__ void k13_step0(const uint64_t* line, uint32_t n, uint64_t* sink, uint32_t* smid) {
    uint64_t p = (uint64_t)line;
#pragma unroll 1
    for (uint32_t i = 0; i < n; ++i) asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(p) : "l"(p) : "memory");
    sink[0] = p;
    smid[0] = read_smid();
}

int main(int argc, char** argv) {
    hb::Run run("13__l2_slice_mapping", argc, argv);
    uint64_t *buf = nullptr, *sink = nullptr;
    uint32_t* smid = nullptr;
    CUDA_CHECK(cudaMalloc(&buf, 2u << 20));
    CUDA_CHECK(cudaMalloc(&sink, 8));
    CUDA_CHECK(cudaMalloc(&smid, 4));
    const uint64_t self = (uint64_t)buf;
    CUDA_CHECK(cudaMemcpy(buf, &self, 8, cudaMemcpyHostToDevice));
    cudaProfilerStart();  // the profiler range for --replay-mode range / app-range (no effect otherwise)
    k13_step0<<<1, 1>>>(buf, 4096, sink, smid);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaProfilerStop();
    uint32_t s = 0;
    CUDA_CHECK(cudaMemcpy(&s, smid, 4, cudaMemcpyDeviceToHost));
    std::printf("k13_step0: 4096 dependent reads of line 0x%llx on SM %u\n", (unsigned long long)self, s);
    run.complete();
    return 0;
}
