/*
 * tma_common.hpp — pieces shared by experiment 06's binaries (tma_chase, tma_completion, tma_issue, tma_boxes).
 *
 *   Wait forms      W_TRY  = mbarrier.try_wait.parity (no hint), W_HINT = try_wait.parity with suspendTimeHint,
 *                   W_TEST = mbarrier.test_wait.parity (non-blocking). wait_once<W> is one call (SASS
 *                   SYNCS.PHASECHK.TRANS64[.TRYWAIT]); wait_loop<W> spins until it returns true; wait_loop_count<W>
 *                   also returns the number of calls made (≥ 1).
 *   lds_u64         ld.shared.u64 as volatile PTX (ptxas may not hoist it out of a loop).
 *   mad_u32         mad.lo.u32 as volatile PTX: the glue of dependent chains (ptxas fuses plain add chains).
 *
 *   k_pred<W, kBaseline>(n, hint, complete, zero, out)   predicate latency of one wait call. One thread. A barrier is
 *       initialised with count 1; complete = 1 arrives once, so a wait on parity 0 returns true at once. Timed: n
 *       dependent calls (unrolled by 16), the barrier address of call i+1 = mad(ok_i, zero, bar), so each call waits
 *       for the previous predicate. kBaseline replaces the wait by setp.ne + selp on the address (same glue): the
 *       wait's own latency is (full − baseline)/n + the setp/selp latency it replaces (a few cycles; stated as such).
 *       complete = 0 measures the time one call takes to return false on an incomplete phase (the poll period).
 *   k_expect<kDependent>(n, zero, out)   cost of mbarrier.arrive.expect_tx(16 B) on a barrier with count 2^20−1
 *       (never completes): kDependent = the returned state feeds the next call's address (latency); otherwise
 *       n independent back-to-back calls (issue throughput).
 *   k_timer_floor(reps, out)   clock64 → clock64 with nothing between: the floor of every single-sample timing.
 *   PredOut         {cycles, ok_count, smid}.
 *
 *   Host: encode_tiled() wraps cuTensorMapEncodeTiled (driver API, libcuda linked through the stubs); element strides
 *   1, no interleave, OOB fill none. Returns the CUresult so that unsupported box/swizzle combinations are recorded.
 *   median() of a vector.
 */
#pragma once
#include <cuda.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "harness.cuh"
#include "host.hpp"

enum WaitForm { W_TRY = 0, W_HINT = 1, W_TEST = 2 };

template <int W>
__device__ __forceinline__ bool wait_once(uint32_t bar, uint32_t parity, uint32_t hint_ns) {
    if constexpr (W == W_TRY) return mbar_try_wait_parity(bar, parity);
    else if constexpr (W == W_HINT) return mbar_try_wait_parity_hint(bar, parity, hint_ns);
    else return mbar_test_wait_parity(bar, parity);
}
template <int W>
__device__ __forceinline__ void wait_loop(uint32_t bar, uint32_t parity, uint32_t hint_ns) {
    while (!wait_once<W>(bar, parity, hint_ns)) { }
}
template <int W>
__device__ __forceinline__ uint32_t wait_loop_count(uint32_t bar, uint32_t parity, uint32_t hint_ns) {
    uint32_t calls = 1;
    while (!wait_once<W>(bar, parity, hint_ns)) ++calls;
    return calls;
}

__device__ __forceinline__ uint64_t lds_u64(uint32_t a) {
    uint64_t v; asm volatile("ld.shared.u64 %0, [%1];" : "=l"(v) : "r"(a) : "memory"); return v; }
__device__ __forceinline__ uint32_t mad_u32(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r; asm volatile("mad.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
// The baseline's stand-in for a wait call: a predicate computed from the address, turned into 0/1 as the wait's selp.
__device__ __forceinline__ uint32_t setp_selp(uint32_t a) {
    uint32_t r;
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %1, 0;\n\tselp.u32 %0, 1, 0, p;\n\t}" : "=r"(r) : "r"(a));
    return r;
}

struct PredOut { uint64_t cycles; uint32_t ok_count, smid; };

template <int W, bool kBaseline>
__global__ void __launch_bounds__(32) k_pred(uint32_t n, uint32_t hint_ns, int complete, uint32_t zero, PredOut* out) {
    __shared__ alignas(8) uint64_t bar;
    if (threadIdx.x != 0) return;
    const uint32_t b = smem_u32(&bar);
    mbar_init(b, 1);
    fence_mbarrier_init();
    if (complete) mbar_arrive(b);  // phase 0 completes: a wait on parity 0 is true from now on
    uint32_t a = b, okc = 0;
    const uint64_t t0 = read_clock64();
#pragma unroll 1
    for (uint32_t i = 0; i < n / 16; ++i) {
#pragma unroll
        for (int u = 0; u < 16; ++u) {
            const uint32_t ok = kBaseline ? setp_selp(a) : (uint32_t)wait_once<W>(a, 0, hint_ns);
            a = mad_u32(ok, zero, b);
            okc += ok;
        }
    }
    const uint64_t t1 = read_clock64_after(a, zero);
    out->cycles = t1 - t0;
    out->ok_count = okc;
    out->smid = read_smid();
}

template <bool kDependent>
__global__ void __launch_bounds__(32) k_expect(uint32_t n, uint32_t zero, PredOut* out) {
    __shared__ alignas(8) uint64_t bar;
    if (threadIdx.x != 0) return;
    const uint32_t b = smem_u32(&bar);
    mbar_init(b, (1u << 20) - 1);  // never completes within n ≤ 4096 arrivals; tx-count stays ≤ 16·n < 2^20
    fence_mbarrier_init();
    uint32_t a = b;
    const uint64_t t0 = read_clock64();
#pragma unroll 1
    for (uint32_t i = 0; i < n / 16; ++i) {
#pragma unroll
        for (int u = 0; u < 16; ++u) {
            if constexpr (kDependent) {
                uint64_t st;
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], 16;" : "=l"(st) : "r"(a) : "memory");
                a = mad_u32((uint32_t)st, zero, b);
            } else {
                mbar_arrive_expect_tx(b, 16);
            }
        }
    }
    const uint64_t t1 = kDependent ? read_clock64_after(a, zero) : read_clock64();
    out->cycles = t1 - t0;
    out->ok_count = 0;
    out->smid = read_smid();
}

__global__ void __launch_bounds__(32) k_timer_floor(uint32_t reps, uint64_t* out) {
    if (threadIdx.x != 0) return;
#pragma unroll 1
    for (uint32_t r = 0; r < reps; ++r) {
        const uint64_t t0 = read_clock64();
        const uint64_t t1 = read_clock64();
        out[r] = t1 - t0;
    }
}

// ---------------------------------------------------------------- host
inline CUresult encode_tiled(CUtensorMap* m, CUtensorMapDataType dt, uint32_t rank, void* gptr, const cuuint64_t* gdim,
                             const cuuint64_t* gstride_bytes, const cuuint32_t* box, CUtensorMapSwizzle sw,
                             CUtensorMapL2promotion prom = CU_TENSOR_MAP_L2_PROMOTION_NONE) {
    cuuint32_t es[5] = {1, 1, 1, 1, 1};
    return cuTensorMapEncodeTiled(m, dt, rank, gptr, gdim, gstride_bytes, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, sw, prom,
                                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

inline double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// Predicate floor (cycles per call, glue subtracted) of wait form W with hint_ns, median over `launches` launches.
template <int W>
inline double measure_pred_floor(uint32_t hint_ns, int launches = 5, uint32_t n = 1024) {
    PredOut* d;
    CUDA_CHECK(cudaMalloc(&d, sizeof(PredOut)));
    std::vector<double> v;
    for (int l = 0; l < launches + 2; ++l) {  // 2 warm-up launches
        PredOut full, base;
        k_pred<W, false><<<1, 32>>>(n, hint_ns, 1, 0u, d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&full, d, sizeof full, cudaMemcpyDeviceToHost));
        k_pred<W, true><<<1, 32>>>(n, hint_ns, 1, 0u, d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&base, d, sizeof base, cudaMemcpyDeviceToHost));
        if (l >= 2) v.push_back((double(full.cycles) - double(base.cycles)) / n);
    }
    cudaFree(d);
    return median(v);
}
