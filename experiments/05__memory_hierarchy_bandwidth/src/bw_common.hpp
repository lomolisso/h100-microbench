/*
 * bw_common.hpp — shared pieces of experiment 05's L2 and HBM benchmarks (bw_l2.cu, bw_dram.cu).
 *   kPartitionOfGpc / partition_of_sm  L2 partition of every SM: GPC (03 map) → partition {G0,G4,G5,G6} = 0,
 *                                     {G1,G2,G3,G7} = 1 (measured on this card in experiment 04; re-checked by bw_l2's
 *                                     home labelling, which probes from one SM of every GPC).
 *   BlockRec                          per-block record written at the end of a timed kernel: %clock64 and %globaltimer
 *                                     at the start and end of the block's timed region, bytes moved, %smid.
 *   V4, ld_v4<C>, st_v4<C>            16-byte global accesses with an explicit cache operator (inline PTX):
 *                                     C_CA ld.global.ca, C_CG ld/st.global.cg, C_CS ld/st.global.cs, C_LU ld.global.lu,
 *                                     C_CV ld.global.cv, C_WB st.global (default).
 *   PlacementCheck                    blocks per SM after a launch (host).
 *   CleanL2                           hb::L2Flusher (128 MB read-modify-write) followed by a read of another 128 MB
 *                                     buffer: the L2 then holds only clean lines of neither buffer under test, so no
 *                                     write-back of the flush's dirty lines overlaps the next timed kernel.
 */
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "../../04__memory_hierarchy_latency/src/gpc_map.hpp"
#include "harness.cuh"
#include "host.hpp"

constexpr int kPartitionOfGpc[8] = {0, 1, 1, 1, 0, 0, 0, 1};

inline std::vector<int> partition_of_sm(const std::vector<int>& gpc) {
    std::vector<int> p(gpc.size());
    for (size_t s = 0; s < gpc.size(); ++s) p[s] = kPartitionOfGpc[gpc[s]];
    return p;
}

struct BlockRec {
    uint64_t c0, c1, n0, n1, bytes;
    uint32_t smid, part, ticket, flags;
};

struct V4 { uint32_t x, y, z, w; };
__device__ __forceinline__ uint32_t xor4(const V4& v) { return v.x ^ v.y ^ v.z ^ v.w; }
// acc ^= x ^ y ^ z ^ w as volatile PTX: it keeps its place after the (volatile) loads of the same step, so the compiler
// cannot interleave consumption with the loads and in-order issue never waits on a load before the step's last one.
__device__ __forceinline__ void xor_into(uint32_t& acc, const V4& v) {
    asm volatile("{\n\t.reg .b32 t;\n\txor.b32 t, %1, %2;\n\txor.b32 t, t, %3;\n\txor.b32 t, t, %4;\n\txor.b32 %0, %0, t;\n\t}"
                 : "+r"(acc) : "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
}

enum CacheOp { C_CA = 0, C_CG = 1, C_CS = 2, C_WB = 3, C_LU = 4, C_CV = 5 };
inline const char* cache_name(int c) { static const char* n[] = {"ca", "cg", "cs", "wb", "lu", "cv"}; return n[c]; }

template <int C>
__device__ __forceinline__ V4 ld_v4(uint64_t a) {
    V4 v;
    if constexpr (C == C_CA)
        asm volatile("ld.global.ca.v4.f32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(a) : "memory");
    else if constexpr (C == C_CG)
        asm volatile("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(a) : "memory");
    else if constexpr (C == C_CS)
        asm volatile("ld.global.cs.v4.f32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(a) : "memory");
    else if constexpr (C == C_LU)
        asm volatile("ld.global.lu.v4.f32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(a) : "memory");
    else
        asm volatile("ld.global.cv.v4.f32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(a) : "memory");
    return v;
}
template <int C>
__device__ __forceinline__ void st_v4(uint64_t a, uint32_t d) {
    if constexpr (C == C_CG) asm volatile("st.global.cg.v4.f32 [%0], {%1, %1, %1, %1};" :: "l"(a), "r"(d) : "memory");
    else if constexpr (C == C_CS) asm volatile("st.global.cs.v4.f32 [%0], {%1, %1, %1, %1};" :: "l"(a), "r"(d) : "memory");
    else asm volatile("st.global.v4.f32 [%0], {%1, %1, %1, %1};" :: "l"(a), "r"(d) : "memory");
}

// Host: how many blocks ran on each SM.
struct PlacementCheck {
    std::vector<int> per_sm;
    explicit PlacementCheck(const std::vector<BlockRec>& r, int sm_count) : per_sm(sm_count, 0) {
        for (const auto& b : r) if (b.smid < (uint32_t)sm_count) ++per_sm[b.smid];
    }
    // every SM hosts exactly k blocks
    bool exactly(int k) const { for (int n : per_sm) if (n != k) return false; return true; }
    // no SM hosts more than one block
    bool at_most_one() const { for (int n : per_sm) if (n > 1) return false; return true; }
};

static __global__ void clean_read_kernel(const uint4* buf, size_t n, uint32_t* sink) {
    uint32_t acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) acc ^= buf[i].x;
    if (acc == 0x9e3779b9u) *sink = acc;  // practically never; keeps the loads
}
class CleanL2 {
  public:
    CleanL2() {
        CUDA_CHECK(cudaMalloc(&buf_, kBytes));
        CUDA_CHECK(cudaMemset(buf_, 0, kBytes));
        CUDA_CHECK(cudaMalloc(&sink_, sizeof(uint32_t)));
    }
    ~CleanL2() { cudaFree(buf_); cudaFree(sink_); }
    void operator()() {
        flush_();
        clean_read_kernel<<<132 * 4, 1024>>>(reinterpret_cast<const uint4*>(buf_), kBytes / sizeof(uint4), sink_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

  private:
    static constexpr size_t kBytes = 128ull << 20;
    hb::L2Flusher flush_;
    char* buf_ = nullptr;
    uint32_t* sink_ = nullptr;
};
