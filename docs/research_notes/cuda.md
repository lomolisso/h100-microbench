# H100 / Hopper microbenchmark reference: Thread Block Clusters, DSMEM, TMA, mbarriers

Compiled 2026-10-23 from official NVIDIA sources only. Audience: engineers writing CUDA C++ microbenchmarks on H100.

## 0. Provenance and how to read this note

| Tag | Meaning |
|---|---|
| **[HDR]** | Quoted verbatim from NVIDIA-shipped headers: `cuda.h`, `cuda_runtime_api.h`, `cuda_runtime.h`, `driver_types.h`, `cooperative_groups.h`, `cooperative_groups/details/helpers.h`, `crt/sm_90_rt.h` taken from the official NVIDIA PyPI wheels `nvidia-cuda-runtime-cu12==12.9.79` and `nvidia-cuda-nvcc-cu12==12.9.86`. These are the same doxygen comments that generate the CUDA Runtime/Driver API reference manuals. |
| **[CCCL]** | Quoted verbatim from the official `NVIDIA/cccl` GitHub repository, `main` @ `b008f6b` (docs under `docs/libcudacxx/`, headers under `libcudacxx/include/cuda/`). This is the source of https://nvidia.github.io/cccl/libcudacxx/ptx.html. |
| **[WEB]** | Text returned by web search *snippets* of the named docs.nvidia.com / developer.nvidia.com page. The session's egress proxy refused direct fetches of `docs.nvidia.com`, `developer.nvidia.com`, `nvidia.github.io`, `resources.nvidia.com`, `www.nvidia.com` (HTTP 403 CONNECT policy denial), so these pages could not be read in full; only the snippet text is reproduced, and anything not seen in a snippet is listed in §10 "Facts I could not confirm". |

Primary URLs (current, CUDA 13.4-era layout; the "CUDA C++ Programming Guide" was split into a multi-page "CUDA Programming Guide"):

- Thread block clusters (programming model): https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html and cluster launch APIs: https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-host-programming.html
- Distributed shared memory + cooperative groups: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html ; cluster_group API reference: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#class-cluster-group
- Asynchronous Barriers: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html
- Asynchronous Data Copies (cp.async, TMA, bulk copies): https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html
- L2 cache control: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html
- Memory fences / sync functions: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html
- Compute capability tables: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
- Legacy single-page guide (still online): https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- Hopper Tuning Guide: https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html (PDF: https://docs.nvidia.com/cuda/pdf/Hopper_Tuning_Guide.pdf)
- Hopper architecture blog: https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
- Driver API tensor-map group: https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__TENSOR__MEMORY.html
- CCCL `cuda::ptx`: https://nvidia.github.io/cccl/libcudacxx/ptx.html ; `cuda::barrier`: https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/barrier.html
- nvidia-smi manual: https://docs.nvidia.com/deploy/nvidia-smi/index.html

---

## 1. Thread Block Clusters (Programming Guide)

### 1.1 Model statements [WEB, Programming Guide 13.x]

- "Similar to how threads in a thread block are guaranteed to be co-scheduled on a streaming multiprocessor, thread blocks in a cluster are also guaranteed to be co-scheduled on a GPU Processing Cluster (GPC) in the GPU."
- "The number of thread blocks in a cluster can be user-defined, and a maximum of 8 thread blocks in a cluster is supported as a portable cluster size in CUDA. Note that on GPU hardware or MIG configurations which are too small to support 8 multiprocessors the maximum cluster size will be reduced accord."
- Compile-time attribute: `__cluster_dims__([x, [y, [z]]])`; example in the guide uses `__cluster_dims__(2, 1, 1)` and a normal `<<<>>>` launch; "the grid size must be a multiple of the cluster size". If the kernel uses a compile-time cluster size, the cluster size cannot be changed at launch.
- Runtime attribute: `cudaLaunchKernelEx` with `cudaLaunchAttributeClusterDimension` in `config.attrs`/`config.numAttrs`; "grid dimensions must be divisible by the cluster dimensions"; advantage over `__cluster_dims__` is that the value can change between launches.
- Query API named by the guide for architecture-specific max: `cudaOccupancyMaxPotentialClusterSize`.
- CC 10.0+ only (not H100): `cudaLaunchAttributePreferredClusterDimension`, "must be an integer multiple of the minimum cluster dimensions".

### 1.2 Exact host API [HDR]

```c
// driver_types.h
enum cudaFuncAttribute {
    cudaFuncAttributeMaxDynamicSharedMemorySize = 8,          /**< Maximum dynamic shared memory size */
    cudaFuncAttributePreferredSharedMemoryCarveout = 9,       /**< Preferred shared memory-L1 cache split */
    cudaFuncAttributeClusterDimMustBeSet = 10,                /**< Indicator to enforce valid cluster dimension specification on kernel launch */
    cudaFuncAttributeRequiredClusterWidth = 11,
    cudaFuncAttributeRequiredClusterHeight = 12,
    cudaFuncAttributeRequiredClusterDepth = 13,
    cudaFuncAttributeNonPortableClusterSizeAllowed = 14,      /**< Whether non-portable cluster scheduling policy is supported */
    cudaFuncAttributeClusterSchedulingPolicyPreference = 15,  /**< Required cluster scheduling policy preference */
    cudaFuncAttributeMax
};

enum cudaClusterSchedulingPolicy {
    cudaClusterSchedulingPolicyDefault       = 0, /**< the default policy */
    cudaClusterSchedulingPolicySpread        = 1, /**< spread the blocks within a cluster to the SMs */
    cudaClusterSchedulingPolicyLoadBalancing = 2  /**< allow the hardware to load-balance the blocks in a cluster to the SMs */
};
// (13.4 runtime docs additionally list cudaClusterSchedulingPolicyRubinDsmemLocality; driver: CU_CLUSTER_SCHEDULING_POLICY_* [WEB])

enum cudaLaunchAttributeID { ...
  cudaLaunchAttributeAccessPolicyWindow    = 1, /**< Valid for streams, graph nodes, launches. */
  cudaLaunchAttributeCooperative           = 2, /**< Valid for graph nodes, launches. */
  cudaLaunchAttributeClusterDimension                  = 4, /**< Valid for graph nodes, launches. See ::cudaLaunchAttributeValue::clusterDim. */
  cudaLaunchAttributeClusterSchedulingPolicyPreference = 5, /**< Valid for graph nodes, launches. */
  cudaLaunchAttributePreferredClusterDimension = 11, ... };

union cudaLaunchAttributeValue {
    char pad[64];
    struct cudaAccessPolicyWindow accessPolicyWindow;
    int cooperative;
    /* "Value of launch attribute ::cudaLaunchAttributeClusterDimension that represents the desired
       cluster dimensions for the kernel. ... x - The X dimension of the cluster, in blocks. Must be a
       divisor of the grid X dimension." (same for y,z) */
    struct { unsigned int x; unsigned int y; unsigned int z; } clusterDim;
    enum cudaClusterSchedulingPolicy clusterSchedulingPolicyPreference;
    ...
};

typedef struct cudaLaunchAttribute_st { cudaLaunchAttributeID id; char pad[8 - sizeof(cudaLaunchAttributeID)]; cudaLaunchAttributeValue val; } cudaLaunchAttribute;

typedef struct cudaLaunchConfig_st {
    dim3 gridDim; dim3 blockDim; size_t dynamicSmemBytes; cudaStream_t stream;
    cudaLaunchAttribute *attrs; /**< List of attributes; nullable if numAttrs == 0 */
    unsigned int numAttrs;
} cudaLaunchConfig_t;

// cuda_runtime.h (C++), cuda_runtime_api.h (C)
template<typename... ExpTypes, typename... ActTypes>
static __inline__ __host__ cudaError_t cudaLaunchKernelEx(const cudaLaunchConfig_t *config, void (*kernel)(ExpTypes...), ActTypes &&... args);
extern __host__ cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *config, const void *func, void **args);
extern __host__ cudaError_t cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr, int value);
extern __host__ cudaError_t cudaOccupancyMaxActiveClusters(int *numClusters, const void *func, const cudaLaunchConfig_t *launchConfig);
extern __host__ cudaError_t cudaOccupancyMaxPotentialClusterSize(int *clusterSize, const void *func, const cudaLaunchConfig_t *launchConfig);
```

`cudaFuncSetAttribute` doc text [HDR]:
- `cudaFuncAttributeMaxDynamicSharedMemorySize` - "The requested maximum size in bytes of dynamically-allocated shared memory. The sum of this value and the function attribute ::sharedSizeBytes cannot exceed the device attribute ::cudaDevAttrMaxSharedMemoryPerBlockOptin."
- `cudaFuncAttributePreferredSharedMemoryCarveout` - "On devices where the L1 cache and shared memory use the same hardware resources, this sets the shared memory carveout preference, in percent of the total shared memory. ... This is only a hint, and the driver can choose a different ratio if required to execute the function."
- `cudaFuncAttributeRequiredClusterWidth/Height/Depth`: "The width, height, and depth values must either all be 0 or all be positive. The validity of the cluster dimensions is checked at launch time. If the value is set during compile time, it cannot be set at runtime. Setting it at runtime will return cudaErrorNotPermitted."
- `cudaFuncAttributeNonPortableClusterSizeAllowed`: "Indicates whether the function can be launched with non-portable cluster size. 1 is allowed, 0 is disallowed."
- `cudaFuncAttributeClusterSchedulingPolicyPreference`: "The block scheduling policy of a function. The value type is cudaClusterSchedulingPolicy."
- Carveout helper enum: `cudaSharedmemCarveoutDefault = -1`, `cudaSharedmemCarveoutMaxShared = 100`, `cudaSharedmemCarveoutMaxL1 = 0`.

`cudaOccupancyMaxActiveClusters` doc text [HDR]: "return the maximum number of clusters that could co-exist on the target device ... If the function has required cluster size already set (see ::cudaFuncGetAttributes), the cluster size from config must either be unspecified or match the required size. Without required sizes, the cluster size must be specified in config, else the function will return an error. ... Runtime environment may affect how the hardware schedules the clusters, so the calculated occupancy is not guaranteed to be achievable."

`cudaOccupancyMaxPotentialClusterSize` doc text [HDR]: "The cluster dimensions in config are ignored. ... By default this function will always return a value that's portable on future hardware. A higher value may be returned if the kernel function allows non-portable cluster sizes. This function will respect the compile time launch bounds."

Portable vs non-portable [WEB, Hopper Tuning Guide]: "the maximum portable cluster size supported is 8; however, NVIDIA Hopper H100 GPU allows for a nonportable cluster size of 16 by opting in. ... launching a kernel with a nonportable cluster size requires setting the cudaFuncAttributeNonPortableClusterSizeAllowed function attribute. ... using larger cluster sizes may reduce the maximum number of active blocks across the GPU."

Device attribute [HDR]: `cudaDevAttrClusterLaunch = 120 /**< Indicates device supports cluster launch */`; `cudaDeviceProp::clusterLaunch /**< Indicates device supports cluster launch */`.

### 1.3 Minimal launch snippet (assembled from the API above)

```cpp
__global__ void __cluster_dims__(2,1,1) k_static(...);            // compile-time cluster; launch with <<<grid, block>>>, grid % 2 == 0

__global__ void k_dyn(...);                                        // runtime cluster
cudaFuncSetAttribute(k_dyn, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);   // needed for 16 on H100
cudaLaunchAttribute attr[2];
attr[0].id = cudaLaunchAttributeClusterDimension;  attr[0].val.clusterDim = {16,1,1};
attr[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
attr[1].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicySpread;
cudaLaunchConfig_t cfg = {}; cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem;
cfg.stream = s; cfg.attrs = attr; cfg.numAttrs = 2;
int nclus = 0; cudaOccupancyMaxActiveClusters(&nclus, (void*)k_dyn, &cfg);
cudaLaunchKernelEx(&cfg, k_dyn, args...);
```

### 1.4 Device-side cluster API: `cooperative_groups::cluster_group` [HDR, cooperative_groups.h 12.9]

Header comment: "a cluster consists of at most 8 blocks by default (although the user is allowed to opt-in to non-standard sizes,) and clustered grids are subject to additional occupancy limitations due to per-cluster hardware resource consumption. In exchange, a block cluster is guaranteed to be a cooperative group ... Constructed via this_cluster_group();" (the actual factory is `cg::this_cluster()`).

| Member (all `static`) | Implementation (helpers.h → crt/sm_90_rt.h intrinsics) |
|---|---|
| `void sync()` | `barrier_arrive(); barrier_wait();` |
| `arrival_token barrier_arrive()` | `__cluster_barrier_arrive()` (there is also `__cluster_barrier_arrive_relaxed()`) |
| `void barrier_wait()` / `barrier_wait(arrival_token&&)` | `__cluster_barrier_wait()` |
| `unsigned query_shared_rank(const void*)` | `__cluster_query_shared_rank(addr)` |
| `T* map_shared_rank(T* addr, int rank)` | `static_cast<T*>(__cluster_map_shared_rank(addr, rank))` (PTX `mapa`) |
| `dim3 block_index()` | `__clusterRelativeBlockIdx()` |
| `unsigned block_rank()` | `__clusterRelativeBlockRank()` |
| `dim3 dim_blocks()` | `__clusterDim()` |
| `unsigned num_blocks()` | `__clusterSizeInBlocks()` |
| `dim3 dim_threads()` | `dim_blocks() * blockDim` |
| `unsigned num_threads()` | `num_blocks() * blockDim` |
| `unsigned thread_rank()` | `block_rank() * cta::num_threads() + cta::thread_rank()` |
| `dim3 thread_index()` | block_index * blockDim + threadIdx |

Raw sm_90 intrinsics (crt/sm_90_rt.h) [HDR]: `__isCtaShared`, `__isClusterShared`, `__cluster_map_shared_rank(const void*, unsigned target_block_rank)`, `__cluster_query_shared_rank`, `__cluster_map_shared_multicast(const void*, unsigned cluster_cta_mask) -> uint2`, `__clusterDimIsSpecified`, `__clusterDim`, `__clusterRelativeBlockIdx`, `__clusterGridDimInClusters`, `__clusterIdx`, `__clusterRelativeBlockRank`, `__clusterSizeInBlocks`, `__cluster_barrier_arrive`, `__cluster_barrier_arrive_relaxed`, `__cluster_barrier_wait`, `__threadfence_cluster`.

### 1.5 Exit rule and existence rule [WEB, Programming Guide]

- "Accessing data in distributed shared memory requires all the thread blocks to exist. A user can guarantee that all thread blocks have started executing using cluster.sync() from Cluster Group API."
- "The user also needs to ensure that all distributed shared memory operations happen before the exit of a thread block." / "the program needs to ensure that the shared memory read by the remote thread block is completed before it can exit" — the guide's histogram example ends with `cluster.sync()` for exactly this reason, "so no block exits while others still access distributed shared memory".
- PTX-level statement [WEB, PTX ISA]: "the peer CTA must exist and hasn't exited before completing the shared memory operation."

---

## 2. Distributed Shared Memory (DSMEM)

- Programming Guide [WEB]: "Thread block clusters on compute capability 9.0+ let threads read, write, and perform atomics in shared memory across all blocks in the cluster ... the shared memory of all blocks in the cluster, which is referred to as distributed shared memory." "The combined shared memory of all thread blocks in the cluster forms this space, and blocks can read, write, and run atomics on any address in it."
- Hopper Tuning Guide [WEB]: "A thread block can read from, write to, and perform atomics in shared memory of other thread blocks within its cluster. This is known as Distributed Shared Memory." It "can act as an intermediate step between shared memory and global memory." For cluster occupancy it recommends `cudaOccupancyMaxActiveClusters`.
- Whitepaper/blog [WEB]: "Distributed Shared Memory allows direct SM-to-SM communications for loads, stores, and atomics across multiple SM shared memory blocks."
- Addressing: obtain a remote pointer with `cluster.map_shared_rank(&x, rank)` (PTX `mapa`), check ownership with `cluster.query_shared_rank(ptr)`; ordinary `ld/st/atom` on the generic pointer go to the remote SM's shared memory (PTX state space `.shared::cluster`). The CCCL docs show the canonical pattern [CCCL, docs/libcudacxx/ptx/instructions/mapa.rst]:

```cuda
#include <cooperative_groups.h>
__cluster_dims__(2)
__global__ void kernel() {
    __shared__ int x;
    x = 1;
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    cluster.sync();
    // Get address of remote shared memory value:
    unsigned int other_block_rank = cluster.block_rank() ^ 1;
    int * remote_x = cluster.map_shared_rank(&x, other_block_rank);
    // Write to remote value:
    *remote_x = 2;
}
```
(CCCL notes `mapa` "can currently not be implemented by libcu++" (cccl issue #1414); use the cooperative-groups API or raw `asm("mapa.shared::cluster.u32 ...")`.)

- Async DSMEM operations available in PTX / `cuda::ptx` (see §9): `cp.async.bulk shared::cta -> shared::cluster` (with remote mbarrier), `cp.reduce.async.bulk ... shared::cluster`, `st.async`, `red.async` (all signal a remote `mbarrier` via `complete_tx::bytes`), `cp.async.bulk(.tensor).multicast::cluster` (SM_90a).
- Cluster-scoped mbarrier operations exist: `mbarrier.arrive{.expect_tx}.release.cluster.shared::cluster` (remote arrive, no state returned), `mbarrier.try_wait ... .cluster` scope, and `fence.mbarrier_init.release.cluster` [CCCL].

---

## 3. Asynchronous barriers (`cuda::barrier`, mbarrier)

### 3.1 Programming Guide "Asynchronous Barriers" [WEB]
- "The second parameter of init() is the expected arrival count, which is how many arrive calls must occur before a waiting thread is released." Initialization "has to finish before any thread joins the barrier" (one thread calls `init(&bar, blockDim.x)`, then `__syncthreads()`).
- Phases: "an asynchronous barrier counts down from the expected arrival count toward zero as participating threads arrive, and when it reaches zero the barrier is complete for the current phase. The reset happens automatically on the last arrival, setting the countdown back to the expected count and advancing to the next phase."
- "A token object of class cuda::barrier::arrival_token, as returned from token=bar.arrive(), is associated with the current phase of the barrier." "token=bar.arrive() must occur during the cuda::barrier's current phase", `wait()` in the same or next phase. `bar.arrive_and_wait()` "is equivalent to bar.wait(bar.arrive())".
- Phase completion step "is executed once per phase, after the last thread arrives and before any thread is unblocked from the wait. Memory operations performed by the threads that arrived at the barrier during the phase are visible to the thread executing the [completion step ...]".
- Scope table: block-scope barrier lives in shared memory, supported CC 8.0+.
- Transaction count: "The transaction count to be tracked by the current phase can be set on arrival with cuda::device::barrier_arrive_tx() or directly with cuda::device::barrier_expect_tx()."
- Explicit parity waiting: "it allows only a single thread to arrive on the barrier and set the transaction count, while other threads only wait for a parity-based phase flip" (via `cuda::ptx::mbarrier_try_wait_parity`).
- Low-level C intrinsics (`<cuda_awbarrier_primitives.h>`) [HDR]: `__mbarrier_init(__mbarrier_t*, uint32_t expected_count)` ("sets the expected arrival count for the current and next phase"), `__mbarrier_arrive`, `__mbarrier_test_wait`, `__mbarrier_test_wait_parity`, `__mbarrier_try_wait(..., uint32_t max_sleep_nanosec)`, `__mbarrier_try_wait_parity`.

### 3.2 `cuda::barrier` semantics [CCCL, docs barrier.rst]
```cuda
template <cuda::thread_scope Scope, typename CompletionFunction = /* unspecified */>
class cuda::barrier;   // <cuda/barrier>
```
- "`Scope` shall not be `cuda::thread_scope_cluster`." (i.e. no cluster-scope `cuda::barrier`; use `cuda::ptx` mbarrier functions with `scope_cluster` for cluster semantics.)
- Modified standard wording for `thread_scope_block` in shared memory: "each phase allows at most an expected number of threads to block until the expected number of threads **and the expected number of transaction-based asynchronous operations** arrive at the barrier. ... 1. The expected count is decremented by each call to arrive, arrive_and_drop, or cuda::device::barrier_arrive_tx. 2. The transaction count is incremented by each call to cuda::device::barrier_arrive_tx and decremented by the completion of transaction-based asynchronous operations such as cuda::memcpy_async_tx. 3. Exactly once after both the expected count and the transaction count reach zero, a thread executes the completion step ... 4. When the completion step finishes, the expected count is reset ... and the next phase starts."
- "The tx-count tracks the number of asynchronous transactions, in units specified by the asynchronous memory operation (typically bytes), that are outstanding and yet to be complete. This capability is exposed, starting with the Hopper architecture (CUDA Compute Capability 9)."
- `barrier<thread_scope_block>::max() == (1 << 20) - 1`.
- CC 8+: `arrive` on a `__shared__` block barrier "performs a reduction of the arrival count among coalesced threads followed by the arrival operation in one thread."
- `__shared__ cuda::barrier` needs `#pragma nv_diag_suppress static_var_with_dynamic_init` and `init(&bar, count)` by one thread.

```cuda
__device__ cuda::barrier<cuda::thread_scope_block>::arrival_token
cuda::device::barrier_arrive_tx(cuda::barrier<cuda::thread_scope_block>& bar,
                                ptrdiff_t arrive_count_update, ptrdiff_t transaction_count_update);
// Preconditions: barrier object in shared memory; 1 <= arrive_count_update <= (1<<20)-1; 0 <= transaction_count_update <= (1<<20)-1
// "decrements the arrival count by arrive_count_update and increments the expected transaction count by transaction_count_update.
//  This function executes atomically. The call to this function strongly happens-before the start of the phase completion step for the current phase."
// Feature flag: __cccl_lib_local_barrier_arrive_tx. CC 9.0+ only.

__device__ void cuda::device::barrier_expect_tx(cuda::barrier<cuda::thread_scope_block>& bar, ptrdiff_t transaction_count_update);
// "Increments the expected transaction count ... executes atomically." CC 9.0+ only.

__device__ cuda::barrier<cuda::thread_scope_block>::native_handle_type*   // uint64_t*
cuda::device::barrier_native_handle(cuda::barrier<cuda::thread_scope_block>& bar);  // use with cuda::ptx::mbarrier_* 
```

### 3.3 `cuda::device::memcpy_async_tx` and `cuda::memcpy_async` [CCCL]
```cuda
template <typename T, size_t Alignment>
inline __device__ void cuda::device::memcpy_async_tx(T* dest, const T* src, cuda::aligned_size_t<Alignment> size,
                                                     cuda::barrier<cuda::thread_scope_block>& bar);
// "Copies size bytes from global memory src to shared memory dest and decrements the transaction count of bar by size bytes."
// Preconditions: src, dest 16-byte aligned, size multiple of 16 (Alignment >= 16); bar in shared memory. CC 9.0+ only, no fallback.

// <cuda/barrier>
template <typename Shape, cuda::thread_scope Scope, typename CompletionFunction>
__host__ __device__ void cuda::memcpy_async(void* destination, void const* source, Shape size, cuda::barrier<Scope, CompletionFunction>& barrier);
template <typename Group, typename Shape, cuda::thread_scope Scope, typename CompletionFunction>
__host__ __device__ void cuda::memcpy_async(Group const& group, void* destination, void const* source, Shape size, cuda::barrier<Scope, CompletionFunction>& barrier);
// <cuda/pipeline>: same two overloads taking cuda::pipeline<Scope>& pipeline.
```
Implementation note [CCCL]: "On Hopper+ GPUs, the overloads taking a barrier may use the Tensor Memory Accelerator (TMA) via the cp.async.bulk instruction to perform the copy if: the barrier resides in shared memory, the data is aligned to 16 bytes, the source is global memory, the destination is shared memory. ... On Ampere+ GPUs, the cp.async instruction may be used ... if the data is aligned to at least 4 bytes, the source is global memory, the destination is shared memory."

Canonical 1D TMA load, single elected thread, parity wait [CCCL barrier.rst]:
```cuda
#include <cuda/barrier>
#include <cuda/ptx>
__device__ bool elect_one() {
  const unsigned int tid = threadIdx.x; const unsigned int warp_id = tid / 32;
  const unsigned int uniform_warp_id = __shfl_sync(0xFFFFFFFF, warp_id, 0);
  return uniform_warp_id == 0 && cuda::ptx::elect_sync(0xFFFFFFFF);
}
__global__ void example_kernel(int* gmem1, double* gmem2) {
  constexpr int tile_size = 1024;
  __shared__ alignas(16)    int smem1[tile_size];
  __shared__ alignas(16) double smem2[tile_size];
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__  cuda::barrier<cuda::thread_scope_block> bar;
  if (elect_one()) {
    init(&bar, 1);
    cuda::device::memcpy_async_tx(smem1, gmem1, cuda::aligned_size_t<16>(tile_size * sizeof(int)   ), bar);
    cuda::device::memcpy_async_tx(smem2, gmem2, cuda::aligned_size_t<16>(tile_size * sizeof(double)), bar);
    (void)cuda::device::barrier_arrive_tx(bar, 1, tile_size * (sizeof(int) + sizeof(double)));
  }
  __syncthreads();
  bar.wait_parity(0);          // CCCL >= 3.2; older: while(!cuda::ptx::mbarrier_try_wait_parity(cuda::device::barrier_native_handle(bar), 0));
}
```

### 3.4 `cuda::pipeline` and `cp.async` (short) [CCCL, HDR]
- `<cuda/pipeline>`: `template <cuda::thread_scope Scope> class cuda::pipeline` with `producer_acquire()`, `producer_commit()`, `consumer_wait()`, `consumer_wait_for()`, `consumer_wait_until()`, `consumer_release()`, `quit()`; built via `cuda::make_pipeline(...)` with a `cuda::pipeline_shared_state<Scope, N>` for non-thread scopes; "Pipelines can be either unified or partitioned."
- Primitives (`<cuda_pipeline_primitives.h>`): `__pipeline_memcpy_async(void* dst_shared, const void* src_global, size_t size_and_align, ...)`, `__pipeline_commit()`, `__pipeline_wait_prior(size_t prior)` = PTX `cp.async` (4/8/16-byte, Ampere+). 
- `cuda::ptx` also exposes `cp_async_mbarrier_arrive` / `cp_async_mbarrier_arrive_noinc` (PTX `cp.async.mbarrier.arrive`).

---

## 4. TMA: tensor maps and bulk asynchronous copies

### 4.1 `CUtensorMap` and `cuTensorMapEncodeTiled` [HDR, cuda.h 12.9 — verbatim doc comment]

```c
#define CU_TENSOR_MAP_NUM_QWORDS 16
typedef struct CUtensorMap_st { alignas(64) cuuint64_t opaque[CU_TENSOR_MAP_NUM_QWORDS]; } CUtensorMap;   // "Requires compiler support for aligning to 64 bytes."

CUresult cuTensorMapEncodeTiled(CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank,
    void *globalAddress, const cuuint64_t *globalDim, const cuuint64_t *globalStrides, const cuuint32_t *boxDim,
    const cuuint32_t *elementStrides, CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill);
```
"Creates a descriptor for Tensor Memory Access (TMA) object specified by the parameters describing a tiled region and returns it in tensorMap. Tensor map objects are only supported on devices of compute capability 9.0 or higher. Additionally, a tensor map object is an opaque value, and, as such, should only be accessed through CUDA APIs and PTX."

| Parameter | Documented requirement (verbatim) |
|---|---|
| `tensorMap` | "address must be aligned to 64 bytes." |
| `tensorDataType` | enum `CUtensorMapDataType`: `UINT8`(1B) `UINT16`(2B) `UINT32`(4B) `INT32`(4B) `UINT64`(8B) `INT64`(8B) `FLOAT16`(2B) `FLOAT32`(4B) `FLOAT64`(8B) `BFLOAT16`(2B) `FLOAT32_FTZ`(4B) `TFLOAT32`(4B) `TFLOAT32_FTZ`(4B) `16U4_ALIGN8B`(4 bit) `16U4_ALIGN16B`(4 bit) `16U6_ALIGN16B`(6 bit), all prefixed `CU_TENSOR_MAP_DATA_TYPE_`. |
| `tensorRank` | "must be non-zero and less than or equal to the maximum supported dimensionality of 5. If interleave is not CU_TENSOR_MAP_INTERLEAVE_NONE, then tensorRank must additionally be greater than or equal to 3." |
| `globalAddress` | "must be 16 byte aligned." "When interleave is CU_TENSOR_MAP_INTERLEAVE_32B, globalAddress must be 32 byte aligned." (also 32B for 16U6_ALIGN16B / 16U4_ALIGN16B) |
| `globalDim` | "specifies tensor size of each of the tensorRank dimensions, must be non-zero and less than or equal to 2^32." (packed types: globalDim[0] multiple of 128 for 16U6/16U4_ALIGN16B, multiple of 2 for 16U4_ALIGN8B) |
| `globalStrides` | "specifies tensor stride of each of the lower tensorRank - 1 dimensions in bytes, must be a multiple of 16 and less than 2^40." "When interleave is CU_TENSOR_MAP_INTERLEAVE_32B, the strides must be a multiple of 32." Formula given: `globalStrides[0] = globalDim[0] * elementSizeInBytes(tensorDataType) + padding[0]; for (i = 1; i < tensorRank - 1; i++) globalStrides[i] = globalStrides[i – 1] * (globalDim[i] + padding[i]); assert(globalStrides[i] >= globalDim[i]);` |
| `boxDim` | "specifies number of elements to be traversed along each of the tensorRank dimensions, must be non-zero and less than or equal to 256." "When interleave is CU_TENSOR_MAP_INTERLEAVE_NONE, { boxDim[0] * elementSizeInBytes( tensorDataType ) } must be a multiple of 16 bytes." (16U6/16U4_ALIGN16B: boxDim[0] must be 128) |
| `elementStrides` | "specifies the iteration step along each of the tensorRank dimensions, must be non-zero and less than or equal to 8. Note that when interleave is CU_TENSOR_MAP_INTERLEAVE_NONE, the first element of this array is ignored since TMA doesn't support the stride for dimension zero. When all elements of elementStrides array is one, boxDim specifies the number of elements to load. However, if the elementStrides[i] is not equal to one, then TMA loads ceil( boxDim[i] / elementStrides[i]) number of elements along i-th dimension. To load N elements along i-th dimension, boxDim[i] must be set to N * elementStrides[i]." |
| `interleave` | `CU_TENSOR_MAP_INTERLEAVE_NONE = 0`, `_16B`, `_32B`. "TMA supports interleaved layouts like NC/8HWC8 where C8 utilizes 16 bytes in memory assuming 2 byte per channel or NC/16HWC16 where C16 uses 32 bytes." "When interleave is CU_TENSOR_MAP_INTERLEAVE_32B, swizzle must be CU_TENSOR_MAP_SWIZZLE_32B. Other interleave modes can have any swizzling pattern." |
| `swizzle` | `CU_TENSOR_MAP_SWIZZLE_NONE = 0`; `_32B // Swizzle 16B chunks within 32B span`; `_64B // Swizzle 16B chunks within 64B span`; `_128B // Swizzle 16B chunks within 128B span`; `_128B_ATOM_32B // Swizzle 32B chunks within 128B span`; `_128B_ATOM_32B_FLIP_8B // ... additionally swap lower 8B with upper 8B within each 16B for every alternate row`; `_128B_ATOM_64B // Swizzle 64B chunks within 128B span`. **Constraint:** "When interleave is CU_TENSOR_MAP_INTERLEAVE_NONE and swizzle is not CU_TENSOR_MAP_SWIZZLE_NONE, the bounding box inner dimension (computed as boxDim[0] multiplied by element size derived from tensorDataType) must be less than or equal to the swizzle size. CU_TENSOR_MAP_SWIZZLE_32B requires the bounding box inner dimension to be <= 32. CU_TENSOR_MAP_SWIZZLE_64B ... <= 64. CU_TENSOR_MAP_SWIZZLE_128B* ... <= 128." Rationale text: "Data are organized in a specific order in global memory; however, this may not match the order in which the application accesses data in shared memory. This difference in data organization may cause bank conflicts when shared memory is accessed. In order to avoid this problem, data can be loaded to shared memory with shuffling across shared memory banks." |
| `l2Promotion` | "specifies L2 fetch size which indicates the byte granurality at which L2 requests is filled from DRAM": `CU_TENSOR_MAP_L2_PROMOTION_NONE = 0`, `_L2_64B`, `_L2_128B`, `_L2_256B`. |
| `oobFill` | "indicates whether zero or a special NaN constant should be used to fill out-of-bound elements": `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0`, `CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA` ("can only be used when tensorDataType represents a floating-point data type"). |

Returns `CUDA_SUCCESS`, `CUDA_ERROR_DEINITIALIZED`, `CUDA_ERROR_NOT_INITIALIZED`, `CUDA_ERROR_INVALID_CONTEXT`, `CUDA_ERROR_INVALID_VALUE`. See also `cuTensorMapEncodeIm2col`, `cuTensorMapEncodeIm2colWide`, `cuTensorMapReplaceAddress`.

Shared-memory destination alignment [WEB, Programming Guide]: "TMA requires shared memory aligned to 128 bytes"; "with the 128-byte swizzle mode, the shared memory destination should be 1024-byte aligned"; for 1D bulk copies "The destination shared memory buffer of a bulk operation should be 16 byte aligned."

Passing the map [WEB, Programming Guide]: "The recommended approach is to pass the tensor map as a const __grid_constant__ parameter to a kernel. The other possibilities are copying the tensor map into device __constant__ memory using cudaMemcpyToSymbol or accessing it via global memory." (`__grid_constant__` "annotates a const-qualified __global__ function parameter of non-reference type", CC 7.0+.) Get the driver entry point without linking `-lcuda` via `cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", ...)` (guide example "Get a function pointer to the cuTensorMapEncodeTiled driver API").

```cpp
#include <cuda.h>  // CUtensorMap
__global__ void k(const __grid_constant__ CUtensorMap tmap) {
  __shared__ alignas(1024) bf16 tile[128][64];                 // 128*2B = 128B inner span -> SWIZZLE_128B ok
  ...
  cuda::ptx::cp_async_bulk_tensor(cuda::ptx::space_cluster, cuda::ptx::space_global, &tile, &tmap, {x, y}, bar_handle);
}
```

### 4.2 `cuTensorMapEncodeIm2col` [HDR]
```c
CUresult cuTensorMapEncodeIm2col(CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank,
    void *globalAddress, const cuuint64_t *globalDim, const cuuint64_t *globalStrides, const int *pixelBoxLowerCorner,
    const int *pixelBoxUpperCorner, cuuint32_t channelsPerPixel, cuuint32_t pixelsPerColumn, const cuuint32_t *elementStrides,
    CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill);
```
Differences from tiled (verbatim): "tensorRank ... must be 3, 4, or 5." `pixelBoxLowerCorner`/`pixelBoxUpperCorner`: "coordinate offsets {D, H, W} of the bounding box from top/left/front corner" / "from bottom/right/back corner"; ranges: rank 3 → one signed offset in [-32768, 32767]; rank 4 → two in [-128, 127]; rank 5 → three in [-16, 15]; "must have non-zero area." `channelsPerPixel` "must be less than or equal to 256"; `pixelsPerColumn` "must be less than or equal to 1024"; swizzle inner-dim constraint uses `channelsPerPixel * elementSize`. Also exists: `cuTensorMapEncodeIm2colWide` (`CUtensorMapIm2ColWideMode`: `CU_TENSOR_MAP_IM2COL_WIDE_MODE_W`, `_W128`).

### 4.3 Device-side wrappers for TMA (what the Programming Guide examples use)
- Older (12.x) guide examples use `namespace cde = cuda::device::experimental;` with `cde::cp_async_bulk_tensor_2d_global_to_shared(&smem_buffer, &tensor_map, x, y, bar)`, `cde::cp_async_bulk_tensor_{1..5}d_{global_to_shared,shared_to_global}`, `cde::cp_async_bulk_global_to_shared` / `cp_async_bulk_shared_to_global`, `cde::fence_proxy_async_shared_cta()`, `cde::cp_async_bulk_commit_group()`, `cde::cp_async_bulk_wait_group_read<N>()`; then `token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer)); bar.wait(std::move(token));` [WEB, Programming Guide 12.5–12.9 examples].
- In current CCCL these are gated by `__cccl_lib_experimental_ctk12_cp_async_exposure` and marked `CCCL_DEPRECATED_BECAUSE("Use cuda::ptx::cp_async_bulk[_tensor] / fence_proxy_async / cp_async_bulk_commit_group / cp_async_bulk_wait_group_read instead")` [CCCL, libcudacxx/include/cuda/barrier]. **Use `cuda::ptx` (§9).**
- Current guide's 1D example [WEB]: before a shared→global bulk store it calls `fence_proxy_async` on shared space plus `__syncthreads()` "so the async proxy sees the writes", then `cp.async.bulk` shared→global, `cp_async_bulk_commit_group`, `cp_async_bulk_wait_group_read` ("wait until the group has finished reading shared memory"). The guide: "fence.proxy.async.shared::cta PTX instruction ... makes a thread's shared memory writes visible to the async proxy, including later bulk copies."

### 4.4 Minimal TMA tile load / store with `cuda::ptx` (assembled from §9 signatures)
```cuda
#include <cuda/ptx>
namespace ptx = cuda::ptx;
__shared__ alignas(1024) __nv_bfloat16 sA[128*64];   // 16 KiB tile, SWIZZLE_128B
__shared__ alignas(8) uint64_t bar;
if (threadIdx.x == 0) { ptx::mbarrier_init(&bar, 1); ptx::fence_mbarrier_init(ptx::sem_release, ptx::scope_cluster); }
__syncthreads();
if (threadIdx.x == 0) {
  ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &bar, 128*64*2);
  ptx::cp_async_bulk_tensor(ptx::space_cluster, ptx::space_global, sA, &tmap, {col0, row0}, &bar);   // load, 2D
}
while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta, &bar, 0 /*phase*/)) {}
/* ... compute into sA ... */
ptx::fence_proxy_async(ptx::space_shared);   // generic-proxy smem writes -> async proxy
__syncthreads();
if (threadIdx.x == 0) {
  ptx::cp_async_bulk_tensor(ptx::space_global, ptx::space_shared, &tmap_out, {col0, row0}, sA);  // store, bulk_group
  ptx::cp_async_bulk_commit_group();
  ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>{});   // smem may be reused; wait_group<0> for global visibility
}
```

### 4.5 Multicast
- PTX forms `cp.async.bulk{.tensor}...multicast::cluster [dstMem], ..., [smem_bar], ctaMask` with `uint16_t ctaMask` (bit i = CTA rank i receives the data at the same CTA-relative smem offset and the same CTA-relative mbarrier offset). CCCL: "In earlier versions, cp_async_bulk_multicast / cp_async_bulk_tensor_multicast was enabled for SM_90. This has been changed to SM_90a." (i.e. compile with `-arch=sm_90a`) [CCCL].
- The cooperative-groups intrinsic `__cluster_map_shared_multicast(const void *ptr, unsigned cluster_cta_mask)` exists in crt/sm_90_rt.h [HDR].
- Hopper Tuning Guide [WEB]: TMA "allows applications to transfer 1D and up to 5D tensors between global memory and shared memory, in both directions, as well as between the shared memory regions of different SMs in the same cluster."
- No multicast code example was seen in the Programming Guide snippets (see §10).

---

## 5. Memory fences, synchronization functions, cooperative launch

- Programming Guide [WEB]: "Memory fence and synchronization functions enforce a sequentially consistent ordering of memory accesses. These functions differ in the thread scope in which orderings are enforced." Recommends `cuda::atomic_thread_fence` from libcu++ "for safety and portability reasons".
  - `__threadfence_block()` "is equivalent to cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_block)"; "All writes to all memory made by the calling thread before the call to __threadfence_block() are observed by all threads in the calling thread's block" (plus the symmetric read ordering).
  - `__threadfence()` "is equivalent to cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device)".
  - `__threadfence_system()` "is equivalent to cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_system)" (visibility to host threads and peer devices).
  - Hopper adds `__threadfence_cluster()` intrinsic (crt/sm_90_rt.h) [HDR]; PTX `fence.sc/acq_rel.cluster` is exposed as `cuda::ptx::fence(sem, scope_cluster)` [CCCL].
- `__syncthreads()` "acts as a barrier at which all threads in the block must wait before any is allowed to proceed"; `__syncthreads_count(pred)`, `__syncthreads_and(pred)`, `__syncthreads_or(pred)` are "identical to __syncthreads() but with a predicate evaluated across the block's non-exited threads" (count returns number of threads with non-zero predicate). `__syncwarp(mask)` is the "warp-wide barrier between steps where threads exchange data through global or shared memory" [WEB].
- Named barriers: PTX `bar.sync a{, b}` / `barrier.sync` (16 named barriers per CTA, ids 0–15, thread count `b` multiple of warp size) — not retrievable from the PTX ISA page in this session; see §10.
- Cooperative launch [HDR, cuda_runtime_api.h]: `cudaLaunchCooperativeKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream)`: "The device on which this kernel is invoked must have a non-zero value for the device attribute ::cudaDevAttrCooperativeLaunch. The total number of blocks launched cannot exceed the maximum number of blocks per multiprocessor as returned by ::cudaOccupancyMaxActiveBlocksPerMultiprocessor ... times the number of multiprocessors as specified by the device attribute ::cudaDevAttrMultiProcessorCount. The kernel cannot make use of CUDA dynamic parallelism." Error: `cudaErrorCooperativeLaunchTooLarge`. Equivalent launch attribute: `cudaLaunchAttributeCooperative` (`val.cooperative = 1`) with `cudaLaunchKernelEx`.
- `cooperative_groups.h` [HDR]: "class grid_group; Threads within this this group are guaranteed to be co-resident on the same device within the same launched kernel. To use this group, the kernel must have been launched with cuLaunchCooperativeKernel (or the CUDA Runtime equivalent), and the device must support it (queryable device attribute). Constructed via this_grid();" → `cg::this_grid().sync()`. Multi-device (`this_multi_grid`) is deprecated/removed in CUDA 13 [WEB].

---

## 6. Compute capability 9.0 (H100) limits

### 6.1 From the Programming Guide "Compute Capabilities" table and the Hopper Tuning Guide [WEB]

| Item | CC 9.0 value | Source |
|---|---|---|
| Max shared memory per SM | 228 KB ("Maximum amount of shared memory per SM ... 228 KB") | PG table; HTG: "shared memory capacity per SM is 228 KB, a 39% increase compared to A100's capacity of 164 KB" |
| Max shared memory per thread block | 227 KB (opt-in) | PG table; HTG: "CUDA reserves 1 KB of shared memory per thread block. Hence, the H100 GPU enables a single thread block to address up to 227 KB of shared memory." |
| Static `__shared__` limit | 48 KB; "an explicit opt-in is also required to enable dynamic allocations above this limit" (`cudaFuncAttributeMaxDynamicSharedMemorySize`) | HTG |
| Unified L1/tex/smem per SM | 256 KB ("increases the maximum capacity of the combined L1 cache, texture cache, and shared memory to 256 KB" vs 192 KB on A100) | HTG |
| Carveout options | "The NVIDIA H100 GPU supports shared memory capacities of 0, 8, 16, 32, 64, 100, 132, 164, 196 and 228 KB per SM." selected via `cudaFuncSetAttribute(..., cudaFuncAttributePreferredSharedMemoryCarveout, pct)` | HTG |
| Shared memory banks | 32 banks; "successive 32-bit words map to successive banks" (4 B/bank/cycle) | PG |
| 32-bit registers per SM | 64 K | PG table |
| Max registers per thread | 255 | HTG / PG |
| Max resident blocks per SM | 32 ("the maximum number of thread blocks per SM is 32 for devices of compute capability 9.0") | HTG / PG table |
| Max resident warps per SM | 64 ("remains the same as in NVIDIA Ampere GPU architecture (that is, 64)") | HTG / PG table |
| Max resident threads per SM | 2048 | PG table |
| Max threads per block | 1024 | PG (unchanged since CC 2.x) |
| Portable / max cluster size | 8 / 16 (opt-in) | PG / HTG |

### 6.2 Runtime queries [HDR, driver_types.h]
`cudaDeviceProp` fields: `int clockRate /**< Deprecated, Clock frequency in kilohertz */`, `int multiProcessorCount`, `int memoryClockRate /**< Deprecated, Peak memory clock frequency in kilohertz */`, `int memoryBusWidth /**< Global memory bus width in bits */`, `int l2CacheSize /**< Size of L2 cache in bytes */`, `int persistingL2CacheMaxSize /**< Device's maximum l2 persisting lines capacity setting in bytes */`, `int maxThreadsPerMultiProcessor`, `size_t sharedMemPerMultiprocessor`, `int regsPerMultiprocessor`, `size_t sharedMemPerBlockOptin /**< Per device maximum shared memory per block usable by special opt in */`, `int maxBlocksPerMultiProcessor`, `int accessPolicyMaxWindowSize`, `size_t reservedSharedMemPerBlock /**< Shared memory reserved by CUDA driver per block in bytes */` (1 KB on H100 per HTG), `int clusterLaunch`.
Attributes: `cudaDevAttrL2CacheSize = 38`, `cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81`, `cudaDevAttrMaxSharedMemoryPerBlockOptin = 97`, `cudaDevAttrMaxBlocksPerMultiprocessor = 106`, `cudaDevAttrMaxPersistingL2CacheSize = 108`, `cudaDevAttrMaxAccessPolicyWindowSize = 109`, `cudaDevAttrReservedSharedMemoryPerBlock = 111`, `cudaDevAttrClusterLaunch = 120`. (`clockRate`/`memoryClockRate` are deprecated; use `cudaDeviceGetAttribute(cudaDevAttrClockRate / cudaDevAttrMemoryClockRate)` or NVML.)

---

## 7. H100 silicon facts (Hopper Tuning Guide, Hopper blog, whitepaper)

### 7.1 Chip configuration [WEB — "NVIDIA Hopper Architecture In-Depth" blog / "NVIDIA H100 Tensor Core GPU Architecture" whitepaper]

| | Full GH100 | H100 SXM5 | H100 PCIe |
|---|---|---|---|
| GPCs | 8 | 8 | "Either seven or eight GPCs are active" |
| TPCs | 72 (9 per GPC) | 66 | — |
| SMs (2 per TPC) | 144 | 132 ("Only 132 out of the 144 SMs are active") | 114 ("only 114 of the 144 SMs are active across those GPCs") |
| HBM stacks | 6 (HBM3 or HBM2e) | 5 HBM3 ("only five out of six of the HBM3 memory banks and associated memory controllers are working") | 5 HBM2e (80 GB) |
| Memory controllers | 12 × 512-bit | 10 × 512-bit (=5120-bit) | 10 × 512-bit |
| L2 | 60 MB | 50 MB | 50 MB |
| Memory bandwidth | — | 3.35 TB/s (nvidia.com H100 product page: "GPU Memory Bandwidth 3.35TB/s"); HTG: "HBM3 memory system supports up to 3 TB/s memory bandwidth, a 93% increase over the 1.55 TB/s on A100-40GB" | 2 TB/s (product/datasheet tables) |
| MIG | — | "Up to 7 MIG instances" at 10 GB each | (datasheet row blank/unclear) |

- L2 [WEB, blog]: "Using a partitioned crossbar structure, the L2 cache localizes and caches data for memory accesses from SMs in GPCs directly connected to the partition." HTG: "The NVIDIA Hopper architecture increases the L2 cache capacity from 40 MB in the A100 GPU to 50 MB in the H100 GPU. Along with the increased capacity, the bandwidth of the L2 cache to the SMs is also increased." and "allows CUDA users to control the persistence of data in L2 cache similar to the NVIDIA Ampere GPU Architecture."
- HTG memory [WEB]: "The NVIDIA H100 GPU has support for HBM3 and HBM2e memory, with capacity up to 80 GB."
- MIG [WEB, whitepaper]: "Up to seven GPU Instances can be securely isolated from each other ... on-chip crossbar ports, L2 cache banks, memory controllers, and DRAM address busses are all assigned uniquely to an individual instance." Each instance has dedicated NVDEC/NVJPG. Second-gen MIG: "approximately 3x more compute capacity and nearly 2x more memory bandwidth per GPU instance compared to A100." (The inference "7 instances → 7 GPCs usable" is consistent with the PCIe statement "seven or eight GPCs active" but no official sentence equating a 1g MIG slice with exactly one GPC was seen — see §10.)
- Clusters [WEB, whitepaper]: "A cluster is a group of thread blocks that are guaranteed to be concurrently scheduled, and enable efficient cooperation and data sharing for threads across multiple SMs."
- DSMEM [WEB, whitepaper via mirrored copy]: "Distributed Shared Memory allows direct SM-to-SM communications for loads, stores, and atomics across multiple SM shared memory blocks." and DSMEM "accelerates data exchange between Thread Blocks by about 7x compared with using global memory" (the only official-document instance of the 7x figure; the blog snippets did not contain it).
- TMA [WEB, blog]: "a new Tensor Memory Accelerator (TMA) unit that can transfer large blocks of data efficiently between global memory and shared memory. TMA also supports asynchronous copies between thread blocks in a cluster." "A new asynchronous transaction barrier for doing atomic data movement and synchronization." "only a small number of CUDA threads are now required to manage the full memory bandwidth of H100."
- SM-to-SM network: no official sentence retrieved (see §10).
- Boost clocks 1980 MHz (SXM5) / 1755 MHz (PCIe): only third-party spec tables were returned; not confirmed from an NVIDIA page (see §10).

---

## 8. Benchmark hygiene: L2 persistence, clocks, timing

### 8.1 L2 set-aside / access policy windows [HDR + WEB]
```c
enum cudaLimit { ... cudaLimitMaxL2FetchGranularity = 0x05, /**< A value between 0 and 128 that indicates the maximum fetch granularity of L2 (in Bytes). This is a hint */
                 cudaLimitPersistingL2CacheSize = 0x06 /**< A size in bytes for L2 persisting lines cache size */ };
cudaError_t cudaDeviceSetLimit(enum cudaLimit limit, size_t value);   // "controls size in bytes available for persisting L2 cache. This is purely a performance hint and it can be ignored or clamped depending on the platform."
enum cudaAccessProperty { cudaAccessPropertyNormal = 0, cudaAccessPropertyStreaming = 1 /* "less likely to persist" */, cudaAccessPropertyPersisting = 2 /* "more likely to persist in cache" */ };
struct cudaAccessPolicyWindow { void *base_ptr; /* "CUDA driver may align it" */ size_t num_bytes; /* "CUDA driver may restrict the maximum size and alignment" */
                                float hitRatio; /* "percentage of lines assigned hitProp, rest are assigned missProp" */
                                enum cudaAccessProperty hitProp; enum cudaAccessProperty missProp; /* "Must be either NORMAL or STREAMING" */ };
cudaError_t cudaCtxResetPersistingL2Cache(void);   // "Resets all persisting lines in cache to normal status. Takes effect on function return."
```
Apply per stream (`cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr)`), per kernel node, or per launch via `cudaLaunchAttributeAccessPolicyWindow` in `cudaLaunchKernelEx`. Programming Guide "L2 Cache Control" [WEB]: example sets `size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize)`; "hitRatio 1.0 ... the hardware will attempt to cache the whole 32KB window in the set-aside L2 cache area. Since the set-aside area is smaller than the window, cache lines will be evicted to keep the most recently used 16KB"; two streams with hitRatio 1.0 "might evict each others' cache lines ... if both accessPolicyWindows have a hitRatio value of 0.5, they will be less likely to evict their own or each others'." Max window: `cudaDevAttrMaxAccessPolicyWindowSize` / `prop.accessPolicyMaxWindowSize`.

### 8.2 Clock locking (nvidia-smi manual) [WEB]
- `nvidia-smi -lgc <min,max>` / `--lock-gpu-clocks`: "takes a min,max pair in MHz (or a single value), is supported on Volta and later, and requires root"; optional `--mode` (1 = closed-loop controllers, "better performance per watt but slightly less accurate frequencies than the default mode 0"). Reset: `nvidia-smi -rgc`.
- `nvidia-smi -lmc` / `--lock-memory-clocks`: requires root; "This option does not work on GPUs based on NVIDIA Hopper architectures" → use `--lock-memory-clocks-deferred`. Reset: `-rmc`.
- `nvidia-smi -ac <mem,gfx>` / `--applications-clocks`: "deprecated and will be removed in a future CUDA release. It recommends using -lmc for memory clocks and -lgc for graphics clocks."
- CUPTI/Nsight guidance: `nvidia-smi --lock-gpu-clocks=tdp,tdp` for profiling consistency.

### 8.3 Event timing [HDR]
`cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)`: "Computes the elapsed time between two events (in milliseconds with a resolution of around 0.5 microseconds). If either event was last recorded in a non-NULL stream, the resulting time may be greater than expected ... Any number of other different stream operations could execute in between the two measured events." Events created with `cudaEventDisableTiming` return `cudaErrorInvalidResourceHandle`. For in-kernel cycle counts use `clock64()` / `%globaltimer` (`cuda::ptx::get_sreg_globaltimer()`, `get_sreg_clock64()` in `<cuda/ptxs/get_sreg.h>`).

---

## 9. CCCL `cuda::ptx` wrappers relevant to clusters / DSMEM / TMA / mbarrier [CCCL @ b008f6b]

Header: `<cuda/ptx>` (all), or per-instruction `<cuda/ptxs/<name>.h>` (e.g. `<cuda/ptxs/cp_async_bulk_tensor.h>`, `<cuda/ptxs/mbarrier_wait.h>` for test_wait/try_wait, `<cuda/ptxs/get_sreg.h>`). Convention: function name = PTX opcode with dots → underscores; qualifiers passed as tag *arguments* (`cuda::ptx::sem_release`, `scope_cta`, `scope_cluster`, `space_global`, `space_shared` (= `.shared::cta`), `space_cluster` (= `.shared::cluster`), `op_add`, `n32_t<N>`). "The API does not guarantee stability of template parameters ... Use arguments to drive overload resolution." Each binding comments the PTX ISA / SM requirement; guard with `#if __cccl_ptx_isa >= 800` and `NV_IF_TARGET(NV_PROVIDES_SM_90, (...))`. All functions are `__device__ static inline`.

| PTX instruction | `cuda::ptx` function(s) | Key overloads (ISA/SM) |
|---|---|---|
| `cp.async.bulk` G→S (mbarrier) | `cp_async_bulk(space_cluster, space_global, void* dstMem, const void* srcMem, const uint32_t& size, uint64_t* smem_bar)` (ISA 80, SM_90); also `(space_shared, space_global, ...)` (ISA 86). "Both srcMem and dstMem must be 16-byte aligned, and size must be a multiple of 16." | |
| `cp.async.bulk` S→cluster S (DSMEM) | `cp_async_bulk(space_cluster, space_shared, void* dstMem, const void* srcMem, const uint32_t& size, uint64_t* rdsmem_bar)` (ISA 80, SM_90) | remote mbarrier in dst CTA |
| `cp.async.bulk` S→G | `cp_async_bulk(space_global, space_shared, void* dstMem, const void* srcMem, const uint32_t& size)` (bulk_group, ISA 80, SM_90) | |
| `cp.async.bulk ... multicast::cluster` | `cp_async_bulk(space_cluster, space_global, dstMem, srcMem, size, uint64_t* smem_bar, const uint16_t& ctaMask)` (ISA 80, **SM_90a**) | |
| `cp.async.bulk.tensor.{1..5}d` G→S | `cp_async_bulk_tensor(space_cluster, space_global, void* dstMem, const void* tensorMap, const int32_t (&tensorCoords)[N], uint64_t* smem_bar)` (ISA 80, SM_90); `(space_shared, space_global, ...)` (ISA 86) | tile mode, `mbarrier::complete_tx::bytes` |
| `cp.async.bulk.tensor.{1..5}d` S→G | `cp_async_bulk_tensor(space_global, space_shared, const void* tensorMap, const int32_t (&tensorCoords)[N], const void* srcMem)` (bulk_group) | |
| `cp.async.bulk.tensor ... multicast::cluster` | same + `const uint16_t& ctaMask` (ISA 80, **SM_90a**) | gather/scatter variants in `cp_async_bulk_tensor_gather_scatter.rst` (SM_100a+) |
| `cp.async.bulk.commit_group` | `cp_async_bulk_commit_group()` | |
| `cp.async.bulk.wait_group{.read} N` | `cp_async_bulk_wait_group(n32_t<N>)`, `cp_async_bulk_wait_group_read(n32_t<N>)` | |
| `cp.async.bulk.prefetch.L2` / `.tensor` | `cp_async_bulk_prefetch(const void* srcMem, uint32_t size, uint64_t cache_policy = 0x10F0000000000000)` (ISA 80, SM_90); `cp_async_bulk_prefetch_L2_evict_last(const void* srcMem, uint32_t size)` (SM_107); `cp_async_bulk_prefetch_tensor(...)` | |
| `cp.reduce.async.bulk` | `cp_reduce_async_bulk(space_cluster, space_shared, op_t, T* dstMem, const T* srcMem, uint32_t size, uint64_t* rdsmem_bar)` (S→DSMEM) and `(space_global, space_shared, op_t, dstMem, srcMem, size)` (S→G). Ops: `op_add`, `op_min`, `op_max`, `op_inc`, `op_dec`, `op_and_op`, `op_or_op`, `op_xor_op`; `.s64 add` emulated; f16/bf16 variants (`cp_reduce_async_bulk_f16/bf16.rst`) | ISA 80, SM_90 |
| `cp.reduce.async.bulk.tensor.{1..5}d` | `cp_reduce_async_bulk_tensor(space_global, space_shared, op_t<Op>, const void* tensorMap, const int32_t (&tensorCoords)[N], const void* srcMem)` (bulk_group) | ISA 80, SM_90 |
| `mbarrier.init` | `mbarrier_init(uint64_t* addr, const uint32_t& count)` (ISA 70, SM_80) | |
| `mbarrier.inval` | `mbarrier_inval(uint64_t* addr)` | |
| `mbarrier.arrive` | `uint64_t mbarrier_arrive(uint64_t* addr)`; `mbarrier_arrive(uint64_t* addr, const uint32_t& count)` (ISA 78, SM_90); `mbarrier_arrive(sem_release, scope_t<cta|cluster>, space_shared, addr[, count]) -> uint64_t`; `void mbarrier_arrive(sem_release, scope_cluster, space_cluster, addr[, count])` (remote CTA barrier, no state) (ISA 80); `sem_relaxed` variants (ISA 86) | |
| `mbarrier.arrive.noComplete` | `mbarrier_arrive_no_complete(uint64_t* addr, const uint32_t& count)` | |
| `mbarrier.arrive.drop` | `mbarrier_arrive_drop(...)` | |
| `mbarrier.arrive.expect_tx` | `uint64_t mbarrier_arrive_expect_tx(sem_release, scope_t<Scope>, space_shared, uint64_t* addr, const uint32_t& tx_count)`; `void mbarrier_arrive_expect_tx(sem_release, scope_cluster, space_cluster, addr, tx_count)` (ISA 80, SM_90); `sem_relaxed` variants (ISA 86) | |
| `mbarrier.expect_tx` | `mbarrier_expect_tx(sem_relaxed, scope_t<Scope>, space_shared|space_cluster, uint64_t* addr, uint32_t txCount)` (ISA 80, SM_90) | |
| `mbarrier.complete_tx` | `mbarrier_complete_tx(sem_relaxed, scope_t<Scope>, space_shared|space_cluster, uint64_t* addr, uint32_t txCount)` | |
| `mbarrier.test_wait{.parity}` | `bool mbarrier_test_wait(uint64_t* addr, const uint64_t& state)`; `mbarrier_test_wait(sem_acquire|sem_relaxed, scope_t<Scope>, addr, state)`; `mbarrier_test_wait_parity(addr, const uint32_t& phaseParity)` + sem/scope forms | |
| `mbarrier.try_wait{.parity}` | `bool mbarrier_try_wait(uint64_t* addr, const uint64_t& state[, const uint32_t& suspendTimeHint])`; `mbarrier_try_wait(sem_acquire|sem_relaxed, scope_t<Scope>, addr, state[, suspendTimeHint])`; `bool mbarrier_try_wait_parity(uint64_t* addr, const uint32_t& phaseParity[, suspendTimeHint])`; `mbarrier_try_wait_parity(sem_acquire|sem_relaxed, scope_t<Scope>, addr, phaseParity[, suspendTimeHint])` (ISA 78/80/86, SM_90) | |
| `mbarrier.pending_count` | `uint32_t mbarrier_pending_count(uint64_t state)` | |
| `fence{.sc,.acq_rel,.acquire,.release}.{cta,cluster,gpu,sys}` | `fence(sem_sc|sem_acq_rel, scope_t<Scope>)` (ISA 60, SM_70); `fence(sem_sc|sem_acq_rel, scope_cluster)` (ISA 78, SM_90); `fence(sem_acquire|sem_release, scope_t<Scope>)` (ISA 86) | |
| `fence.mbarrier_init.release.cluster` | `fence_mbarrier_init(sem_release, scope_cluster)` (ISA 80, SM_90) | |
| `fence.proxy.async{.space}` | `fence_proxy_async()`; `fence_proxy_async(space_t<Space>)` e.g. `fence_proxy_async(space_shared)` ≡ `fence.proxy.async.shared::cta` (ISA 80, SM_90) | |
| `fence.proxy.tensormap::generic` | `fence_proxy_tensormap_generic(sem_release, scope)`, `(sem_acquire, scope, const void* addr, n32_t<N> size)` (ISA 83, SM_90) | |
| `fence.proxy.alias` / `fence...sync_restrict` | `fence_proxy_alias()`, `fence_proxy_alias(sem_t<Sem>)`, `fence_sync_restrict(...)` (ISA 86) | |
| `barrier.cluster.arrive{.release,.relaxed}` / `.wait{.acquire}` | `barrier_cluster_arrive()`, `barrier_cluster_wait()` (ISA 78, SM_90); `barrier_cluster_arrive(sem_release)`, `barrier_cluster_arrive(sem_relaxed)`, `barrier_cluster_wait(sem_acquire)` (ISA 80, SM_90) | |
| `mapa` | **not implemented** in `cuda::ptx` (cccl #1414); use `cg::cluster_group::map_shared_rank` / `__cluster_map_shared_rank` or inline asm | |
| `getctarank` | `uint32_t getctarank(space_cluster, const void* addr)` (ISA 78, SM_90) | |
| `st.async` | `st_async(Type* addr, const Type& value, uint64_t* remote_bar)`; `st_async(Type* addr, const Type (&value)[2], remote_bar)`; `st_async(B32* addr, const B32 (&value)[4], remote_bar)` — `st.async.weak.shared::cluster.mbarrier::complete_tx::bytes` (ISA 81, SM_90) | |
| `red.async` | `red_async(op_inc|op_dec|op_min|op_max|op_add, uint32_t* dest, const uint32_t& value, uint64_t* remote_bar)`; `op_min|op_max|op_add` for `int32_t`; `op_and_op|op_or_op|op_xor_op` for `B32`; `op_add` for `uint64_t` — `red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes` (ISA 81, SM_90) | |
| `elect.sync` | `bool elect_sync(const uint32_t& membermask)` (ISA 80, SM_90) | |
| `tensormap.replace` / `tensormap.cp_fenceproxy` | `tensormap_replace_global_address`, `_rank`, `_box_dim`, `_global_dim`, `_global_stride`, `_element_size`, `_element_stride`, `_elemtype`, `_interleave_layout`, `_swizzle_mode`, `_swizzle_atomicity`, `_fill_mode`; `tensormap_cp_fenceproxy(...)` (SM_90a) | |
| special registers | `get_sreg_clock()`, `get_sreg_clock64()`, `get_sreg_globaltimer()`, `get_sreg_smid()`, `get_sreg_nsmid()`, `get_sreg_clusterid_{x,y,z}()`, `get_sreg_nclusterid_{x,y,z}()`, `get_sreg_cluster_ctaid_{x,y,z}()`, `get_sreg_cluster_nctaid_{x,y,z}()`, `get_sreg_cluster_ctarank()`, `get_sreg_cluster_nctarank()`, `get_sreg_is_explicit_cluster()`, `get_sreg_dynamic_smem_size()`, `get_sreg_total_smem_size()`, `get_sreg_aggr_smem_size()`, `get_sreg_laneid()`, `get_sreg_warpid()`, `get_sreg_lanemask_*()` (`<cuda/ptxs/get_sreg.h>`) | |
| others in the list | `applypriority_async_bulk`, `clusterlaunchcontrol.*` (SM_100), `cp_async_mbarrier_arrive{_noinc}`, `ldmatrix`/`stmatrix`, `multimem.*`, `setmaxnreg`, `st.bulk`, `tcgen05.*` (SM_100), `exit`, `trap`, `prmt`, `bfind`, `bmsk`, `shfl_sync`, `shl`, `shr` | |

Raw-asm vs `cuda::ptx` choice: `cuda::ptx` covers every TMA/mbarrier/fence/cluster-barrier/DSMEM-async instruction needed on sm_90 except `mapa` (use CG). Multicast variants require `-arch=sm_90a`. For `bar.sync`/named barriers and `mapa` with explicit `.shared::cluster` 32-bit addressing, use inline PTX.

---

## 10. Facts I could not confirm from official docs (this session)

1. **Full text of any docs.nvidia.com / developer.nvidia.com page.** Direct fetches were refused by the session proxy (403); everything tagged [WEB] is a search-snippet excerpt. The Programming Guide's complete Thread Block Cluster, DSMEM (`clusterHist_kernel` body), TMA, and Async Barrier sections were therefore not read end-to-end; only the quoted sentences are confirmed.
2. **"DSMEM about 7x faster than global memory"**: seen only in a third-party-hosted copy of the NVIDIA whitepaper ("accelerates data exchange between Thread Blocks by about 7x compared with using global memory"); not seen in snippets of the Hopper blog itself. No official DSMEM bandwidth/latency number was found.
3. **"SM-to-SM network"** wording and any statement that it is dedicated/per-GPC: not retrieved from an NVIDIA page (only academic papers).
4. **L2 "two partitions"**: the blog confirms "partitioned crossbar structure"; the number of partitions (2 × 25 MB) and that 4 GPCs attach to each come only from third-party sources. No official L2 bandwidth figure was retrieved.
5. **Boost clocks 1980 MHz (SXM5) / 1755 MHz (PCIe)**: only third-party spec tables; the whitepaper table was not readable. A forum post claims the whitepaper's Table 3 uses 1755 MHz for the SXM5 throughput math.
6. **PCIe memory bandwidth 2.0 TB/s and HBM2e, PCIe MIG support**: from reseller/datasheet mirrors, not an NVIDIA page read directly. The nvidia.com H100 page snippet confirmed only the SXM 3.35 TB/s and "Up to 7 MIG instances at 10 GB".
7. **MIG ↔ GPC mapping** ("7 MIG instances → 7 GPCs usable"): the whitepaper confirms 7 instances and per-instance L2 banks/memory controllers; no sentence equating one 1g instance with one GPC was found.
8. **Hopper Tuning Guide SM counts (132/114)**: the 132/114/144 numbers were confirmed from the blog/whitepaper snippets; whether the Tuning Guide itself states them was not verified.
9. **Compute capability table entries** (2048 threads/SM, 64 warps/SM, 32 blocks/SM, 64 K regs, 32 banks): read from a search summary of the table whose cell alignment the summarizer flagged as ambiguous; HTG sentences independently confirm 32 blocks/SM, 64 warps/SM, 228/227 KB, 255 regs/thread.
10. **Named barriers (`bar.sync id, count`, 16 per CTA)** and the PTX ISA text for `mapa`, `st.async`, `red.async`, `barrier.cluster`, `.shared::cluster` and the "peer CTA must exist" rule: PTX ISA page not fetchable; only the one snippet quoted in §1.5.
11. **Programming Guide multicast example**: none appeared in snippets; multicast API facts come from CCCL/PTX wrapper signatures and the crt intrinsic `__cluster_map_shared_multicast`.
12. **`cudaClusterSchedulingPolicy` semantics beyond the one-line enum comments** (when hardware honors Spread vs LoadBalancing, interaction with cluster size): no further official text found; 13.4 docs add `cudaClusterSchedulingPolicyRubinDsmemLocality` (not described).
13. **Exact alignment rules for TMA smem destination (128 B base, 1024 B for SWIZZLE_128B)**: from 12.8.1-era guide snippets; the cuda.h comment itself does not state shared-memory alignment.
14. **`cuda::ptx` list is from CCCL `main` (post-3.2)**; a CUDA 12.x toolkit's bundled libcu++ may lack newer overloads (e.g. `sem_relaxed` forms, `space_shared` dst for `cp_async_bulk_tensor` requiring PTX ISA 86). Check `__cccl_ptx_isa`.
