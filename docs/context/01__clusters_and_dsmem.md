# Thread Block Clusters and Distributed Shared Memory — programming model, APIs, hardware guarantees

Sources: CUDA Programming Guide (clusters, DSMEM, cooperative groups), Hopper Tuning Guide, CUDA 12.9 runtime/driver headers (verbatim doc comments), CCCL `cuda::ptx`, CUTLASS `cluster_sm90.hpp` (canonical PTX). Everything here is official unless tagged **[LIT]** (measured by prior work) or **[INF]** (our inference).

## 1. Model

- A **cluster** is a group of thread blocks (CTAs) "guaranteed to be co-scheduled on a GPU Processing Cluster (GPC)". All CTAs of a cluster are launched **all-or-nothing** (hardware refuses to launch any CTA of a CGA unless there is room for all) [PAT US 12,333,311 B2].
- Max portable size **8**; H100 allows **16** after `cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1)`. Grid dimensions must be divisible by cluster dimensions. Cluster shape may be 1D/2D/3D; rank = linearised position `x + y·dimx + z·dimx·dimy`.
- **DSMEM**: "threads read, write, and perform atomics in shared memory across all blocks in the cluster"; the combined shared memory of the cluster forms one logical address space (`.shared::cluster` PTX state space) that is *physically distributed* — every byte lives in exactly one SM's SRAM.
- Existence rule: "Accessing data in distributed shared memory requires all the thread blocks to exist" → `cluster.sync()` (or equivalent arrive/wait) **before** the first remote access, and "all distributed shared memory operations [must] happen before the exit of a thread block" → `cluster.sync()` **after** the last remote access. PTX: "the peer CTA must exist and hasn't exited before completing the shared memory operation." Violations give undefined results (in practice silent garbage or an illegal-address error).
- Scheduling policy (`cudaClusterSchedulingPolicy`): `Default = 0`, `Spread = 1` ("spread the blocks within a cluster to the SMs"), `LoadBalancing = 2` ("allow the hardware to load-balance the blocks in a cluster to the SMs"). Observed [LIT, Luo]: Default and Spread never put two CTAs of one cluster on the same SM; LoadBalancing may co-locate them (then "remote" accesses are local). Observed [LIT, Lühnen]: with Default, rank→SM assignment is stable across launches and CTAs of one cluster never share an SM.

## 2. Host API (CUDA 12.x/13.x runtime) — exact names

```cpp
// Compile-time cluster shape (cannot be changed at launch):
__global__ void __cluster_dims__(2, 1, 1) k_static(...);   // launch with <<<grid, block>>>, grid.x % 2 == 0

// Runtime cluster shape:
cudaFuncSetAttribute((void*)k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);     // only needed for 9..16
cudaFuncSetAttribute((void*)k, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);    // > 48 KB dynamic smem
cudaLaunchAttribute attrs[2];
attrs[0].id = cudaLaunchAttributeClusterDimension;                  // = 4
attrs[0].val.clusterDim = {CS, 1, 1};                               // each must divide the grid dim
attrs[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference; // = 5
attrs[1].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicyDefault; // or Spread / LoadBalancing
cudaLaunchConfig_t cfg{};
cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = bytes; cfg.stream = stream;
cfg.attrs = attrs; cfg.numAttrs = 2;
int maxClusters = 0;
cudaOccupancyMaxActiveClusters(&maxClusters, (void*)k, &cfg);  // "not guaranteed to be achievable"
cudaLaunchKernelEx(&cfg, k, args...);
```

Other relevant runtime items: `cudaOccupancyMaxPotentialClusterSize`, `cudaFuncAttributeRequiredClusterWidth/Height/Depth` (10–13; cannot be set if `__cluster_dims__` was used), `cudaFuncAttributeClusterDimMustBeSet`, `cudaDevAttrClusterLaunch` (= 120), `cudaDeviceProp::clusterLaunch`, `cudaLaunchAttributeCooperative` (grid sync via `cg::this_grid().sync()`, requires all blocks co-resident).

## 3. Device API — cooperative groups `cluster_group` and the PTX underneath

| CG member (`cg::this_cluster()`) | PTX / intrinsic | Notes |
|---|---|---|
| `block_rank()` | `mov.u32 %r, %cluster_ctarank;` | 0 … num_blocks()-1 |
| `num_blocks()` | `%cluster_nctarank` | |
| `block_index()` / `dim_blocks()` | `%cluster_ctaid.{x,y,z}` / `%cluster_nctaid.{x,y,z}` | |
| cluster id in grid | `%clusterid.{x,y,z}`, `%nclusterid.{x,y,z}` | |
| `map_shared_rank(T* p, unsigned rank)` | `mapa.shared::cluster.u32 r, addr32, rank;` (or `mapa.u64` on generic addresses) | returns a **generic** pointer into the target CTA's smem; dereference with ordinary `ld/st/atom` → compiler emits `.shared::cluster` or generic-space SASS `LD/ST` |
| `query_shared_rank(const void*)` | `getctarank.shared::cluster.u32` | owner rank of a DSMEM address |
| `barrier_arrive()` | `barrier.cluster.arrive.aligned;` (release semantics by default; `.relaxed` variant via `__cluster_barrier_arrive_relaxed()`) | non-blocking |
| `barrier_wait()` | `barrier.cluster.wait.aligned;` (acquire) | blocks until **all threads of all CTAs** arrived |
| `sync()` | arrive + wait | ≈ 850–1300 cycles [LIT] |
| — | `%is_explicit_cluster` | 1 if launched with explicit cluster dims |
| `__threadfence_cluster()` | `fence.acq_rel.cluster;` | cluster-scope fence |

Canonical inline PTX (CUTLASS `cluster_sm90.hpp`, verified strings):

```cpp
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint32_t mapa_u32(uint32_t smem_addr, uint32_t rank) {
  uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(smem_addr), "r"(rank)); return r; }
__device__ __forceinline__ uint32_t cluster_ctarank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ void cluster_arrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
__device__ __forceinline__ void cluster_arrive()         { asm volatile("barrier.cluster.arrive.aligned;"         ::: "memory"); }
__device__ __forceinline__ void cluster_wait()           { asm volatile("barrier.cluster.wait.aligned;"           ::: "memory"); }
// DSMEM load/store with explicit 32-bit cluster-shared addresses (bypasses generic-address resolution):
__device__ __forceinline__ uint32_t ld_dsmem_u32(uint32_t a) { uint32_t v; asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory"); return v; }
__device__ __forceinline__ void st_dsmem_u32(uint32_t a, uint32_t v) { asm volatile("st.shared::cluster.u32 [%0], %1;" :: "r"(a), "r"(v) : "memory"); }
```

Notes on address forms: `mapa.shared::cluster.u32` takes a `.shared::cta` 32-bit address and returns a `.shared::cluster` 32-bit address (valid for `ld/st/atom/red.shared::cluster`, `mbarrier.*.shared::cluster`, `st.async`, `red.async`, `cp.async.bulk ... .shared::cluster`). `mapa.u64` maps a generic address to a generic address of the peer's smem. A `.shared::cluster` address whose owner is the executing CTA is a local access routed through the DSMEM interface (≈33 cycles vs ≈29 cycles for `ld.shared::cta`) [LIT, Luo].

## 4. What hardware does on a DSMEM access (for interpreting measurements)

- Generic-proxy `ld/st.shared::cluster` to a remote CTA: the request leaves the SM's LSU, is routed through the GPC's SM-to-SM network to the owner SM, is serviced by the owner's shared-memory pipeline (it competes with the owner's local `LDS/STS` for bank cycles — local SMEM bandwidth drops from ≈120 to 71–103 B/cycle while being read remotely [LIT, ClusterSim]), and the response returns the same way. Loads are blocking for the issuing warp (scoreboard), stores are posted (fire-and-forget) — hence the asymmetry: remote store issue rate ≈ 21 B/cycle vs remote load rate ≈ 4.8 B/cycle per block with limited MLP [LIT, CREDIT], and the **width of a store does not change its cost** (uint8 → uint64 halves time per doubling) [LIT, Lühnen].
- Atomics `atom.shared::cluster` / `red.shared::cluster`: executed at the owner SM's shared-memory atomic unit; return value (for `atom`) travels back. No published latency [LIT survey] → `docs/experiments/12__dsmem_atomics_and_async_stores.md`.
- Visibility of a remote store to the owner: a plain remote `st` followed by the owner polling its own smem shows ≈625 cycles store-visibility latency on H200 [LIT, FlashGPU-sim] (much larger than the 190-cycle load latency; the posted store must traverse the fabric and be committed before the owner's load can observe it). `st.async`/`red.async` instead signal a remote **mbarrier** with `complete_tx` so the consumer waits on the barrier rather than polling data.
- Cluster barrier: `barrier.cluster.arrive` ≈ 600–1050 cycles, `.wait` ≈ 45–90 cycles when everyone has already arrived [LIT]. The arrive is implemented with a hardware CGA barrier at GPC level [PAT]. This is the dominant fixed cost of any DSMEM protocol; measure it per cluster size and per thread count (`docs/experiments/07__synchronization_primitives.md`).

## 5. Occupancy and placement facts the experiments must respect

- Each CTA of a cluster is placed on a distinct SM under Default/Spread; the cluster consumes `CS` SMs of **one GPC**. With 1 CTA per SM (e.g. ≥115 KB dynamic smem, or 1024 threads with enough registers), `cudaOccupancyMaxActiveClusters` ≈ Σ_GPC ⌊SMs_in_GPC / CS⌋ (e.g. B200 example 15×8, 33×4, 7×16 [LIT]).
- With cluster size 16 only GPCs with ≥16 enabled SMs can host a cluster; use this as the primary GPC-discovery probe (`03__sm_gpc_topology`).
- Rank→SM assignment within a GPC is deterministic for a given launch configuration [LIT]; the SM *set* chosen for a given cluster id varies with what else is resident. Always record `%smid` per rank.
- A kernel that is launched with clusters pays launch overhead when the grid exceeds one wave: ≈0 % (CS 2), ≤5 % (CS 4, 8), up to 22 % (CS 16) [LIT, Lühnen] → `docs/experiments/02__cluster_launch_and_occupancy.md`.

## 6. Minimal correct DSMEM kernel skeleton (template for every cluster experiment)

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

extern __shared__ __align__(1024) unsigned char dyn_smem[];

__global__ void dsmem_kernel(unsigned* out_smid, ... ) {
  cg::cluster_group cluster = cg::this_cluster();
  const unsigned rank = cluster.block_rank(), n = cluster.num_blocks();
  unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  if (threadIdx.x == 0) out_smid[blockIdx.x] = smid;          // every experiment records placement

  // 1. local initialisation of this CTA's smem
  ...
  // 2. make initialisation visible cluster-wide and guarantee all CTAs exist
  cluster.sync();                                              // barrier.cluster.arrive(release) + wait(acquire)
  // 3. remote accesses
  unsigned peer = (rank + 1) % n;
  T* remote = cluster.map_shared_rank(reinterpret_cast<T*>(dyn_smem), peer);
  ...
  // 4. no CTA may exit while a peer can still touch its smem
  cluster.sync();
}
```

Launch with `cudaLaunchKernelEx` as in §2; grid = `num_clusters * CS` blocks.
