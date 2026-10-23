# Synchronization primitives on H100: what each one guarantees, what it costs, when to use it

All syntax verified with ptxas 12.9 for sm_90a. Costs are prior-work medians (`08__prior_work_reference_numbers.md`); experiment `07__synchronization_primitives` re-measures every row on our card and the DSMEM/TMA experiments report the share of each primitive in their protocols.

## 1. Decision table

| Need | Primitive | Scope of ordering | Who participates | Expected cost (cycles) |
|---|---|---|---|---|
| Order memory ops among threads of one warp | `__syncwarp(mask)` / `bar.warp.sync` | warp (not a memory-model scope) | lanes in mask | ~few |
| Block-wide barrier + visibility of generic smem/global writes within the CTA | `__syncthreads()` = `bar.sync 0` (`BAR.SYNC.DEFER_BLOCKING`); named `bar.sync id, nthreads` / `bar.arrive id, nthreads` (16 barriers/CTA) | `.cta` | all (or `nthreads`) threads of the CTA | ≈ 14–25 |
| Producer/consumer inside a CTA with asynchronous (TMA/`cp.async`) data | mbarrier (`04__mbarrier.md`) | `.cta` (or `.cluster`) + async completion | arriving threads + async engines | arrive ≈ 10s; wait predicate ≈ 42; completion-to-observation quantised by polling |
| Cluster-wide barrier + visibility of generic writes (incl. DSMEM stores) across CTAs | `barrier.cluster.arrive{.release|.relaxed}.aligned; barrier.cluster.wait{.acquire}.aligned;` (`UCGABAR_ARV/WAIT`; `cluster.sync()`) | `.cluster` | all non-exited threads of all CTAs | arrive ≈ 600–1050 (release form includes `MEMBAR.ALL.GPU; ERRBAR; CGAERRBAR`), wait ≈ 45–90 when everyone arrived; sync ≈ 850–1300 |
| Signal "data landed" from one CTA/engine to another CTA without a full cluster barrier | remote `mbarrier.arrive.release.cluster.shared::cluster` (one arrive per producer) or `st.async`/`red.async`/bulk copy with `complete_tx` on the consumer's barrier; consumer `mbarrier.try_wait.parity.acquire.cluster` | `.cluster` | producers + one waiting CTA | arrive: one fabric message; wait ≈ 42 per poll — expected ≪ cluster.sync (hypothesis for `07`) |
| Flag-based point-to-point without barriers | `st.release.cluster.shared::cluster` / `ld.acquire.cluster.shared::cluster` (or `.global` with `.gpu` scope) | `.cluster` / `.gpu` | 2 threads | store visibility ≈ 300–650 cycles one-way on DSMEM [LIT 625 on H200]; `ld.acquire` adds a fence after the load |
| Make generic smem writes visible to the async proxy before a bulk/TMA store or put | `fence.proxy.async.shared::cta` (`FENCE.VIEW.ASYNC.S`) | proxy fence | writer threads (then a CTA barrier before the issuer) | ~10s (measure) |
| Make a freshly initialised mbarrier visible to peer CTAs | `fence.mbarrier_init.release.cluster` + cluster barrier | `.cluster`, op-restricted | initialising thread | small |
| Device-wide ordering (e.g., bulk store results for other CTAs/kernels) | `fence.acq_rel.gpu` / `__threadfence()` (`MEMBAR.ALL.GPU; ERRBAR; CGAERRBAR; CCTL.IVALL`), `fence.sc.gpu` (`MEMBAR.SC.GPU`), `st.release.gpu`/`ld.acquire.gpu` | `.gpu` | issuing thread | 100s (L1 invalidate `CCTL.IVALL` included) |
| Cluster-scope fence without a barrier | `fence.acq_rel.cluster` / `__threadfence_cluster()` | `.cluster` | issuing thread | similar to gpu fence (same SASS pattern observed) |
| Grid-wide barrier | `cg::this_grid().sync()` (cooperative launch) | `.gpu` | all blocks (must be co-resident) | ≈ 2200 |
| Wait for own bulk stores | `cp.async.bulk.commit_group; cp.async.bulk.wait_group{.read} N` (`UTMACMDFLUSH; DEPBAR.LE SB0,N`) | thread's own async groups | issuing thread | ~10s + transfer time |
| Wait for own `cp.async` (LDGSTS) | `cp.async.commit_group; cp.async.wait_group N` (`LDGDEPBAR; DEPBAR`) | thread | issuing thread | small |
| Back-off in spin loops | `nanosleep.u32 t` (`NANOSLEEP`, actual ∈ [0, 2t], ≤ 1 ms) | — | thread | t ns |
| Elect one issuing thread | `elect.sync` (`ELECT`) | warp | warp | ~few |

## 2. Memory-model facts that decide placement of fences
- Scopes: `.cta` ⊂ `.cluster` ⊂ `.gpu` ⊂ `.sys`. Two operations synchronise only if each specifies a scope that includes the other thread. For DSMEM data exchanged between CTAs, `.cluster` scope is the minimum: `st.release.cluster` ↔ `ld.acquire.cluster`, `mbarrier.arrive.release.cluster` ↔ `try_wait.acquire.cluster`, `barrier.cluster.arrive.release` ↔ `barrier.cluster.wait.acquire`.
- `barrier.cluster.wait` and mbarrier acquire **exclude asynchronous operations**: a TMA/bulk copy or `st.async` is only guaranteed visible through *its own* completion mechanism (the mbarrier it names, or `wait_group`). A `cluster.sync()` after issuing a TMA put does **not** make the put visible to the destination.
- A bulk op's `complete_tx` is ordered only with that op's accesses — it does not publish the issuing thread's earlier generic writes; pair it with a release arrive if those must be visible too.
- `barrier.cluster.arrive.relaxed` drops ordering (no `MEMBAR`), so a relaxed arrive + wait is a pure rendezvous; use it when data were already published by an mbarrier or by `fence.proxy.async`+completion. Measuring relaxed vs release arrive isolates the fence cost from the rendezvous cost (`07`).
- `membar.cta/gl/sys` = `fence.sc.cta/gpu/sys` on sm_70+. `fence` without `.sem` is `.acq_rel`. `fence.sc` is strictly stronger and slower.
- Cache operators (`.ca/.cg/.cs/.cv`) cannot be combined with `.relaxed/.acquire/.release/.volatile`; `ld.volatile` ≡ relaxed system-scope.
- `__syncthreads()` guarantees visibility of generic smem/global writes within the CTA; it is not a cluster-scope operation.
- Exited threads: `barrier.cluster` waits only for non-exited threads; `bar.sync` with a count must not be deadlocked by exited warps; mbarrier expected counts must be reduced with `arrive_drop` by threads that leave.

## 3. Templates (compilable inline PTX; constraints: `"r"` 32-bit, `"l"` 64-bit, `"h"` 16-bit)

```cpp
// --- block ---
__device__ __forceinline__ void bar_sync0()                  { asm volatile("bar.sync 0;" ::: "memory"); }
__device__ __forceinline__ void bar_sync(int id, int n)      { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id, int n)    { asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(n) : "memory"); }  // producer side of a named barrier
// --- cluster barrier ---
__device__ __forceinline__ void cluster_arrive_release()     { asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory"); }
__device__ __forceinline__ void cluster_arrive_relaxed()     { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
__device__ __forceinline__ void cluster_wait_acquire()       { asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory"); }
__device__ __forceinline__ void cluster_sync()               { cluster_arrive_release(); cluster_wait_acquire(); }
// split-phase use: arrive early, do independent work, wait late — hides the ~1000-cycle arrive latency
// --- fences ---
__device__ __forceinline__ void fence_acq_rel_cluster()      { asm volatile("fence.acq_rel.cluster;" ::: "memory"); }
__device__ __forceinline__ void fence_acq_rel_gpu()          { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ void fence_sc_gpu()               { asm volatile("fence.sc.gpu;" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async_smem()     { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async_dsmem()    { asm volatile("fence.proxy.async.shared::cluster;" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async_all()      { asm volatile("fence.proxy.async;" ::: "memory"); }
__device__ __forceinline__ void fence_mbarrier_init()        { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
// --- flags over DSMEM (addresses from mapa) ---
__device__ __forceinline__ void st_release_cluster(uint32_t a, uint32_t v) { asm volatile("st.release.cluster.shared::cluster.u32 [%0], %1;" :: "r"(a), "r"(v) : "memory"); }
__device__ __forceinline__ uint32_t ld_acquire_cluster(uint32_t a)         { uint32_t v; asm volatile("ld.acquire.cluster.shared::cluster.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory"); return v; }
__device__ __forceinline__ uint32_t ld_relaxed_cluster(uint32_t a)         { uint32_t v; asm volatile("ld.relaxed.cluster.shared::cluster.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory"); return v; }
__device__ __forceinline__ void spin_until(uint32_t a, uint32_t v)         { while (ld_acquire_cluster(a) != v) { } }
// --- atomics over DSMEM ---
__device__ __forceinline__ uint32_t atom_add_acq_rel_cluster(uint32_t a, uint32_t v) { uint32_t o; asm volatile("atom.acq_rel.cluster.shared::cluster.add.u32 %0, [%1], %2;" : "=r"(o) : "r"(a), "r"(v) : "memory"); return o; }
__device__ __forceinline__ void red_release_cluster(uint32_t a, uint32_t v)          { asm volatile("red.release.cluster.shared::cluster.add.u32 [%0], %1;" :: "r"(a), "r"(v) : "memory"); }
// --- misc ---
__device__ __forceinline__ void nanosleep_ns(uint32_t ns)    { asm volatile("nanosleep.u32 %0;" :: "r"(ns)); }
__device__ __forceinline__ bool elect_one()                   { uint32_t p; asm volatile("{ .reg .pred q; elect.sync _|q, 0xffffffff; selp.u32 %0, 1, 0, q; }" : "=r"(p)); return p; }
```
mbarrier wrappers (`mbar_init`, `mbar_arrive_expect_tx`, `mbar_remote_arrive`, `mbar_try_wait_parity_acq_cluster`, `mbar_wait_parity`) are in `04__mbarrier.md` §4; TMA/bulk wrappers in `03__tma_and_bulk_async_copies.md`.

## 4. Protocol recipes used by the DSMEM experiments (so that every experiment uses the same, correct sequence)

1. **Setup**: each CTA writes its smem (generic) → `fence_mbarrier_init()` if barriers were initialised → `cluster_sync()` (publishes smem and guarantees existence).
2. **Generic push (st.shared::cluster) + cluster barrier**: stores → `cluster_sync()` → consumer reads. Cost = stores + 1 cluster sync.
3. **Generic push + flag**: stores → `st_release_cluster(flag@dst, epoch)`; consumer `spin_until(flag, epoch)` → reads. Cost = stores + release store + one-way visibility.
4. **st.async push**: consumer `mbar_arrive_expect_tx(bar, bytes)` in setup; producer `st_async_*` into consumer with consumer's barrier; consumer `try_wait.parity.acquire.cluster`. Cost = stores + barrier observation; no fence needed on the producer side.
5. **TMA put / reduce-put**: producer generic-writes source tile → `fence_proxy_async_smem()` → `bar_sync0()` → elected thread issues `cp.async.bulk.shared::cluster.shared::cta...` with consumer's barrier (consumer pre-armed `expect_tx`). Consumer waits as in 4.
6. **Pull (ld.shared::cluster)**: producer writes its own smem → `cluster_sync()` (or release-flag) → consumer loads remotely. Reads are blocking; use ILP ≥ 4 and 16 B vectors.
7. **Teardown**: `cluster_sync()` before any CTA exits.

Each experiment reports the time of the protocol *and* the time of the pure data movement (from the affine fit), so the synchronisation overhead is the difference.
