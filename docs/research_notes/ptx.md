# PTX ISA reference note — Hopper (sm_90 / sm_90a) clusters, DSMEM, TMA, mbarrier, synchronization

Target audience: H100 microbenchmarking. Everything below is for `sm_90`/`sm_90a` unless marked otherwise.

## 0. Provenance and how to read this note

* **Primary source**: NVIDIA PTX ISA, https://docs.nvidia.com/cuda/parallel-thread-execution/index.html . Section anchors below are of the form `#<anchor>` on that page (anchor names are the ones the document itself uses in cross-links, e.g. `#data-movement-and-conversion-instructions-cp-async-bulk-tensor`).
* **Access caveat**: from this sandbox `docs.nvidia.com` (and every other `*.nvidia.com` / `nvidia.github.io` host, plus the Wayback Machine) was blocked at the network policy level (HTTP 403 on CONNECT; DNS failure in WebFetch). The ISA text quoted here therefore comes from a **verbatim markdown conversion of PTX ISA 9.1 (Jan 2026)** hosted on GitHub (`technillogue/ptx-isa-markdown`, scraped from the official HTML, section numbering and anchors preserved). The live official document is currently titled "PTX ISA 9.4"; nothing in the sm_90 feature set has changed between 9.1 and 9.4 as far as the NVIDIA CCCL generated instruction tables (which track PTX ISA 9.4, see next bullet) show.
* **Secondary official source**: NVIDIA CCCL `docs/libcudacxx/ptx/instructions/generated/*.rst` (auto-generated from the PTX ISA; each line gives the exact instruction form plus `PTX ISA xx, SM_yy` gates). Quoted as `// ... // PTX ISA 80, SM_90` lines.
* **Empirical validation**: every instruction form cited was assembled with `ptxas` 12.9 (`Cuda compilation tools, release 12.9, V12.9.86`, max `.version 8.8`) for both `-arch=sm_90` and `-arch=sm_90a`; the result matrix is in section 10. All inline-asm snippets in this note were compiled as one CUDA C++ file with `clang -x cuda --cuda-gpu-arch=sm_90a` and the resulting PTX (patched to `.version 8.8`) was assembled by `ptxas -arch=sm_90a` without error.
* **Numeric limits** that the PTX ISA does not state (TMA box limits, portable cluster size) come from the CUDA Driver API reference (`cuTensorMapEncodeTiled`, `CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED`), same mirror (CUDA 13.1 docs).
* Text in `> quote` blocks is verbatim from the ISA. Statements marked **[interp]** are my interpretation; **[empirical]** are measured/ptxas-derived, not from the spec.

---

## 1. Memory-model basics (ISA ch. 8, `#memory-consistency-model`)

### 1.1 Proxies (`#proxies`, 8.6)

> A _memory proxy_, or a _proxy_ is an abstract label applied to a method of memory access. When two memory operations use distinct methods of memory access, they are said to be different _proxies_. Memory operations as defined in Operation types use _generic_ method of memory access, i.e. a _generic proxy_. Other operations such as textures and surfaces all use distinct methods of memory access, also distinct from the _generic_ method. A _proxy fence_ is required to synchronize memory operations across different _proxies_. Although virtual aliases use the _generic_ method of memory access, since using distinct virtual addresses behaves as if using different _proxies_, they require a _proxy fence_ to establish memory ordering.

**Async proxy** (`#async-proxy`, 9.7.9.25.2, and 9.7.15.4 for wgmma):

> The `cp{.reduce}.async.bulk` operations are performed in the _asynchronous proxy_ (or _async proxy_). Accessing the same memory location across multiple proxies needs a cross-proxy fence. For the _async proxy_, `fence.proxy.async` should be used to synchronize memory between _generic proxy_ and the _async proxy_. The completion of a `cp{.reduce}.async.bulk` operation is followed by an implicit _generic-async_ proxy fence. So the result of the asynchronous operation is made visible to the generic proxy as soon as its completion is observed. _Async-group_ OR _mbarrier-based_ completion mechanism must be used to wait for the completion of the `cp{.reduce}.async.bulk` instructions.

`wgmma.mma_async` is also in the async proxy (9.7.15.4) with the same implicit fence at completion.

**Which proxy each instruction uses** (collected from the per-instruction text):

| Instruction | Data accesses | mbarrier operand | tensor-map operand |
|---|---|---|---|
| `ld`, `st`, `atom`, `red` (incl. `.shared::cluster`), `st.async`, `red.async`, `cp.async` (sm_80 style) | generic proxy | generic (st.async/red.async: "accesses its mbarrier operand using generic-proxy") | – |
| `cp.async.bulk`, `cp.reduce.async.bulk`, `cp.async.bulk.prefetch` | **async proxy** | generic ("This instruction accesses its mbarrier operand using generic-proxy") | – |
| `cp.async.bulk.tensor`, `cp.reduce.async.bulk.tensor`, `cp.async.bulk.prefetch.tensor` | **async proxy** | generic | **tensormap proxy** ("The tensorMap is accessed in tensormap proxy") |
| `tensormap.replace` | "treated as a weak memory operation, on the entire 1024-bit opaque tensor-map object" (generic) | – | – |
| `wgmma.mma_async` | async proxy | – | – |

### 1.2 Official rules for when `fence.proxy.async` is needed **[ISA text + interp]**

1. Generic-proxy writes to shared memory that a subsequent **bulk store / bulk reduce / TMA store** (async proxy *reads* smem) must see: `fence.proxy.async.shared::cta` after the `st.shared` and before issuing `cp.async.bulk.global.shared::cta...` (if different threads wrote vs. issue, also a CTA barrier between them). This is the ISA's "Accessing the same memory location across multiple proxies needs a cross-proxy fence". CUTLASS's `tma_store_fence()` is exactly `fence.proxy.async.shared::cta`.
2. After a **TMA / bulk load completes** and that completion has been *observed* (mbarrier `try_wait`/`test_wait` returned true, or `cp.async.bulk.wait_group` returned): **no `fence.proxy.async` is required** before generic-proxy `ld.shared` of the data — "The completion of a cp{.reduce}.async.bulk operation is followed by an implicit generic-async proxy fence."
3. Reusing a smem buffer: generic-proxy accesses (reads *or* writes) to a buffer followed by an async-proxy **write** into the same buffer (next TMA load into a just-consumed stage) are cross-proxy accesses to the same location; the ISA rule literally requires a proxy fence **[interp: a WAR hazard across proxies; CUTLASS pipelines rely on the mbarrier release/acquire chain plus the producer's `fence.proxy.async` only for stores. Treat as an open question for the benchmark, section 11]**.
4. `fence.proxy.async` with no state space orders all state spaces; `.shared::cta`, `.shared::cluster`, `.global` restrict it ("The memory ordering is limited only to operations performed on objects in the state space specified").
5. `barrier.cluster.wait` and `mbarrier` acquire do **not** cover async operations: "memory accesses (except asynchronous operations) requested ... prior to the preceding barrier.cluster.arrive ... are complete and visible" and mbarrier item 1 below says "(except async operations)". Async ops are only made visible by their own completion mechanism.
6. Tensor-map modification by generic-proxy writes (`tensormap.replace`, `st.global`) → consumption by `cp.async.bulk.tensor` needs the **uni-directional tensormap proxy fence** (section 5.9).
7. mbarrier init → remote arrive: `fence.mbarrier_init.release.cluster` (section 4.1).

### 1.3 Asynchronous operations in program order (`#program-order-async-operations`, 8.9.1.1) — verbatim

> Some PTX instructions (all variants of `cp.async`, `cp.async.bulk`, `cp.reduce.async.bulk`, `wgmma.mma_async`) perform operations that are asynchronous to the thread that executed the instruction. These asynchronous operations are ordered after prior instructions in the same thread (except in the case of `wgmma.mma_async`), but they are not part of the program order for that thread. Instead, they provide weaker ordering guarantees as documented in the instruction description. [...] The implicit mbarrier complete-tx operation that is part of all variants of `cp.async.bulk` and `cp.reduce.async.bulk` instructions is ordered only with respect to the memory operations performed by the same asynchronous instruction, and in particular it does not transitively establish ordering with respect to prior instructions from the issuing thread.

**Consequence [interp]**: a bulk copy's complete-tx does *not* publish the issuing thread's earlier generic writes to the waiter. If the waiter must also see those, the issuer needs its own release (e.g. `mbarrier.arrive.expect_tx` is `.release` by default) that the waiter's `.acquire` wait synchronizes with.

### 1.4 Scopes (`#scope`, 8.5) — verbatim table

| Scope | Description |
|---|---|
| `.cta` | The set of all threads executing in the same CTA as the current thread. |
| `.cluster` | The set of all threads executing in the same cluster as the current thread. |
| `.gpu` | The set of all threads in the current program executing on the same compute device as the current thread. This also includes other kernel grids invoked by the host program on the same compute device. |
| `.sys` | The set of all threads in the current program, including all kernel grids invoked by the host program on all compute devices, and all threads constituting the host program itself. |

> Note that the warp is not a _scope_; the CTA is the smallest collection of threads that qualifies as a _scope_ in the memory consistency model.

State spaces (8.3): "the synchronizing effect of the PTX instruction `ld.relaxed.shared.sys` is identical to that of `ld.relaxed.shared.cluster`, since no thread outside the same cluster can execute an operation that accesses the same memory location."

### 1.5 Release / acquire patterns (`#release-and-acquire-patterns`, 8.8)

Release pattern on M: (1) a release op on M (`st.release`, `atom.release`, `mbarrier.arrive.release`); (2) release/acq_rel op on M followed by a strong write on M; (3) release/acq_rel **fence** followed by a strong write on M (`fence.release; st.relaxed [M]`). Acquire pattern on M: (1) acquire op on M (`ld.acquire`, `atom.acquire`, `mbarrier.test_wait.acquire`); (2) strong read on M then acquire op on M; (3) strong read on M then acquire fence. "Note that while atomic reductions conceptually perform a strong read as part of its read-modify-write sequence, this strong read does not form an acquire pattern. E.g.: `red.add [M], 1; fence.acquire;` is not an acquire pattern."

Synchronizes-with (8.9.4): fence.sc ↔ fence.sc in Fence-SC order; `bar{.cta}` ops on same barrier; "A `barrier.cluster.arrive` operation synchronizes with a `barrier.cluster.wait` operation"; release pattern X ↔ acquire pattern Y when a write in X precedes a read in Y in observation order and the first op of X / last op of Y are morally strong.

### 1.6 `membar` / `fence` (`#parallel-synchronization-and-communication-instructions-membar`, 9.7.13.4)

Syntax (verbatim):
```
fence{.sem}.scope;
fence.acquire.sync_restrict::shared::cluster.cluster;
fence.release.sync_restrict::shared::cta.cluster;
fence.op_restrict.release.cluster;                       // .op_restrict = { .mbarrier_init }
fence.proxy.proxykind;                                   // bi-directional
fence.proxy.to_proxykind::from_proxykind.release.scope;  // uni-directional
fence.proxy.to_proxykind::from_proxykind.acquire.scope  [addr], size;
fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster;
fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;
membar.level;  membar.proxy.proxykind;
.sem = { .sc, .acq_rel, .acquire, .release };  .scope = { .cta, .cluster, .gpu, .sys };
.level = { .cta, .gl, .sys };
.proxykind = { .alias, .async, .async.global, .async.shared::{cta, cluster} };
.to_proxykind::from_proxykind = {.tensormap::generic};
```
Key statements:
* "If the optional `.sem` qualifier is absent, `.acq_rel` is assumed by default." `fence.sc` "is a slower fence that can restore sequential consistency"; "Instances of fence.sc with sufficient scope always synchronize by forming a total order per scope, determined at runtime."
* "On sm_70 and higher membar is a synonym for fence.sc, and the membar levels cta, gl and sys are synonymous with the fence scopes cta, gpu and sys respectively." `membar.proxy` ≡ `fence.proxy`.
* `.op_restrict = .mbarrier_init`: "the synchronizing effect of the fence only applies to the prior `mbarrier.init` operations executed by the same thread on mbarrier objects in `.shared::cta` state space."
* `.sync_restrict::shared::cta` requires `.release`; `::shared::cluster` requires `.acquire`; both require `.scope = .cluster` (PTX 8.6, sm_90).
* `fence.proxy.alias`: "memory accesses performed using virtually aliased addresses to the same memory location."
* `fence.proxy.async[.space]`: bi-directional generic↔async ordering, optionally restricted to a state space.
* `fence.proxy.tensormap::generic.acquire.scope [addr], 128`: "The only supported value for the size operand is 128 ... the address specified by the operand addr must fall within the .global state space."
* Versions: `fence` PTX 6.0/sm_70; `fence.proxy` 7.5/sm_70; `.cluster` scope 7.8/sm_90; `.op_restrict` 8.0/sm_90; `fence.proxy.async` 8.0/sm_90; `tensormap::generic` 8.3/sm_90; `.acquire/.release` on `fence` and `.sync_restrict` 8.6/sm_90.

ISA examples worth copying (verbatim):
```
// Acquire remote barrier state via async proxy.
barrier.cluster.wait.acquire;
fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster;
// Release local barrier state via generic proxy.
mbarrier.init [bar];
fence.mbarrier_init.release.cluster;
barrier.cluster.arrive.relaxed;
// Acquire local shared memory via generic proxy.
mbarrier.try_wait.relaxed.cluster.shared::cta.b64 complete, [addr], parity;
fence.acquire.sync_restrict::shared::cluster.cluster;
// Release local shared memory via generic proxy.
fence.release.sync_restrict::shared::cta.cluster;
mbarrier.arrive.relaxed.cluster.shared::cluster.b64 state, [bar];
```
CCCL forms: `fence.sem.scope; // PTX ISA 60, SM_70` (`.sc/.acq_rel` × `.cta/.gpu/.sys`), `// PTX ISA 78, SM_90` for `.cluster`, `fence.acquire/.release` `// PTX ISA 86, SM_90`; `fence.mbarrier_init.release.cluster; // PTX ISA 80, SM_90`; `fence.proxy.alias; // PTX ISA 75, SM_70`; `fence.proxy.async{.space}; // PTX ISA 80, SM_90`; `fence.proxy.async::generic.sem.sync_restrict::space.scope; // PTX ISA 86, SM_90`; `fence.proxy.tensormap::generic.release.scope; // PTX ISA 83, SM_90`.

```cpp
asm volatile("fence.proxy.async.shared::cta;" ::: "memory");          // before TMA/bulk *store* of generic-written smem
asm volatile("fence.proxy.async.shared::cluster;" ::: "memory");
asm volatile("fence.proxy.async.global;" ::: "memory");
asm volatile("fence.proxy.async;" ::: "memory");                       // all state spaces
asm volatile("fence.proxy.alias;" ::: "memory");
asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");     // after mbarrier.init, before cluster sync
asm volatile("fence.acq_rel.cluster;" ::: "memory");
asm volatile("fence.sc.cluster;" ::: "memory");
asm volatile("fence.release.sync_restrict::shared::cta.cluster;" ::: "memory");       // PTX 8.6+
asm volatile("fence.acquire.sync_restrict::shared::cluster.cluster;" ::: "memory");   // PTX 8.6+
asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"(tmap_gptr) : "memory");
```

### 1.7 `ld`/`st` with `.relaxed`/`.acquire`/`.release` and `.cluster` scope (9.7.9.8 / 9.7.9.11)

Verbatim syntax:
```
ld{.weak}{.ss}{.cop}{.level::cache_hint}{.level::prefetch_size}{.vec}.type  d, [a]{.unified}{, cache-policy};
ld{.weak}{.ss}{.level1::eviction_priority}{.level2::eviction_priority}{.level::cache_hint}{.level::prefetch_size}{.vec}.type  d, [a]{.unified}{, cache-policy};
ld.volatile{.ss}{.level::prefetch_size}{.vec}.type  d, [a];
ld.relaxed.scope{.ss}{.level1::eviction_priority}{.level2::eviction_priority}{.level::cache_hint}{.level::prefetch_size}{.vec}.type  d, [a]{, cache-policy};
ld.acquire.scope{.ss}{.level1::eviction_priority}{.level2::eviction_priority}{.level::cache_hint}{.level::prefetch_size}{.vec}.type  d, [a]{, cache-policy};
ld.mmio.relaxed.sys{.global}.type  d, [a];
.ss = { .const, .global, .local, .param{::entry, ::func}, .shared{::cta, ::cluster} };
.cop = { .ca, .cg, .cs, .lu, .cv };
.level1::eviction_priority = { .L1::evict_normal, .L1::evict_unchanged, .L1::evict_first, .L1::evict_last, .L1::no_allocate };
.level2::eviction_priority = { .L2::evict_normal, .L2::evict_first, .L2::evict_last };   // sm_100+ (needs .v8.b32/.v4.b64)
.level::cache_hint = { .L2::cache_hint };  .level::prefetch_size = { .L2::64B, .L2::128B, .L2::256B }
.scope = { .cta, .cluster, .gpu, .sys };  .vec = { .v2, .v4, .v8 };
.type = { .b8, .b16, .b32, .b64, .b128, .u8, .u16, .u32, .u64, .s8, .s16, .s32, .s64, .f32, .f64 };

st{.weak}{.ss}{.cop}{.level::cache_hint}{.vec}.type   [a], b{, cache-policy};
st{.weak}{.ss}{.level1::eviction_priority}{.level2::eviction_priority}{.level::cache_hint}{.vec}.type [a], b{, cache-policy};
st.volatile{.ss}{.vec}.type                           [a], b;
st.relaxed.scope{.ss}{.level1::eviction_priority}{.level2::eviction_priority}{.level::cache_hint}{.vec}.type [a], b{, cache-policy};
st.release.scope{.ss}{.level1::eviction_priority}{.level2::eviction_priority}{.level::cache_hint}{.vec}.type [a], b{, cache-policy};
st.mmio.relaxed.sys{.global}.type         [a], b;
.ss = { .global, .local, .param{::func}, .shared{::cta, ::cluster} };  .cop = { .wb, .cg, .cs, .wt };
```
Rules: "`.relaxed` and `.acquire`: May be used with `.global`, `.shared` spaces, or with generic addressing where the address points to `.global` or `.shared` space. Cache operations are not allowed." "The `.weak`, `.volatile`, `.relaxed` and `.acquire` qualifiers are mutually exclusive. When none of these is specified, the `.weak` qualifier is assumed by default." Same for `st` with `.release`. `.cluster` scope and `::cta/::cluster` sub-qualifiers: PTX 7.8, sm_90. Volatile (8.4.2): "equivalent to a relaxed memory operation with system-scope" plus implementation constraints; "prefer ... `ld.relaxed.sys` or `st.relaxed.sys` for Inter-Thread Synchronization".

ISA examples: `ld.global.relaxed.cluster.u32 %r2, [gbl];`, `ld.shared::cluster.u32 %r3, [sh + 8];`, `st.global.relaxed.cluster.u32 [gbl], %r2;`, `st.shared::cluster.u32 [sh + 8], %r1;`.

```cpp
__device__ __forceinline__ uint32_t ld_relaxed_cluster_dsmem(uint32_t a) { uint32_t v; asm volatile("ld.relaxed.cluster.shared::cluster.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory"); return v; }
__device__ __forceinline__ uint32_t ld_acquire_cluster_dsmem(uint32_t a) { uint32_t v; asm volatile("ld.acquire.cluster.shared::cluster.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory"); return v; }
__device__ __forceinline__ void st_relaxed_cluster_dsmem(uint32_t a, uint32_t v) { asm volatile("st.relaxed.cluster.shared::cluster.u32 [%0], %1;" :: "r"(a), "r"(v) : "memory"); }
__device__ __forceinline__ void st_release_cluster_dsmem(uint32_t a, uint32_t v) { asm volatile("st.release.cluster.shared::cluster.u32 [%0], %1;" :: "r"(a), "r"(v) : "memory"); }
__device__ __forceinline__ uint32_t ld_relaxed_cluster_global(const uint32_t* p) { uint32_t v; asm volatile("ld.relaxed.cluster.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void st_release_cluster_global(uint32_t* p, uint32_t v) { asm volatile("st.release.cluster.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
```

---

## 2. Clusters

### 2.1 Programming-model statements (2.2.2, 2.3, 5.1.7)

> Cluster is a group of CTAs that run concurrently or in parallel and can synchronize and communicate with each other via shared memory. The executing CTA has to make sure that the shared memory of the peer CTA exists before communicating with it via shared memory and the peer CTA hasn't exited before completing the shared memory operation.

> Specifying cluster level during launch time is optional. If the user specifies the cluster dimensions at launch time then it will be treated as explicit cluster launch, otherwise it will be treated as implicit cluster launch with default dimension 1x1x1. PTX provides read-only special register `%is_explicit_cluster` to differentiate between explicit and implicit cluster launch.

Shared state space (5.1.7): "The shared (`.shared`) state space is a memory that is owned by an executing CTA and is accessible to the threads of all the CTAs within a cluster. [...] The addresses in the `.shared::cta` window also fall within the `.shared::cluster` window. If no sub-qualifier is specified with the `.shared` state space, then it defaults to `::cta`. [...] Instruction `mapa` gives the `.shared::cluster` address of the corresponding variable in another CTA in the cluster."

**[interp]** The two sentences in 2.2.2 are the ISA's only statement about DSMEM lifetime: you must `barrier.cluster` (or equivalent) after launch before touching a peer's smem and again before exiting; the peer must not exit while remote ops (including in-flight `st.async`/bulk copies into it) are pending.

### 2.2 `barrier.cluster` (`#parallel-synchronization-and-communication-instructions-barrier-cluster`, 9.7.13.3)

```
barrier.cluster.arrive{.sem}{.aligned};
barrier.cluster.wait{.acquire}{.aligned};
.sem = {.release, .relaxed}
```
Verbatim semantics:
* "`barrier.cluster.arrive` instruction marks warps' arrival at barrier without causing executing thread to wait for threads of other participating warps. `barrier.cluster.wait` instruction causes the executing thread to wait for all non-exited threads of the cluster to perform `barrier.cluster.arrive`. In addition, `barrier.cluster` instructions cause the executing thread to wait for all non-exited threads from its warp."
* "When all non-exited threads in the cluster have executed `barrier.cluster.arrive`, the barrier completes and is automatically reinitialized. After using `barrier.cluster.wait` to detect completion of the barrier, a thread may immediately arrive at the barrier once again. Each thread must arrive at the barrier only once before the barrier completes."
* "The `barrier.cluster.wait` instruction guarantees that when it completes the execution, memory accesses (except asynchronous operations) requested, in program order, prior to the preceding `barrier.cluster.arrive` by all threads in the cluster are complete and visible to the executing thread."
* "There is no memory ordering and visibility guarantee for memory accesses requested by the executing thread, in program order, after `barrier.cluster.arrive` and prior to `barrier.cluster.wait`."
* "The optional `.relaxed` qualifier on `barrier.cluster.arrive` specifies that there are no memory ordering and visibility guarantees provided for the memory accesses performed prior to `barrier.cluster.arrive`." Defaults: arrive → `.release`, wait → `.acquire`.
* `.aligned`: "all threads in the warp must execute the same `barrier.cluster` instruction."
* PTX 7.8 / sm_90; `.acquire/.relaxed/.release` PTX 8.0. CCCL: `barrier.cluster.arrive.sem; // PTX ISA 80, SM_90`, `barrier.cluster.wait.sem; // PTX ISA 80, SM_90`.

```cpp
asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");   // pair with explicit fence.acq_rel.cluster if ordering needed
asm volatile("barrier.cluster.wait.acquire.aligned;"   ::: "memory");
```

### 2.3 Address mapping: `mapa`, `getctarank`, `isspacep`, `cvta`

`mapa` (9.7.9.23, `#data-movement-and-conversion-instructions-mapa`, PTX 7.8, sm_90):
```
mapa{.space}.type          d, a, b;
mapa.shared::cluster.type  d, a, b;        // Maps shared memory address in register a into CTA b.
mapa.shared::cluster.type  d, sh, b;       // Maps shared memory variable into CTA b.
mapa.shared::cluster.type  d, sh + imm, b;
mapa.type                  d, a, b;        // Maps generic address in register a into CTA b.
.space = { .shared::cluster }   .type = { .u32, .u64 }
```
"When space is `.shared::cluster`, source `a` is either a shared memory variable or a register containing a valid shared memory address and register `d` contains a shared memory address. When the optional qualifier `.space` is not specified, both `a` and `d` are registers containing generic addresses pointing to shared memory. `b` is a 32-bit integer operand representing the rank of the target CTA."

`getctarank` (9.7.9.24, PTX 7.8, sm_90): `getctarank{.space}.type d, a;` `.space = { .shared::cluster }`, `.type = { .u32, .u64 }`; "Destination `d` is always a 32-bit register which holds the rank of the CTA."

`isspacep` (9.7.9.19): `isspacep.space p, a;` `.space = { const, .global, .local, .shared{::cta, ::cluster}, .param{::entry} }`. "`ispacep.shared::cluster` will return 1 for every shared memory address that is accessible to the threads in the cluster, whereas `ispacep.shared::cta` will return 1 only if the address is of a variable declared in the executing CTA." `::cluster` requires sm_90.

`cvta` (9.7.9.20): `cvta.space.size p, a;` / `cvta.to.space.size p, a;` with `.space = { .const, .global, .local, .shared{::cta, ::cluster}, .param{::entry} }`, `.size = { .u32, .u64 }`. "For `cvta` with `.shared` state space, the address must belong to the space specified by `::cta` or `::cluster` sub-qualifier, otherwise the behavior is undefined."

```cpp
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ uint32_t mapa_smem(uint32_t local_smem_addr, uint32_t cta_rank) {
  uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(local_smem_addr), "r"(cta_rank)); return r; }
__device__ __forceinline__ void* mapa_generic(void* p, uint32_t cta_rank) {
  void* r; asm volatile("mapa.u64 %0, %1, %2;" : "=l"(r) : "l"(p), "r"(cta_rank)); return r; }
__device__ __forceinline__ uint32_t getctarank_smem(uint32_t dsmem_addr) {
  uint32_t r; asm volatile("getctarank.shared::cluster.u32 %0, %1;" : "=r"(r) : "r"(dsmem_addr)); return r; }
__device__ __forceinline__ bool isspacep_shared_cluster(const void* p) {
  uint32_t r; asm volatile("{ .reg .pred p; isspacep.shared::cluster p, %1; selp.u32 %0, 1, 0, p; }" : "=r"(r) : "l"(p)); return r; }
```

### 2.4 Special registers (ch. 10)

| Register | Type | Semantics (verbatim excerpts) | PTX/SM |
|---|---|---|---|
| `%clusterid` (`.x/.y/.z`, `.v4.u32`) | u32 | "cluster identifier in a grid in each dimension. Each cluster in a grid has a unique identifier." 4th element always 0. `0 <= %clusterid.x < %nclusterid.x` | 7.8 / sm_90 |
| `%nclusterid` | u32 | "number of clusters in each grid dimension"; max values → CUDA Programming Guide | 7.8 / sm_90 |
| `%cluster_ctaid` | u32 | "CTA identifier in a cluster in each dimension"; `0 <= %cluster_ctaid.x < %cluster_nctaid.x` | 7.8 / sm_90 |
| `%cluster_nctaid` | u32 | "number of CTAs in a cluster in each dimension" | 7.8 / sm_90 |
| `%cluster_ctarank` | u32 | "CTA rank within a cluster across all dimensions"; `0 <= %cluster_ctarank < %cluster_nctarank` | 7.8 / sm_90 |
| `%cluster_nctarank` | u32 | "number of CTAs within a cluster across all dimensions" | 7.8 / sm_90 |
| `%is_explicit_cluster` | pred | "whether the cluster launch is explicitly specified by user" | 7.8 / sm_90 |
| `%smid` | u32 | "returns the processor (SM) identifier on which a particular thread is executing. The SM identifier ranges from 0 to %nsmid-1. **The SM identifier numbering is not guaranteed to be contiguous.**" Note: "`%smid` returns the location of a thread at the moment when read, but its value may change during execution, e.g. due to rescheduling of threads following preemption. `%smid` is intended mainly to enable profiling and diagnostic code" | 1.3 / all |
| `%nsmid` | u32 | "returns the maximum number of SM identifiers. The SM identifier numbering is not guaranteed to be contiguous, so `%nsmid` may be larger than the physical number of SMs in the device." | 2.0 / sm_20 |
| `%clock`, `%clock_hi` | u32 | "unsigned 32-bit read-only cycle counters that wrap silently" | 1.0 / 5.0 |
| `%clock64` | u64 | "unsigned 64-bit read-only cycle counter that wraps silently"; low 32 = `%clock`, high 32 = `%clock_hi` | 2.0 / sm_20 |
| `%globaltimer`, `%globaltimer_lo/_hi` | u64 / u32 | "A predefined, 64-bit global nanosecond timer." "Special registers intended for use by NVIDIA tools. The behavior is target-specific and may change or be removed in future GPUs. When JIT-compiled to other targets, the value of these registers is unspecified." | 3.1 / sm_30 |

```cpp
__device__ __forceinline__ uint32_t cluster_ctarank()  { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;"  : "=r"(r)); return r; }
__device__ __forceinline__ uint32_t cluster_nctarank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ uint32_t cluster_ctaid_x()  { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctaid.x;"  : "=r"(r)); return r; }
__device__ __forceinline__ uint32_t clusterid_x()      { uint32_t r; asm volatile("mov.u32 %0, %%clusterid.x;"      : "=r"(r)); return r; }
__device__ __forceinline__ uint32_t nclusterid_x()     { uint32_t r; asm volatile("mov.u32 %0, %%nclusterid.x;"     : "=r"(r)); return r; }
__device__ __forceinline__ bool is_explicit_cluster()  { uint32_t r; asm volatile("{ .reg .pred p; mov.pred p, %%is_explicit_cluster; selp.u32 %0, 1, 0, p; }" : "=r"(r)); return r; }
__device__ __forceinline__ uint32_t smid()  { uint32_t r; asm volatile("mov.u32 %0, %%smid;"  : "=r"(r)); return r; }
__device__ __forceinline__ uint32_t nsmid() { uint32_t r; asm volatile("mov.u32 %0, %%nsmid;" : "=r"(r)); return r; }
__device__ __forceinline__ uint64_t globaltimer() { uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint64_t clock64_()    { uint64_t t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t)); return t; }
```

### 2.5 Cluster directives (11.7, 11.8.1)

| Directive | Syntax | Semantics (verbatim excerpts) |
|---|---|---|
| `.reqnctapercluster` | `.reqnctapercluster nx{, ny{, nz}}` on `.entry` | "runtime will use the specified values for configuring the launch if the same are not specified at launch time. If cluster dimension is explicitly specified at launch time, it should be equal to the values specified in this directive." |
| `.explicitcluster` | `.explicitcluster` | "must be launched with cluster dimension explicitly specified (either at launch time or via .reqnctapercluster), otherwise program will fail with runtime error or kernel launch failure." |
| `.maxclusterrank` | `.maxclusterrank n` | "Product of the number of CTAs in each cluster dimension specified in any invocation of the kernel is required to be less or equal to that specified in this directive. [...] The .maxclusterrank directive cannot be used in conjunction with the .reqnctapercluster directive." |
| `.blocksareclusters` | `.blocksareclusters` (PTX 9.0) | launch config counts clusters instead of blocks; requires `.reqntid` and `.reqnctapercluster`. |

All PTX 7.8 / sm_90. ptxas 12.9 accepts `.reqnctapercluster 16` and `.maxclusterrank 16` for sm_90(a) **[empirical]**. CUDA C++ equivalents: `__cluster_dims__(x,y,z)` → `.reqnctapercluster`; `cudaFuncAttributeNonPortableClusterSizeAllowed` for >8.

---

## 3. Generic-proxy DSMEM access

### 3.1 `ld.shared::cluster` / `st.shared::cluster`
Covered by section 1.7: any `ld`/`st` form with `.ss = .shared::cluster` (weak, `.volatile`, `.relaxed.scope`, `.acquire.scope`/`.release.scope`), any `.vec`/`.type` (including `.v4.b32`, `.b128`). Cache operators (`.ca` ...) and `L1::`/`L2::` hints are global-only. Address operand is a 32-bit (`.u32`) or 64-bit shared::cluster-window address obtained from `mapa`, or a generic address (no `.ss`) that lies in the `.shared::cluster` window.

```cpp
__device__ __forceinline__ uint32_t ld_dsmem_u32(uint32_t a) { uint32_t v; asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory"); return v; }
__device__ __forceinline__ void     st_dsmem_u32(uint32_t a, uint32_t v) { asm volatile("st.shared::cluster.u32 [%0], %1;" :: "r"(a), "r"(v) : "memory"); }
__device__ __forceinline__ uint4 ld_dsmem_v4(uint32_t a) { uint4 v; asm volatile("ld.shared::cluster.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "r"(a) : "memory"); return v; }
```

### 3.2 `atom` / `red` on `.shared::cluster` (9.7.13.5 / 9.7.13.6)

```
atom{.sem}{.scope}{.space}.op{.level::cache_hint}.type d, [a], b{, cache-policy};
atom{.sem}{.scope}{.space}.op.type d, [a], b, c;                       // .cas
atom{.sem}{.scope}{.space}.cas.b16 d, [a], b, c;   atom{...}.cas.b128 d, [a], b, c;  atom{...}.exch.b128 ...
atom{.sem}{.scope}{.space}.add.noftz{.level::cache_hint}.f16 / .f16x2 / .bf16 / .bf16x2  d, [a], b;
.space = { .global, .shared{::cta, ::cluster} };  .sem = { .relaxed, .acquire, .release, .acq_rel };
.scope = { .cta, .cluster, .gpu, .sys };
.op = { .and, .or, .xor, .cas, .exch, .add, .inc, .dec, .min, .max };
.type = { .b32, .b64, .u32, .u64, .s32, .s64, .f32, .f64 };

red{.sem}{.scope}{.space}.op{.level::cache_hint}.type [a], b{, cache-policy};   // .sem = {.relaxed, .release}; same .space/.scope/.op/.type
red{.sem}{.scope}{.space}.add.noftz{...}.f16 / .f16x2 / .bf16 / .bf16x2 [a], b;
```
* Vector forms (`.v2/.v4/.v8` f16/bf16/f32) are **`.global` only**.
* Defaults: `.sem` → `.relaxed`; `.scope` → `.gpu`. `::cluster` sub-qualifier and `.cluster` scope: PTX 7.8, sm_90. `.level::cache_hint` global only.
* "Two atomic operations (`atom` or `red`) are performed atomically with respect to each other only if each operation specifies a scope that includes the other."
* "`atom.add.f32` on shared memory supports subnormal inputs and results and doesn't flush them to zero" (global flushes).
* ISA examples: `red.shared::cluster.max.u32 [x+4],0;`, `atom.add.shared::cluster.noftz.f16 hd, [ha], hb;`, `atom.global.cluster.relaxed.add.u32 d, [a], 1;`.
* ptxas-verified on sm_90a **[empirical]**: `atom.shared::cluster.{add.u32, add.u64, add.f32, add.f64, cas.b32, exch.b32, add.noftz.f16}`, `atom.acq_rel.cluster.shared::cluster.add.u32`, `red.shared::cluster.{add.u32, max.s32, add.f32}`, `red.release.cluster.shared::cluster.add.u32`.

```cpp
__device__ __forceinline__ uint32_t atom_add_dsmem(uint32_t a, uint32_t v) { uint32_t o; asm volatile("atom.shared::cluster.add.u32 %0, [%1], %2;" : "=r"(o) : "r"(a), "r"(v) : "memory"); return o; }
__device__ __forceinline__ uint32_t atom_add_dsmem_acq_rel(uint32_t a, uint32_t v) { uint32_t o; asm volatile("atom.acq_rel.cluster.shared::cluster.add.u32 %0, [%1], %2;" : "=r"(o) : "r"(a), "r"(v) : "memory"); return o; }
__device__ __forceinline__ void red_add_dsmem(uint32_t a, uint32_t v) { asm volatile("red.shared::cluster.add.u32 [%0], %1;" :: "r"(a), "r"(v) : "memory"); }
```

### 3.3 `st.async` (`#data-movement-and-conversion-instructions-st-async`, 9.7.9.12) — PTX 8.1, sm_90

```
st.async{.weak}{.ss}.completion_mechanism{.vec}.type [a], b, [mbar];
st.async{.scope}{.ss}.completion_mechanism{.vec}.type [a], b, [mbar];
.scope = { .cluster };  .ss = { .shared::cluster };
.type = { .b32, .b64, .u32, .u64, .s32, .s64, .f32, .f64 };  .vec = { .v2, .v4 };
.completion_mechanism = { .mbarrier::complete_tx::bytes };
// (sm_100+ only) st.async{.mmio}.release.scope{.global}.type [a], b;
```
Verbatim conditions (`.sem` = `.weak`, the only sm_90 form):
* "This is a weak store to shared memory, which signals its completion through an mbarrier object. The store operation is treated as a weak memory operation. The complete-tx operation on the mbarrier has `.release` semantics at `.cluster` scope."
* "Requires: The shared memory addresses of destination operand `a` and the _mbarrier object_ `mbar` belong to the same CTA within the same cluster as the executing thread. The number of CTAs within the cluster is strictly greater than one; `%cluster_nctarank > 1` is true. Otherwise, the behavior is undefined." (Yes — the ISA text literally says "same CTA ... as the executing thread"; the `red.async` text says "different from the CTA of the executing thread". The st.async wording is widely read as "a and mbar in the same CTA as each other" — treat the precise st.async-to-own-CTA case as **open question**, section 11; CUTLASS only uses it for remote CTAs.)
* "`.completion_mechanism` must be specified and must be `.mbarrier::complete_tx::bytes`." complete-tx `completeCount` = bytes stored. mbarrier operand accessed via generic proxy.
* `.v4` only with 32-bit types; `.v2.b64` accepted, `.v4.b64` rejected ("Vector type too large, exceeds 128 bit limit") **[empirical]**. CCCL: `st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.{type|.v2.type|.v4.b32} [addr], value, [remote_bar]; // PTX ISA 81, SM_90`.

```cpp
__device__ __forceinline__ void st_async_b32(uint32_t remote_addr, uint32_t v, uint32_t remote_mbar) {
  asm volatile("st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" :: "r"(remote_addr), "r"(v), "r"(remote_mbar) : "memory"); }
__device__ __forceinline__ void st_async_v4_b32(uint32_t remote_addr, uint4 v, uint32_t remote_mbar) {
  asm volatile("st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1,%2,%3,%4}, [%5];" :: "r"(remote_addr), "r"(v.x),"r"(v.y),"r"(v.z),"r"(v.w), "r"(remote_mbar) : "memory"); }
__device__ __forceinline__ void st_async_b64(uint32_t remote_addr, uint64_t v, uint32_t remote_mbar) {
  asm volatile("st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.b64 [%0], %1, [%2];" :: "r"(remote_addr), "l"(v), "r"(remote_mbar) : "memory"); }
```

### 3.4 `red.async` (`#parallel-synchronization-and-communication-instructions-red-async`, 9.7.13.7) — PTX 8.1, sm_90

```
red.async.sem.scope{.ss}.completion_mechanism.op.type [a], b, [mbar];
.sem = { .relaxed };  .scope = { .cluster };  .ss = { .shared::cluster };
.completion_mechanism = { .mbarrier::complete_tx::bytes };
.op/.type:  .inc,.dec → .u32 ;  .min,.max → .u32,.s32 ;  .and,.or,.xor → .b32 ;  .add → .u32,.s32,.u64
// (sm_100+ only) red.async{.mmio}.release.scope{.global}.add.type [a], b;
```
* "The reduce operation is a relaxed memory operation. The complete-tx operation on the mbarrier has `.release` semantics at `.cluster` scope."
* "The shared-memory addresses of the destination operand `a` and the mbarrier operand `mbar` must meet all of the following conditions: They belong to the same CTA. **The CTA to which they belong is different from the CTA of the executing thread, but must be within the same cluster.**"
* "If `.completion_mechanism` is not specified, it defaults to `.mbarrier::complete_tx::bytes`." `.inc/.dec` result in `[0..b]`. No floating-point ops (ptxas: `add.f32` rejected) **[empirical]**.

```cpp
__device__ __forceinline__ void red_async_add_u32(uint32_t remote_addr, uint32_t v, uint32_t remote_mbar) {
  asm volatile("red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u32 [%0], %1, [%2];" :: "r"(remote_addr), "r"(v), "r"(remote_mbar) : "memory"); }
__device__ __forceinline__ void red_async_add_u64(uint32_t remote_addr, uint64_t v, uint32_t remote_mbar) {
  asm volatile("red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u64 [%0], %1, [%2];" :: "r"(remote_addr), "l"(v), "r"(remote_mbar) : "memory"); }
```

---

## 4. mbarrier (`#parallel-synchronization-and-communication-instructions-mbarrier`, 9.7.13.15)

### 4.1 Object, counts, phases (verbatim)

* "An mbarrier object is an opaque object with the following type and alignment requirements: `.b64`, 8 bytes, `.shared`." Must be in `.shared::cta` of the owning CTA (all ops: "If the address specified by addr does not fall within the address window of .shared::cta state space then the behavior is undefined", except arrive/expect_tx/complete_tx which also accept `.shared::cluster`).
* Contents: current phase; pending arrival count; expected arrival count for the next phase; tx-count.
* Valid ranges: **Expected arrival count 1 .. 2^20−1; Pending arrival count 0 .. 2^20−1; tx-count −(2^20−1) .. 2^20−1.** `mbarrier.init` count ∈ [1, 2^20−1].
* "Starting with the Hopper architecture (sm_9x), mbarrier object supports a new count, called tx-count, which is used for tracking the completion of asynchronous memory operations or transactions. tx-count tracks the number of asynchronous transactions, in units specified by the asynchronous memory operation, that are outstanding and yet to be complete." expect-tx increases tx-count by `expectCount`; complete-tx decrements by `completeCount` and may complete the phase.
* Phase completion: "The count of the pending arrivals has reached zero. The tx-count has reached zero." Then atomically: phase → next, pending count ← expected count (tx-count is 0 by construction). "The current phase is incomplete and all prior phases are complete."
* "For each phase of the mbarrier object, at least one test_wait or try_wait operation must be performed which returns True for waitComplete before an arrive-on operation in the subsequent phase."
* **[interp]** Because tx-count may go negative, complete-tx (data landing) may precede the matching expect-tx within a phase; the phase just cannot complete until both counts are zero.
* Lifecycle: init → use → `mbarrier.inval` before reusing memory ("Performing any mbarrier operation except mbarrier.init on an uninitialized mbarrier object results in undefined behavior. Performing any non-mbarrier or mbarrier.init operations on an initialized mbarrier object results in undefined behavior.").
* **Support by location (verbatim table)**:

| mbarrier operation | `.shared::cta` | `.shared::cluster` |
|---|---|---|
| `mbarrier.arrive` | Supported | Supported, cannot return result |
| `mbarrier.expect_tx` | Supported | Supported |
| `mbarrier.complete_tx` | Supported | Supported |
| Other mbarrier operations | Supported | Not supported |

i.e. **waiting on a remote CTA's mbarrier is impossible** ("threads can perform only arrive operations but not *_wait on an mbarrier located in shared::cluster space"). ptxas: `mbarrier.try_wait.parity.shared::cluster` → "Illegal modifier '::cluster'"; `mbarrier.arrive.release.cluster.shared::cluster.b64 %rd, [..]` → "Sink '_' is expected" **[empirical]**.

### 4.2 Instruction syntax (verbatim) + versions

```
mbarrier.init{.shared{::cta}}.b64 [addr], count;                                   // PTX 7.0, sm_80
mbarrier.inval{.shared{::cta}}.b64 [addr];                                         // PTX 7.0, sm_80
mbarrier.expect_tx{.sem.scope}{.space}.b64 [addr], txCount;                        // PTX 8.0, sm_90; .sem={.relaxed} .scope={.cta,.cluster} .space={.shared{::cta},.shared::cluster}
mbarrier.complete_tx{.sem.scope}{.space}.b64 [addr], txCount;                      // PTX 8.0, sm_90; same qualifiers; "only simulates the completion of an asynchronous memory operation"
mbarrier.arrive{.sem.scope}{.shared{::cta}}.b64           state, [addr]{, count};
mbarrier.arrive{.sem.scope}{.shared::cluster}.b64         _, [addr] {,count}
mbarrier.arrive.expect_tx{.sem.scope}{.shared{::cta}}.b64 state, [addr], txCount;
mbarrier.arrive.expect_tx{.sem.scope}{.shared::cluster}.b64   _, [addr], txCount;
mbarrier.arrive.noComplete{.release.cta}{.shared{::cta}}.b64  state, [addr], count;
   .sem = { .release, .relaxed }   .scope = { .cta, .cluster }
mbarrier.arrive_drop{.sem.scope}{.shared{::cta}}.b64 state, [addr]{, count};
mbarrier.arrive_drop{.sem.scope}{.shared::cluster}.b64 _, [addr] {,count};
mbarrier.arrive_drop.expect_tx{.sem.scope}{.shared{::cta}}.b64 state, [addr], tx_count;
mbarrier.arrive_drop.expect_tx{.sem.scope}{.shared::cluster}.b64 _, [addr], tx_count;
mbarrier.arrive_drop.noComplete{.release.cta}{.shared{::cta}}.b64 state, [addr], count;
mbarrier.test_wait{.sem.scope}{.shared{::cta}}.b64        waitComplete, [addr], state;
mbarrier.test_wait.parity{.sem.scope}{.shared{::cta}}.b64 waitComplete, [addr], phaseParity;
mbarrier.try_wait{.sem.scope}{.shared{::cta}}.b64         waitComplete, [addr], state {, suspendTimeHint};
mbarrier.try_wait.parity{.sem.scope}{.shared{::cta}}.b64  waitComplete, [addr], phaseParity {, suspendTimeHint};
   .sem = { .acquire, .relaxed }   .scope = { .cta, .cluster }
mbarrier.pending_count.b64 count, state;                                            // PTX 7.0, sm_80
cp.async.mbarrier.arrive{.noinc}{.shared{::cta}}.b64 [addr];                         // PTX 7.0, sm_80 (legacy cp.async tracking)
```
Version gates: `::cta` 7.8; `count` without `.noComplete` 7.8 / sm_90; `::cluster`, `.expect_tx`, `.sem/.scope` 8.0 / sm_90; `.relaxed` on arrive/arrive_drop/test_wait/try_wait 8.6 / sm_90; `mbarrier.try_wait` 7.8 / sm_90; `.parity` 7.1. CCCL one-liners: `mbarrier.arrive.sem.scope.space.b64 state, [addr]{, count}; // PTX ISA 80, SM_90`, `mbarrier.arrive.expect_tx.sem.scope.space.b64 state|_, [addr], txCount; // PTX ISA 80, SM_90`, `mbarrier.try_wait.parity.sem.scope.shared::cta.b64 waitComplete, [addr], phaseParity{, suspendTimeHint}; // PTX ISA 80, SM_90`, `mbarrier.expect_tx.sem.scope.space.b64 [addr], txCount; // PTX ISA 80, SM_90`, `mbarrier.complete_tx.sem.scope.space.b64 [addr], txCount; // PTX ISA 80, SM_90`.

Semantics (verbatim excerpts):
* arrive: "`mbarrier.arrive` operation on an mbarrier object located in `.shared::cta` returns an opaque 64-bit register capturing the phase of the mbarrier object prior to the arrive-on operation in the destination operand `state`. Contents of the `state` operand are implementation specific. Optionally, sink symbol '_' can be used." "`mbarrier.arrive` operation on an mbarrier object located in `.shared::cluster` but not in `.shared::cta` cannot return a value. Sink symbol '_' is mandatory." "If the `.sem` qualifier is absent, `.release` is assumed by default." "If the `.scope` qualifier is not specified then it defaults to `.cta`. In contrast, the `.shared::<scope>` indicates the state space where the mbarrier resides." "Qualifiers `.sem` and `.scope` must be specified together." With `.expect_tx`: expect-tx first, then arrive with count 1. `.noComplete`: "must not cause the mbarrier to complete its current phase, otherwise the behavior is undefined." "Note: for sm_8x, when the argument count is specified, the modifier .noComplete is required."
* arrive_drop: decrements expected count by `count` (default 1) "for all the subsequent phases", then arrive-on; for threads exiting / opting out.
* test_wait / try_wait: "`mbarrier.test_wait` is a non-blocking instruction which tests for the completion of the phase. `mbarrier.try_wait` is a potentially blocking instruction [...] If the phase is not complete, the executing thread may be suspended. Suspended thread resumes execution when the specified phase completes OR before the phase completes following a system-dependent time limit. The optional 32-bit unsigned integer operand `suspendTimeHint` specifies the time limit, in nanoseconds, that may be used for the time limit instead of the system-dependent limit." "An even phase has integer parity 0 and an odd phase has integer parity of 1." Valid only for "the current incomplete phase, for which waitComplete returns False" and "the immediately preceding phase, for which waitComplete returns True". Default `.sem` = `.acquire`, default scope `.cta`.
* **Acquire ordering when wait returns True (verbatim list)**: (1) "All memory accesses (except async operations) requested prior, in program order, to `mbarrier.arrive` having release semantics during the completed phase by the participating threads of the CTA are performed and are visible to the executing thread." (2) "All `cp.async` operations requested prior, in program order, to `cp.async.mbarrier.arrive` during the completed phase by the participating threads of the CTA are performed and made visible to the executing thread." (3) "All `cp.async.bulk` asynchronous operations using the same mbarrier object requested prior, in program order, to `mbarrier.arrive` having release semantics during the completed phase by the participating threads of the CTA are performed and made visible to the executing thread." (4) later accesses are not visible to accesses before the release arrive of other threads; (5) "There is no ordering and visibility guarantee for memory accesses requested by the thread after `mbarrier.arrive` having release semantics and prior to `mbarrier.test_wait`, in program order."
* pending_count: state "must be the result of a prior `mbarrier.arrive.noComplete` or `mbarrier.arrive_drop.noComplete` instruction. Otherwise, the behavior is undefined."
* ISA Example 4 (cluster-wide arrive on CTA0's barrier, verbatim):
```
cvta.shared.u64 addr, shMem;   mapa.u64 remAddr, addr, 0;       // CTA0's shMem instance
@p0 mbarrier.init.shared::cta.b64 [shMem], N;  // N = no of cluster threads
barrier.cluster.arrive;  barrier.cluster.wait;
mbarrier.arrive.release.cluster.b64 _, [remAddr];               // Entire cluster
waitLoop:                                                       // Only CTA0
mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 complete, [shMem], 0;
@!complete bra waitLoop;
```
**Required init publication [ISA membar example + CUTLASS]**: `mbarrier.init` → `fence.mbarrier_init.release.cluster` (op-restricted release at cluster scope) → `barrier.cluster.arrive{.relaxed}` / `wait` before any other CTA arrives on it. (CUTLASS `fence_barrier_init()` = `fence.mbarrier_init.release.cluster;`.)

```cpp
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory"); }
__device__ __forceinline__ void mbar_inval(uint32_t bar) { asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(bar) : "memory"); }
__device__ __forceinline__ uint64_t mbar_arrive(uint32_t bar) { uint64_t s; asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];" : "=l"(s) : "r"(bar) : "memory"); return s; }
__device__ __forceinline__ uint64_t mbar_arrive_expect_tx(uint32_t bar, uint32_t tx) { uint64_t s; asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;" : "=l"(s) : "r"(bar), "r"(tx) : "memory"); return s; }
__device__ __forceinline__ void mbar_remote_arrive(uint32_t remote_bar) { asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(remote_bar) : "memory"); }   // no state returned
__device__ __forceinline__ void mbar_remote_arrive_expect_tx(uint32_t remote_bar, uint32_t tx) { asm volatile("mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64 _, [%0], %1;" :: "r"(remote_bar), "r"(tx) : "memory"); }
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t tx) { asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(tx) : "memory"); }
__device__ __forceinline__ void mbar_complete_tx(uint32_t bar, uint32_t tx) { asm volatile("mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(tx) : "memory"); }
__device__ __forceinline__ bool mbar_try_wait_parity(uint32_t bar, uint32_t parity) { uint32_t ok; asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }" : "=r"(ok) : "r"(bar), "r"(parity) : "memory"); return ok; }
__device__ __forceinline__ bool mbar_try_wait_parity_acq_cluster(uint32_t bar, uint32_t parity, uint32_t ns_hint) { uint32_t ok; asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2, %3; selp.u32 %0, 1, 0, p; }" : "=r"(ok) : "r"(bar), "r"(parity), "r"(ns_hint) : "memory"); return ok; }
__device__ __forceinline__ void mbar_wait_parity(uint32_t bar, uint32_t parity) {           // spin entirely in PTX
  asm volatile("{ .reg .pred p; WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra WAIT_%=; }" :: "r"(bar), "r"(parity) : "memory"); }
__device__ __forceinline__ bool mbar_test_wait(uint32_t bar, uint64_t state) { uint32_t ok; asm volatile("{ .reg .pred p; mbarrier.test_wait.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }" : "=r"(ok) : "r"(bar), "l"(state) : "memory"); return ok; }
__device__ __forceinline__ uint32_t mbar_pending_count(uint64_t state) { uint32_t c; asm volatile("mbarrier.pending_count.b64 %0, %1;" : "=r"(c) : "l"(state)); return c; }
```

---

## 5. Bulk asynchronous copies (`#data-movement-and-conversion-instructions-asynchronous-copy`, 9.7.9.25)

### 5.1 General rules (verbatim)
* "The amount of data handled by a bulk asynchronous operation must be a multiple of 16 bytes."
* "Once an asynchronous copy operation is initiated, modifying the source memory location or tensor descriptor or reading from the destination memory location before the asynchronous operation completes, exhibits undefined behavior."
* Completion mechanisms: **async-group** (`commit`/`wait`; "The thread issuing the asynchronous operation must create separate async-groups for bulk and non-bulk asynchronous operations." "All async-groups committed by an executing thread always complete in the order in which they were committed. There is no ordering between asynchronous operations within an async-group.") and **mbarrier-based** ("For a bulk asynchronous operation, the mbarrier object must be specified in the asynchronous operation, whereas for non-bulk operations, it can be specified after the asynchronous operation.").
* All forms: `size` multiple of 16, `dstMem`/`srcMem` 16-byte aligned, ranges must not overflow their spaces; `.L2::cache_hint` only when src or dst is `.global`; copy is a *weak* memory op; complete-tx on the mbarrier has `.release` semantics at `.cluster` scope.

### 5.2 Master table: every bulk-copy form, sm_90 status

Legend: mbar = `.mbarrier::complete_tx::bytes` (complete-tx of `bytes` on the mbarrier, release.cluster); group = `.bulk_group` (`cp.async.bulk.commit_group` / `cp.async.bulk.wait_group{.read}`); ✔ = assembles with ptxas 12.9 for sm_90 and sm_90a; (adv) = assembles for sm_90 with ptxas advisory "should be used on .target sm_90a".

| # | Instruction | src → dst | tensor / reduce | multicast | completion | sm_90? | PTX |
|---|---|---|---|---|---|---|---|
| 1 | `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes{.L2::cache_hint} [dst],[src],size,[mbar]{,policy}` | global → shared::cluster (own or peer CTA) | non-tensor | no | mbar | ✔ | 8.0 |
| 2 | `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster{.L2::cache_hint} [dst],[src],size,[mbar],ctaMask{,policy}` | global → shared of every CTA in 16-bit `ctaMask` (same CTA-relative offset; mbarrier signal multicast to same offset of `mbar`) | non-tensor | **yes** | mbar (one complete-tx per destination CTA) | ✔ (adv); sm_90a advised | 8.0 |
| 3 | `cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes{.L2::cache_hint} [dst],[src],size,[mbar]{,policy}` | global → own CTA smem ("strictly non-remote") | non-tensor | no | mbar | ✔ (PTX ≥ 8.6) | 8.6 |
| 4 | `cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [dst],[src],size,[mbar]` | own smem → **a different CTA's** smem (DSMEM push); `mbar` is the remote CTA's barrier | non-tensor | no | mbar (remote) | ✔ | 8.0 |
| 5 | `cp.async.bulk.global.shared::cta.bulk_group{.L2::cache_hint} [dst],[src],size{,policy}` | own smem → global | non-tensor | – | group | ✔ | 8.0 |
| 5b | `... .bulk_group{.L2::cache_hint}.cp_mask [dst],[src],size{,policy},byteMask` | same, 16-bit per-byte mask per 16-B chunk | non-tensor | – | group | ✘ "requires .target sm_100" | 8.6 |
| – | global→global, shared::cta→shared::cta, shared::cluster as *source* | – | – | – | – | ✘ not forms (ptxas: "State space incorrect") | – |
| 6 | `cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.redOp.type [dst],[src],size,[mbar]` | own smem → peer CTA smem, element-wise reduce | reduce | no | mbar (remote) | ✔ | 8.0 |
| 7 | `cp.reduce.async.bulk.global.shared::cta.bulk_group{.L2::cache_hint}.redOp.type [dst],[src],size{,policy}` (+ `.add.noftz.{f16,bf16}`) | own smem → global, reduce | reduce | – | group | ✔ | 8.0 |
| – | `cp.reduce.async.bulk` global → shared::cluster | – | – | – | listed in the ISA's completion table but **absent from the syntax** and rejected by ptxas (doc inconsistency) | ✘ | – |
| 8 | `cp.async.bulk.prefetch.L2.global{.L2::cache_hint} [src],size{,policy}` | global → L2 | non-tensor | – | none (weak op; see 5.8) | ✔ | 8.0 |
| 9 | `cp.async.bulk.tensor.{1d..5d}.shared::cluster.global{.tile|.im2col}.mbarrier::complete_tx::bytes{.L2::cache_hint} [dst],[tmap,{coords}],[mbar]{,{im2colOffs}}{,policy}` | global tensor → smem (own or peer CTA) | tensor | no | mbar | ✔ (im2col needs ≥3d) | 8.0 |
| 10 | `... .mbarrier::complete_tx::bytes.multicast::cluster{.L2::cache_hint} [dst],[tmap,{coords}],[mbar]{,{im2colOffs}},ctaMask{,policy}` | global tensor → all CTAs in ctaMask | tensor | **yes** | mbar (multicast) | ✔ (adv); sm_90a advised | 8.0 |
| 11 | `cp.async.bulk.tensor.{1d..5d}.shared::cta.global{.tile|.im2col}.mbarrier::complete_tx::bytes{...}` | global tensor → own CTA smem | tensor | no | mbar | ✔ (PTX ≥ 8.6) | 8.6 |
| 12 | `cp.async.bulk.tensor.{1d..5d}.global.shared::cta{.tile|.im2col_no_offs}.bulk_group{.L2::cache_hint} [tmap,{coords}],[src]{,policy}` | own smem → global tensor | tensor | – | group | ✔ | 8.0 |
| 13 | `cp.reduce.async.bulk.tensor.{1d..5d}.global.shared::cta.redOp{.tile|.im2col_no_offs}.bulk_group{.L2::cache_hint} [tmap,{coords}],[src]{,policy}` | own smem → global tensor, reduce (`.add .min .max .inc .dec .and .or .xor`) | reduce-tensor | – | group | ✔ | 8.0 |
| 14 | `cp.async.bulk.prefetch.tensor.{1d..5d}.L2.global{.tile|.im2col}{.L2::cache_hint} [tmap,{coords}]{,{im2colOffs}}{,policy}` | global tensor → L2 | tensor | – | none | ✔ | 8.0 |
| – | `.tile::gather4` (load/prefetch), `.tile::scatter4` (store), `.im2col::w`, `.im2col::w::128`, `.cta_group::1/2`, `.multicast::cluster::32b`, `.mbarrier::report::*`, `.override::global_address` | – | – | – | **sm_100 / sm_100a / sm_107a+ only** (PTX 8.6 / 9.4); ptxas 12.9 for sm_90a: "not supported on .target" | ✘ | 8.6+ |

Note on the `ctaMask` bit meaning: the ISA says "each bit position in the 16-bit `ctaMask` operand corresponds to the `%ctaid` of the destination CTA" — read as the CTA's rank within the cluster (`%cluster_ctarank`), which is what CUTLASS uses **[interp]**.

### 5.3 `cp.async.bulk` (non-tensor) — verbatim syntax block (`#data-movement-and-conversion-instructions-cp-async-bulk`)

```
// global -> shared::cta
cp.async.bulk.dst.src.completion_mechanism{.level::cache_hint} [dstMem], [srcMem], size, [mbar] {, cache-policy}
  .dst = { .shared::cta }  .src = { .global }  .completion_mechanism = { .mbarrier::complete_tx::bytes }  .level::cache_hint = { .L2::cache_hint }
// global -> shared::cluster
cp.async.bulk.dst.src.completion_mechanism{.multicast}{.level::cache_hint} [dstMem], [srcMem], size, [mbar] {, ctaMask} {, cache-policy}
  .dst = { .shared::cluster }  .src = { .global }  .completion_mechanism = { .mbarrier::complete_tx::bytes }  .multicast = { .multicast::cluster }
// shared::cta -> shared::cluster
cp.async.bulk.dst.src.completion_mechanism [dstMem], [srcMem], size, [mbar]
  .dst = { .shared::cluster }  .src = { .shared::cta }  .completion_mechanism = { .mbarrier::complete_tx::bytes }
// shared::cta -> global
cp.async.bulk.dst.src.completion_mechanism{.level::cache_hint}{.cp_mask} [dstMem], [srcMem], size {, cache-policy} {, byteMask}
  .dst = { .global }  .src = { .shared::cta }  .completion_mechanism = { .bulk_group }
```
Verbatim semantics: "When the destination of the copy is `.shared::cta` the destination address has to be in the shared memory of the executing CTA within the cluster, otherwise the behavior is undefined. When the source of the copy is `.shared::cta` and the destination is `.shared::cluster`, the destination has to be in the shared memory of a different CTA within the cluster." Multicast: "Operand `ctaMask` specifies the destination CTAs in the cluster such that each bit position in the 16-bit `ctaMask` operand corresponds to the `%ctaid` of the destination CTA. The source data is multicast to the same CTA-relative offset as `dstMem` in the shared memory of each destination CTA. The mbarrier signal is also multicast to the same CTA-relative offset as `mbar` in the shared memory of the destination CTA." "The copy operation in `cp.async.bulk` is treated as a weak memory operation and the complete-tx operation on the mbarrier has `.release` semantics at the `.cluster` scope." Notes: "`.multicast::cluster` qualifier is optimized for target architecture `sm_90a`/`sm_100f`/... and may have substantially reduced performance on other targets". `.cp_mask`: PTX 8.6, sm_100. CCCL gates: `cp.async.bulk.shared::cluster.global... // PTX ISA 80, SM_90`; `.shared::cta.global // PTX ISA 86, SM_90`; `.shared::cluster.shared::cta // PTX ISA 80, SM_90`; `.global.shared::cta.bulk_group // PTX ISA 80, SM_90`; `.multicast::cluster // PTX ISA 80, SM_90a, SM_100a, ...` (CCCL changelog: "In earlier versions, cp_async_bulk_multicast was enabled for SM_90. This has been changed to SM_90a.").

```cpp
__device__ __forceinline__ void bulk_g2s(uint32_t dst_smem, const void* src_gmem, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(dst_smem), "l"(src_gmem), "r"(bytes), "r"(bar) : "memory"); }
__device__ __forceinline__ void bulk_g2s_cta(uint32_t dst_smem, const void* src_gmem, uint32_t bytes, uint32_t bar) {       // PTX 8.6+
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(dst_smem), "l"(src_gmem), "r"(bytes), "r"(bar) : "memory"); }
__device__ __forceinline__ void bulk_g2s_multicast(uint32_t dst_smem, const void* src_gmem, uint32_t bytes, uint32_t bar, uint16_t cta_mask) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], %4;" :: "r"(dst_smem), "l"(src_gmem), "r"(bytes), "r"(bar), "h"(cta_mask) : "memory"); }
__device__ __forceinline__ void bulk_g2s_hint(uint32_t dst_smem, const void* src_gmem, uint32_t bytes, uint32_t bar, uint64_t policy) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" :: "r"(dst_smem), "l"(src_gmem), "r"(bytes), "r"(bar), "l"(policy) : "memory"); }
__device__ __forceinline__ void bulk_s2s_dsmem(uint32_t dst_dsmem, uint32_t src_smem, uint32_t bytes, uint32_t remote_bar) {   // dst and remote_bar via mapa, must be a different CTA
  asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(dst_dsmem), "r"(src_smem), "r"(bytes), "r"(remote_bar) : "memory"); }
__device__ __forceinline__ void bulk_s2g(void* dst_gmem, uint32_t src_smem, uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" :: "l"(dst_gmem), "r"(src_smem), "r"(bytes) : "memory"); }
__device__ __forceinline__ void bulk_prefetch_L2(const void* src_gmem, uint32_t bytes) { asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(src_gmem), "r"(bytes) : "memory"); }
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void bulk_wait()      { asm volatile("cp.async.bulk.wait_group %0;"      :: "n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void bulk_wait_read() { asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(N) : "memory"); }
```
Canonical producer sequence for a bulk/TMA load tracked by an mbarrier **[ISA + CUTLASS]**: `mbarrier.arrive.expect_tx[.release.cta/.cluster] _, [bar], bytes;` (one elected thread) → issue `cp.async.bulk...mbarrier::complete_tx::bytes` with the same `bar` and total `bytes` → consumers spin on `mbarrier.try_wait.parity[.acquire.scope]`. Order of expect_tx vs. issue does not matter for correctness (tx-count may go negative) but the expect must land in the same phase.

### 5.4 `cp.reduce.async.bulk` — verbatim (`#data-movement-and-conversion-instructions-cp-reduce-async-bulk`)
```
cp.reduce.async.bulk.dst.src.completion_mechanism.redOp.type [dstMem], [srcMem], size, [mbar]
  .dst = { .shared::cluster }  .src = { .shared::cta }  .completion_mechanism = { .mbarrier::complete_tx::bytes }
  .redOp = { .and, .or, .xor, .add, .inc, .dec, .min, .max }   .type = { .b32, .u32, .s32, .b64, .u64 }
cp.reduce.async.bulk.dst.src.completion_mechanism{.level::cache_hint}.redOp.type [dstMem], [srcMem], size{, cache-policy}
  .dst = { .global }  .src = { .shared::cta }  .completion_mechanism = { .bulk_group }
  .type = { .f16, .bf16, .b32, .u32, .s32, .b64, .u64, .s64, .f32, .f64 }
cp.reduce.async.bulk.dst.src.completion_mechanism{.level::cache_hint}.add.noftz.type [dstMem], [srcMem], size{, cache-policy}   // .type = { .f16, .bf16 }
```
Valid (`.dst`, `.redOp`, type) combinations (verbatim table):

| `.dst` | `.redOp` | Element type |
|---|---|---|
| `.shared::cluster` | `.add` | `.u32`, `.s32`, `.u64` |
| | `.min`, `.max` | `.u32`, `.s32` |
| | `.inc`, `.dec` | `.u32` |
| | `.and`, `.or`, `.xor` | `.b32` |
| `.global` | `.add` | `.u32`, `.s32`, `.u64`, `.f32`, `.f64`, `.f16`, `.bf16` |
| | `.min`, `.max` | `.u32`, `.s32`, `.u64`, `.s64`, `.f16`, `.bf16` |
| | `.inc`, `.dec` | `.u32` |
| | `.and`, `.or`, `.xor` | `.b32`, `.b64` |

"The `.inc` and `.dec` operations return a result in the range `[0..x]` where `x` is the value at the source state space." "The floating point operation `.add` rounds to the nearest even. The current implementation of `cp.reduce.async.bulk.add.f32` flushes subnormal inputs and results to sign-preserving zero. The `cp.reduce.async.bulk.add.f16` and `cp.reduce.async.bulk.add.bf16` operations require `.noftz` qualifier." "Each reduction operation performed by the `cp.reduce.async.bulk` has individually `.relaxed.gpu` memory ordering semantics. The load operations ... are treated as weak memory operation and the complete-tx operation on the mbarrier has `.release` semantics at the `.cluster` scope." PTX 8.0, sm_90. ptxas confirms `.shared::cluster` dst rejects `.add.f32` and `.max.u64` **[empirical]**.

```cpp
__device__ __forceinline__ void bulk_red_s2s_add_u32(uint32_t dst_dsmem, uint32_t src_smem, uint32_t bytes, uint32_t remote_bar) {
  asm volatile("cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.add.u32 [%0], [%1], %2, [%3];" :: "r"(dst_dsmem), "r"(src_smem), "r"(bytes), "r"(remote_bar) : "memory"); }
__device__ __forceinline__ void bulk_red_s2g_add_f32(void* dst_gmem, uint32_t src_smem, uint32_t bytes) {
  asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;" :: "l"(dst_gmem), "r"(src_smem), "r"(bytes) : "memory"); }
__device__ __forceinline__ void bulk_red_s2g_add_bf16(void* dst_gmem, uint32_t src_smem, uint32_t bytes) {
  asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 [%0], [%1], %2;" :: "l"(dst_gmem), "r"(src_smem), "r"(bytes) : "memory"); }
```

### 5.5 `cp.async.bulk.tensor` — verbatim (`#data-movement-and-conversion-instructions-cp-async-bulk-tensor`)
```
// global -> shared::cta
cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.cta_group}{.level::cache_hint}
                                   [dstMem], [tensorMap, tensorCoords], [mbar]{, im2colInfo} {, cache-policy}
  .dst = { .shared::cta }  .src = { .global }  .dim = { .1d, .2d, .3d, .4d, .5d }
  .completion_mechanism = { .mbarrier::complete_tx::bytes }  .cta_group = { .cta_group::1, .cta_group::2 }
  .load_mode = { .tile, .tile::gather4, .im2col, .im2col::w, .im2col::w::128 }  .level::cache_hint = { .L2::cache_hint }
// global -> shared::cluster
cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.multicast}{.cta_group}{.level::cache_hint}
                                   [dstMem], [tensorMap, tensorCoords], [mbar]{, im2colInfo} {, ctaMask} {, cache-policy}
  .dst = { .shared::cluster }  ...  .multicast = { .multicast::cluster }
// shared::cta -> global
cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.level::cache_hint}
                                   [tensorMap, tensorCoords], [srcMem] {, cache-policy}
  .dst = { .global }  .src = { .shared::cta }  .completion_mechanism = { .bulk_group }  .load_mode = { .tile, .tile::scatter4, .im2col_no_offs }
```
sm_90 subset: `.dim` 1d–5d, `.load_mode` ∈ {`.tile` (default), `.im2col` (loads, ≥3d), `.im2col_no_offs` (stores)}, `.multicast::cluster`, `.L2::cache_hint`, `.shared::cluster` dst (PTX 8.0) and `.shared::cta` dst (PTX 8.6). Everything else in the syntax (gather4/scatter4, im2col::w, cta_group) is sm_100+.

Verbatim semantics: "When `.dst` is specified as `.shared::cta`, the address `dstMem` must be in the shared memory of the executing CTA within the cluster, otherwise the behavior is undefined. When `.dst` is specified as `.shared::cluster`, the address `dstMem` can be in the shared memory of any of the CTAs within the current cluster." "The operand `tensorMap` is the generic address of the opaque tensor-map object which resides in `.param` space or `.const` space or `.global` space. [...] The `tensorMap` is accessed in tensormap proxy." "The individual tensor coordinates in `tensorCoords` are of type `.s32`." tensorCoords = `{d0, .., dn}` for n = `.dim` (gather4/scatter4: `{col_idx, row_idx0..3}`). "Upon the completion of the asynchronous copy operation, the complete-tx operation, with `completeCount` argument equal to amount of data copied in bytes, will be performed on the mbarrier object specified by the operand `mbar`. This instruction accesses its `mbarrier` operand using generic-proxy." `.im2col` → `im2colInfo` = `{ i2cOffW , i2cOffH , i2cOffD }` "A vector of im2col offsets whose vector size is two less than number of dimensions .dim" (16-bit registers in the ISA examples). Multicast: same text as non-tensor ("each bit position in the 16-bit ctaMask ... The source data is multicast to the same offset as dstMem in the shared memory of each destination CTA. [cta_group::1:] The mbarrier signal is also multicasted to the same offset as mbar in the shared memory of the destination CTA."). "The copy operation in `cp.async.bulk.tensor` is treated as a weak memory operation and the complete-tx operation on the mbarrier has `.release` semantics at the `.cluster` scope." PTX 8.0 / sm_90; `.shared::cta` dst, gather4/scatter4, im2col::w, cta_group PTX 8.6.

The completion table in the ISA additionally says the `.bulk_group` mechanism "optionally can be used for the completion of reading of the tensormap object" for the mbarrier-tracked load forms — i.e. **[interp]** a TMA *load* can be followed by `cp.async.bulk.commit_group; cp.async.bulk.wait_group.read 0` to know the descriptor has been consumed (CUTLASS does not rely on this).

Restrictions on `.global.shared::cta` (stores), from "Restriction on Tensor Copy instructions": "Starting co-ordinates for Bounding Box (`tensorCoords`) must be non-negative. The bounding box along the D, W and H dimensions must stay within the tensor boundaries." (The sub-byte-type, 96B-swizzle, sm_120a, sm_103a items in that section are not applicable to sm_90.)

CCCL one-liners: `cp.async.bulk.tensor.{1..5}d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 80, SM_90` (`.dst = .shared::cluster`) and `// PTX ISA 86, SM_90` (`.dst = .shared::cta`); `...tile.mbarrier::complete_tx::bytes.multicast::cluster [...], [...], [smem_bar], ctaMask; // PTX ISA 80, SM_90a, SM_100a, ...`; `cp.async.bulk.tensor.{1..5}d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90`; `.tile::gather4 ... // PTX ISA 86, SM_100`; `.tile::scatter4.bulk_group // PTX ISA 86, SM_100a, ...`. CUTLASS (`cute/arch/copy_sm90_tma.hpp`) uses `cp.async.bulk.tensor.Nd.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint`, `...multicast::cluster.L2::cache_hint`, `...im2col.mbarrier::complete_tx::bytes`, `cp.async.bulk.tensor.Nd.global.shared::cta.bulk_group`, `...im2col_no_offs.bulk_group`, `cp.reduce.async.bulk.tensor.Nd.global.shared::cta.add.bulk_group`, `cp.async.bulk.prefetch.tensor.Nd.L2.global{.im2col}`.

```cpp
__device__ __forceinline__ void tma_load_2d(uint32_t dst_smem, const void* tmap, int c0, int c1, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" :: "r"(dst_smem), "l"(tmap), "r"(c0), "r"(c1), "r"(bar) : "memory"); }
__device__ __forceinline__ void tma_load_2d_hint(uint32_t dst_smem, const void* tmap, int c0, int c1, uint32_t bar, uint64_t policy) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;" :: "r"(dst_smem), "l"(tmap), "r"(c0), "r"(c1), "r"(bar), "l"(policy) : "memory"); }
__device__ __forceinline__ void tma_load_2d_multicast(uint32_t dst_smem, const void* tmap, int c0, int c1, uint32_t bar, uint16_t cta_mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;" :: "r"(dst_smem), "l"(tmap), "r"(c0), "r"(c1), "r"(bar), "h"(cta_mask) : "memory"); }
__device__ __forceinline__ void tma_load_3d_im2col(uint32_t dst_smem, const void* tmap, int c0, int c1, int c2, uint32_t bar, uint16_t offW) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5], {%6};" :: "r"(dst_smem), "l"(tmap), "r"(c0), "r"(c1), "r"(c2), "r"(bar), "h"(offW) : "memory"); }
__device__ __forceinline__ void tma_store_2d(const void* tmap, int c0, int c1, uint32_t src_smem) {   // precede with fence.proxy.async.shared::cta (+ CTA barrier)
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%1, %2}], [%3];" :: "l"(tmap), "r"(c0), "r"(c1), "r"(src_smem) : "memory"); }
__device__ __forceinline__ void tma_reduce_add_2d(const void* tmap, int c0, int c1, uint32_t src_smem) {
  asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2}], [%3];" :: "l"(tmap), "r"(c0), "r"(c1), "r"(src_smem) : "memory"); }
__device__ __forceinline__ void tma_prefetch_2d(const void* tmap, int c0, int c1) {
  asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"(tmap), "r"(c0), "r"(c1) : "memory"); }
```
(`tmap` is the generic address of a 128-B `CUtensorMap`, normally a `const __grid_constant__ CUtensorMap` kernel parameter → `.param` space.)

### 5.6 `cp.reduce.async.bulk.tensor` — verbatim (`#data-movement-and-conversion-instructions-cp-reduce-async-bulk-tensor`)
```
cp.reduce.async.bulk.tensor.dim.dst.src.redOp{.load_mode}.completion_mechanism{.level::cache_hint} [tensorMap, tensorCoords], [srcMem] {,cache-policy}
  .dst = { .global }  .src = { .shared::cta }  .dim = { .1d, .2d, .3d, .4d, .5d }  .completion_mechanism = { .bulk_group }
  .load_mode = { .tile, .im2col_no_offs }  .redOp = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
```
Element type comes from the tensor-map: `.add` → u32,s32,u64,f32,f16,bf16; `.min/.max` → u32,s32,u64,s64,f16,bf16; `.inc/.dec` → u32; `.and/.or/.xor` → b32,b64. "Each reduction operation ... has individually `.relaxed.gpu` memory ordering semantics." Only shared::cta → global; PTX 8.0, sm_90. ISA example: `cp.reduce.async.bulk.tensor.2d.global.shared::cta.and.bulk_group.L2::cache_hint [tensorMap1, {tc0, tc1}], [sMem1] , policy;`.

### 5.7 Prefetch forms
* `cp.async.bulk.prefetch.L2.src{.level::cache_hint} [srcMem], size {, cache-policy}` `.src = { .global }` — "may initiate an asynchronous prefetch ... to the L2 cache"; size multiple of 16, 16-B aligned; "treated as a weak memory operation". PTX 8.0, sm_90. CCCL: `cp.async.bulk.prefetch.L2.global.L2::cache_hint [srcMem], size, cache_policy; // PTX ISA 80, SM_90`.
* `cp.async.bulk.prefetch.tensor.dim.L2.src{.load_mode}{.level::cache_hint} [tensorMap, tensorCoords] {, im2colInfo } {, cache-policy}` — sm_90 `.load_mode` = `.tile`, `.im2col`; weak op; PTX 8.0, sm_90.
* `prefetch{.tensormap_space}.tensormap [a]` with `.tensormap_space = { .const, .param }` — "brings the cache line containing the specified address in the .const or .param memory state space for subsequent use by the cp.async.bulk.tensor instruction." PTX 8.0, sm_90. (`prefetch.tensormap [generic]` also assembles.)

### 5.8 Completion: `cp.async.bulk.commit_group` / `cp.async.bulk.wait_group{.read} N` (verbatim)
> `cp.async.bulk.commit_group` instruction creates a new per-thread _bulk async-group_ and batches all prior `cp{.reduce}.async.bulk.{.prefetch}{.tensor}` instructions satisfying the following conditions into the new _bulk async-group_: The prior ... instructions use _bulk_group_ based completion mechanism, and They are initiated by the executing thread but not committed to any _bulk async-group_. [...] There is no memory ordering guarantee provided between any two ... operations within the same _bulk async-group_.

> `cp.async.bulk.wait_group` instruction will cause the executing thread to wait until only N or fewer of the most recent _bulk async-groups_ are pending and all the prior _bulk async-groups_ committed by the executing threads are complete. [...] By default, `cp.async.bulk.wait_group` instruction will cause the executing thread to wait until completion of all the bulk async operations in the specified _bulk async-group_. A bulk async operation includes the following: Optionally, reading from the tensormap. Reading from the source locations. Writing to their respective destination locations. Writes being made visible to the executing thread. The optional `.read` modifier indicates that the waiting has to be done until all the bulk async operations in the specified _bulk async-group_ have completed: reading from the tensormap, the reading from their source locations.

`.read` is what you use before overwriting the smem source of a TMA store; plain `wait_group 0` is what you use before another agent reads the global destination. Both PTX 8.0, sm_90; per-thread groups (the issuing thread must wait).

### 5.9 Tensor-map manipulation: `tensormap.replace`, `tensormap.cp_fenceproxy`, `fence.proxy.tensormap::generic`
* Tensor-map (5.5.8): "The tensor-map is a 128-byte opaque object either in `.const` space or `.param` (kernel function parameter) space or `.global` space".
* `tensormap.replace.mode.field{1,2,3}{.ss}.b1024.type [addr], {ord,} new_val;` `.mode = { .tile }`, `.field1 = { .global_address, .rank }`, `.field2 = { .box_dim, .global_dim, .global_stride, .element_stride }`, `.field3 = { .elemtype, .interleave_layout, .swizzle_mode, .swizzle_atomicity, .fill_mode }`, `.ss = { .global, .shared::cta }`, `.type = { .b32, .b64 }` (b64 for global_address/global_stride). "`tensormap.replace` is treated as a weak memory operation, on the entire 1024-bit opaque tensor-map object". **sm_90a only** (ptxas for sm_90: "not supported on .target 'sm_90'") **[empirical, matches ISA target list]**; PTX 8.3. `.swizzle_mode` values: 0 none, 1 32B, 2 64B, 3 128B, 4 96B (sm_103a); `.elemtype` 0 u8 ... 6 f16, 7 f32, 8 f32.ftz, 9 f64, 10 bf16, 11 tf32, 12 tf32.ftz ("do not correspond to the values of the CUtensorMapDataType enum").
* `tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.scope.sync.aligned [dst], [src], size;` `size` = 128; "Copies data ... from the location specified by the address operand src in shared memory to the location specified by the address operand dst in the global memory, in the generic proxy. Establishes a uni-directional proxy release pattern on the ordering from the copy operation to the subsequent access performed in the tensormap proxy on the address dst." PTX 8.3, sm_90.
* Consumer side: `fence.proxy.tensormap::generic.acquire.scope [addr], 128;` before the `cp.async.bulk.tensor` that reads the modified map. ISA example sequence (verbatim):
```
tensormap.replace.tile.global_address.shared.b1024.b64   [sMem], new_addr;
tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [gbl], [sMem], 128;
fence.proxy.tensormap::generic.acquire.gpu [gbl], 128;
cp.async.bulk.tensor.1d.shared::cluster.global.tile  [addr0], [gbl, {tc0}], [mbar0];
```
or, when the global map is written directly by generic stores: `tensormap.replace...global...; fence.proxy.tensormap::generic.release.gpu; cvta.global.u64 tmap, gbl; fence.proxy.tensormap::generic.acquire.gpu [tmap], 128;`.

```cpp
__device__ __forceinline__ void prefetch_tensormap(const void* tmap) { asm volatile("prefetch.tensormap [%0];" :: "l"(tmap) : "memory"); }
__device__ __forceinline__ void tensormap_replace_gaddr_smem(uint32_t tmap_smem, const void* new_gaddr) {   // sm_90a only
  asm volatile("tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;" :: "r"(tmap_smem), "l"(new_gaddr) : "memory"); }
__device__ __forceinline__ void tensormap_cp_fenceproxy(void* dst_gmem_tmap, uint32_t src_smem_tmap) {
  asm volatile("tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [%0], [%1], 128;" :: "l"(dst_gmem_tmap), "r"(src_smem_tmap) : "memory"); }
```

---

## 6. Legacy (sm_80) non-bulk `cp.async` for comparison (`#data-movement-and-conversion-instructions-cp-async`)

```
cp.async.ca.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size} [dst], [src], cp-size{, src-size}{, cache-policy} ;
cp.async.cg.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size} [dst], [src], 16{, src-size}{, cache-policy} ;
cp.async.ca... [dst], [src], cp-size{, ignore-src}{, cache-policy} ;   cp.async.cg... [dst], [src], 16{, ignore-src}{, cache-policy} ;
.level::cache_hint = { .L2::cache_hint }  .level::prefetch_size = { .L2::64B, .L2::128B, .L2::256B }  cp-size = { 4, 8, 16 }
cp.async.commit_group ;      cp.async.wait_group N;      cp.async.wait_all ;
cp.async.mbarrier.arrive{.noinc}{.shared{::cta}}.b64 [addr];
```
* Per-thread, **generic proxy**, global→shared::cta only, 4/8/16 B (`.cg` only 16: ptxas "expected to be 16"); `src-size < cp-size` zero-fills; `ignore-src` predicate zero-fills. "`.cg` ... caching of data only at global level cache L2 and not at L1 whereas `.ca` ... at all levels including L1". "`cp.async` is treated as a weak memory operation". No ordering between cp.async ops unless `wait_group`/`wait_all`/mbarrier. "The thread issuing the asynchronous operation must create separate async-groups for bulk and non-bulk asynchronous operations" (so `cp.async.commit_group` and `cp.async.bulk.commit_group` are independent group stacks).
* Writes visible to the issuing thread only after `wait_all`, `wait_group` on its group, or `mbarrier.test_wait` returning True on a tracking mbarrier. "`cp.async.wait_group` and `cp.async.wait_all` does not provide any ordering and visibility guarantees for any other memory operation apart from `cp.async`."
* `cp.async.mbarrier.arrive`: "Causes an arrive-on operation to be triggered by the system on the mbarrier object upon the completion of all prior cp.async operations initiated by the executing thread. [...] When `.noinc` modifier is not specified, the pending count of the mbarrier object is incremented by 1 prior to the asynchronous arrive-on operation. This results in a zero-net change for the pending count"; with `.noinc` the init count must account for it. PTX 7.0, sm_80. Note: no tx-count/bytes involved — it is an arrive, not a complete-tx.
* Versions: PTX 7.0 / sm_80; `.L2::cache_hint`, `.L2::{64,128,256}B` 7.4; `ignore-src` 7.5; `::cta` 7.8.

```cpp
__device__ __forceinline__ void cp_async_cg_16(uint32_t dst_smem, const void* src_gmem) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_smem), "l"(src_gmem) : "memory"); }
__device__ __forceinline__ void cp_async_ca_4(uint32_t dst_smem, const void* src_gmem)  { asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"  :: "r"(dst_smem), "l"(src_gmem) : "memory"); }
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void cp_async_mbar_arrive(uint32_t bar) { asm volatile("cp.async.mbarrier.arrive.shared::cta.b64 [%0];" :: "r"(bar) : "memory"); }
```

---

## 7. Cache operators, eviction priorities, `createpolicy`, `discard`, `applypriority`, `prefetch`

### 7.1 Cache operators (`#cache-operators`, 9.7.9.1) — verbatim meanings; "treated as performance hints only"
| Load op | Meaning |
|---|---|
| `.ca` | "Cache at all levels, likely to be accessed again. The default load instruction cache operation is ld.ca, which allocates cache lines in all levels (L1 and L2) with normal eviction policy. Global data is coherent at the L2 level, but multiple L1 caches are not coherent for global data." |
| `.cg` | "Cache at global level (cache in L2 and below, not L1). Use `ld.cg` to cache loads only globally, bypassing the L1 cache, and cache only in the L2 cache." |
| `.cs` | "Cache streaming, likely to be accessed once. The `ld.cs` load cached streaming operation allocates global lines with evict-first policy in L1 and L2 to limit cache pollution by temporary streaming data" |
| `.lu` | "Last use. [...] performs a load cached streaming operation (`ld.cs`) on global addresses." |
| `.cv` | "Don't cache and fetch again (consider cached system memory lines stale, fetch again). The ld.cv load operation applied to a global System Memory address invalidates (discards) a matching L2 line and re-fetches the line on each new load." |

| Store op | Meaning |
|---|---|
| `.wb` | default: "writes back cache lines of coherent cache levels with normal eviction policy" |
| `.cg` | "Cache at global level (cache in L2 and below, not L1)" |
| `.cs` | "allocates cache lines with evict-first policy" |
| `.wt` | "Cache write-through (to system memory). The `st.wt` store write-through operation applied to a global System Memory address writes through the L2 cache." |

Constraints: cache operators cannot be combined with `.relaxed/.acquire/.release/.volatile` ("Cache operations are not allowed"), nor with `L1::eviction_priority` (ptxas: "Modifier '.evict_last' cannot be combined with modifier '.cg'") **[empirical]**. `ld.global.nc` (read-only/non-coherent path, 9.7.9.9) accepts `.cop = { .ca, .cg, .cs }` or the eviction-priority qualifiers.

### 7.2 Eviction priority hints (`#cache-eviction-priority-hints`, 9.7.9.2) — PTX 7.4, sm_70; `.global` only
`evict_normal` (default), `evict_first` ("suitable for streaming data"), `evict_last` ("suitable for data that should remain persistent in cache"), `evict_unchanged`, `no_allocate` ("Do not allocate data to cache"). On `ld`/`st`: `.L1::{evict_normal, evict_unchanged, evict_first, evict_last, no_allocate}`. **`.L2::{evict_first,evict_last,evict_normal}` on `ld`/`st` is sm_100+ and only with `.v8.b32`/`.v4.b64` (256-bit)**; on sm_90 the L2 priority must go through `createpolicy` + `.L2::cache_hint` (ptxas: "Instruction 'ld' requires '.v8.b32/.v4.b64' type with '.L2::evict_last' modifier"). `.L2::{64B,128B,256B}` prefetch-size hints: PTX 7.4, sm_75 (256B/cache_hint sm_80); global only.

### 7.3 `createpolicy` (9.7.9.18) — PTX 7.4, sm_80
```
createpolicy.range{.global}.level::primary_priority{.level::secondary_priority}.b64 cache-policy, [a], primary-size, total-size;
createpolicy.fractional.level::primary_priority{.level::secondary_priority}.b64 cache-policy{, fraction};
createpolicy.cvt.L2.b64 cache-policy, access-property;
.level::primary_priority = { .L2::evict_last, .L2::evict_normal, .L2::evict_first, .L2::evict_unchanged };
.level::secondary_priority = { .L2::evict_first, .L2::evict_unchanged };
```
Range policy: primary range `[a, a+primary-size)`, trailing secondary `[a+primary-size, a+total-size)`, preceding secondary `[a-(total-size-primary-size), a)`; outside → unspecified; "Maximum allowed value of total-size is 4GB". Fractional: fraction ∈ (0.0, 1.0], default 1.0 = probability of primary priority; secondary default `evict_unchanged`. The resulting opaque 64-bit policy is consumed by any `.L2::cache_hint` operand (`ld`, `st`, `atom`, `red`, `cp.async`, all `cp.async.bulk*` / `cp.reduce.async.bulk*` / prefetch forms with a `.global` side).

### 7.4 `discard`, `applypriority`, `prefetch`
* `discard{.global}.L2 [a], 128;` — "Semantically, this behaves like a weak write of an unstable indeterminate value [...] hints to the implementation that data in the specified cache .level can be destructively discarded without writing it back to memory." size must be 128, `a` 128-B aligned. PTX 7.4, sm_80.
* `applypriority{.global}.L2::evict_normal [a], 128;` — applies the priority to `[a, a+128)` in L2 (only `evict_normal`, i.e. "demote a persisting line"); 128-B aligned. PTX 7.4, sm_80.
* `prefetch{.space}.level [a]` (`.space = {.global,.local}`, `.level = {.L1,.L2}`); `prefetch.global.L2::{evict_last,evict_normal} [a]` (PTX 7.4, sm_80); `prefetchu.L1 [a]`; "A prefetch to a shared memory location performs no operation."

```cpp
__device__ __forceinline__ uint32_t ld_cg(const uint32_t* p) { uint32_t v; asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }       // bypass L1
__device__ __forceinline__ uint32_t ld_cv(const uint32_t* p) { uint32_t v; asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }       // refetch
__device__ __forceinline__ uint32_t ld_cs(const uint32_t* p) { uint32_t v; asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ uint32_t ld_L1_no_allocate(const uint32_t* p) { uint32_t v; asm volatile("ld.global.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ uint32_t ld_L1_evict_last(const uint32_t* p)  { uint32_t v; asm volatile("ld.global.L1::evict_last.u32 %0, [%1];"  : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ uint32_t ld_L2_256B(const uint32_t* p) { uint32_t v; asm volatile("ld.global.L2::256B.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ uint64_t createpolicy_evict_last()  { uint64_t pol; asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;"  : "=l"(pol)); return pol; }
__device__ __forceinline__ uint64_t createpolicy_evict_first() { uint64_t pol; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol)); return pol; }
__device__ __forceinline__ uint64_t createpolicy_range(const void* base, uint32_t primary, uint32_t total) { uint64_t pol; asm volatile("createpolicy.range.global.L2::evict_last.L2::evict_first.b64 %0, [%1], %2, %3;" : "=l"(pol) : "l"(base), "r"(primary), "r"(total)); return pol; }
__device__ __forceinline__ uint32_t ld_cache_hint(const uint32_t* p, uint64_t policy) { uint32_t v; asm volatile("ld.global.L2::cache_hint.u32 %0, [%1], %2;" : "=r"(v) : "l"(p), "l"(policy) : "memory"); return v; }
__device__ __forceinline__ void st_cg(uint32_t* p, uint32_t v) { asm volatile("st.global.cg.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void discard_L2(const void* p)          { asm volatile("discard.global.L2 [%0], 128;" :: "l"(p) : "memory"); }
__device__ __forceinline__ void applypriority_normal(const void* p) { asm volatile("applypriority.global.L2::evict_normal [%0], 128;" :: "l"(p) : "memory"); }
__device__ __forceinline__ void prefetch_L2_evict_last(const void* p) { asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(p)); }
```

---

## 8. `nanosleep`, clocks, `elect.sync`
* `nanosleep.u32 t;` (PTX 6.3, sm_70): "Suspends the thread for a sleep duration approximately close to the delay `t`, specified in nanoseconds. [...] The sleep duration is approximated, but guaranteed to be in the interval `[0, 2*t]`. The maximum sleep duration is 1 millisecond. The implementation may reduce the sleep duration for individual threads within a warp such that all sleeping threads in the warp wake up together."
* `%clock64`: free-running SM cycle counter (64-bit, wraps silently); `%clock` = low 32 bits. The ISA does not specify the clock domain or whether it is per-SM; it is not synchronized across SMs **[interp: do not compare %clock64 values across SMs/CTAs of a cluster]**.
* `%globaltimer`: "64-bit global nanosecond timer", "intended for use by NVIDIA tools", "target-specific and may change". **[empirical, prior work, not from the ISA]** on Hopper its value advances in steps of ~32 ns (i.e. ~31 MHz update rate) even though its unit is ns, so single-event latencies below ~100 ns cannot be resolved with it; use `%clock64` deltas within one SM and convert with the measured SM clock. It is the only time base the ISA offers that is comparable across SMs.
* `mbarrier.try_wait` `suspendTimeHint` is also in ns (section 4.2).
* `elect.sync d|p, membermask;` (PTX 8.0, sm_90): "elects one predicated active leader thread [...] Election of a leader thread happens deterministically, i.e. the same leader thread is elected for the same membermask every time." Use to pick the single TMA-issuing thread.

```cpp
__device__ __forceinline__ void nanosleep_(uint32_t ns) { asm volatile("nanosleep.u32 %0;" :: "r"(ns)); }
__device__ __forceinline__ bool elect_one() { uint32_t pred; asm volatile("{ .reg .pred p; elect.sync _|p, 0xffffffff; selp.u32 %0, 1, 0, p; }" : "=r"(pred)); return pred; }
```

---

## 9. Explicit numeric limits

| Item | Value | Source |
|---|---|---|
| Bulk copy / reduce / prefetch `size` | multiple of 16 B; `srcMem`,`dstMem` 16-B aligned; 32-bit operand | ISA 9.7.9.25 (every bulk form) |
| Multicast `ctaMask` | 16-bit → at most 16 destination CTAs per cluster (`.b16` register / `"h"` constraint) | ISA |
| `.cp_mask` byteMask | 16-bit, per byte of each 16-B chunk (sm_100+) | ISA |
| mbarrier object | `.b64`, 8-B aligned, in `.shared::cta`; init count ∈ [1, 2^20−1]; pending ∈ [0, 2^20−1]; tx-count ∈ [−(2^20−1), 2^20−1] | ISA 9.7.13.15.1–2 |
| Remote mbarrier ops | only `arrive`, `arrive.expect_tx`, `arrive_drop`, `expect_tx`, `complete_tx` (no returned state); no remote wait | ISA 9.7.13.15.8 |
| `st.async` payload | `.b32/.b64/.u32/.u64/.s32/.s64/.f32/.f64`, `.v2`, `.v4` (≤128 bit total); requires `%cluster_nctarank > 1` | ISA 9.7.9.12 |
| `red.async` ops | add(u32,s32,u64), min/max(u32,s32), inc/dec(u32), and/or/xor(b32); target CTA ≠ executing CTA | ISA 9.7.13.7 |
| Portable cluster size | 8 CTAs ("The portable cluster size for sm_90 is 8 blocks per cluster"); larger needs `CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED` / `cudaFuncAttributeNonPortableClusterSizeAllowed`; query with `cudaOccupancyMaxPotentialClusterSize` | CUDA Driver/Runtime API docs |
| Non-portable max on H100 | 16 CTAs (consistent with 16-bit ctaMask; `.reqnctapercluster 16` / `.maxclusterrank 16` accepted by ptxas) | CUDA Programming Guide (not re-verified here) + [empirical ptxas] |
| Cluster dims | `%cluster_ctaid/%cluster_nctaid/%clusterid/%nclusterid` are `.v4.u32`, 4th element always 0; max `%nclusterid`/`%cluster_nctaid` deferred to Programming Guide | ISA ch. 10 |
| Tensor map | 128-byte opaque object in `.param`/`.const`/`.global`; `tensormap.replace` type `.b1024`; proxy fence / `cp_fenceproxy` size operand = 128 | ISA 5.5.8, 9.7.9.28, 9.7.13.4 |
| TMA rank | 1–5 (`tensorRank` "non-zero and less than or equal to the maximum supported dimensionality of 5"); im2col/interleave need ≥3 | cuTensorMapEncodeTiled |
| TMA `boxDim[i]` | "must be non-zero and less than or equal to 256"; with no interleave, `boxDim[0] * elementSizeInBytes` "must be a multiple of 16 bytes" | cuTensorMapEncodeTiled |
| TMA `globalAddress` | 16-B aligned (32-B with `CU_TENSOR_MAP_INTERLEAVE_32B`) | cuTensorMapEncodeTiled |
| TMA `globalDim[i]` | non-zero, ≤ 2^32 | cuTensorMapEncodeTiled |
| TMA `globalStrides[i]` (rank−1 entries, bytes) | "must be a multiple of 16 and less than 2^40" (multiple of 32 with 32B interleave) | cuTensorMapEncodeTiled |
| TMA `elementStrides[i]` | "non-zero and less than or equal to 8"; entry 0 ignored without interleave ("TMA doesn't support the stride for dimension zero"); TMA loads ceil(boxDim[i]/elementStrides[i]) elements | cuTensorMapEncodeTiled |
| TMA swizzle vs inner box | inner dim (`boxDim[0]`·elemsize) ≤ 32 B for SWIZZLE_32B, ≤ 64 B for 64B, ≤ 128 B for 128B*; `INTERLEAVE_32B` requires `SWIZZLE_32B` | cuTensorMapEncodeTiled |
| Swizzle pattern period / smem base alignment | 128B swizzle: pattern repeats at 1024-B boundary, base offset = (dstMem/128)%8; 64B: 512-B, (dstMem/128)%4; 32B (and 96B): 256-B, (dstMem/128)%2 | ISA 5.5.7 |
| Swizzle modes (sm_90) | none, 32B, 64B, 128B (16-B atomicity); 96B and 128B with 32B/32B+8B-flip/64B atomicity are sm_100+ | ISA 5.5.7 / 9.7.9.28 |
| Interleave | none, 16B (NC/8DHWC8), 32B (NC/16HWC16); 3D–5D only | ISA 5.5.6 |
| OOB fill | zero or `OOB-NaN` | ISA 5.5.3.3 |
| im2col bounding-box corners | 3D [−2^15, 2^15−1]; 4D [−2^7, 2^7−1]; 5D [−2^4, 2^4−1]; im2col offsets 3D [0, 2^16−1], 4D [0, 2^8−1], 5D [0, 2^5−1]; channelsPerPixel ≤ 256; pixelsPerColumn ≤ 1024 | ISA 5.5.4.1 + cuTensorMapEncodeIm2col |
| L2 promotion (tensor map) | NONE / 64B / 128B / 256B | cuTensorMapEncodeTiled |
| `cp.async` (sm_80) | cp-size 4/8/16; `.cg` only 16; prefetch-size 64/128/256 B | ISA 9.7.9.25.3 |
| `createpolicy.range` | total-size ≤ 4 GB; fraction ∈ (0,1] | ISA 9.7.9.18 |
| `discard` / `applypriority` | size 128, 128-B aligned, `.L2` only | ISA |
| `nanosleep` | actual ∈ [0, 2t]; max 1 ms | ISA |
| `ld`/`st` 256-bit (`.v8.b32`/`.v4.b64`) and `.L2::evict_*` on ld/st | sm_100+ | ISA / ptxas |
| DSMEM address window | not sized by the ISA; `.shared::cta` window ⊂ `.shared::cluster` window; `mapa.u32` / `.u64` both valid; 32-bit shared addresses used by all `[%r]` operands above | ISA 5.1.7 |

---

## 10. ptxas 12.9 validation matrix (`.version 8.8`; identical results for `sm_90` and `sm_90a` unless noted)

Assembled OK (200 forms, sm_90a; 198 on sm_90): all forms in rows 1–14 of the master table (1d–5d tile, 3d/4d/5d im2col, multicast, L2::cache_hint, cta dst, DSMEM push, bulk_group stores, reduce forms per the type tables, prefetch forms, commit/wait/wait.read), `prefetch.tensormap`, `tensormap.cp_fenceproxy`, all `fence.*` forms in section 1.6 (incl. `sync_restrict`, `async::generic`, `tensormap::generic` release/acquire), `barrier.cluster.*` with all `.sem/.aligned` combos, `mapa.{shared::cluster.u32,u64}`, `getctarank.{shared::cluster.u32,u64}`, `isspacep.shared::{cta,cluster}`, `cvta.shared::cluster.u64`, `cvta.to.shared::cluster.u64`, all cluster/timer special registers, `ld/st.shared::cluster` (weak, `.v4`, `.volatile`, `.relaxed.cluster`, `.acquire.cluster`, `.release.cluster`), `ld.relaxed.cluster.global`, `st.release.cluster.global`, `atom/red.shared::cluster.*` listed in 3.2, `st.async` (`.b32/.u32/.b64/.f32/.v2.b32/.v4.b32/.v2.b64`, with and without `.weak`, generic-address form), `red.async` (`add.u32/u64`, `min.s32`, `and.b32`, `inc.u32`), every mbarrier form in 4.2 (local and remote, `.relaxed`, `.expect_tx`, `.noComplete`, `arrive_drop`, `pending_count`, `try_wait` with hint), `cp.async.mbarrier.arrive{.noinc}`, legacy `cp.async.{ca,cg}` incl. `L2::128B`, `L2::cache_hint`, src-size, `ld/st` cache ops `.ca .cg .cs .lu .cv .wb .wt`, `L1::{no_allocate,evict_last,evict_first,evict_unchanged}`, `L2::{64B,128B,256B}`, `.nc` variants, `L2::cache_hint`, `createpolicy.{fractional,range}`, `discard`, `applypriority`, `prefetch.global.{L1,L2,L2::evict_last,L2::evict_normal}`, `nanosleep`, `elect.sync` (via the compiled snippet file), directives `.reqnctapercluster 2,1,1` / `16`, `.maxclusterrank 8` / `16`, `.explicitcluster`.

Rejected (message):
* `cp.async.bulk.global.shared::cta.bulk_group.cp_mask` — "Feature '.cp_mask' requires .target sm_100 or higher".
* `cp.async.bulk.global.global...`, `cp.async.bulk.shared::cta.shared::cta...`, `cp.reduce.async.bulk.shared::cluster.global...` — "State space incorrect".
* `cp.reduce.async.bulk.shared::cluster.shared::cta...add.f32` / `...max.u64` — "Arguments mismatch" / "Illegal reduction operation".
* `.tile::gather4` (shared::cluster dst), `.tile::scatter4`, `.cta_group::2` — "not supported on .target 'sm_90a'".
* `tensormap.replace.*` — OK on sm_90a, **"not supported on .target 'sm_90'"**.
* `.multicast::cluster` on sm_90 — accepted with advisory warning only.
* `st.async ... .v4.b64` — "Vector type too large, exceeds 128 bit limit"; `red.async ... add.f32` — "Unexpected instruction types".
* `mbarrier.arrive.release.cluster.shared::cluster.b64 %rd, [..]` — "Sink '_' is expected as a destination operand"; `mbarrier.try_wait.parity.shared::cluster` — "Illegal modifier '::cluster'".
* `cp.async.cg.shared.global [..],[..],4` — "expected to be 16".
* `ld.global.L2::evict_last.u32` — "requires '.v8.b32/.v4.b64' type"; `ld.global.L2::evict_last.v8.f32` — "'256 bit wide load/store' requires .target sm_100"; `ld.global.cg.L1::evict_last` — cannot combine; `ld.relaxed.gpu.global.cg` — cannot combine.
* PTX-version gates seen when the snippet file was assembled with clang's default `.version 8.3`: `.sync_restrict` and `mbarrier.arrive.relaxed` need PTX ≥ 8.6; `cp.async.bulk` with `.shared::cta` destination needs PTX ≥ 8.6 (nvcc 12.8+/13.x emit ≥ 8.7, so fine in practice).

---

## 11. Open questions / things the ISA does not specify

1. **Where multicast fan-out happens.** The ISA only says data and mbarrier signal are "multicast to the same CTA-relative offset" in each destination CTA. Whether the L2 read happens once (fan-out in the L2/crossbar) or per-CTA, and whether the delivered bytes count once per destination mbarrier (they do: each destination mbarrier gets its own complete-tx of `size` bytes, so every destination must `expect_tx(size)`) is implied but not stated. Latency/bandwidth cost of multicast vs. N unicast loads: measure.
2. **Ordering of DSMEM stores from different SMs.** Generic `st.shared::cluster` from two CTAs to the same remote location are ordinary weak/strong ops under the memory model; the ISA gives no statement about the interconnect (SM-to-SM network) ordering, write combining, or whether remote stores are posted. `st.async` completion is only observable via the remote mbarrier.
3. **`st.async` "same CTA as the executing thread" wording** (section 3.3) vs. `red.async` "different from the CTA of the executing thread": the ISA text is inconsistent; whether `st.async` to one's own CTA is defined is open (ptxas accepts it syntactically). CUTLASS only uses it remotely.
4. **Cross-proxy WAR on smem buffer reuse** (section 1.2 item 3): whether a `fence.proxy.async.shared::cta` is required between the last generic-proxy read of a stage and the next TMA load into it, or whether the mbarrier release/acquire chain suffices. The ISA's literal rule says a proxy fence is needed; production code (CUTLASS pipelines) does not emit one for loads.
5. **Does `cp.async.bulk.prefetch{.tensor}` join a bulk async-group?** The commit_group text includes `.prefetch` in the batched set but the prefetch instruction has no `.completion_mechanism`; what `wait_group` means for a prefetch (and whether `wait_group.read` returns after the L2 fill) is unspecified.
6. **`ctaMask` bit ↔ CTA identity**: ISA says "%ctaid of the destination CTA"; cluster rank (`%cluster_ctarank`) is the only consistent reading for a 16-bit mask.
7. **`cp.reduce.async.bulk` global → shared::cluster** appears in the ISA's completion-mechanism table but not in the syntax; ptxas rejects it. Treat as a documentation error.
8. **Which mbarrier scope is needed for TMA loads consumed by another CTA.** complete-tx is "release at .cluster scope"; the ISA does not say whether a `.acquire.cta` wait in the *destination* CTA (the normal multicast pattern: issuer in CTA A, data+signal land in CTA B, B waits with `.acquire.cta`) is formally sufficient. CUTLASS uses `mbarrier.try_wait.parity.shared::cta.b64` (default `.acquire.cta`) for this and it works in practice; a strictly model-conformant choice is `.acquire.cluster`.
9. **Timer resolution**: `%globaltimer` has no stated resolution/update rate; `%clock64` has no stated clock domain, cross-SM relation, or behavior under clock throttling.
10. **DSMEM capacity/addressing**: no statement about the size of the `.shared::cluster` window, how ranks map into it, or the maximum distance/number of hops; `%smid` explicitly non-contiguous and not stable, so SM↔GPC placement of a cluster's CTAs must be inferred empirically (e.g., from `%smid` sampled once per CTA).
11. **mbarrier `state` encoding** ("implementation specific") and what `mbarrier.pending_count` reports after an `expect_tx`-only phase.
12. **Async-group completion granularity**: `wait_group N` waits on groups "in the order they were committed"; the ISA does not say whether operations *within* a group complete in issue order (explicitly "no ordering").
13. **Interaction of `.L2::cache_hint` policies with TMA** (whether the policy applies per 16-B sector, per line, or per box) and of `discard`/`applypriority` with lines fetched by the async proxy — unspecified ("hint ... may not always be respected").
14. **Non-portable cluster size 16**: the ISA/driver docs only guarantee 8; 16 is SKU-specific and must be probed with `cudaOccupancyMaxPotentialClusterSize` after setting the non-portable attribute.
