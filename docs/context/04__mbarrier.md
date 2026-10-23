# mbarrier — the asynchronous transaction barrier

Sources: PTX ISA §9.7.13.15 (verbatim where quoted), CUDA `cuda::barrier` docs (CCCL), CUTLASS `barrier.h`, NVIDIA patent US 2023/0289242 A1 ("Hardware accelerated synchronization with asynchronous transaction support"), ptxas 12.9 validation. Measured costs: `08__prior_work_reference_numbers.md` (P5).

## 1. Object and state
- 64-bit, 8-byte aligned object in **`.shared::cta`** of the owning CTA. State = current phase (parity), pending arrival count, expected arrival count (for the next phase), and **tx-count** (Hopper+): "tracks the number of asynchronous transactions, in units specified by the asynchronous memory operation [bytes], that are outstanding and yet to be complete."
- Ranges: expected/pending arrivals 1…2^20−1 / 0…2^20−1; tx-count −(2^20−1)…2^20−1 (so a `complete_tx` may precede its `expect_tx` inside a phase).
- Phase completes when pending arrivals == 0 **and** tx-count == 0; then phase parity flips, pending ← expected. "For each phase ... at least one test_wait or try_wait operation must be performed which returns True ... before an arrive-on operation in the subsequent phase."
- Lifecycle: `mbarrier.init` (one thread) → publish → use → `mbarrier.inval` before the memory is reused for anything else. Any other op on an uninitialised object, or any non-mbarrier access to an initialised one, is undefined.
- Hardware: the barrier is implemented by a synchronisation circuit at the SM operating on the shared-memory word; async engines (TMA) and remote CTAs update it through the SM-to-SM network ("receives arrival updates from CTAs of the same CGA running on different SMs"; "zero-balanced counter") [PAT].

## 2. Operations (sm_90 syntax, defaults, what is allowed remotely)

| Operation | Syntax (local form) | Remote (`.shared::cluster`) form | Default sem/scope |
|---|---|---|---|
| init | `mbarrier.init.shared::cta.b64 [bar], count;` | not allowed | — |
| inval | `mbarrier.inval.shared::cta.b64 [bar];` | not allowed | — |
| arrive | `mbarrier.arrive{.sem.scope}.shared::cta.b64 state, [bar]{, count};` | `mbarrier.arrive{.sem.scope}.shared::cluster.b64 _, [rbar]{, count};` (sink `_` mandatory) | `.release.cta` |
| arrive + expect | `mbarrier.arrive.expect_tx{.sem.scope}.shared::cta.b64 state, [bar], tx;` | `...shared::cluster.b64 _, [rbar], tx;` | `.release.cta` |
| expect only | `mbarrier.expect_tx{.relaxed.scope}.shared::cta.b64 [bar], tx;` | `.shared::cluster` allowed | `.relaxed.cta` |
| complete (simulate async completion) | `mbarrier.complete_tx{.relaxed.scope}.shared::cta.b64 [bar], tx;` | `.shared::cluster` allowed | `.relaxed.cta` |
| arrive and leave | `mbarrier.arrive_drop{.sem.scope}.shared::cta.b64 state, [bar]{, count};` (+ `.expect_tx`) | `.shared::cluster` allowed (sink) | `.release.cta` |
| arrive without completing | `mbarrier.arrive.noComplete.shared::cta.b64 state, [bar], count;` | not allowed | — |
| test (non-blocking) | `mbarrier.test_wait{.sem.scope}.shared::cta.b64 p, [bar], state;` / `.parity ... p, [bar], parity;` | **not allowed** | `.acquire.cta` |
| try (may suspend) | `mbarrier.try_wait{.sem.scope}.shared::cta.b64 p, [bar], state{, hintNs};` / `.parity ... p, [bar], parity{, hintNs};` | **not allowed** | `.acquire.cta` |
| pending count | `mbarrier.pending_count.b64 c, state;` (state from `.noComplete` arrives only) | — | — |
| legacy cp.async hook | `cp.async.mbarrier.arrive{.noinc}.shared::cta.b64 [bar];` | — | — |

`.sem ∈ {.release, .relaxed}` for arrive-type ops, `{.acquire, .relaxed}` for waits; `.scope ∈ {.cta, .cluster}`; `.sem` and `.scope` must be given together. **Only the owning CTA can wait**; peers can only arrive/expect/complete. State values are opaque; `.parity` waits use the phase parity (0 for the first phase) and are the only form usable by threads that did not arrive.

Acquire guarantee when a wait returns true (verbatim list, abridged): all memory accesses *except async operations* before the participating threads' release arrives are visible; all `cp.async` tracked by `cp.async.mbarrier.arrive` are visible; **all `cp.async.bulk` operations using the same mbarrier object requested before the release arrive are visible**. Async operations tracked by a *different* barrier are not covered. The complete-tx of a bulk op is "release at .cluster scope" but ordered only with that op's own memory accesses.

## 3. Publication of a barrier to other CTAs
```
mbarrier.init.shared::cta.b64 [bar], N;         // one thread
fence.mbarrier_init.release.cluster;            // op-restricted release (CUTLASS fence_barrier_init)
barrier.cluster.arrive.relaxed.aligned;  barrier.cluster.wait.aligned;   // or cluster.sync()
// now peers may: mapa → mbarrier.arrive.release.cluster.shared::cluster.b64 _, [rbar];
```
Before reusing the memory: `mbarrier.inval`. A CTA must not exit while peers (or TMA engines working on their behalf) can still arrive on its barrier → trailing `cluster.sync()`.

## 4. Templates

```cpp
// 4.1 Producer/consumer in one CTA, TMA load, parity wait (one elected producer)
__shared__ alignas(8) uint64_t full_bar;
if (elect_one()) { mbar_init(smem_u32(&full_bar), 1); }
asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); __syncthreads();
for (int stage = 0, parity = 0; stage < N; ++stage, parity ^= 1) {
  if (elect_one()) { mbar_arrive_expect_tx(smem_u32(&full_bar), BYTES); bulk_g2s(dst, src + stage*BYTES, BYTES, smem_u32(&full_bar)); }
  mbar_wait_parity(smem_u32(&full_bar), parity);      // everyone
  /* consume */ __syncthreads();                       // or an "empty" barrier for pipelining
}

// 4.2 Cluster-wide arrive on rank 0's barrier (ISA example): every thread of the cluster arrives, rank 0 waits
if (rank == 0 && threadIdx.x == 0) mbar_init(smem_u32(&bar), cluster_threads);
asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); cluster.sync();
mbar_remote_arrive(mapa_smem(smem_u32(&bar), 0));     // mbarrier.arrive.release.cluster.shared::cluster.b64 _, [rbar]
if (rank == 0) mbar_wait_parity_acq_cluster(smem_u32(&bar), 0);

// 4.3 st.async / red.async producer → consumer (2 CTAs), hardware-signalled
// consumer (rank 1): init(bar,1); arrive_expect_tx(bar, 4*count) ; fence.mbarrier_init ; cluster.sync()
// producer (rank 0): for i: st_async_b32(mapa(dst)+4*i, v, mapa(bar)); cluster.sync() at end
// consumer: while(!mbar_try_wait_parity_acq_cluster(bar, 0, 0)); read data

// 4.4 Polling costs (expected, verify in 07): test_wait / try_wait(no hint) predicate ready ≈ 42 cycles [LIT];
//     try_wait with suspendTimeHint ≈ 150 cycles when complete, sleeps up to the hint otherwise; observed completion
//     is quantised by the polling loop (≈147-cycle steps with yield/sleep loops) [LIT].
```

## 5. `cuda::barrier` (C++) mapping
`cuda::barrier<cuda::thread_scope_block>` in shared memory = an mbarrier; `init(&bar, n)`; `bar.arrive()` → `mbarrier.arrive`; `bar.wait(token)` → `test_wait` loop; `cuda::device::barrier_arrive_tx(bar, 1, bytes)` → `mbarrier.arrive.expect_tx`; `cuda::device::barrier_expect_tx` → `mbarrier.expect_tx`; `cuda::device::memcpy_async_tx` → `cp.async.bulk` G→S; `bar.wait_parity(p)` (CCCL ≥ 3.2) or `cuda::ptx::mbarrier_try_wait_parity(cuda::device::barrier_native_handle(bar), p)`. `cuda::barrier` cannot have cluster scope — use `cuda::ptx::mbarrier_*` with `scope_cluster`/`space_cluster` for remote arrives. In benchmarks prefer the raw PTX wrappers so that the exact instruction and polling loop are under control.
