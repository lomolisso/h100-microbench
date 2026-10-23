# Taxonomy of memory accesses on a single H100: proxy × path × completion mechanism

This file is the authoritative list of **every hardware-supported way to move or reduce data** between the memory levels of one H100, with the instruction that performs it, the proxy that executes it, how its completion is observed, and the ordering guarantees attached to that completion. The microbenchmarks measure latency and bandwidth for each row; `docs/experiments/18__access_type_summary.md` collects them into one table.

Sources: PTX ISA (memory-consistency model ch. 8; §9.7.9.25 asynchronous copies; §9.7.13 synchronization) quoted verbatim where marked ">", CCCL `cuda::ptx` generated tables (ISA/SM gates), ptxas 12.9 validation (which forms assemble for sm_90/sm_90a).

## 1. Proxies and completion mechanisms (definitions)

- **Generic proxy**: ordinary `ld`, `st`, `atom`, `red` (any state space, including `.shared::cluster`), `st.async`, `red.async`, and the Ampere-style `cp.async`. Executed by the SM's LSU; completion = instruction retirement (loads: scoreboard; stores: posted) or, for `st.async`/`red.async`, an mbarrier `complete_tx` in the *destination* CTA.
- **Async proxy**: `cp.async.bulk`, `cp.reduce.async.bulk`, their `.tensor` forms, and `cp.async.bulk.prefetch{.tensor}` (also `wgmma`). Executed by the TMA unit of the issuing SM. > "The completion of a cp{.reduce}.async.bulk operation is followed by an implicit generic-async proxy fence. So the result of the asynchronous operation is made visible to the generic proxy as soon as its completion is observed." The mbarrier operand of these instructions is accessed through the generic proxy; the tensor map through the **tensormap proxy**.
- Two completion mechanisms for async-proxy operations:
  1. **mbarrier-based** (`.mbarrier::complete_tx::bytes`): when the operation finishes, the hardware performs `complete_tx(bytes)` on the named mbarrier (which must live in the *destination* CTA's `.shared::cta`); the complete-tx "has `.release` semantics at the `.cluster` scope" but "is ordered only with respect to the memory operations performed by the same asynchronous instruction" (it does not publish the issuer's earlier generic writes). Consumers wait with `mbarrier.try_wait{.parity}[.acquire.{cta,cluster}]`.
  2. **bulk async-group** (`.bulk_group`): the issuing thread batches operations with `cp.async.bulk.commit_group` and waits with `cp.async.bulk.wait_group{.read} N`; only the issuing thread can wait; `.read` = source smem may be reused; without `.read` = destination writes are visible to the issuing thread. Groups complete in commit order; no ordering inside a group.
- Cross-proxy rule: > "Accessing the same memory location across multiple proxies needs a cross-proxy fence." → `fence.proxy.async{.shared::cta|.shared::cluster|.global}` between generic-proxy **writes** to smem and an async-proxy **read** of that smem (bulk/TMA store, bulk reduce, TMA put). Not needed after an observed async-proxy write (implicit fence). Reuse of a buffer by the next TMA load after generic reads is formally a cross-proxy WAR (open question; CUTLASS does not fence it).

## 2. The master table

Legend: G = global memory (L2/HBM), S = own CTA shared memory (`.shared::cta`), R = a peer CTA's shared memory in the same cluster (`.shared::cluster`, address from `mapa`), L2 = cache only. "sm_90a" = requires `-arch=sm_90a` for full performance (assembles on sm_90 with a ptxas advisory). All bulk forms: size multiple of 16 B, 16-B aligned addresses. Each row names the experiment that measures it.

| # | Path | Instruction (exact form) | Proxy | Completion observed by | Ordering of completion | Experiment |
|---|---|---|---|---|---|---|
| 1 | S → reg (local smem load) | `ld.shared{::cta}.T` (`LDS`) | generic | scoreboard (dependent use) | program order | 04 |
| 2 | reg → S (local smem store) | `st.shared{::cta}.T` (`STS`) | generic | posted; `bar.sync`/fence for other threads | — | 05 |
| 3 | S via DSMEM window, own rank | `ld/st.shared::cluster.T` with `mapa(addr, own_rank)` (`LD.E/ST.E`) | generic | as 1/2 (≈ +4 cycles) | — | 09 |
| 4 | R → reg (remote smem load) | `ld{.weak|.volatile|.relaxed.cluster|.acquire.cluster}.shared::cluster{.v2,.v4}.T` | generic | scoreboard; round trip over the SM-to-SM network | `.acquire.cluster` forms an acquire pattern; weak otherwise | 09, 10, 11 |
| 5 | reg → R (remote smem store) | `st{.weak|.volatile|.relaxed.cluster|.release.cluster}.shared::cluster.T` | generic | posted; visible to owner after fabric delivery; publish with `st.release.cluster` + `ld.acquire.cluster` flag, or `barrier.cluster.arrive.release/wait.acquire` | release/acquire at `.cluster` scope | 09 (visibility), 11 |
| 6 | reg → R with hardware completion signal | `st.async.weak.shared::cluster.mbarrier::complete_tx::bytes{.v2,.v4}.T [a], b, [mbar]` (`STAS`); requires `%cluster_nctarank > 1` | generic (data) | `complete_tx(bytes)` on the **destination CTA's** mbarrier; consumer waits `try_wait.parity.acquire.cluster` | complete-tx = release.cluster, ordered only with the store itself | 09, 12 |
| 7 | reg ⊕ R (remote atomic with return) | `atom{.sem}{.scope}.shared::cluster.op.T d, [a], b` (`ATOM.E`); ops and/or/xor/cas/exch/add/inc/dec/min/max; 32/64-bit int, f32, f64, f16/bf16 add | generic | return value (scoreboard) | `.acq_rel.cluster` etc. | 12 |
| 8 | reg ⊕ R without return | `red{.sem}{.scope}.shared::cluster.op.T [a], b` (`RED`) | generic | posted | `.release.cluster` available | 12 |
| 9 | reg ⊕ R with hardware completion | `red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.op.T [a], b, [mbar]` (`REDAS`); add(u32,s32,u64) min/max(u32,s32) inc/dec(u32) and/or/xor(b32); target CTA ≠ issuer | generic | `complete_tx(bytes)` on the destination CTA's mbarrier | relaxed op; complete-tx release.cluster | 12 |
| 10 | G → S (bulk, own CTA) | `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes{.L2::cache_hint} [dst], [src], size, [mbar]` (`UBLKCP.S.G`); PTX 8.6 also `.shared::cta` dst | async | mbarrier in own CTA (`expect_tx(size)` + `try_wait`) | implicit proxy fence on completion | 06, 16 |
| 11 | G → R (bulk, peer CTA) | same instruction with `dst`/`mbar` mapped by `mapa` to a peer rank | async | mbarrier in the **peer** CTA | as 10 | 16 |
| 12 | G → {R_i} multicast (bulk) | `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [dst], [src], size, [mbar], ctaMask` (16-bit mask of ranks; same CTA-relative offsets for data and mbarrier; each destination gets its own `complete_tx(size)`) — sm_90a | async | one mbarrier per destination CTA | as 10 | 16 |
| 13 | G tensor → S/R (TMA tile/im2col, 1–5D) | `cp.async.bulk.tensor.{1..5}d.shared::cluster.global{.tile|.im2col}.mbarrier::complete_tx::bytes{.L2::cache_hint} [dst], [tmap, {coords}], [mbar]{, {im2colOffs}}` (`UTMALDG`) | async (+ tensormap proxy for `tmap`) | mbarrier in destination CTA | as 10 | 06, 16 |
| 14 | G tensor → {R_i} multicast | row 13 + `.multicast::cluster ... , ctaMask` — sm_90a | async | one mbarrier per destination | as 10 | 16 |
| 15 | S → R (DSMEM push, bulk) | `cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [dst], [src], size, [mbar]` (`UBLKCP.S.S`); destination must be a **different** CTA | async | mbarrier in the destination CTA | as 10; source smem must be fenced with `fence.proxy.async.shared::cta` if written by generic stores | 11, 16 |
| 16 | S ⊕→ R (reduce-push, bulk) | `cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.op.T [dst], [src], size, [mbar]` (`UBLKRED`); add(u32,s32,u64) min/max(u32,s32) inc/dec(u32) and/or/xor(b32) — **no floating point to DSMEM** | async | mbarrier in destination CTA | each element op `.relaxed.gpu`; complete-tx release.cluster | 16, 17 |
| 17 | S → G (bulk store) | `cp.async.bulk.global.shared::cta.bulk_group{.L2::cache_hint} [dst], [src], size` (`UBLKCP.G.S`) | async | `commit_group` / `wait_group{.read}` by the issuing thread; other agents need the issuer's `wait_group 0` + a release | — | 16 |
| 18 | S ⊕→ G (bulk reduce to global) | `cp.reduce.async.bulk.global.shared::cta.bulk_group{.L2::cache_hint}.op.T [dst], [src], size` (`UBLKRED.G.S`); add(u32,s32,u64,f32,f64,f16/bf16 `.add.noftz`) min/max(u32,s32,u64,s64,f16,bf16) inc/dec(u32) and/or/xor(b32,b64); reduction performed **at L2** (atomic units), each op `.relaxed.gpu` | async | bulk group | — | 16, 17 |
| 19 | S tensor → G (TMA store) | `cp.async.bulk.tensor.{1..5}d.global.shared::cta{.tile|.im2col_no_offs}.bulk_group{.L2::cache_hint} [tmap, {coords}], [src]` (`UTMASTG`) | async | bulk group | — | 16 |
| 20 | S tensor ⊕→ G (TMA reduce-store) | `cp.reduce.async.bulk.tensor.{1..5}d.global.shared::cta.op{.tile|.im2col_no_offs}.bulk_group [tmap, {coords}], [src]` (`UTMAREDG`); element type from the tensor map | async | bulk group | — | 16, 17 |
| 21 | G → L2 prefetch (bulk / tensor) | `cp.async.bulk.prefetch.L2.global [src], size` (`UBLKPF.L2`); `cp.async.bulk.prefetch.tensor.{1..5}d.L2.global{.tile|.im2col} [tmap, {coords}]` (`UTMAPF.L2`) | async | none (weak op; wait_group semantics unspecified) | — | 06 |
| 22 | G → S (non-bulk `cp.async`, per thread, 4/8/16 B) | `cp.async.{ca,cg}.shared.global [dst], [src], 16{, src-size}` (`LDGSTS`) + `cp.async.commit_group` / `cp.async.wait_group N` or `cp.async.mbarrier.arrive{.noinc}` | generic (async to the thread) | per-thread groups or mbarrier arrive | — | 16 (reference) |
| 23 | G → reg through L1 / bypassing L1 / refetch | `ld.global{.ca|.cg|.cs|.lu|.cv}`, `ld.global.nc`, `ld.global.L1::no_allocate`, `.L2::{64B,128B,256B}` prefetch hints, `.L2::cache_hint` with `createpolicy` | generic | scoreboard | `.relaxed/.acquire.{cta,cluster,gpu,sys}` variants cannot carry cache ops | 04, 05, 08, 14 |
| 24 | reg → G | `st.global{.wb|.cg|.cs|.wt}`, `.release.scope` | generic | posted | — | 05 |
| 25 | reg ⊕ G | `atom/red.global.op.T` (executed at L2), `.scope` cta/cluster/gpu/sys | generic | return value / posted | — | 12 (reference) |
| 26 | L2 line management | `prefetch.global.{L1,L2,L2::evict_last}`, `discard.global.L2 [a],128`, `applypriority.global.L2::evict_normal [a],128`, `createpolicy.{fractional,range}` + `.L2::cache_hint` operands on `ld/st/atom/red/cp.async.bulk*` | generic | — | hints only | 08, 13 |

Forms that are **not** available on sm_90 (do not plan experiments around them): `cp.async.bulk` global→global or `shared::cta`→`shared::cta`; `cp.reduce.async.bulk` global→`shared::cluster` (listed in an ISA table but rejected by ptxas); `.cp_mask`, `.tile::gather4/scatter4`, `.im2col::w`, `.cta_group::2` (sm_100+); `ld/st.L2::evict_*` without 256-bit vectors (sm_100+); floating-point `cp.reduce.async.bulk` or `red.async` into DSMEM; waiting on a remote CTA's mbarrier (only `arrive`, `arrive.expect_tx`, `arrive_drop`, `expect_tx`, `complete_tx` are allowed remotely, with sink `_`).

## 3. Hardware collectives (the only ones in scope)

| Collective | Hardware primitive(s) | What the hardware does | Software composition needed |
|---|---|---|---|
| Broadcast (1 → N smems) from global | row 12 / 14 (`.multicast::cluster`) | one L2 read, fan-out in the GPC crossbar / LRC to ≤16 CTAs, one `complete_tx` per destination [PAT] | none |
| Broadcast (1 → N) from one CTA's smem | N × row 15 (bulk put) or N × row 6 (`st.async`) | no hardware fan-out for smem sources (`.multicast` exists only with a `.global` source) | issuer loops over destinations |
| Reduce (N → 1) into one CTA's smem | N × row 16 (bulk reduce-push, integer only) or N × row 9 (`red.async`) or N × row 7/8 (remote atomics) | element-wise atomic reduction performed in the owner SM's shared-memory atomic unit; completion counted in bytes on the owner's mbarrier | the owner `expect_tx(N·bytes)` once; nothing else |
| Reduce (N → L2/global) | N × row 18 / 20 (bulk reduce to global, incl. fp) | reduction at the L2 atomic units | consumer waits for the issuers' bulk groups + a global release/acquire |
| Gather / All-gather | rows 15/6 from every CTA to every CTA | none beyond point-to-point | N(N−1) puts, one mbarrier per destination with `expect_tx((N−1)·bytes)` |
| All-reduce | reduce-scatter (row 16 to owner of each slice) + all-gather (row 15), or N reduce-pushes to every CTA (replicated) | none beyond point-to-point reductions | see `docs/experiments/17__dsmem_collectives.md` |
| All-to-all | N(N−1) row 15/6 puts | none | one mbarrier per destination |

## 4. Per-row measurement recipe (what "latency" and "bandwidth" mean for each row)

- **Latency, generic loads (rows 1, 3, 4, 23):** dependent pointer chase, 1 thread, `%clock64`, minus loop overhead from `01`.
- **Latency, generic stores (rows 2, 5, 24):** not directly observable; measure (a) issue cost (back-to-back stores per cycle) and (b) **visibility latency** = time until a consumer observes the stored value (flag ping-pong, `09` K2).
- **Latency, async-proxy transfers (rows 6, 9–21):** time from issue (`%clock64` before the instruction) to the first successful `mbarrier.try_wait` (or `wait_group 0` return) on the consumer, minus the measured polling granularity (`07`), using 16 B payloads for the latency chase (TMA p-chase: payload = next address) and single transfers for stores.
- **Bandwidth:** bytes in flight per SM × SMs, affine fit of time vs payload (`11`), reported as B/cycle/SM and TB/s GPU-wide; completion overhead reported separately as the intercept.
- **Completion overhead:** for rows 10–20 measure both mechanisms where the ISA allows (G→S: mbarrier only; S→G: bulk_group only; S→R: mbarrier only) and the cost of the consumer's wait primitive in isolation (`07`).
