# 12 — Atomics on every path (local SMEM, DSMEM window, remote DSMEM, L2 near/far, HBM, host memory), `red.async`, `st.async`, and store visibility

> **Device configuration:** runs under [`docs/context/10__device_config.md`](../context/10__device_config.md): H100 NVL (GPU 2), 132 SMs, 60 MiB L2, HBM3 3.94 TB/s, SM clock locked at 1635 MHz, power limit 400 W, ECC on, CUDA 12.9.1 `sm_90a`. Reference numbers quoted in this spec come from **other chips and configurations** (H800 PCIe, H100 PCIe, H100 SXM, H200, RTX 5090; see `10` §4). They are kept unchanged for the agent's sanity checks only, never cited in the notebook and not expectations for this device.

## Goal
**Latency and throughput of atomic memory operations on every path an atomic can take on this GPU**, plus the DSMEM signalling primitives built on them:

| target | instruction (generic proxy) | where the atomic executes | taxonomy row |
|---|---|---|---|
| T1 own CTA's smem | `atom/red.shared{::cta}` (`ATOMS`) | own SM's shared-memory atomic unit | 27 |
| T2 own smem through the DSMEM window | `atom/red.shared::cluster` at `mapa(addr, own rank)` (`ATOM.E`/`RED.E`) | own SM, via the DSMEM interface | 28 |
| T3 a peer CTA's smem (DSMEM) | `atom/red.shared::cluster` at `mapa(addr, peer)`; `red.async` | the owner SM's atomic unit, over the SM-to-SM network | 7, 8, 9 |
| T4 global, L2 hit, line homed in the issuer's L2 partition | `atom/red.global` (`ATOMG`/`REDG`) | L2 atomic units | 25 |
| T5 global, L2 hit, line homed in the other partition | same | L2 (which partition: to be measured) | 25 |
| T6 global, L2 miss (HBM), near- and far-homed lines | same | L2 after an HBM fill | 25 |
| T7 host memory (pinned, mapped), `.sys` scope | `atom/red.global` on a `cudaHostAlloc` pointer | over PCIe (reference point only) | 25 |

For each target: dependent latency, per-SM throughput, and (T3–T6) GPU-wide throughput, without contention and under same-word / same-line / hot-spot contention. The atomic numbers are compared with the plain load/store numbers of the same path from `04`, `05`, `09` and `11`.

Why this matters on this device (results of `04`/`05`):
- Every L2 line has an address-determined **home partition**. A *read* leaves a copy in the reader's partition, but a *store* always travels to the home partition, over a partition link that carries ≈ 1.11 TB/s per direction.
- Global atomics execute in the L2. If they behave like stores, half of all global atomics of a random buffer pay the far latency (04: near 270 / far 426 cycles for loads) and share that link. A single "L2-resident atomic" number would hide this split.
- Each GPC reaches the L2 through a port of ≈ 1.16 TB/s, and many SMs reading the same lines are served by request merging before the L2 tags. Whether same-address atomics from many SMs are merged or serialised is the hot-spot question below.

No published numbers exist for most of these [lit survey]; these are new measurements.

## Background
`docs/context/01__clusters_and_dsmem.md` §4; `docs/context/02__memory_access_taxonomy.md` rows 6–9, 25, 27, 28; `docs/context/04__mbarrier.md` §4.3; `docs/context/06__l2_nuca_and_physical_topology.md` §1.1. Store-visibility reference 625 cycles [P5]; remote store issue rate ≈ 21 B/cycle [P4]. `04`: L2 near/far latency, HBM near/far latency, line homing and copies. `05`: line-home labelling by timing (`experiments/05__memory_hierarchy_bandwidth/src/bw_l2.cu`, `label_homes`; homes are constant per 4 KiB and change within every 8 KiB), partition-link bandwidth, per-GPC L2 bandwidth, `CleanL2`, and the harness pitfalls (ptxas hoists loop-invariant global loads and plain `LDS`, merges identical global loads; 64-bit shared `atomicAdd` from every thread costs tens of µs per block).

Operations swept (each verified to assemble with ptxas 12.9 for `sm_90a`; drop and record any form that does not):
- integer: `add.u32`, `add.u64`, `inc.u32`, `min.s32`, `max.u32`, `and.b32`, `exch.b32`, `cas.b32`, `cas.b64`;
- floating point: `add.f32`, `add.f64`, `add.noftz.f16x2`, `add.noftz.bf16x2`;
- vector global atomics (`atom/red.global.v2/.v4.f32.add`, `.v2/.v4.f16x2`) if ptxas accepts them for `sm_90a`;
- semantics and scope: `.relaxed` (default), `.acq_rel`; `.cta`, `.cluster`, `.gpu`, `.sys` (global); `.cluster` (DSMEM);
- `atom` (with return) versus `red` (no return) for every target.

## Kernels (1 CTA per SM unless stated; clusters via `cudaLaunchKernelEx`, addresses via `mapa`; `%smid`, cluster id and rank recorded for every block)

### K0 — home labels for global targets
Reuse `05`'s labelling (flush, whole-line write from an SM of partition 0, timed read from an SM of partition 1, and the reverse) on the global buffers of K1–K3. Because homes are constant per 4 KiB, large buffers (T6) may be labelled per 4 KiB page from one line per page; verify on one chunk that every line of a page agrees. Lists of near- and far-homed lines per issuing partition drive K1–K3.

### K1 — dependent atomic latency (one thread, all targets)
One thread, `atom.<op>` where the returned value feeds the next address (`a = base + (old & mask)·stride`, chain built so that every hop reaches a new line for global targets and a new word for smem). Placement: claim-pinned one-warp block (`04`'s `claim_target_sm`) on the first SM of each GPC; for T3 the peer rank's SM is recorded and binned by (smid_src, smid_dst).
- T1, T2, T3: 4096 hops, slope between 4096 and 16 384 hops (as `04` K1); T3 for CS 2 (TPC mate), CS 4, CS 16 and per pair of the `10` placement.
- T4/T5: footprint 8 MiB of lines homed in the issuer's / the other partition (K0 lists), lines made L2-resident first. **Two passes**: pass 1 on lines last touched only by their home partition (flush, `CleanL2`, home touch as in `05`'s first-pass test), pass 2 immediately after (does an atomic leave a copy near the issuer, like a load, or always go home, like a store?).
- T6: footprint ≥ 256 MiB, flushed and cleaned L2, per-access timing (`%clock64` around each atomic, as `04` K3), ≥ 131 072 samples per launch; classes by 1-D K-means; near/far by the K0 home of each line.
- T7: 1024 hops on host-pinned memory, `.sys` scope (reference point).
- Ops × semantics × scopes as listed above; for T4–T6 also `ld.global.cg` and `st.global` + `ld` of the same chain in the same binary as same-clock references.

### K2 — per-SM throughput and intra-SM contention (one SM, then 132 SMs at once)
`T` ∈ {128, 256, 512, 1024} threads × ILP ∈ {1, 2, 4, 8} independent `red` (and `atom`) per thread and step (the ILP results are consumed before the next step, as in `05`), ≥ 4096 steps, `%clock64` per block. Address patterns:
- (a) distinct words, warp-coalesced (32 consecutive words per warp instruction; smem: conflict-free);
- (b) distinct words, one per 32-B sector / one per 128-B line (global: one sector or line per lane);
- (c) same bank, different words (smem only: 32-way bank conflict);
- (d) 32 lanes of a warp → one word (warp-level same address);
- (e) all threads of the block → one word.
Targets T1, T2, T4 (near list), T5 (far list), T6 (≥ 256 MiB, streaming), T3 (peer in CS 2). Report ops/cycle/SM and payload B/cycle/SM; for 132 SMs at once also the per-SM distribution.

### K3 — GPU-wide global atomic throughput (T4–T6)
2 × 132 blocks × 1024 threads, ILP 4, cudaEvents + per-block timers (`05`'s streaming structure; addresses from the K0 line lists so that "local" / "remote" refer to the line's home partition):
- (a) **footprint sweep** 1 MiB … 256 MiB, distinct words, patterns (a) and (b) of K2, `red` and `atom`, `add.u32`, `add.f32`, `add.u64`; variants mixed / local / remote (as in `05`). Compare with `05`'s `ld` and `st` curves at the same footprints.
- (b) **partition link**: remote-homed lines from growing GPC sets of one partition, then both partitions (`05` link mode): does the atomic rate saturate at the link bandwidth like stores?
- (c) **hot spots**: all SMs → 1 word; → 1 line (32 words); → N distinct lines with N ∈ {1, 2, 4, …, 1024} (how many L2 atomic units work in parallel; whether same-address requests from many SMs are merged before the L2); also versus the number of active SMs 1…132.
- (d) per GPC: one GPC alone (≈ 1.16 TB/s load port in `05`) for T4.

### K4 — DSMEM fire-and-forget throughput (T3)
1024 threads issue `red.shared::cluster.add.u32` (a) each to a distinct word (conflict-free, 32-bank spread), (b) 32 threads per word, (c) all to one word; 4096 iterations; also `atom` (with return) versions; also `red.async.add.u32` with the destination's mbarrier expecting 4·(ops) bytes and `st.async.b32`/`.v4.b32` (pure stores with hardware completion). Per-SM B/cycle and ops/cycle; completion time for the `.async` forms (issue → destination's `try_wait` returns). GPU-wide: every cluster of CS 2/4/8/16 runs the ring pattern at once (each rank atomics into the next rank), total ops/s.

### K5 — DSMEM contention from N sources (T3)
N ∈ {1, 2, 4, 8, 15} CTAs (CS 16) hammer rank 0's smem: same word / same bank / spread; measure per-source ops/cycle and total ops/cycle at the destination, with the destination idle or running local `atom.shared` on the same words (local vs remote arbitration).

### K6 — visibility latency matrix (extends `09` K2)
One-way latency for: plain `st.shared::cluster`; `st.release.cluster`; `st.async` (mbarrier); `red.async` (mbarrier); `red.shared::cluster`; `atom.shared::cluster` (return = acknowledgement); and for global flags `red.release.gpu.global` / `atom.acq_rel.gpu.global` between SMs of the same and of different L2 partitions (flag line homed near the consumer / near the producer). Measured as round-trip/2 with `%clock64` and one-way with `%globaltimer`; per pair across one GPC (reuse `10` placement) to see whether atomics/st.async show the same flatness as loads.

### K7 — width independence of remote stores
`st.shared::cluster.{u8,u16,u32,u64,v2.u32,v4.u32}` streaming 256 KB from 32 threads (reproduce [P2] Table II: time halves per width doubling) and the same for `st.async.{b32,b64,v2.b32,v4.b32}`.

Every configuration: 2 warm-up and ≥ 10 recorded launches (K1 per-access runs: ≥ 3); placement and clock checks and discards as in `05`; crash-safe CSVs (`09` §6). Dump SASS and check that every timed loop contains the intended `ATOMS` / `ATOM.E` / `RED.E` / `ATOMG` / `REDG` / `REDAS` / `STAS` and that no atomic was hoisted or merged (repeated identical addresses).

## CSV columns
`12__atomics__latency.csv`: `op, sem, scope, target ∈ {smem, smem_dsmem_if, dsmem, l2_near, l2_far, hbm, sysmem}, pass, cs, smid_src, smid_dst, gpc_src, hops, total_cycles, cycles_per_op`.
`12__atomics__fine.csv` (K1 per-access, T5/T6 and T4 pass 1/2): `op, target, smid, line_offset, home_partition, pass, hop_idx, cycles` (store exact histograms instead if the row count would reach GiB scale; state the choice in the plan).
`12__atomics__throughput_sm.csv`: `op, kind ∈ {atom, red}, target, pattern, threads, ilp, placement, block, smid, ops, cycles, ops_per_cycle, bpc`.
`12__atomics__throughput_gpu.csv`: `op, kind, pattern, variant ∈ {mixed, local, remote}, footprint_bytes, n_hot_lines, reader_gpc_mask, blocks, threads, ilp, ops, ms_event, gops, tbps_payload, sm_mhz_blocks, discard` (+ per-block file `12__atomics__throughput_gpu_blocks.csv` as in `05`).
`12__dsmem_atomics__throughput.csv`: `op, pattern, threads, cs, n_sources, dest_busy, ops, cycles, ops_per_cycle, bpc, completion_cycles`.
`12__dsmem_atomics__visibility.csv`: `primitive, scope, smid_src, smid_dst, trial, one_way_ns, round_trip_cycles`.
`12__dsmem_atomics__width.csv`: `primitive, width_bytes, bytes, cycles, bpc`.
`12__atomics__summary.csv`: one row per (target, op, kind) with median latency, per-SM and GPU-wide throughput (consumed by `18`).

## Figures
1. **Atomic latency ladder**: dependent latency per target (T1…T7) and op, with the plain load latency of the same path (`04`, `09`) as a marker next to each bar.
2. Per-access histograms of global atomics (L2 near/far, pass 1/2; HBM near/far) with the K-means classes.
3. Per-SM atomic throughput vs threads (lines per ILP), one panel per target and pattern; 128 B/cycle (`05`) dotted for smem.
4. GPU-wide global atomic throughput vs footprint (`red`/`atom`, u32/f32/u64, local/remote), `05`'s load and store curves dotted.
5. Hot spots: Gops/s vs number of distinct hot lines, and vs number of active SMs.
6. Remote-homed atomics vs number of GPCs (with `05`'s link bandwidth dotted).
7. DSMEM: throughput vs threads for red / atom / red.async / st.async, conflict-free vs same-word; destination ops/cycle vs contending sources; visibility latency per primitive; width independence (log-log).

## Sanity checks (agent-only)
Expected ranges for judging plausibility during result analysis. They are never presented as pass/fail and never quoted in the notebook. If a value falls outside, investigate (placement, clocks, SASS, cache state, ncu replay) and re-run. A value that survives is a finding about this device; describe it on its own terms.
- Remote atomic latency ≈ remote load latency + atomic-unit time (expect 200–300 cycles) — report; `red.async`/`st.async` completion within 2× of plain store visibility; width independence reproduced (slope ≈ −1 on log-log). Figure 7 width plot vs P2 Table II (time halves per width doubling).
- An atomic's latency should not be below the load latency of the same path (`04`/`09`); `red` throughput should not exceed the store throughput of the same path (`05`: 32 B/cycle/SM leaving an SM, 4.0 TB/s into L2, 1.11 TB/s per direction over the partition link) unless the counters show merging.
- Global atomics counted by ncu (`lts__t_sectors_op_atom`, `lts__t_sectors_op_red`) must match the issued operations; a same-word hot spot faster than one operation per L2 cycle per hot line implies merging before the L2 — confirm with the counters before reporting it.

## NVIDIA tools
- DSMEM: `ncu --metrics sm__sass_inst_executed_op_dshared_atom.sum,sm__sass_inst_executed_op_dshared_redas.sum,sm__sass_inst_executed_op_dshared_stas.sum,l1tex__t_requests_pipe_lsu_mem_dshared_op_atom.sum,l1tex__t_requests_pipe_lsu_mem_dshared_op_redas.sum,l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_atom.sum,l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_redas.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_atom.sum,l1tex__t_set_conflicts_pipe_lsu_mem_dshared_op_atom.sum --clock-control none` (rollups only: per-SM values are not exported on this setup, `07` §8; use `.max/.avg` of the byte metrics for the hot-spot factor and one-rank-active launches when one SM's own count is needed).
- Local smem atomics: `smsp__inst_executed_op_shared_atom.sum`, `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_atom.sum`, `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_atom.sum`.
- Global atomics (L2): `lts__t_sectors_op_atom.sum`, `lts__t_sectors_op_red.sum`, `lts__t_requests_srcunit_tex_op_atom.sum` / `_op_red.sum`, `lts__t_sectors_srcunit_ltcfabric.sum` (and its `_op_atom`/`_op_red` variants if listed), `lts__t_sector_hit_rate.pct`, `dram__bytes_{read,write}.sum`, `smsp__inst_executed_op_global_{atom,red}.sum`, `l1tex__t_requests_pipe_lsu_mem_global_op_{atom,red}.sum`, for the hot-spot runs the `.max` and `.avg` rollups of the `lts__` atomic counters (`.max == .sum` ⇔ one slice executes them; the slice identity, if needed, via the `13` oracle). Verify every name with `ncu --query-metrics` first and drop (and record) the ones not listed; profile an all-exit launch as the baseline (`05`); use `--replay-mode application --cache-control none` for L2-resident runs.
- SASS: `ATOM.E` / `RED.E` (DSMEM and DSMEM window) vs `ATOMS` (local) vs `ATOMG` / `REDG` (global), `STAS`, `REDAS`.
