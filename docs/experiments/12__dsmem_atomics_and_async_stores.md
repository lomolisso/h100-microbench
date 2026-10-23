# 12 — DSMEM atomics, `red.async`, `st.async`, and store visibility (generic-proxy reductions into remote smem)

> **Device configuration:** runs under [`docs/context/10__device_config.md`](../context/10__device_config.md): H100 NVL (GPU 2), 132 SMs, 60 MiB L2, HBM3 3.94 TB/s, SM clock locked at 1635 MHz, power limit 400 W, ECC on, CUDA 12.9.1 `sm_90a`. Reference numbers quoted in this spec come from **other chips and configurations** (H800 PCIe, H100 PCIe, H100 SXM, H200, RTX 5090; see `10` §4). They are kept unchanged for the agent's sanity checks only, never cited in the notebook and not expectations for this device.

## Goal
Latency and throughput of rows 6–9 of the taxonomy (`atom.shared::cluster`, `red.shared::cluster`, `red.async`, `st.async`) — the primitives on which hardware-assisted DSMEM reductions rest — under no contention, same-address contention from N CTAs, and bank-spread patterns; compared with local smem atomics and global atomics (at L2). No published numbers exist for these [lit survey] — these are new measurements.

## Background
`docs/context/01__clusters_and_dsmem.md` §4; `docs/context/02__memory_access_taxonomy.md` rows 6–9, 25; `docs/context/04__mbarrier.md` §4.3. Store-visibility reference 625 cycles [P5]; remote store issue rate ≈ 21 B/cycle [P4].

## Kernels (CS 2 by default; CS 4/8/16 for contention; 1 CTA per SM; addresses via `mapa`)

### K1 — dependent atomic latency
One thread, 4096 dependent `atom.shared::cluster.add.u32` where the returned value feeds the next address (`a = base + (old & mask)*4`): remote vs local (`atom.shared.add.u32`, `ATOMS`) vs global (`atom.global.add.u32`, L2-resident line). Variants: `.exch`, `.cas`, `.add.u64`, `.add.f32`, `.min.s32`; semantics `.relaxed.cluster` / `.acq_rel.cluster` / `.release.cluster` on `red`.

### K2 — fire-and-forget throughput
1024 threads issue `red.shared::cluster.add.u32` (a) each to a distinct word (conflict-free, 32-bank spread), (b) 32 threads per word, (c) all to one word; 4096 iterations; also `atom` (with return) versions; also `red.async.add.u32` with the destination's mbarrier expecting 4·(ops) bytes and `st.async.b32`/`.v4.b32` (pure stores with hardware completion). Per-SM B/cycle and ops/cycle; completion time for the `.async` forms (issue → destination's `try_wait` returns).

### K3 — contention from N sources
N ∈ {1, 2, 4, 8, 15} CTAs (CS 16) hammer rank 0's smem: same word / same bank / spread; measure per-source ops/cycle and total ops/cycle at the destination, with the destination idle or running local `atom.shared` on the same words (local vs remote arbitration).

### K4 — visibility latency matrix (extends `09` K2)
One-way latency for: plain `st.shared::cluster`; `st.release.cluster`; `st.async` (mbarrier); `red.async` (mbarrier); `red.shared::cluster`; `atom.shared::cluster` (return = acknowledgement); measured as round-trip/2 with `%clock64` and one-way with `%globaltimer`; per pair across one GPC (reuse `10` placement) to see whether atomics/st.async show the same flatness as loads.

### K5 — width independence of remote stores
`st.shared::cluster.{u8,u16,u32,u64,v2.u32,v4.u32}` streaming 256 KB from 32 threads (reproduce [P2] Table II: time halves per width doubling) and the same for `st.async.{b32,b64,v2.b32,v4.b32}`.

## CSV columns
`12__dsmem_atomics__latency.csv`: `op, sem, target ∈ {local, remote, global}, cs, smid_src, smid_dst, cycles_per_op`.
`12__dsmem_atomics__throughput.csv`: `op, pattern, threads, cs, n_sources, dest_busy, ops, cycles, ops_per_cycle, bpc, completion_cycles`.
`12__dsmem_atomics__visibility.csv`: `primitive, smid_src, smid_dst, trial, one_way_ns, round_trip_cycles`.
`12__dsmem_atomics__width.csv`: `primitive, width_bytes, bytes, cycles, bpc`.

## Figures
1. Dependent latency bars: local / remote / global × op.
2. Throughput vs threads for red / atom / red.async / st.async, conflict-free vs same-word.
3. Destination ops/cycle vs number of contending sources (fairness: per-source share).
4. Visibility latency per primitive (box plots) with the load round-trip/2 reference from `09`.
5. Width-independence plot (time vs width, log-log).

## Sanity checks (agent-only)
Expected ranges for judging plausibility during result analysis. They are never presented as pass/fail and never quoted in the notebook. If a value falls outside, investigate (placement, clocks, SASS, cache state, ncu replay) and re-run. A value that survives is a finding about this device; describe it on its own terms.
Remote atomic latency ≈ remote load latency + atomic-unit time (expect 200–300 cycles) — report; `red.async`/`st.async` completion within 2× of plain store visibility; width independence reproduced (slope ≈ −1 on log-log).
- Figure 5 vs P2 Table II (time halves per width doubling).

## NVIDIA tools
`ncu --metrics sm__sass_inst_executed_op_dshared_atom.sum,sm__sass_inst_executed_op_dshared_redas.sum,sm__sass_inst_executed_op_dshared_stas.sum,l1tex__t_requests_pipe_lsu_mem_dshared_op_atom.sum,l1tex__t_requests_pipe_lsu_mem_dshared_op_redas.sum,l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_atom.sum,l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_redas.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_atom.sum,l1tex__t_set_conflicts_pipe_lsu_mem_dshared_op_atom.sum --print-metric-instances values --clock-control none`; SASS: `ATOM.E` / `RED.E` (DSMEM) vs `ATOMS` (local) vs `ATOMG`/`REDG` (global), `STAS`, `REDAS`.
