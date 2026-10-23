# 07 — Cost of every synchronization primitive (block, cluster, mbarrier, fences, flags) vs cluster size and thread count

> **Device configuration:** runs under [`docs/context/10__device_config.md`](../context/10__device_config.md): H100 NVL (GPU 2), 132 SMs, 60 MiB L2, HBM3 3.94 TB/s, SM clock locked at 1635 MHz, power limit 400 W, ECC on, CUDA 12.9.1 `sm_90a`. Reference numbers quoted in this spec come from **other chips and configurations** (H800 PCIe, H100 PCIe, H100 SXM, H200, RTX 5090; see `10` §4). They are kept unchanged as intuition and reproduction targets, not as expectations for this device.

## Goal
Isolated latency (cycles) of each primitive in `docs/context/05__synchronization_primitives.md` §1 so that the DSMEM/TMA protocol costs in later experiments can be decomposed; identify which stall reason absorbs each wait; and quantify the cheapest correct way to signal "data ready" between two CTAs of a cluster.

## Background
References [P2]: `barrier.cluster.arrive` ≈ 1050 (1 thread/block) rising to ≈ 1200 at CS 16; `.wait` ≈ 45–87; `cluster.sync()` ≈ 1300; `__syncthreads()` 14; `grid.sync()` ≈ 2200; CREDIT: cluster.sync 851 [P4]; ClusterSim calibration arrive 610 / wait 60 [P3]; mbarrier predicate 42 / 150 with hint [P5]; store visibility 625 [P5].

## Kernels
All timed with `%clock64` around a single instruction (or a 64× unrolled sequence, divided), ≥ 1000 repetitions, median and p5/p95; block sizes {1 warp (32 threads, 1 active thread for cluster-level tests where meaningful), 128, 512, 1024}; CS ∈ {1, 2, 4, 8, 16} for cluster primitives.

### K1 — intra-block
`bar.sync 0` (`__syncthreads`), named `bar.sync 1, N` with N ∈ {32, 128, 512, 1024}, `bar.arrive`+`bar.sync` producer/consumer split (2 warps), `__syncwarp`, `fence.acq_rel.cta`, `fence.acq_rel.gpu`, `fence.sc.gpu`, `fence.acq_rel.cluster`, `fence.proxy.async.shared::cta`, `fence.proxy.async` (all spaces), `fence.mbarrier_init.release.cluster`, `__threadfence()`, `nanosleep 0/100/1000`.

### K2 — cluster barrier components
Per CS and block size: (a) `barrier.cluster.arrive.release.aligned` alone (all CTAs arrive simultaneously after a preceding `cluster.sync()`; time the arrive instruction); (b) `barrier.cluster.arrive.relaxed.aligned` alone (isolates the `MEMBAR.ALL.GPU; ERRBAR; CGAERRBAR` cost); (c) `barrier.cluster.wait.acquire.aligned` when all others have arrived ≥ 5000 cycles earlier (pure wait cost); (d) `wait` when the last arriver is delayed by D ∈ {0, 500, 2000} cycles (measures propagation: wait returns ≈ D + propagation); (e) full `cluster.sync()`; (f) split-phase: arrive, 2000 cycles of independent ALU work, wait — total vs sum. Also measure the propagation latency of the arrive itself: CTA A records `%globaltimer` at arrive, CTA B (already waiting) records `%globaltimer` when wait returns → one-way barrier propagation.

### K3 — mbarrier
Local: `mbarrier.init`, `arrive` (1 thread; 32 threads of one warp — coalesced arrive; 1024 threads), `arrive.expect_tx`, `expect_tx`, `complete_tx`, `test_wait` (complete / incomplete), `try_wait.parity` (complete; incomplete with hint 0/100/1000/10000 ns — measures actual suspension and wake-up), `arrive_drop`, `inval`. Remote (CS 2): `mbarrier.arrive.release.cluster.shared::cluster` issue cost, and the one-way propagation (rank 0 arrives at `%globaltimer` t0; rank 1's `try_wait.parity.acquire.cluster` returns at t1; also round-trip ping-pong via two barriers with `%clock64` on one SM). Compare `.release.cluster` vs `.relaxed.cluster` remote arrive.

### K4 — flag ping-pong matrix (2 CTAs, CS 2, same measurement loop for all)
Round-trip latency (`%clock64` on rank 0, 1000 iterations) for: (1) `st.shared::cluster` + `ld.volatile.shared::cluster` spin; (2) `st.release.cluster` + `ld.acquire.cluster`; (3) `st.relaxed.cluster` + `ld.relaxed.cluster` + `fence.acq_rel.cluster`; (4) `red.release.cluster.shared::cluster.add` + acquire load; (5) `atom.acq_rel.cluster.shared::cluster.exch`; (6) `st.async` + `mbarrier.try_wait.parity.acquire.cluster`; (7) `red.async` + mbarrier; (8) remote `mbarrier.arrive.release.cluster` + wait; (9) global-memory flag `st.release.gpu` / `ld.acquire.gpu` (L2 round trip, the "via GMEM" reference, expect ≈ 1100 cycles per store+load [P1]); (10) `cluster.sync()` as the signal.

### K5 — grid sync reference
`cg::this_grid().sync()` with 1, 8, 66, 132 blocks of 1024 threads (cooperative launch).

## Stall attribution
Run K2(e) and K3 `try_wait` loops under `ncu --section WarpStateStats,SourceCounters,InstructionStats --print-metric-instances details` and record which `smsp__pcsamp_warps_issue_stalled_*` reason dominates at the `UCGABAR_WAIT` and `SYNCS.PHASECHK` PCs.

## CSV columns
`07__sync__primitives.csv`: `kernel, primitive, cs, block_threads, variant, rep, cycles`.
`07__sync__propagation.csv`: `primitive, cs, smid_src, smid_dst, one_way_ns, round_trip_cycles`.
`07__sync__stalls.csv`: `primitive, pc_opcode, stall_reason, samples`.

## Figures
1. Arrive / wait / sync vs CS, lines per block size (reproduce P2 Fig. 4a–h).
2. Relaxed vs release arrive (fence cost) bar.
3. Ping-pong round trip for the 10 signalling methods, sorted; dotted line = via-GMEM reference.
4. mbarrier operation costs bar; try_wait suspension actual vs hint.
5. Stall-reason breakdown for the two spin loops.

## Pass criteria
`__syncthreads` 10–30; arrive 500–1300; wait 40–100; sync 800–1400; mbarrier predicate 30–60; via-GMEM round trip 900–1300; st.async/mbarrier signalling faster than plain-store polling (report either way).

## NVIDIA tools
ncu as above plus `sass__inst_executed_per_opcode` to count `UCGABAR_ARV/WAIT`, `SYNCS.*`, `MEMBAR.*`, `CGAERRBAR`, `FENCE.VIEW.ASYNC.*` per kernel; `cuobjdump -sass` of each primitive's kernel in the notebook.
