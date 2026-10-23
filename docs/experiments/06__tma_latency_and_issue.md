# 06 — TMA / bulk-copy latency, issue cost, and concurrency limits (async proxy, global → shared)

## Goal
For the async-proxy load path (rows 10, 13, 21 of `docs/context/02__memory_access_taxonomy.md`): (1) end-to-end latency of one bulk copy from L2 and from HBM as a function of footprint (the async twin of `04` K2), (2) the fixed costs: issue gap per command, mbarrier polling granularity, completion overhead vs payload size, (3) the per-SM concurrency limit (transactions in flight) and the per-SM read bandwidth limit, (4) tensor vs non-tensor and box-shape effects on latency.

## Background
`docs/context/03__tma_and_bulk_async_copies.md` §1, §4; `docs/context/04__mbarrier.md`. References: TMA p-chase latency ≈ generic latency + ~170 cycles, no L1 plateau [P1]; issue gap 44 cycles, 16 admitted transactions per SM, 32 B/cycle per SM read limit, staircase +49 cycles / 1536 B, mbarrier poll 42 cycles [P5, RTX 5090]; per-command cost ≈ 54.5 cycles (128 B commands → 2.35 B/cycle) [blog, H200].

## Kernels (1 block, 1 warp unless stated; elected thread issues; `%clock64`)

### K1 — TMA pointer chase (latency vs footprint)
Reuse the `04` K2 permutation (64-bit pointers at 128 B granularity). Each hop: `mbar_arrive_expect_tx(bar, 16)`; `bulk_g2s(smem, ptr, 16, bar)`; `mbar_wait_parity(bar, phase)` (tight `try_wait.parity` loop, no hint); read `ptr = smem[0]`; `phase ^= 1`. ITERS ≥ 4096; footprint sweep as in `04` (1 KB … 2 GB). Variants: (a) non-tensor bulk 16 B; (b) tensor 1-D box of 2 elements × 8 B (`cp.async.bulk.tensor.1d`), (c) 2-D box 4×4 floats = 64 B (coordinates derived from the pointer); (d) `.L2::cache_hint` evict_last policy; (e) with `cp.async.bulk.prefetch.L2` of the *next* line issued speculatively (measures prefetch benefit); (f) consumer wait = `try_wait.parity` with `suspendTimeHint` = 0 / 100 / 1000 ns; (g) `test_wait` loop. Report cycles/hop minus the polling floor measured in K2.

### K2 — polling floor and completion overhead vs payload
Payload sweep 16 B, 128 B, 1 KB, 4 KB, 16 KB, 64 KB from an L2-resident buffer (4 MB, warmed): time issue→observed completion with (a) tight `try_wait.parity` loop, (b) `test_wait` loop, (c) `try_wait` with hint. Separately time a `try_wait`/`test_wait` call on a barrier that is already complete (predicate latency, expected ≈ 42) and the cost of `mbarrier.arrive.expect_tx` alone (`07` reuses this). Fit `T(payload) = T0 + payload/BW`; T0 = fixed completion overhead, BW = per-SM TMA read bandwidth (expect ≈ 32 B/cycle). Identify the staircase (step size in bytes and cycles).

### K3 — issue gap and concurrency limit
Unrolled batches of N ∈ {1…32} independent 4 KB bulk copies (distinct destinations, one shared barrier with `expect_tx(N·4096)`), timed from first issue to last issue (`issue_span_cycles`) and to completion. Issue gap = slope of issue span vs N for N ≤ 16; the N at which issue span jumps (expected 17) = admission limit. Repeat with issuers in 1, 2, 4 warps on different sub-partitions (parallel issue) and with 2, 4, 8 blocks on different SMs (independent units). Also with sources in HBM (1 GB buffer) to see whether the limit is in requests or transactions.

### K4 — tensor map and box-shape latency
16 KB loads via tensor maps with boxes 256×16×1, 64×64×1, 16×256×1, 16×16×16, 4×4×256 (floats) and swizzle NONE / 128B (box inner ≤ 128 B for swizzle); latency of a single load from L2 (warm) and HBM (cold), plus `prefetch.tensormap` warm vs cold descriptor (first use after `cudaMemcpyToSymbol`/param).

### K5 — generic-proxy twin
Same chase with `ld.global.cg.u64` in the same binary (and `cp.async.cg` 16 B + `cp.async.wait_group 0`) for the "TMA minus generic" delta.

## CSV columns
`06__tma__chase.csv`: `variant, footprint_bytes, hops, cycles_per_hop, poll_floor_cycles, smid`.
`06__tma__completion.csv`: `wait_variant, payload_bytes, rep, cycles_issue_to_complete, cycles_predicate_only`.
`06__tma__issue.csv`: `n_in_flight, issuers, sms, source ∈ {l2, hbm}, issue_span_cycles, completion_cycles`.
`06__tma__boxes.csv`: `box, swizzle, source, descriptor_state ∈ {cold, prefetched}, cycles`.

## Figures
1. Latency vs footprint: TMA (a–c) vs generic (K5) on one plot (reproduce P1 Fig. 3 "H800 (by TMA)").
2. Completion time vs payload with the fitted T0 and 1/BW; staircase zoom.
3. Issue span vs N (knee = admission limit) and completion vs N.
4. Box-shape latency bars.

## Pass criteria
TMA − generic ≈ 120–220 cycles for L2 and DRAM regimes; no inflection at the L1 size; per-SM TMA BW ≈ 32 B/cycle ±15 %; admission limit = 16 (if different, report); issue gap 40–60 cycles.

## NVIDIA tools
`ncu --metrics sm__sass_inst_executed_op_tma_ld.sum,l1tex__tmain_requests.sum,l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum,l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.avg.per_cycle_active,l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.avg.peak_sustained,sm__pipe_tma_cycles_active.avg,lts__t_sectors_srcunit_tex_op_read.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum (must be 0 for TMA variants) --clock-control none --cache-control none`; `cuobjdump -sass` showing `UBLKCP.S.G` / `UTMALDG` / `SYNCS.ARRIVE.TRANS64` / `SYNCS.PHASECHK.TRANS64.TRYWAIT` in the loop.
