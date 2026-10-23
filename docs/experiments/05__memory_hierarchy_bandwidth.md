# 05 — Generic-proxy bandwidth of SMEM, L1, L2 and HBM (per SM and GPU-wide)

## Goal
Attained throughput in B/cycle/SM and TB/s GPU-wide for ordinary loads/stores at each level, with enough SMs/warps/ILP to saturate. These are the dotted GPU-wide reference lines for the DSMEM bandwidth figures and the denominators for "fraction of peak".

## Background
References: SMEM/L1 ≈ 124–128 B/cycle/SM; L2 ≈ 3900–4500 B/cycle chip (H800, 7–8 TB/s at 1755 MHz; Chips&Cheese > 5.5 TB/s near-partition on PCIe); DRAM ≈ 90 % of 3.35 TB/s [P1, lit]. Plain `LDG` throughput falls with > 2 loads in flight per thread on H100 (arXiv 2608.15764) — so sweep ILP.

## Kernels

### K1 — SMEM bandwidth (per SM)
1 block × 1024 threads, 64 KB dynamic smem; each thread issues `R = 4096` iterations of `ILP` independent `ld.shared.v4.f32` (conflict-free addressing: thread t reads `16·t + 16·1024·k`), accumulate XOR into a register to keep loads alive; `%clock64` per block; also `st.shared.v4` and 32-bit variants (`ld.shared.f32`). Report B/cycle = bytes / cycles; sweep threads ∈ {128, 256, 512, 1024}, ILP ∈ {1, 2, 4, 8}. Launch 132 blocks (one per SM) simultaneously and report per-SM distribution.

### K2 — L1 bandwidth (per SM)
Same as K1 but global array of 64 KB per block, warmed with `ld.global.ca`, chase-free vectorised `ld.global.ca.v4.f32` (addresses +0/+128/+256/+384 B pattern of the accel-sim ubench).

### K3 — L2 bandwidth (GPU-wide and per partition)
Working set 16 MB (fits in one partition's share) pre-warmed, grid = 2 × SMs blocks × 1024 threads, `ld.global.cg.v4.f32` streaming with ILP 4, 2048 repetitions; time with cudaEvents (GPU-wide TB/s) and with `%globaltimer` per block. Variants: (a) all SMs read the same 16 MB (lines homed by whichever SM touched first — randomised); (b) **partition-local**: each block only reads lines whose home partition equals its own SM's partition (requires the partition labels from `08`/`13`; until available use lines first-touched by the block's own SM); (c) partition-remote: the opposite. (d) footprint sweep 1–48 MB to see the ≈25 MB drop.

### K4 — DRAM bandwidth (GPU-wide)
2 GB float array, grid = 4 × SMs blocks × 1024 threads, each thread `float4` reads ×5 + write ×1 (Luo recipe), cudaEvents; also read-only and write-only variants; also `ld.global.cs` (streaming) vs `.cg`. Sweep active SM count (1, 2, 4, …, 132 blocks of 1024 threads with per-block 64 KB in flight) to get the per-SM linear region slope (≈ 29 GB/s per SM on 5090 [P5]; H100 expected higher) and the saturation knee.

### K5 — stores and read/write mix
Same as K3/K4 with `st.global.v4` and 50/50 mixes; records `lts__t_sectors_op_write`.

## CSV columns
`05__bandwidth__smem_l1.csv`: `level ∈ {smem, l1}, op ∈ {ld, st}, width_bytes, threads, ilp, bytes, cycles, bpc_per_sm, smid`.
`05__bandwidth__l2.csv`: `variant ∈ {mixed, local, remote}, footprint_bytes, blocks, threads, ilp, bytes, ms_event, tbps, bpc_chip (= tbps / sm_hz)`.
`05__bandwidth__dram.csv`: `op ∈ {read, write, mix}, cache_op, active_sms, bytes, ms_event, tbps, gbps_per_sm`.

## Figures
1. B/cycle/SM vs threads for SMEM and L1, lines per ILP.
2. L2 TB/s vs footprint (1–48 MB) with the partition-local / remote variants.
3. DRAM TB/s vs active SMs (knee annotated), dotted line at 3.35 TB/s (SXM5) or 2.0 TB/s (PCIe).
4. Summary bar: SMEM, L1, L2, DRAM in TB/s GPU-wide (SMEM/L1 = per-SM × SM count × clock).

## Pass criteria
SMEM/L1 ≥ 120 B/cycle/SM; L2 ≥ 5 TB/s (SXM5, expected 7–8 TB/s); DRAM ≥ 85 % of peak.

## NVIDIA tools
`ncu --metrics l1tex__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_bytes.sum.per_second,dram__bytes.sum.per_second,sm__cycles_elapsed.avg.per_second --clock-control none` (with locked clocks) on K3/K4 to cross-check the event-based TB/s; `lts__t_sectors_srcunit_ltcfabric.sum` on K3 variants (b) vs (c) to prove partition locality (verify the metric name with `ncu --query-metrics`).
