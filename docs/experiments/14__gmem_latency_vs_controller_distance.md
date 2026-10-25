# 14 — DRAM (HBM3) latency vs partition, memory controller, and channel/bank structure: does SM↔controller distance matter?

> **Device configuration:** runs under [`docs/context/10__device_config.md`](../context/10__device_config.md): H100 NVL (GPU 2), 132 SMs, 60 MiB L2, HBM3 3.94 TB/s, SM clock locked at 1635 MHz, power limit 400 W, ECC on, CUDA 12.9.1 `sm_90a`. Reference numbers quoted in this spec come from **other chips and configurations** (H800 PCIe, H100 PCIe, H100 SXM, H200, RTX 5090; see `10` §4). They are kept unchanged for the agent's sanity checks only, never cited in the notebook and not expectations for this device.

> **Method note (OI-1, decided):** Nsight Compute on this setup exports no per-instance counter values. Controller (FBPA) identity of a line comes from `13`'s map (`fbpa_id`, derived with the `fbpa__dram_read_sectors.max` oracle, `13` Step 4) or, for lines not in that map, from the same oracle run here; per-controller imbalance is read from the `.max/.avg/.min` rollups. Nothing below needs instances.

## Goal
Decompose the DRAM-miss latency seen by each SM into (a) the L2 partition term (near/far home, +162 cycles on this device, `08`), (b) a per-memory-controller / per-stack term (physical distance between the L2 partition and the HBM stack), and (c) DRAM-internal terms (row-buffer hit/miss, bank conflicts). Also recover as much of the channel/bank interleave as the timing allows, measure the miss penalty per GPC (Jin et al. saw it bimodal by GPC on H100), and the HBM loaded latency vs injection rate (the GPUNetBench HBM_LAT-BW measurement). Output: an SM × memory-controller latency matrix, the channel-bit structure, the per-GPC miss-penalty table and the HBM latency–throughput curve.

## Background
`docs/context/06__l2_nuca_and_physical_topology.md` §1.4, §2.5; `13` (slice and FBPA map, oracle); `08` (home hash; HBM-miss latency classes 505/681 from SMs 0 and 2, by latency not by home); `04` (misses with known home: near/far 505/667). References: near/far miss 555/744 cycles [P1]; DRAM remote penalty ≈ 174 [DGNA]; miss penalty bimodal by GPC ≈ 250–270 vs 440–470 on H100 PCIe [P6]; no public HBM3 mapping; GPUHammer row-conflict method (256 B chunks rotate banks; conflict step ≈ 25–30 ns on GDDR6). This device: 12 FBPAs, 6016-bit bus = 94 × 64-bit channels; HBM3 16 channels × 2 pseudo-channels per stack, 16 banks / 4 bank groups each [INF]. GPUNetBench reference for K5: `docs/research_notes/gpunetbench.md` §2 (HBM_LAT-BW) and §3.

## Kernels (K1–K4: one thread; `ld.global.cg.u64` with footprint ≫ L2 so every access misses; flush L2 between repetitions with `CleanL2`)

### K1 — per-line miss latency with partition and controller label
Fine-grained p-chase (`04` K3 style, `read_clock64_after`) over 1 GB in random order, 100 k samples per probe SM; each sampled line's home partition from `08`'s hash and its **FBPA id from `13`** (lines of the 256 mapped pages; for other lines run the `.max` oracle of `13` Step 4 in batches of 64 lines with distinct repeat counts). Probe from 8 SMs (one per GPC from `03`, including one of SMs 124–131). Fit `L(sm, line) = μ + a(sm) + b(partition) + c(fbpa) + ε`; report the size of `c` relative to `b` and the residual.

### K2 — row-buffer conflict detection (bank/row mapping)
Pairs of lines (A, B): alternate dependent loads A, B, A, B … with L2 bypass (both lines evicted between accesses by touching a 64 MB eviction set, or `discard.global.L2` + reload; verify the miss with `lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum`). Classify pair latency as "same bank, different row" (conflict: +tRP+tRCD ≈ 30–60 ns) vs "different bank" vs "same row" (hit). Sweep B = A ⊕ (1 << b) for b ∈ [8, 30] and B = A + 256·j; from the conflict pattern derive candidate bank/channel bits (XOR sets) within bits 8–20 and statistically above; threshold from the bimodal histogram. Cross-check: lines of one FBPA (from `13`) must never show a "different controller" class.

### K3 — bandwidth-side confirmation (rollups)
Stream 256 MB with all SMs vs the SMs of one GPC vs one partition, over (i) all lines, (ii) lines of one FBPA, (iii) lines of k FBPAs (from `13`), reading `fbpa__dram_read_bytes.{sum,max,avg,min}`: for (ii) `.max == .sum`, for (iii) `.max ≈ .sum/k` — confirms the controller labels and shows the imbalance under address-restricted streams; per-SM DRAM bandwidth vs FBPA affinity (does an SM stream faster from a "near" controller? Jin et al.: bandwidth uniform).

### K4 — refresh signature (optional)
Long dependent chain to one line with periodic outliers → tREFI estimate (GPUHammer: 1407 ns on GDDR6); report if visible.

### K5 — HBM loaded latency vs injection rate (GPUNetBench HBM_LAT-BW design, our timers)
Grid = 132 × 2 CTAs (2 per SM), threads per CTA ∈ {1, 32, 64, 128, 256, 512, 1024}; a 16 GiB buffer, `CleanL2` first, every element read **once** (stream pattern: thread t of the grid reads 64 B at `(t + i·N)·64` for iteration i = 0…4; random pattern: random 64 B-aligned start per thread, then the same stride); each thread times **one dependent load** at iteration 4 with `%clock64` and `read_clock64_after`, keeps the value in a register sink and stores `(smid, warp, lane, cycles)` after the loop; variants with a per-warp (or per-thread) random delay of 0…31 dependent `add` steps before each load (the artifact's `RANDOM_DELAY_STEPS`). Throughput of the same launch from `cudaEvent`s and `dram__bytes_read.sum.per_second`; 5 launches per point. Output: latency histogram per injection rate and the latency–throughput curve (throughput on a log₂ axis).

### K6 — miss penalty per GPC
Probe SM = first SM of each GPC (8) and SMs 124/126/128/130; 1024 random misses into lines homed in partition 0 and 1024 into partition 1 (`08` hash), footprint 1 GB, flush between repetitions, 5 repetitions → 12 × 2 matrix of median miss latency and, subtracting `08`'s hit latency of the same (SM, home) class, the miss *penalty* (memory → L2 part).

## CSV columns
`14__dram__samples.csv`: `probe_smid, gpc, line_offset, partition, fbpa_id, cycles`.
`14__dram__pairs.csv`: `probe_smid, line_a, line_b, flipped_bit, cycles_a, cycles_b, class ∈ {hit, bank_conflict, other}`.
`14__dram__model.csv`: `term, estimate_cycles, stderr, r2`.
`14__dram__bw_affinity.csv`: `sm_set, fbpa_set, bytes, ms, gbps, fbpa_bytes_sum, fbpa_bytes_max, fbpa_bytes_avg, fbpa_bytes_min`.
`14__dram__loaded.csv`: `threads_per_cta, ctas_per_sm, pattern, delay_mode, delay_steps, launch_idx, smid, warp, lane, cycles, gbps_event, gbps_ncu`.
`14__dram__gpc_miss.csv`: `probe_smid, gpc, home_partition, rep, median_cycles, mad_cycles, hit_cycles_08, penalty_cycles`.

## Figures
1. SM × FBPA heatmap of median miss latency (rows ordered by GPC/partition).
2. Histogram of pair latencies (K2) with thresholds; bit-participation chart for bank/channel bits.
3. Term sizes from the additive model (partition vs controller vs residual).
4. Per-FBPA bandwidth imbalance (`.max/.avg`) under restricted streams.
5. HBM loaded-latency histograms per injection rate and the latency–throughput curve (K5), no-delay vs random-delay.
6. Miss penalty per GPC and home partition (K6).

## Sanity checks (agent-only)
Expected ranges for judging plausibility during result analysis. They are never presented as pass/fail and never quoted in the notebook. If a value falls outside, investigate (placement, clocks, SASS, cache state, ncu replay) and re-run. A value that survives is a finding about this device; describe it on its own terms.
Partition term ≈ 150–200 cycles (`08`: 162 on hits, 505 vs 667 on misses with known home (`04`)); the notebook states whether the controller term is distinguishable from 0 at 95 % confidence (either outcome is a result); bank-conflict step detected (≥ 20 ns bimodality) or explicitly reported as not observable; K3 `.max == .sum` for single-FBPA streams; K5 median latency at 1 thread/CTA ≈ the unloaded miss latency of `04` and grows with injection, throughput approaches `05`'s 3.70 TB/s at 1024 threads; K6 penalty bimodal by home partition (P6: ≈ 250 vs ≈ 450 on H100 PCIe) and constant per GPC.

## NVIDIA tools
Rollups only: `fbpa__dram_read_sectors.{sum,max}`, `fbpa__dram_read_bytes.{sum,max,avg,min}`, `lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum`, `lts__d_sectors_fill_device.sum`, `dram__bytes_read.sum.per_second`, `dram__cycles_elapsed.avg.per_second` (memory clock), `gpu__time_duration.sum`, with `--cache-control all --clock-control none --csv --page raw --print-units base`; verify each `fbpa__` name with `ncu --query-metrics` and record which exist.
