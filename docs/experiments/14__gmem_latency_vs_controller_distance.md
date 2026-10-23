# 14 — DRAM (HBM3) latency vs partition, memory controller, and channel/bank structure: does SM↔controller distance matter?

> **Device configuration:** runs under [`docs/context/10__device_config.md`](../context/10__device_config.md): H100 NVL (GPU 2), 132 SMs, 60 MiB L2, HBM3 3.94 TB/s, SM clock locked at 1635 MHz, power limit 400 W, ECC on, CUDA 12.9.1 `sm_90a`. Reference numbers quoted in this spec come from **other chips and configurations** (H800 PCIe, H100 PCIe, H100 SXM, H200, RTX 5090; see `10` §4). They are kept unchanged for the agent's sanity checks only, never cited in the notebook and not expectations for this device.

## Goal
Decompose the DRAM-miss latency seen by each SM into (a) the L2 partition term (near/far, known ≈ +175 cycles), (b) a per-memory-controller / per-stack term (physical distance between the L2 partition and the HBM stack), and (c) DRAM-internal terms (row-buffer hit/miss, bank conflicts). Also recover as much of the channel/bank interleave as the timing allows. Output: an SM × memory-controller latency matrix and the channel-bit structure.

## Background
`docs/context/06__l2_nuca_and_physical_topology.md` §1.4, §2.5. References: near/far miss 555/744 cycles [P1]; DRAM remote penalty ≈ 174 [DGNA]; no public HBM3 mapping; GPUHammer row-conflict method (256 B chunks rotate banks; conflict step ≈ 25–30 ns on GDDR6). H100 SXM5: 5 stacks × 2 × 512-bit controllers = 10 FBPAs; HBM3 16 channels × 2 pseudo-channels per stack, 16 banks / 4 bank groups each [INF].

## Kernels (one thread; `ld.global.cg.u64` with footprint ≫ L2 so every access misses; or `ld.global.cv`; flush L2 between repetitions)

### K1 — per-line miss latency with controller label
Fine-grained p-chase (`04` K3 style) over 1 GB in random order, 100 k samples per probe SM; each sampled line's **FBPA id** obtained once via ncu (`fbpa__dram_read_sectors.sum --print-metric-instances details` on a kernel that reads that line after an L2 flush — batch 64 lines per launch with distinct repeat counts as in `13`) and its partition from `13`. Probe from 8 SMs (one per GPC from `03`). Fit `L(sm, line) = μ + a(sm) + b(partition) + c(fbpa) + ε`; report the size of `c` relative to `b` and the residual.

### K2 — row-buffer conflict detection (bank/row mapping)
Pairs of lines (A, B): alternate dependent loads A, B, A, B … with L2 bypass (both lines evicted between accesses by touching a 64 MB eviction set, or `discard.global.L2` + reload; verify miss with `lts__t_sectors_srcunit_tex_op_read_lookup_miss`). Classify pair latency as "same bank, different row" (conflict: +tRP+tRCD ≈ 30–60 ns) vs "different bank" vs "same row" (hit). Sweep B = A ⊕ (1 << b) for b ∈ [8, 30] and B = A + 256·j; from the conflict pattern derive candidate bank/channel bits (XOR sets) within bits 8–20 and statistically above. Threshold chosen from the bimodal histogram.

### K3 — bandwidth-side confirmation
Stream 256 MB with all SMs vs with only the SMs of one GPC vs one partition, while reading `fbpa__dram_read_bytes` per instance: imbalance across the 10 FBPAs under address-restricted streams (lines selected by the channel bits from K2) confirms the channel mapping; per-SM DRAM bandwidth vs FBPA affinity (does an SM stream faster from a "near" controller? expect no [Jin: uniform BW]).

### K4 — refresh signature (optional)
Long dependent chain to one line with periodic outliers → tREFI estimate (GPUHammer: 1407 ns on GDDR6); report if visible.

## CSV columns
`14__dram__samples.csv`: `probe_smid, gpc, line_offset, partition, fbpa_id, cycles`.
`14__dram__pairs.csv`: `probe_smid, line_a, line_b, flipped_bit, cycles_a, cycles_b, class ∈ {hit, bank_conflict, other}`.
`14__dram__model.csv`: `term, estimate_cycles, stderr, r2`.
`14__dram__bw_affinity.csv`: `sm_set, fbpa_id, bytes, ms, gbps`.

## Figures
1. SM × FBPA heatmap of median miss latency (rows ordered by GPC/partition).
2. Histogram of pair latencies (K2) with thresholds; bit-participation chart for bank/channel bits.
3. Term sizes from the additive model (partition vs controller vs residual).
4. Per-FBPA bandwidth under restricted streams.

## Sanity checks (agent-only)
Expected ranges for judging plausibility during result analysis. They are never presented as pass/fail and never quoted in the notebook. If a value falls outside, investigate (placement, clocks, SASS, cache state, ncu replay) and re-run. A value that survives is a finding about this device; describe it on its own terms.
Partition term ≈ 150–200 cycles; the notebook states whether the controller term is distinguishable from 0 at 95 % confidence (either outcome is a result); bank-conflict step detected (≥ 20 ns bimodality) or explicitly reported as not observable.

## NVIDIA tools
ncu per-instance `fbpa__dram_read_sectors.sum`, `fbpa__dram_read_bytes.sum`, `lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum`, `lts__d_sectors_fill_device.sum`, `dram__cycles_elapsed.avg.per_second` (memory clock) with `--cache-control none --clock-control none --print-metric-instances details`.
