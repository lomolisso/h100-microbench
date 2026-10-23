# 13 — Address → L2 slice / partition mapping via per-slice hardware counters, and the slice hash

## Goal
Recover, without timing, which of the 80 L2 slices (and which partition) each 128 B line maps to, using Nsight Compute's per-instance `lts__` counters; derive the partition hash and as much of the slice hash as the accessible address bits allow; produce `results/13__l2_slice_map.csv` (line offset → slice id → partition) used by `15` to choose target slices and by `08`/`14` to label latency classes.

## Background
`docs/context/06__l2_nuca_and_physical_topology.md` §1.1, §2.2 (counter method); `docs/context/07__profiling_tools.md` §1 (`--print-metric-instances details`), §2 (L2 metrics). Facts: 80 slices (verify `device__attribute_l2s_count`), 2 partitions, partition = XOR hash of bits ≥ 12 (14 bits on H100 PCIe) [Fan et al.]; slice hash unknown (non-power-of-two; accel-sim uses IPOLY-modulo); L2 line 128 B, sector 32 B; `cudaMalloc` ≥ 2 MB → 2 MB pages so bits 0–20 are physical.

## Method

### Step 1 — probe kernel
`touch_line(ptr, k)`: one thread, `k = 1024` dependent `ld.global.cg.u64` of the same line (self-pointing). Launch once per line to be mapped. Profile with
`ncu --metrics lts__t_sectors_op_read.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_ltcfabric.sum,lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum --print-metric-instances details --csv --page raw --cache-control none --clock-control none -k touch_line`.
The instance (slice id) whose `lts__t_sectors_srcunit_tex_op_read.sum` increments by ≈ k sectors (±2 %) is the owner; other slices show ≈ 0. If `ltcfabric` sectors also appear on a second slice, the line was accessed via the far partition (the probing SM's partition ≠ the home) — record both.
To amortise ncu overhead, instead map **many lines per launch**: kernel touches line *i* exactly `k_i = 1024 + 7·i` times for i = 0 … 63 (distinct counts) in sequence; per-slice counts are then decoded by matching the count signature (two lines never share a count); use one kernel launch per 64 lines.

### Step 2 — address sweep
Within one 2 MB page: all 16384 lines (128 B) → complete map of bits 7–20. Across pages: 256 random 2 MB-aligned pages, first 64 lines each → statistics for bits 21–30 (virtual→physical unknown above bit 20; look for consistency of the *within-page* pattern across pages and for how many distinct page-level offsets exist).

### Step 3 — hash recovery
- Partition bit set: test linearity `part(a ⊕ b) == part(a) ⊕ part(b)` over all pairs within a page; solve GF(2) for the 14-bit mask within bits 7–20; compare with Fan et al. (lowest bit 12).
- Slice within partition: try models (i) linear XOR map to a 6-bit index (fails if 40 is not a power of two — 40 = 8 × 5), (ii) IPOLY (accel-sim polynomials) + modulo 40 / modulo 5 on a hashed group index, (iii) decision-tree / brute-force search over XOR-of-bit-subsets followed by mod. Report the best model's accuracy on held-out lines; if < 100 %, publish the lookup table instead of a formula (the table is what `15` needs anyway).
- Memory-controller grouping: slices 8k…8k+7 are expected to belong to one 512-bit controller [INF]; test whether lines that map to the same `fbpa__dram_read_bytes` instance (DRAM-miss variant of the probe: 1 GB buffer, `ld.global.cv`) share a slice group.

### Step 4 — cross-check with timing
For 64 lines per partition, run `08` K1 from a known-near SM: counter-derived partition must agree with the near/far latency class ≥ 99 %.

## CSV columns
`13__l2_slice_map.csv`: `page_base, line_offset, slice_id, partition, fbpa_id, sectors_counted, probe_smid`.
`13__l2_hash_fit.csv`: `model, bits_used, accuracy_pct, formula_or_table_path`.

## Figures
1. Slice id vs line offset within a 2 MB page (scatter, 16384 points) — shows interleave period.
2. Bit-participation matrix for partition and slice index bits (heatmap of GF(2) coefficients).
3. Histogram of lines per slice (balance) and per partition.
4. Agreement with the timing classes from `08`.

## Pass criteria
Exactly one slice increments per line; all 80 slices used; partition hash matches Fan et al. on bit 12; timing agreement ≥ 99 %.

## NVIDIA tools
ncu only (no in-kernel timing); `device__attribute_l2s_count`, `device__attribute_num_l2s_per_fbp` (collection `device`) recorded in the CSV header; `nvidia-smi -q` serial number.
