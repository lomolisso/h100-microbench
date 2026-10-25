# 13 — Address → L2 slice / partition / FBPA map with rollup counters (busiest-unit oracle), and the slice hash

> **Device configuration:** runs under [`docs/context/10__device_config.md`](../context/10__device_config.md): H100 NVL (GPU 3), 132 SMs, 60 MiB L2, HBM3 3.94 TB/s, SM clock locked at 1635 MHz, power limit 400 W, ECC on, CUDA 12.9.1 `sm_90a`. Reference numbers quoted in this spec come from **other chips and configurations** (H800 PCIe, H100 PCIe, H100 SXM, H200, RTX 5090; see `10` §4). They are kept unchanged for the agent's sanity checks only, never cited in the notebook and not expectations for this device.

> **Method note (resolves OI-1, AGENTS.md):** Nsight Compute 2025.2.1 on this setup exports no per-instance (per-slice / per-FBPA / per-SM) counter values, only the `.sum/.avg/.min/.max` rollups across units. This spec therefore identifies the slice of a line with the **busiest-unit (`.max`) oracle** (one exact hardware count per profiled launch) and cross-checks it with the profiler-free **contention oracle** of Jin et al. (MICRO 2024) and with the timing classes of `08`. Step 0 re-checks the instance export once; if instances turn out to be available, Step 2 uses them directly and everything else stays the same.

## Goal
Recover, for every 128 B line of one complete 2 MiB page and of 256 sampled pages, (a) its **home slice** (the `lts` unit that serves it when the reading SM is in the line's home partition), (b) its **copy slice** in the other partition (the unit that holds the copy left by a read from there, `08`), (c) its home partition (confirming `08`'s in-page parity and the measured per-page term) and (d) its **FBPA** (memory controller); settle the active slice count (94 vs 96); derive the partition hash and as much of the slice hash as the accessible address bits allow, otherwise publish the lookup table; produce `results/13__l2_slice_map.csv`, which `15` uses to choose target slices and `14` to label controllers.

## Background
`docs/context/06__l2_nuca_and_physical_topology.md` §1.1 (homes, copies, 96/94 slices), §2.2 (this method); `docs/context/07__profiling_tools.md` §1–2 (rollup tricks), §8; `08` results: partition = XOR of PA bits 12,13,15,17,19 ⊕ a physical-page term (PA bits 21,22 inferred, ≥ 23 unresolved: measure the term per page), home constant per 4 KiB, any read or write from partition P leaves a copy in P (partner sub-node of the home sub-node), a far first read is looked up in the near partition (miss, 2 sectors) and at home; `docs/context/08` P6 (Jin et al.: no per-slice counters on A100/H100 either; contention oracle; 6 of 80 slices mapped). Facts: `device__attribute_l2s_count` = 96 = 12 FBPs × 8, `lts__*.avg = .sum/94`, 640 KiB per slice, 2 partitions; L2 line 128 B, sector 32 B; `cudaMalloc` ≥ 2 MiB → 2 MiB pages, bits 0–20 physical. Research note: `docs/research_notes/gpunetbench.md` §1.5.

## Method

### Step 0 — instance-export re-check (≤ 1 h of GPU, result recorded in `07` §8 and in the notebook's setup section)
One launch of K1 (one line, 4096 reads) profiled four ways: (i) `--replay-mode kernel --metrics lts__t_sectors_srcunit_tex_op_read.sum --print-metric-instances details`; (ii) the same under `--replay-mode application`; (iii) `--set full -o rep` then `ncu -i rep --page raw --csv --print-metric-instances details` and the `ncu_report` Python module (`metric.num_instances()` for `lts__t_sectors_srcunit_tex_op_read.sum`, `fbpa__dram_read_sectors.sum`, `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`, `sm__cycles_active.sum`); (iv) `--metrics` with the unsuffixed base name `lts__t_sectors_srcunit_tex_op_read` (all rollups at once). If any form yields per-unit instances, Step 2 reads the slice id from the instance that holds ≈ k lookups (the original counter method) and records `method = instances`; otherwise `method = max_oracle` below. Also record whether `fbpa__*` rollups exist at all.

### Step 1 — oracle calibration (K1 `k13_probe`)
`k13_probe(lines[], counts[], n)`: one thread (claimed on a chosen SM with `claim_target_sm`), for each listed line *i* a **dependent chain of `counts[i]` loads of the same line** (`ld.global.cg.u64` of a self-pointing qword, address = previous value, so ptxas cannot merge or hoist loads); addresses and counts live in registers/constant memory (the list reads themselves are ≤ 64 lookups and are subtracted as a known baseline). Profile with the K1 command of §NVIDIA tools (kernel replay, `--cache-control all`: lookup counts do not depend on cache contents). Calibration launches:
1. one line × 4096 → `lookups_sum` ≈ 4096 + ε (ε = list/TLB reads, expect ≤ 8) and `lookups_max` = `lookups_sum` − ε' (**single-unit proof: `.max == .sum` up to ε**);
2. two lines of different 4 KiB homes (opposite partitions by the in-page parity, same 2 MiB page) × (4096, 2048) → `.max` = 4096 (different units) — probed from an SM of the first line's home partition;
3. the four sectors of one line × (4096, 2048, 1024, 512) → `.max` = 7680 iff all sectors share a slice (expected);
4. the same line probed from one SM per sub-node (4 SMs, `03`/`08` map) → `.max` equal from the two SMs of the home partition; from the other partition the first read is a near miss (2 lookup sectors there) and the rest hit the copy → `.max` = 4096 on the copy slice; record `lts__t_sectors_srcunit_ltcfabric.sum` (2 sectors expected) to show the far first read.

### Step 2 — reference-set bootstrap and per-line decoding
Keep, per partition, a reference set R = {r_1 … r_|R|}: one line per slice found so far. Per profiled launch, from a probe SM of the **home partition** of the test lines (home from the per-page home protocol (`06` §1.1): bits 12,13,15,17,19 of the offset plus the page term measured once for every 2 MiB page used; verified in Step 6): read every r_j exactly 4096 + 8·j times and up to **three test lines** X_a, X_b, X_c exactly 2048, 1024, 512 times. Then `lookups_max` = 4096 + 8·j* + S, where S ∈ {512, 1024, 1536, 2048, 2560, 3072, 3584} is the sum of the counts of the test lines that share slice r_j*'s unit (all seven subset sums are distinct and every one exceeds 8·|R| ≤ 384, so j* and the subset decode uniquely); a test line not in the decoded subset is on another slice; if `lookups_max` = 4096 + 8·|R| (no test line on any reference slice) the undecoded lines are re-probed one at a time, and a line that still shows `.max` = 4096 + 8·|R| is a **new slice**: it joins R (per partition, |R| ≤ 48). Copy slices: the same decoding from a probe SM of the **other** partition, with an untimed warm read of the test lines first (so that the copy exists) and `--replay-mode application --cache-control none`; the reference lines of that partition are read from their home SM in a preceding kernel of the same application (they are home lines there). Record for every line: `home_slice`, `copy_slice_p0`/`copy_slice_p1` (the copy slice when read from each partition; one of them equals the home slice), `lookups_max`, `lookups_sum`, `probe_smid_*`, `method`.

### Step 3 — address sweeps
- **Page-complete:** one 2 MiB-aligned page: all 16384 lines → home slice (bits 7–20 of the hash) and copy slices; probe SMs = one per sub-node (first SMs of G0, G4, G1, G3 from the `03` map), so that the home sub-node of each line (from `08`'s timing classes on a sample) can be related to its slice.
- **Across pages:** 256 random 2 MiB-aligned pages of a 2 GiB buffer, 64 lines each (the first 32 lines and 32 random lines) → bits 21–30 statistics (how many distinct page-level offsets of the slice function exist; consistency of the in-page pattern across pages), plus the `08` page-pair test for the page term of the partition hash.
- Expected profiled launches ≈ 7 k (page) + 6 k (pages) per partition side ≈ 25 k; at ≈ 30–60 ms per profiled launch (one pass, kernel replay) ≈ 15–25 min per side. Use one `ncu` process per 2048 launches (`--launch-count`), never one process per launch.

### Step 4 — FBPA grouping (needs `fbpa__dram_read_sectors.max`; if no `fbpa__` rollup exists, record that and derive controllers from `14` K2 instead)
After `CleanL2`, from the home-partition probe SM, read **once each** 1024 distinct lines of slice *a* and 512 distinct lines of slice *b* (all from the map); each first read is a miss of 64 B = 2 DRAM sectors (`08`) → `fbpa__dram_read_sectors.max` = 3072 iff *a* and *b* share a controller, 2048 otherwise. Bootstrap an FBPA reference set exactly as in Step 2 (12 expected) and assign every slice; expect groups of 8 slices per FBPA with two groups short of one slice each (94 active).

### Step 5 — hash recovery
- Partition: GF(2) linearity test and solve over bits 7–20 on the page-complete data (must reproduce `08`: 12, 13, 15, 17, 19), page term: measured per page (per-page home protocol, `06` §1.1); from the across-pages data report only whether it is consistent with page-index bits 21, 22 (the virtual page index is not the physical one, so no claim about bits ≥ 23).
- Home slice inside a partition (48 units = 16 × 3): try (i) XOR map to a 4-bit index plus a mod-3 term of a second XOR-folded index; (ii) IPOLY polynomials (accel-sim `memory_partition_indexing` family) followed by mod 48; (iii) decision-tree / brute-force search over XOR-of-bit-subsets followed by a modulus; report accuracy on held-out lines; if < 100 %, publish the lookup table (what `15` consumes anyway).
- Copy slice: express `copy_slice` as a function of `home_slice` (expected: a fixed partner unit in the partner sub-node, `08`) and test whether it is one-to-one.
- Sub-node: with `08`'s timing labels on a sample, test whether the home slice determines the home sub-node (24/24 split of the 48 slices expected).
- Controller: slices 8k…8k+7 ↔ one FBPA (Step 4); test whether the FBPA is a function of the slice id only.

### Step 6 — cross-check A: contention oracle (profiler-free, Jin et al.; K3 `k13_contend`)
Stream set: 8 SMs of one GPC (claimed via `%smid`, 1024 threads each, `ld.global.cg.v4`, ILP 4) read 256 distinct lines of slice *a* (from the map, warmed by themselves so that they hit near copies/homes) continuously for 20 ms. Probe: one SM of the same partition, in another GPC, chases 64 dependent hops over 64 lines of slice *b* 1024 times, `%clock64` per hop, idle and under the stream. The probe latency rises (≥ 30 %) iff *a* = *b*. 200 random (a, b) pairs covering every slice at least twice, 25 % with a = b; agreement with the map ≥ 99 %, disagreements listed. Also run 20 pairs with the probe in the other partition (a far reader) to show whether contention is per home slice or per copy slice.

### Step 7 — cross-check B: timing classes
For 64 lines per partition: `08` K1 from a known-near SM → the near/far class must agree with the home column (per-page protocol) ≥ 99 %. (`15` K1 later checks that the per-SM slice signature is constant within a slice class.)

## CSV columns
`13__l2_slice_map.csv`: `page_idx, page_base_va, line_offset, home_partition, home_subnode (from 08 sample or empty), home_slice, copy_slice_p0, copy_slice_p1, fbpa_id, probe_smid_home, probe_smid_copy, lookups_max, lookups_sum, method`.
`13__l2_refset.csv`: `partition, slice_id, ref_page_idx, ref_line_offset, fbpa_id, n_lines_mapped`.
`13__l2_oracle_calib.csv`: `test, probe_smid, line_a, line_b, count_a, count_b, lookups_max, lookups_sum, fabric_sectors, verdict`.
`13__l2_contention_check.csv`: `slice_a, slice_b, stream_gpc, stream_smids, probe_smid, probe_partition, probe_cycles_idle, probe_cycles_loaded, same_slice_map, same_slice_timing`.
`13__l2_hash_fit.csv`: `target ∈ {partition, home_slice, copy_slice, subnode, fbpa}, model, bits_used, accuracy_pct, formula_or_table_path`.

## Figures
1. Home slice id vs line offset within the 2 MiB page (scatter, 16384 points) — interleave period.
2. Copy slice vs home slice (96×96 count matrix) — the copy-placement rule.
3. Bit-participation matrix (GF(2) coefficients) for partition, slice index and FBPA.
4. Lines per slice (balance) and per partition; the number of populated slices (94 vs 96).
5. Slice → FBPA grouping (sorted bar/tile).
6. Oracle calibration: `lookups_max` vs `lookups_sum` for Step 1 tests.
7. Contention check: probe latency idle vs loaded for same-slice and different-slice pairs; agreement matrix.
8. Agreement with the `08` timing classes.

## Sanity checks (agent-only)
Expected ranges for judging plausibility during result analysis. They are never presented as pass/fail and never quoted in the notebook. If a value falls outside, investigate (placement, clocks, SASS, cache state, ncu replay) and re-run. A value that survives is a finding about this device; describe it on its own terms.
Single-line probes: `.max == .sum` within ≤ 8 lookups; every line decodes (no residual ambiguity after single re-probes); |R| per partition ∈ {47, 48}, total ∈ {94, 96}; partition bits = `08`'s; all lines of one 4 KiB block share home partition; copy slice ≠ home slice; FBPA groups of 8 (two short); contention agreement ≥ 99 % (Jin et al. saw a slice saturate with ≥ 4–8 SMs, so 8 streaming SMs must load it); timing agreement ≥ 99 %.

## NVIDIA tools
- K1/K2 (lookups, kernel replay; omit `--print-metric-name name` with `--page raw` on 2025.2.1): `ncu --clock-control none --cache-control all --replay-mode kernel -k regex:^k13_probe --launch-count <N> --metrics lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_read.max,lts__t_sectors_srcunit_tex_op_read_lookup_hit.max,lts__t_sectors_srcunit_tex_op_read_lookup_miss.max,lts__t_sectors_srcunit_ltcfabric.sum,lts__t_sectors_srcunit_ltcfabric.max --csv --page raw --print-units base`.
- Copy slices: the same metrics with `--replay-mode application --cache-control none` (warm read in a preceding kernel of the same application).
- FBPA: `fbpa__dram_read_sectors.sum,fbpa__dram_read_sectors.max` (verify with `ncu --query-metrics`; fall back to `fbpa__dram_read_bytes`), `--cache-control all`, after `CleanL2`.
- Device attributes recorded in the CSV header: `device__attribute_l2s_count`, `device__attribute_num_l2s_per_fbp`, `device__attribute_fbp_count`; `nvidia-smi -q` serial number.
- Baseline: an all-exit launch under the same command (`05`/`08`: the profiler adds writes but ≈ 0 tex read lookups).
- Step 6 runs without ncu (timing only); the stream kernels and the probe are launched on two streams.
