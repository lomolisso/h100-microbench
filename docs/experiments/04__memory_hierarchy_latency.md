# 04 — Generic-proxy latency of every local memory level: SMEM, L1, L2 (near/far), HBM (near/far)

## Goal
Average dependent-access latency in SM cycles for paths 1, 3, 4, 5 of `docs/context/00__h100_architecture.md` §6 using ordinary `ld` instructions, as a function of working-set size (the classic latency-vs-footprint curve) and with per-access classification of the L2/DRAM near/far cases. These numbers are the dotted reference lines in all DSMEM figures.

## Background
`docs/context/09__measurement_conventions.md`; `docs/context/06__l2_nuca_and_physical_topology.md` §1.1. References: SMEM 29–30, L1 32–33, L2 near 258–295 / far 414–470, DRAM ≈ 656 avg (555 near / 744 far) cycles [P1, DGNA]; L2 inflection points at ≈ ½ L2 and L2 size [P1]; `.cg`-based probes may be contaminated by replication effects [DGNA].

## Kernels (all one thread per block, one block, `%clock64`, loop from `01` with unroll 16)

### K1 — SMEM latency
Dynamic smem array of `N` 32-bit words, `s[i] = (i + stride) mod N` with stride ∈ {1 word, 32 words (same bank), random permutation}; chase 4096 hops with `ld.shared.u32`. Also `ld.volatile.shared.u32`, and the DSMEM-interface local variant (`mapa` to own rank + `ld.shared::cluster.u32`) launched as a cluster of 1 (expected 33 vs 29 cycles [P1]).

### K2 — footprint sweep (L1 → L2 → DRAM)
Global array of 64-bit pointers forming a random cyclic permutation at 128 B granularity (RRZE/Luo style: `skip = 16` qwords), footprint `F` from 1 KB to 2 GB in ×1.25 steps (plus exact points 128 KB, 25 MB, 50 MB); for each F: fill from host, warm by chasing once (not timed), then time `max(F/128, 1e6)`-hop chase, 3 reps, report min and median. Variants: `ld.global.ca.u64` (L1 allowed), `ld.global.cg.u64` (L1 bypass), `ld.global.cv.u64` (L2 bypass/volatile — check in SASS whether it really bypasses; Luo reports cache modifiers are not reliable). Record the SM running the probe (`%smid`); repeat the whole sweep from 4 SMs chosen from different GPCs (map from `03`) — the near/far plateau should move between SMs.

### K3 — fine-grained per-access classification
Footprint 8 MB (hits only) and 256 MB (misses), 128 B stride random permutation, `ld.global.cg.u64`, `%clock64` around **each** load (`t0 = clock64; p = *p; t1 = clock64 (depends on p)`), store `(line_offset, cycles)` into smem then to global (≥ 100 k samples per footprint). First-touch control: variant A — the probe SM itself initialises (touches) every line before timing; variant B — a block on a different-GPC SM (chosen by the `03` map, ideally far partition) touches the lines first, then the probe SM chases. Run from 8 probe SMs (one per GPC).

### K4 — same, TMA-free control for the async proxy comparison
No kernel; `06__tma_latency_and_issue` reuses K2's permutation so that generic vs async latency curves are directly comparable.

## Analysis
- K2: detect inflection points (change-point on log-scale); report plateau medians: L1, L2-near, L2-mixed, DRAM.
- K3: 1-D K-means (k = 2 for 8 MB, k = 4 for 256 MB), report cluster centres, SD, CV; compare variant A vs B (first-touch effect); per probe SM report which cluster dominates → preliminary `partition` label for `03__sm_gpc_topology__map.csv` (an SM whose K3 8 MB modal latency is the low cluster for lines homed by itself is "near its own home" — trivially; the informative statistic is the *fraction* of lines in the low cluster when lines were first touched by SM X: lines touched by X are near for X's partition).

## CSV columns
`04__latency__smem.csv`: `variant, stride, n_words, cycles_per_hop, smid`.
`04__latency__footprint.csv`: `load_variant, footprint_bytes, hops, rep, cycles_per_hop, smid, gpc`.
`04__latency__fine.csv`: `footprint_bytes, first_touch_smid, probe_smid, line_offset, cycles`.
`04__latency__summary.csv`: `level ∈ {smem, smem_dsmem_if, l1, l2_near, l2_far, dram_near, dram_far}, median_cycles, p5, p95, n`.

## Figures
1. Latency vs footprint (log x), one curve per load variant and per probe SM; dotted lines at 128 KB, 25 MB, 50 MB.
2. Histogram of per-access cycles for 8 MB and 256 MB with K-means centres annotated (reproduce P1 Table 4).
3. Variant A vs B per probe SM (first-touch effect).

## Pass criteria
SMEM 28–32; L1 30–35; L2 near 250–300; L2 far 400–480; DRAM 550–760 depending on class; K-means CV < 10 %.

## NVIDIA tools
`ncu --metrics l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum,dram__bytes_read.sum --cache-control none` on K2 at 64 KB / 8 MB / 256 MB to prove which level served the chase (hit rates ≈100 % / ≈100 % L2 / ≈0 %). `cuobjdump -sass` to show `LDG.E.64` with `.STRONG.GPU`/`.CONSTANT` qualifiers per variant.
