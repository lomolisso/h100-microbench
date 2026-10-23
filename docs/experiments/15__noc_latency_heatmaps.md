# 15 — Point-to-point latency heatmaps across the whole memory hierarchy (SM × L2 slice, SM × partition, SM × SM via DSMEM, SM × SM via L2) and the inferred interconnect

## Goal
The sensitivity-analysis dataset for 3D stacking: for every SM, the latency to every L2 slice (hit), to every partition (miss), and to every reachable SM (DSMEM), in one consistent set of units and one consistent placement map; a fitted additive/hierarchical model that says how much of each latency is "distance"; and an explicit statement of the interconnect structure the data support (flat crossbar per GPC; two-level GPC→partition; sub-node tiers).

## Background
`docs/context/06__l2_nuca_and_physical_topology.md` §2.3–2.4; Jin et al. (SM×slice non-uniformity up to 70 %, uniform bandwidth); L40 additive model (R² 0.87/0.98; SM term 57 cycles, slice term 40 cycles); DGNA sub-node +32; ClusterSim DSMEM flat 187–218; AnandTech 3-point caveat (SM→SM through L2 is not a distance).

## Inputs from other experiments
`03` smid→GPC map; `13` line→slice/partition table; `08` per-SM partition/subnode labels; `10` DSMEM SM×SM matrix (reused, not re-measured).

## Kernels

### K1 — SM × slice hit latency
For each SM *i* (132) and each slice *s* (80): 64 lines homed on *s* (from `13`), first touched **by SM i itself** (so no replica ambiguity), chased 1024 dependent hops in random order (lines stay L2-resident; L1 bypassed with `.cg`); median and MAD of the per-hop `%clock64` series; ≥ 5 repetitions. One probe at a time on an idle chip (serialise via a global work queue: block with `%smid == i` runs, others exit). Also a variant where lines were first touched by an SM of the *other* partition (replica/far effect).

### K2 — SM × partition miss latency
Per SM: 1024 random misses into lines of partition α and of β (from `13`), 256 MB footprint, `ld.global.cg`, flush between repetitions. Produces the two-column miss matrix.

### K3 — SM × SM through L2 (the 3-point path, for completeness)
CTA on SM i stores a flag line homed on slice s; CTA on SM j spins and replies through a line homed on s'; round trip measured on i for a few (s, s') choices — demonstrates that the result depends on s, s' (not on i, j) and must not be read as SM distance.

### K4 — contention bandwidth matrix (Jin two-kernel method)
Kernel A: SM set X streams lines homed on slice s at full rate; kernel B: SM j measures hit latency/bandwidth to slice s' while A runs; sweep s' = s / same partition / other partition; shows whether contention is per slice, per partition port, or global.

## Analysis
1. Additive model per matrix: `L(i,s) = μ + a(i) + b(s) + e`; report R², the SM-term spread, the slice-term spread, and a low-rank residual (rank-1 term as in the L40 paper). Cluster `a(i)` → should reproduce GPC groups from `03` and partition/sub-node labels from `08`; cluster `b(s)` → partition and controller groups from `13`.
2. Hierarchical model: `L(i,s) = μ + α·[part(i) ≠ part(s)] + β·[subnode(i) ≠ subnode(s)] + γ(gpc(i)) + δ(slice group(s)) + e`; report each coefficient with CI — these are the "distance" terms the 3D-stacking study will remove or shrink.
3. DSMEM matrix (from `10`) with the same hierarchical treatment (TPC / CPC / GPC levels).
4. Combined "memory-hierarchy distance map": for each SM, a vector (local smem, DSMEM mean/max within GPC, L2 near slice min/mean/max, L2 far, DRAM near/far) — the per-SM profile that the stacking model consumes.

## CSV columns
`15__noc__sm_slice.csv`: `smid, gpc, slice_id, partition_slice, first_touch_smid, rep, median_cycles, mad_cycles, n`.
`15__noc__sm_partition_miss.csv`: `smid, gpc, partition, rep, median_cycles, mad_cycles`.
`15__noc__l2_pingpong.csv`: `smid_i, smid_j, slice_s, slice_s2, round_trip_cycles`.
`15__noc__contention.csv`: `stream_sm_set, stream_slice, probe_smid, probe_slice, probe_latency_cycles, probe_bpc`.
`15__noc__model.csv`: `matrix, term, estimate, ci_low, ci_high, r2`.
`15__noc__sm_profile.csv`: `smid, gpc, partition, subnode, smem, dsmem_mean, dsmem_max, l2_near_min, l2_near_mean, l2_far_mean, dram_near, dram_far`.

## Figures
1. **SM × slice heatmap** (132 × 80; rows ordered by GPC/partition, columns by partition/controller group) — the headline figure.
2. SM × partition miss heatmap (132 × 2) next to the SM×SM DSMEM heatmap from `10` on the same colour scale (cycles).
3. Model coefficients with CIs (partition, sub-node, GPC, slice group).
4. Per-SM distance profile (small multiples or parallel coordinates).
5. Contention: probe latency vs stream target (same slice / same partition / other).

## Pass criteria
Additive R² ≥ 0.85; SM-term clustering recovers the `03` GPC map (adjusted Rand index ≥ 0.9); partition coefficient 120–200 cycles; the notebook writes an explicit "inferred interconnect" paragraph (levels found, latency per level, confidence).

## NVIDIA tools
ncu `lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum --print-metric-instances details` on a sample of K1 runs to prove the chased lines hit only the intended slice; `lts__t_sectors_srcunit_ltcfabric.sum` to prove far accesses cross the fabric; clock locking and `clocks_event_reasons` logging for the multi-hour run.
