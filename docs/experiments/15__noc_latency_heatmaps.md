# 15 — Point-to-point latency heatmaps across the whole memory hierarchy (SM × L2 slice, SM × partition, SM × SM via DSMEM, SM × SM via L2), per-slice bandwidth, NoC input speedup, and the inferred interconnect

> **Device configuration:** runs under [`docs/context/10__device_config.md`](../context/10__device_config.md): H100 NVL (GPU 2), 132 SMs, 60 MiB L2, HBM3 3.94 TB/s, SM clock locked at 1635 MHz, power limit 400 W, ECC on, CUDA 12.9.1 `sm_90a`. Reference numbers quoted in this spec come from **other chips and configurations** (H800 PCIe, H100 PCIe, H100 SXM, H200, RTX 5090; see `10` §4). They are kept unchanged for the agent's sanity checks only, never cited in the notebook and not expectations for this device.

> **Method note (OI-1, decided):** slice identities come from `13`'s map (busiest-unit oracle); the proof that a chase hits only the intended slice is `lts__t_sectors_srcunit_tex_op_read.max == .sum` on a profiled sample; no per-instance counters are used anywhere. K5–K7 are the GPUNetBench measurements (SPEEDUP, PER_SLICE/MP, L2_LAT-BW) re-done with our harness; design details in `docs/research_notes/gpunetbench.md` §2–3.

## Goal
The sensitivity-analysis dataset for 3D stacking: for every SM, the latency to every L2 slice (hit), to every partition (miss), and to every reachable SM (DSMEM), in one consistent set of units and one consistent placement map; the bandwidth from any SM/GPC to any slice and the NoC input speedup per hierarchy level (TPC, CPC, GPC); the L2 latency–throughput curve under load; a fitted additive/hierarchical model that says how much of each latency is "distance"; and an explicit statement of the interconnect structure the data support (flat crossbar per GPC; CPC tier; two-level GPC→partition; sub-node tiers).

## Background
`docs/context/06__l2_nuca_and_physical_topology.md` §1.1–1.3, §2.3–2.4; `08` (2×2 grid 251/280/417/443, per-TPC position term, slice-specific ≈ 390 mode); Jin et al. [P6]: SM×slice non-uniformity by physical position, Pearson-signature placement, uniform per-slice bandwidth (one SM → one slice ≈ 34–45 GB/s, ≥ 4–8 SMs saturate a slice), input speedup (H100 read/write TPC 1.95/2.0, CPC 5.5/4.6, GPC 7.75/9.0 local, 9.1/12.4 global), L2 fabric 2.4–3.5× DRAM; L40 additive model (R² 0.87/0.98; SM term 57 cycles, slice term 40 cycles); DGNA sub-node +32; ClusterSim DSMEM flat 187–218; AnandTech 3-point caveat (SM→SM through L2 is not a distance).

## Inputs from other experiments
`03` smid→GPC map (with the caveat that SMs 124–131 are placed by clock domain); `13` line→home slice / copy slice / partition / FBPA table; `08` per-SM partition/sub-node labels and hit classes; `10` DSMEM SM×SM matrix and CPC grouping (reused, not re-measured); `05` per-GPC and partition-link bandwidths (reference lines).

## Kernels

### K1 — SM × slice hit latency
For each SM *i* (132, claimed one at a time) and each home slice *s* (94 or 96, from `13`): 64 lines homed on *s* and in the probe's own partition, first touched **by SM i itself** after `CleanL2` (so they live only at home and no copy ambiguity exists), chased 1024 dependent hops in random order (lines stay L2-resident; `.cg`); median and MAD of the per-hop `%clock64` series; ≥ 5 repetitions. One probe at a time on an idle chip (serialise via a global work queue: the block with `%smid == i` runs, the others exit). Slices of the *other* partition: the same lines first touched by an SM of their home partition, probed from SM *i* on the first pass only (far hits, 64 lines × 5 repetitions × flush) and on repeated passes (copy slices from `13`). Proof on a 5 % sample under ncu: `lts__t_sectors_srcunit_tex_op_read.max == .sum`.

### K2 — SM × partition miss latency
Per SM: 1024 random misses into lines homed in partition α and in β (`08` hash), 256 MB footprint, `ld.global.cg`, flush between repetitions. Produces the two-column miss matrix (shared with `14` K6).

### K3 — SM × SM through L2 (the 3-point path, for completeness)
CTA on SM i stores a flag line homed on slice s; CTA on SM j spins and replies through a line homed on s'; round trip measured on i for a few (s, s') choices — demonstrates that the result depends on s, s' (not on i, j) and must not be read as SM distance.

### K4 — contention (Jin et al. two-kernel method, formalised in `13` Step 6)
Kernel A: 8 SMs of one GPC stream 256 lines of slice *s* at full rate (two streams, synchronised start); kernel B: SM j (same GPC / other GPC of the same partition / other partition) measures hit latency and single-SM bandwidth to slice *s'* while A runs; sweep s' = s / same FBPA / same partition / other partition. Shows whether contention is per slice, per controller group, per partition port or global.

### K5 — NoC input speedup (GPUNetBench SPEEDUP design)
Within one GPC (G1 and G4 at least; all 8 if time allows): N = 1…18 SMs selected by `%smid` from the `03` map, added in three orders — TPC-sorted (mates together), CPC-sorted (the grouping found in `10`; if `10` found none, TPC index mod 3), and GPC-interleaved (one SM per TPC first); each selected SM runs 2 CTAs × 1024 threads streaming `ld.global.cg.v4` reads (and, separately, `st.global.v4` stores) over a 16 MiB L2-resident buffer whose lines are homed in the GPC's own partition (`08` hash), ILP 4, ≥ 1 GiB per launch, 5 launches; GPU-wide GB/s from `cudaEvent`s and `l1tex__m_xbar2l1tex_read_bytes.sum.per_second`. Speedup(N) = BW(N)/BW(1); report the TPC speedup (N = 2 mates), the CPC speedup (N = 6 of one CPC), GPC local (all TPCs) and GPC global (all SMs of the GPC reading lines of both partitions). Also one CPC of a GPC vs one SM of each of three CPCs (Jin et al.'s "no TPC contention" selection).

### K6 — per-slice bandwidth (GPUNetBench PER_SLICE / MP design)
Senders stream 256 lines of **one** slice (from `13`; each lane a different line, each warp a different 32 B sector, as in the artifact) with `ld.global.cg`, 1024 threads, ILP 4, ≥ 0.5 GiB per launch, 5 launches: (a) one SM → one slice for every (SM, slice) pair of the SM's partition (132 × 48, 10 % random sample of the far-partition pairs for comparison); (b) one GPC (16–18 SMs) → one slice, for every slice; (c) N = 1…18 SMs of one GPC → one slice (saturation curve) for 8 slices spread over the FBPAs; (d) 28 SMs contiguous (2 GPCs) vs distributed (3–4 per GPC) → one slice, one FBPA group of 8 slices, and 4 FBPA groups. GB/s per launch from `cudaEvent`s; `lts__t_sectors_srcunit_tex_op_read.max == .sum` on a sample proves the single-slice target.

### K7 — L2 loaded latency vs injection rate (GPUNetBench L2_LAT-BW design, our timers)
Grid = 132 × 2 CTAs, threads per CTA ∈ {1, 32, 64, 128, 256, 512, 1024}; a 16 MiB buffer warmed by the same SMs (so all hits are near copies) — plus a variant on 16 MiB homed entirely in one partition, warmed from that partition only, read by all SMs (half of them far on the first pass); stream and random patterns as `14` K5; each thread times one dependent load at iteration 4; per-warp random-delay variant (0…31 `add` steps); throughput from `cudaEvent`s and `l1tex__m_xbar2l1tex_read_bytes.sum.per_second`; 5 launches per point. Output: latency histograms per injection rate and the latency–throughput curve.

## Analysis
1. Additive model per matrix: `L(i,s) = μ + a(i) + b(s) + e`; report R², the SM-term spread, the slice-term spread, and a low-rank residual (rank-1 term as in the L40 paper). Cluster `a(i)` → should reproduce GPC groups from `03` and partition/sub-node labels from `08`; cluster `b(s)` → partition and FBPA groups from `13`; `b(s)` must be constant within a `13` slice class (cross-check B of `13`).
2. **Pearson-correlation heatmap** of the per-SM signatures (rows of the SM × slice matrix; Jin et al. Obs 4): correlation ≈ 1 inside a GPC (and inside a CPC if one exists); an independent placement of every SM, including SMs 124–131 (their GPC in `03` rests on clock domains) — report agreement with `03` (adjusted Rand index).
3. Hierarchical model: `L(i,s) = μ + α·[part(i) ≠ part(s)] + β·[subnode(i) ≠ subnode(s)] + γ(gpc(i)) + κ(cpc(i)) + τ(tpc position) + δ(fbpa(s)) + e`; report each coefficient with CI — these are the "distance" terms the 3D-stacking study will remove or shrink.
4. DSMEM matrix (from `10`) with the same hierarchical treatment (TPC / CPC / GPC levels).
5. Bandwidth: per-(SM, slice) bandwidth histogram (uniform?); saturation point (SMs per slice); speedup table per level and GPC; contiguous vs distributed; compare GPC speedup × single-SM bandwidth with `05`'s 1.16 TB/s per GPC and the chip-wide 7.7 TB/s; K7 curve knee vs `05`'s L2 bandwidth.
6. Combined "memory-hierarchy distance map": for each SM, a vector (local smem, DSMEM mean/max within GPC and within CPC, L2 near slice min/mean/max, L2 far, DRAM near/far) — the per-SM profile that the stacking model consumes.

## CSV columns
`15__noc__sm_slice.csv`: `smid, gpc, cpc, slice_id, partition_slice, fbpa_id, first_touch_smid, pass ∈ {home, far_first, copy}, rep, median_cycles, mad_cycles, n`.
`15__noc__sm_partition_miss.csv`: `smid, gpc, partition, rep, median_cycles, mad_cycles`.
`15__noc__l2_pingpong.csv`: `smid_i, smid_j, slice_s, slice_s2, round_trip_cycles`.
`15__noc__contention.csv`: `stream_gpc, stream_smids, stream_slice, probe_smid, probe_relation, probe_slice, probe_latency_idle, probe_latency_loaded, probe_gbps_idle, probe_gbps_loaded`.
`15__noc__speedup.csv`: `gpc, order ∈ {tpc, cpc, interleaved}, n_sms, smids, op ∈ {ld, st}, scope ∈ {local, global}, launch_idx, bytes, ms, gbps, gbps_ncu, speedup`.
`15__noc__slice_bw.csv`: `sender_set ∈ {sm, gpc, n_of_gpc, contiguous, distributed}, smids, n_sms, slice_id, fbpa_id, slice_partition, sender_partition, launch_idx, bytes, ms, gbps`.
`15__noc__loaded.csv`: `threads_per_cta, ctas_per_sm, pattern, buffer_home ∈ {own, partition0, partition1}, delay_mode, delay_steps, launch_idx, smid, warp, lane, cycles, gbps_event, gbps_ncu`.
`15__noc__model.csv`: `matrix, term, estimate, ci_low, ci_high, r2`.
`15__noc__sm_profile.csv`: `smid, gpc, cpc, partition, subnode, smem, dsmem_cpc_mean, dsmem_gpc_mean, dsmem_max, l2_near_min, l2_near_mean, l2_far_mean, dram_near, dram_far`.

## Figures
1. **SM × slice heatmap** (132 × 96; rows ordered by GPC/CPC/partition, columns by partition/FBPA group) — the headline figure; home, far-first and copy panels.
2. SM × partition miss heatmap (132 × 2) next to the SM×SM DSMEM heatmap from `10` on the same colour scale (cycles).
3. Pearson-correlation heatmap of SM signatures with the `03` GPC boundaries drawn.
4. Model coefficients with CIs (partition, sub-node, GPC, CPC, TPC position, FBPA group).
5. Per-SM distance profile (small multiples or parallel coordinates).
6. Contention: probe latency and bandwidth vs stream target (same slice / same FBPA / same partition / other).
7. Input speedup vs N (read and store; TPC-, CPC- and interleaved order) per GPC, with the full-bandwidth line (GPC port from `05`).
8. Per-(SM, slice) bandwidth histogram; GPC → slice bandwidth per slice; saturation curve N SMs → one slice; contiguous vs distributed bars.
9. L2 loaded-latency histograms per injection rate and the latency–throughput curve (own-partition vs single-partition buffer).

## Sanity checks (agent-only)
Expected ranges for judging plausibility during result analysis. They are never presented as pass/fail and never quoted in the notebook. If a value falls outside, investigate (placement, clocks, SASS, cache state, ncu replay) and re-run. A value that survives is a finding about this device; describe it on its own terms.
Additive R² ≥ 0.85; SM-term clustering and the Pearson heatmap recover the `03` GPC map (adjusted Rand index ≥ 0.9); partition coefficient ≈ 150–170 (`08`: 162), sub-node ≈ 29; slice term constant within `13` classes; K4: a slice saturates with ≈ 4–8 streaming SMs [P6]; K5: TPC speedup ≈ 2, CPC ≈ 5–6, GPC ≈ 8–9 read / ≈ 9–12 write [P6], GPC bandwidth ≈ `05`'s 1.16 TB/s; K6: per-(SM, slice) bandwidth uniform within ≈ 10 % and ≈ 30–45 GB/s per SM [P6], far-partition senders lower on the first pass only; K7: latency at 1 thread/CTA ≈ `04`'s 270, throughput at 1024 threads ≈ `05`'s 7.7 TB/s; the notebook writes an explicit "inferred interconnect" paragraph (levels found, latency per level, confidence).

## NVIDIA tools
`ncu --metrics lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_read.max,lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,lts__t_sectors_srcunit_ltcfabric.sum,l1tex__m_xbar2l1tex_read_bytes.sum.per_second,lts__t_bytes.sum.per_second,sm__cycles_elapsed.avg.per_second --clock-control none --cache-control none --replay-mode application --csv --page raw --print-units base` on samples of K1 (`.max == .sum` proves the chased lines hit only the intended slice; `ltcfabric` proves far accesses cross the fabric), K5–K7 (throughput cross-check; `.max/.avg` of `lts__t_sectors_srcunit_tex_op_read` shows the slice imbalance under restricted streams); K4 runs without ncu (two streams) or as one `--replay-mode range`; clock locking and `clocks_event_reasons` logging for the multi-hour run.
