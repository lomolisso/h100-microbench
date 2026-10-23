# 10 — SM×SM DSMEM latency heatmap per GPC: does physical distance inside a GPC matter?

## Goal
A full `smid_src × smid_dst` matrix of remote DSMEM load latency (and one-way store-visibility latency) for every SM pair that can share a cluster, ordered by the GPC map from `03`, to detect any sub-GPC structure (same TPC, same CPC/6-SM group, same switch plane). This is the central input of the 3D-stacking sensitivity analysis: if the fabric is flat, stacking gains come only from the fabric's base latency; if it is hierarchical, the gains are pair-dependent.

## Background
`docs/context/06__l2_nuca_and_physical_topology.md` §2.4; `docs/context/00__h100_architecture.md` §3 (CPC = 3 TPCs = 6 SMs, GX0/GX1 planes [PAT/INF]). References: ClusterSim heatmap 187–218 cycles "fairly uniform" (CS 8, one card) [P3]; H200 directed pairs 31.5–32.0 GB/s with no visible topology [blog]; Blackwell cross-die step > 300 cycles found with exactly this method [SemiAnalysis]. CPU analogues needed 10^5 samples per pair to resolve single-hop differences.

## Method
1. Reuse K1 of `09` (variant (a), 1 thread, 4096-hop chase, 200 KB smem → 1 CTA/SM) with **CS 16** (covers all pairs inside every ≥16-SM GPC) and **CS 8** (covers the 8-SM domain and gives a second sample of 16-SM GPCs); 1 cluster per launch to keep the rest of the chip idle; ≥ 200 launches per CS so that each SM pair inside a GPC is sampled from ≥ 20 different launches (the hardware chooses which 16 of 16–18 SMs; vary nothing but repeat — if some SMs are never chosen, add a second concurrent "blocker" kernel occupying specific SMs via `%smid` filtering to force different subsets).
2. Per pair: ≥ 20 launches × 4096 hops → median cycles/hop and MAD; also the fine-grained 256-hop series per launch for the distribution.
3. Store-visibility one-way (K2 of `09`, `st.async` + mbarrier variant and plain store variant) for every pair, 1000 ping-pongs per pair per launch.
4. Analysis: order rows/cols by GPC then by `smid`; within each GPC test (i) same-TPC pairs `(2k, 2k+1)` vs others, (ii) a 6-SM block structure (try all partitions of the GPC's SMs into groups of 6 consistent with TPC pairs; pick the one minimising within-group variance; report the F-statistic), (iii) symmetry `L(i,j)` vs `L(j,i)`; (iv) difference between the two 18-SM GPCs and the 16-SM GPCs. Additive-model fit `L(i,j) = μ + a(i) + b(j) + e(i,j)` and report whether the residual has block structure.
5. Repeat under light background load (all other SMs running a local-SMEM-only kernel) to check for interference-free behaviour, and under heavy load (`11` ring traffic in other clusters) to see whether the fabric is shared across clusters/GPCs.

## CSV columns
`10__heatmap__pairs.csv`: `cs, launch_idx, cluster_id, src_rank, dst_rank, smid_src, smid_dst, gpc, cycles_per_hop_median, cycles_mad, n_hops, load_condition`.
`10__heatmap__visibility.csv`: `variant, smid_src, smid_dst, gpc, one_way_cycles_median, mad, n`.
`10__heatmap__model.csv`: `gpc, mu, r2_additive, f_stat_tpc, f_stat_cpc, best_cpc_partition`.

## Figures
1. One heatmap per GPC (SM×SM, cycles), colour scale shared across GPCs, with TPC pairs marked.
2. Whole-chip 132×132 heatmap (NaN for pairs that cannot co-schedule) ordered by GPC — shows the block-diagonal structure.
3. Violin plot of pair latencies split by relation: same TPC / same hypothesised CPC / other.
4. Load vs remote latency (idle / light / heavy).

## Pass criteria
Matrix coverage ≥ 95 % of co-schedulable pairs; per-pair MAD ≤ 2 cycles; the notebook must state a conclusion: "flat within ±X cycles" or "structured: …".

## NVIDIA tools
Same ncu checks as `09`; additionally `l1tex__m_l1tex2xbar_req_cycles_active_mem_dshared_loopback.sum` must be 0 (no self-pairs in the matrix runs).
