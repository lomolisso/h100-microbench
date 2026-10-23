# 02 — Cluster launch overhead, achievable cluster occupancy, and scheduling-policy placement

> **Device configuration:** runs under [`docs/context/10__device_config.md`](../context/10__device_config.md): H100 NVL (GPU 2), 132 SMs, 60 MiB L2, HBM3 3.94 TB/s, SM clock locked at 1635 MHz, power limit 400 W, ECC on, CUDA 12.9.1 `sm_90a`. Reference numbers quoted in this spec come from **other chips and configurations** (H800 PCIe, H100 PCIe, H100 SXM, H200, RTX 5090; see `10` §4). They are kept unchanged for the agent's sanity checks only, never cited in the notebook and not expectations for this device.

## Goal
Quantify what it costs to *launch* a cluster kernel relative to a non-cluster kernel with identical grid/block, how many clusters of each size can be co-resident (and how many SMs stay idle), and how `cudaClusterSchedulingPolicy` {Default, Spread, LoadBalancing} changes rank→SM placement. This is the fixed-cost baseline for every DSMEM collective.

## Background
`docs/context/01__clusters_and_dsmem.md` §2, §5. Reference numbers: launch overhead ≈0 % (CS 2), ≤5 % (CS 4/8), ≤22 % (CS 16) when the grid is larger than one wave; CS 16 leaves 18 SMs idle on H100 PCIe, CS 8 leaves 4, CS 4 leaves 4, CS 2 none [P2]; co-resident clusters on B200 15×8, 33×4, 7×16 [lit]; LoadBalancing may co-locate CTAs of one cluster on an SM [P1].

## Kernels

### K1 — `sleep_kernel(cycles)`
Every block spins `__nanosleep`-free on `clock64()` until `cycles` (default 100 000) have elapsed, then thread 0 writes `(blockIdx, %smid, %clusterid.x, %cluster_ctarank, start_globaltimer, end_globaltimer)` to a global record. Block = 1024 threads, dynamic smem = `smem_bytes` (sweep below). Launched (a) without cluster attribute, (b) with CS ∈ {1 (explicit cluster of 1), 2, 4, 8, 16} via `cudaLaunchKernelEx`; CS 16 needs `cudaFuncAttributeNonPortableClusterSizeAllowed`.

### K2 — occupancy probe
Same kernel but `cycles` large (≈ 2 M) so that all co-resident blocks overlap; grid = `cudaOccupancyMaxActiveClusters × CS` and separately grid = 4 × that. From the records compute the number of distinct `(clusterid)` whose blocks all started within the first wave (start time < min end time), the set of SMs used, and the idle SM count.

## Procedure
1. Grid sweep: 1, 8, 16, 32, 64, 128, 132, 264, 528, 1056, 2112 blocks (multiple of CS; pad to the next multiple); smem_bytes ∈ {0, 64 KB, 200 KB} (1 CTA/SM at 200 KB); 5 repetitions each after 2 warm-ups; time with cudaEvents around the launch; subtract the event floor from `01`.
2. Overhead(CS, grid) = T(CS) / T(no cluster) − 1. Also report T per wave.
3. K2 for CS ∈ {2, 4, 8, 16} and policy ∈ {Default, Spread, LoadBalancing}; record `cudaOccupancyMaxActiveClusters` output alongside.
4. For every launch, store the full placement record; the `03` experiment reuses these CSVs.
5. Placement invariants to test from the records: (i) CTAs of one cluster never share an SM under Default/Spread; (ii) under LoadBalancing count the fraction of clusters with ≥2 CTAs on one SM; (iii) rank→smid stability across repeated launches with identical configuration (fraction of clusters whose rank→smid tuple is identical to the previous launch).

## CSV columns
`02__cluster_launch__overhead.csv`: `cs, policy, grid_blocks, smem_bytes, rep, ms_event, ms_floor, overhead_pct`.
`02__cluster_launch__placement.csv`: `cs, policy, grid_blocks, smem_bytes, rep, block_idx, cluster_id, rank, smid, t_start_ns, t_end_ns`.
`02__cluster_launch__occupancy.csv`: `cs, policy, smem_bytes, api_max_active_clusters, measured_first_wave_clusters, sms_used, sms_idle`.

## Figures
1. Overhead % vs grid size, one line per CS.
2. Tile map of SMs (sorted by smid) coloured by cluster id for CS 8 and CS 16 — the notebook must also draw it with SMs ordered by the GPC map from `03` once available.
3. Bar chart: measured first-wave clusters vs API prediction per CS and policy; idle SMs per CS.
4. Table of placement invariants (i)–(iii).

## Sanity checks (agent-only)
Expected ranges for judging plausibility during result analysis. They are never presented as pass/fail and never quoted in the notebook. If a value falls outside, investigate (placement, clocks, SASS, cache state, ncu replay) and re-run. A value that survives is a finding about this device; describe it on its own terms.
Overhead(CS 2) < 2 %; Overhead(CS 16, 2112 blocks) in [10 %, 30 %]; idle SMs at CS 16 reproduced within ±4 of P2 (which card-dependent); Default/Spread never co-locate.
- Figure 1 shape vs P2 Fig. 3a; figure 2 placement pattern vs P2 Fig. 3b/c.

## NVIDIA tools
cudaEvents (primary), `nsys profile --trace cuda --stats=true` on one configuration to show the launch gap directly (`cuda_gpu_kern_sum`, `cuda_api_sum`); `ncu --metrics sm__cycles_active.avg,sm__cycles_elapsed.avg,launch__*` (launch statistics section: `launch__cluster_dim_x`, `launch__cluster_max_active`, `launch__cluster_max_potential_size`, `launch__cluster_scheduling_policy` — verify names with `ncu --query-metrics | grep -i cluster`).
