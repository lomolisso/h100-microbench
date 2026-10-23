# 03 — Mapping logical SM ids to GPCs (and TPC pairs) on this card

> **Device configuration:** runs under [`docs/context/10__device_config.md`](../context/10__device_config.md): H100 NVL (GPU 2), 132 SMs, 60 MiB L2, HBM3 3.94 TB/s, SM clock locked at 1635 MHz, power limit 400 W, ECC on, CUDA 12.9.1 `sm_90a`. Reference numbers quoted in this spec come from **other chips and configurations** (H800 PCIe, H100 PCIe, H100 SXM, H200, RTX 5090; see `10` §4). They are kept unchanged for the agent's sanity checks only, never cited in the notebook and not expectations for this device.

## Goal
Produce `results/03__sm_gpc_topology__map.csv`: for every `%smid` on the device, its co-scheduling domain (GPC id), inferred TPC mate, and (later, from `08`/`15`) its L2 partition and sub-node. Every heatmap in later experiments orders SMs by this map, and the 3D-stacking sensitivity analysis uses it to define "physical distance".

## Background
`docs/context/06__l2_nuca_and_physical_topology.md` §1.2, §2.1; `docs/context/00__h100_architecture.md` §1. Reference: H100 SXM co-scheduling domains `18,18,16,16,16,16,16,8,4,2,2` on one card (ClusterSim), H200: six 16-SM + two 18-SM GPCs on 23/28 cards (Fan et al.), H100 PCIe: 6 × ≥16 + 1 × ≥14 (Fan et al.). `%smid` ↔ physical position is driver-assigned and partly card-specific.

## Method (unprivileged; no nvdebug required)

### Step 1 — cluster co-scheduling probe
Reuse K1 of `02` (`sleep_kernel`, 1024 threads, 200 KB dynamic smem → 1 CTA/SM, 100 000-cycle spin). For CS ∈ {16, 8, 4, 3, 2}:
- grid = `cudaOccupancyMaxActiveClusters × CS` (one wave) and grid = 3 waves; policy Default; 50 launches each; between launches vary `smem_bytes` ∈ {200 KB, 120 KB, 80 KB} and block size ∈ {1024, 512} so that occupancy, and thus the SM subsets chosen, change.
- From every launch record `(cluster_id, rank, smid)`; add an edge between all smids of one cluster; union-find → components = co-scheduling domains. Count, per component, the SM count and how many clusters of each CS were ever hosted there.
- SMs never observed in a cluster of size > 2 are assigned to the component of their size-2 partners; if an SM only ever appears alone (CS 1 launches), mark `gpc = unknown`.

### Step 2 — consistency and sizes
- Expected total = `multiProcessorCount`. Components with 16–18 SMs are GPCs. A component larger than 18 means the union merged two GPCs (impossible if co-scheduling is honoured) → flag as a bug in the probe (e.g. records from different launches merged).
- Count the number of GPCs able to host CS 16 (expected 6–8 on SXM5).

### Step 3 — TPC mates
Hypothesis H_tpc: smids (2k, 2k+1) share a TPC. Test later with `10__dsmem_latency_heatmap` (lowest-latency partner) and with `15` (identical slice signature). Here only record the hypothesis column `tpc_mate_hyp = smid ^ 1`.

### Step 4 — L2 partition affinity (filled by `08`)
Column `partition` ∈ {A, B} and `subnode` ∈ {0,1} are left empty here; `08` fills them (near-hit latency signature per SM); `15` refines.

### Step 5 — optional privileged cross-check
If root is available: `nvidia-smi mig -lgip` and MIG profile SM counts; the `nvdebug` kernel module (GPC/TPC disable masks). Record in the notebook whether it was possible; do not require it.

## CSV columns
`03__sm_gpc_topology__edges.csv`: `launch_id, cs, policy, smem_bytes, block_threads, cluster_id, rank, smid`.
`03__sm_gpc_topology__map.csv`: `smid, gpc, gpc_size, hosts_cs16 (0/1), tpc_mate_hyp, partition, subnode, notes`.

## Figures
1. 132-cell tile map (12 × 11) of smids coloured by GPC, annotated with GPC size.
2. Adjacency heatmap (smid × smid) = number of times two SMs were in the same cluster; block structure shows the GPCs.
3. Bar chart of the GPC sizes found on this card.

## Sanity checks (agent-only)
Expected ranges for judging plausibility during result analysis. They are never presented as pass/fail and never quoted in the notebook. If a value falls outside, investigate (placement, clocks, SASS, cache state, ncu replay) and re-run. A value that survives is a finding about this device; describe it on its own terms.
Σ sizes = SM count; no component > 18; ≥ 6 components of size ≥ 16 on SXM5 (≥ 6 of ≥ 14 on PCIe). The notebook must state the exact composition found on this card, since it is card-specific.
- GPC sizes vs ClusterSim's `18,18,16,16,16,16,16,8,4,2,2` (H100 SXM) and Fan et al.'s H200 `6×16 + 2×18`.

## NVIDIA tools
None beyond the CUDA runtime; `nvidia-smi -q` (serial number, to tie the map to the physical card) recorded in the CSV `notes`.
