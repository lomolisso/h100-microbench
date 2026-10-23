# 08 — L2 partition NUCA: near/far hit latency per SM, first-touch homing, replication, and partition hash bits

> **Device configuration:** runs under [`docs/context/10__device_config.md`](../context/10__device_config.md): H100 NVL (GPU 2), 132 SMs, 60 MiB L2, HBM3 3.94 TB/s, SM clock locked at 1635 MHz, power limit 400 W, ECC on, CUDA 12.9.1 `sm_90a`. Reference numbers quoted in this spec come from **other chips and configurations** (H800 PCIe, H100 PCIe, H100 SXM, H200, RTX 5090; see `10` §4). They are kept unchanged for the agent's sanity checks only, never cited in the notebook and not expectations for this device.

## Goal
For every SM: the near-hit and far-hit L2 latency, which partition it is "near", and the sub-node offset if any; plus confirmation (or refutation) on this card of DGNA's first-touch homing and remote-line replication, and recovery of the XOR partition hash bits accessible through 2 MB pages. Fills the `partition` / `subnode` columns of `03__sm_gpc_topology__map.csv`.

## Background
`docs/context/06__l2_nuca_and_physical_topology.md` §1.1, §2.2. References: near/far hit 258/414 (H800), 295/470 (H100), sub-node +32 cycles (H100), partition hash XOR over bits ≥12 with 14 bits (H100 PCIe), unique capacity ≈ 65 % [LIT].

## Kernels

### K1 — single-line latency probe (the workhorse)
Block of 1 thread pinned to a chosen `%smid` (launch 132 blocks, all exit except the one whose `%smid == target`). For a given 128 B line `L`: (1) *touch phase* (optional, performed by a different block pinned to `first_touch_smid` before this block starts; sequencing by a global flag), (2) *warm*: one load of `L`, (3) *measure*: 64 dependent loads of `L` (`p = *(T**)p` where `L` points to itself), each individually timed with `%clock64` → 64 samples; report median. `ld.global.cg.u64` by default (bypasses L1, so repeated hits are served by L2; confirm with ncu `l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum == 0`); control variant with `ld.global.ca.u64` (repeated hits then come from L1 → expect ≈ 32 cycles, which validates that the `.cg` numbers are L2 numbers).

### K2 — line sweep for hash bits
Allocate 1 GB (`cudaMalloc`, 2 MB pages). Base line `L0` at a 2 MB-aligned offset. For each bit b ∈ [7, 30]: line `L0 ^ (1<<b)`. From one fixed probe SM measure K1 latency (first-touched by the probe SM itself) for `L0` and each flipped line, 32 lines per bit (varying other random bits), 64 samples each. A bit "participates" if the median latency class flips for > 90 % of the pairs. Also measure pairwise XOR combinations of participating bits to test linearity. Repeat from a probe SM of the other partition.

### K3 — first-touch / replication test
Lines first touched by SM A (partition α) are read by SM B (partition β, from `03` candidates): (i) B's first read latency (expected far-hit ≈ 414–470), (ii) B's 2nd…64th reads (if a replica is created in β: drops to near ≈ 258–295; else stays far), (iii) after B's reads, A's read latency (still near?), (iv) after A *writes* the line, B's read latency (replica invalidated?). Use `%globaltimer` + global flags to sequence A and B.

### K4 — effective capacity
From one SM: random-permutation chase at 128 B granularity over footprints 16 … 60 MB (steps 2 MB; on this device extend to 80 MB so the sweep passes its 60 MiB L2, `docs/context/10__device_config.md` §3), 3 reps; hit rate from ncu `lts__t_sector_hit_rate.pct` and latency plateau. Lines first touched by (a) the probe SM only, (b) alternately by an SM of each partition. The footprint at which misses begin tells the usable capacity (expect ≈ 32 MB ≈ 65 % if replication halves the far-partition share).

### K5 — per-SM signature
For every one of the 132 SMs: K1 against a fixed panel of 256 lines (128 known-near-for-partition-α, 128 known-near-for-β, obtained from K2 on two reference SMs, all first-touched by their near SM); report the two medians → `partition` label and the near/far latencies per SM; the residual spread among near-latencies across SMs of the same partition reveals the sub-node (+≈32) structure → `subnode` label.

## CSV columns
`08__l2__probe.csv`: `kernel, probe_smid, first_touch_smid, line_offset, sample_idx, cycles, load_variant`.
`08__l2__hashbits.csv`: `probe_smid, bit, pair_idx, base_offset, flipped_offset, base_median, flipped_median, flipped_class_changed`.
`08__l2__replication.csv`: `a_smid, b_smid, phase ∈ {b_first, b_repeat, a_after, b_after_a_write}, median_cycles`.
`08__l2__capacity.csv`: `touch_pattern, footprint_bytes, cycles_per_hop, lts_hit_rate_pct`.
`08__l2__sm_signature.csv`: `smid, near_median, far_median, partition, subnode_offset`.

## Figures
1. Per-SM near/far latencies as a 132-cell tile map (ordered by GPC from `03`) — two panels.
2. Histogram of all K5 samples (expect 2 or 4 modes).
3. Bit-participation chart for the partition hash (bits 7–30).
4. Replication timeline (K3 phases) for 3 SM pairs.
5. Effective-capacity curve (K4).

## Sanity checks (agent-only)
Expected ranges for judging plausibility during result analysis. They are never presented as pass/fail and never quoted in the notebook. If a value falls outside, investigate (placement, clocks, SASS, cache state, ncu replay) and re-run. A value that survives is a finding about this device; describe it on its own terms.
Two latency modes per SM separated by ≥ 100 cycles; all SMs of a GPC share the same partition label; hash lowest bit = 12 (if 2 MB pages hold); K3 either confirms replication (phase ii near) or not — both are valid results, the notebook states which.
- Partition hash vs Fan et al. (lowest bit 12, 14 bits on H100 PCIe).

## NVIDIA tools
`ncu --metrics lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum,lts__t_sectors_srcunit_ltcfabric.sum,lts__t_sector_hit_rate.pct --print-metric-instances values --cache-control none --clock-control none` on K1/K5 to (a) show that far reads cross the fabric (`ltcfabric` > 0) and (b) see which slice instance serves each line (feeds `13`).
