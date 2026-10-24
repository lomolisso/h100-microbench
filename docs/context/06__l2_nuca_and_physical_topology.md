# L2 NUCA, address hashing, and recovering the physical topology from latency

Goal of this context file: give the coding agents (a) what is known about how H100 addresses map onto L2 partitions/slices and DRAM channels, (b) what is known about how SMs map onto GPCs and partitions, and (c) the proven methodology for turning latency measurements into a topology estimate. The experiments that implement this are `03__sm_gpc_topology`, `08__l2_partitions_and_first_touch`, `13__l2_slice_mapping`, `14__gmem_latency_vs_controller_distance`, `15__noc_latency_heatmaps`.

## 1. Known facts (tagged [NV] official, [LIT] measured by prior work, [INF] to be confirmed)

### 1.1 L2 partitions
- The L2 is a "partitioned crossbar structure"; each partition "localizes and caches data for memory accesses from SMs in GPCs directly connected to the partition" [NV, Hopper blog].
- Two partitions on H100 (≈25 MB each) [LIT]. The far partition costs ≈ +150–175 cycles on a hit (near 258 / far 414 cycles on H800 PCIe, Luo; 295 / 470 on H100, DGNA) and the same offset on a miss (555 / 744 cycles, Luo) [LIT].
- DGNA [LIT]: inside a partition there is a **sub-NUMA** level (+≈32 cycles; two local-hit peaks at ≈280 and ≈312 cycles on H100); topology 2 GPCs per sub-node, 4 GPCs per partition; **first-touch homing** (a line is homed where the SM that first touched it lives) and remote readers keep a **replica** in their local partition (Fan et al.: effective unique capacity ≈ 65 %). → Our p-chase must control *which SM first touches each line*; when a far-partition line is re-read repeatedly from a near SM, the replica may make it look "near" after the first access [INF, must test in `08`].
- L2 line 128 B, sector 32 B, 80 slices ("lts" instances in ncu; 640 KB each), 8 slices per 512-bit memory controller [LIT/INF].
- Partition hash: XOR of physical address bits ≥ 12 (4 KB granularity); 14 participating bits on H100 PCIe, 16 on H200/B200 [LIT, Fan et al.]. Slice hash within a partition: **unknown** (non-power-of-two slice count → likely IPOLY/modulo style as modelled by accel-sim `memory_partition_indexing 6`) [INF].
- Physical vs virtual addresses: `cudaMalloc` allocations ≥ 2 MB use 2 MB pages, so address bits 0–20 of the virtual offset equal the physical bits. Hash bits above bit 20 can only be probed statistically (many 2 MB pages) [INF based on Fan et al. methodology].

### 1.2 SMs and GPCs
- Logical `%smid` is assigned by the driver; the smid→(GPC, TPC) map has a fixed architecture-defined part plus a floor-sweeping-dependent part that differs across cards [LIT, Fan et al., Bakita].
- Cluster co-scheduling (all CTAs of a cluster in one GPC) is the only unprivileged way to recover GPC membership; it agreed 100 % with the privileged `nvdebug` kernel module on Fan et al.'s cards [LIT].
- SMs sharing a TPC share the TPCARB/texture path [PAT]; whether they show a measurable DSMEM latency advantage is [INF].
- Each GPC attaches to one partition through its GPC crossbar port; SMs of the same GPC therefore have identical near/far classification for every address, and the SM-effect in an SM×slice latency matrix should cluster by GPC and by partition [LIT (L40 paper, DGNA), INF for H100].

### 1.3 NoC observations from Jin et al. (MICRO 2024, V100/A100/H100) [LIT]
- SM→L2-slice round-trip latency is **non-uniform by placement, up to ≈70 % spread**; bandwidth from any SM to any partition is approximately uniform; crossing the partition boundary adds a discrete latency step and reduces bandwidth.
- Method: one thread, loads with `ld.global.cg` (L1 bypass), source SM chosen via `%smid`, target slice chosen by address using the profiler's per-slice counters. Caveat raised by DGNA: `.cg`-based numbers may include replica effects; use dependent chains and control first-touch.

### 1.4 DRAM side
- No public channel/bank mapping for H100 HBM3 [LIT survey]. GPUHammer's row-buffer-conflict timing method (GDDR6 A6000: 256 B chunks map to one row/bank, successive chunks rotate banks, threshold ≈25–30 ns between hit and conflict) is the template; HBM3 has 16 channels × 2 pseudo-channels per stack, 5 stacks → 160 pseudo-channels, 16 banks (4 bank groups) each [INF from HBM3 spec; accel-sim config uses `nbk=16`, `nbkgrp=4`].
- Whether SM→memory-controller distance is visible in DRAM latency is [INF]; the partition split already contributes ≈ +175 cycles on a miss [LIT], so the question is whether a *second-order* term (which of the 5 stacks / 10 controllers) exists.

## 2. Methodology (synthesised from Jin et al., Luo et al., DGNA, Fan et al., Alpay & Basaran L40, SemiAnalysis Blackwell, Lord of the Ring(s))

### 2.1 smid → GPC (and TPC) map — `03__sm_gpc_topology`
1. Launch 16-CTA clusters (non-portable) of a kernel that occupies one CTA per SM (≥ 200 KB dynamic smem, 1024 threads) and spins ≈ 100 000 cycles; record `(clusterid, rank, smid)`. Repeat for cluster sizes 8, 4, 3, 2 and for hundreds of launches with different grid sizes so that different SM subsets are used. Union-find on "same cluster id" edges → co-scheduling domains. Cluster size 16 immediately exposes ≥16-SM GPCs; sizes 3/5 catch odd TPC counts; SMs that never appear in clusters > 2 belong to heavily floor-swept GPCs (Fan et al. saw smid 124–131 never in clusters > 2).
2. TPC pairs: with cluster size 2 and 1 CTA per SM, the pair chosen by the hardware is not guaranteed to be a TPC; instead derive TPC pairing from the DSMEM latency matrix (pairs with the lowest latency, if any), and from the fact that `%smid` pairs (2k, 2k+1) are TPC-mates on all NVIDIA GPUs examined so far [INF].
3. Cross-check with MIG if the card allows it (`nvidia-smi mig -lgip`: 1g.10gb = 14 SMs = one GPC) and with the partition signature from §2.3.

### 2.2 address → partition / slice — `13__l2_slice_mapping`, `08__l2_partitions_and_first_touch`
1. **Counter method (exact, no timing):** ⚠ *Not possible as written on our setup:* Nsight Compute exports no per-slice instance values (AGENTS.md "Open issues", OI-1; replacement proposal there). run a kernel that touches exactly one 128 B line *k* times from a single SM under `ncu --metrics lts__t_sectors_op_read.sum,lts__t_sectors_srcunit_tex_op_read.sum --print-metric-instances values`; the slice instance whose counter increments by *k* sectors is the owner. Sweep the line address over bits 7…20 (one bit at a time, plus random pairs) and over several 2 MB pages; solve for the hash: a partition/slice function is linear over GF(2) if `slice(a ⊕ b) = slice(a) ⊕ slice(b)` for all tested pairs; otherwise fit the modulo structure (slice = (XOR-folded index) mod 80 or mod (slices per partition)). Also record `lts__t_sectors_srcunit_ltcfabric` (traffic between partitions) to tell near from far. Expected [INF]: partition = XOR of ≥14 bits starting at bit 12; slice inside partition depends on bits 7–11 plus higher bits.
2. **Timing method (per SM, per line):** fine-grained p-chase (Mei & Chu; Luo) — one dependent load per line, `%clock64` before/after each load, 128 B stride, random permutation, working set 4–16 MB (hits) or ≥ 256 MB (misses); ≥ 100 samples per line; 1-D K-means with k = 2 (hits) or 4 (hits + misses) and report cluster centres, SD, CV (Luo: CV ≤ 9 %). Then verify that the timing labels agree with the counter-derived partition labels.
3. **First-touch control:** for each line, the SM that first touches it is recorded; repeat the measurement with the line first touched by a far-partition SM; if latency changes, DGNA's first-touch homing/replication is confirmed on our card and the notebook must say so.

### 2.3 SM × slice latency matrix — `15__noc_latency_heatmaps`
- For every SM *i* (pinned via `%smid`, one thread) and every slice *s* (addresses from §2.2.1): warm the lines into L2 from the *same* SM, then chase a short chain whose lines all belong to *s*; record median and MAD of ≥ 1000 dependent accesses. Lock clocks; keep one thread active on the chip (background SMs idle) to avoid contention.
- Fit the additive model `L(i, s) = μ + a(i) + b(s)` (+ one rank-1 term), report R²; L40 reference: R² 0.87 / 0.98, SM-term span 57 cycles, slice-term span 40 cycles. The SM term `a(i)` should split by partition (≈ +150 cycles) and sub-node (≈ +32 cycles); hierarchical clustering of the rows gives GPC groups independently of §2.1 (cross-validation).
- Interpretation rule (AnandTech caveat): an SM→SM latency measured *through L2* is a 3-point path (SM_i → home slice → SM_j) and must not be read as a distance between the SMs; use DSMEM for true 2-point SM↔SM latency.

### 2.4 SM × SM DSMEM latency matrix — `10__dsmem_latency_heatmap`
- Cluster size 16 and 8 so that every pair inside a GPC co-occurs; rank *r* chases into rank *r'* via `mapa` + `ld.shared::cluster` (dependent chain, 1 thread), serialised across source ranks with `cluster.sync()`; also measure the 2-point store→flag→load ping-pong (`st.async` + mbarrier) to get one-way latency; ≥ 1000 samples per pair. Bin by `(smid_src, smid_dst)` across launches; plot per GPC with rows/columns ordered by the §2.1 map. Reference: 184–218 cycles, flat [LIT]; H200 one-way 78 cycles [LIT].
- Test for structure: same-TPC pairs, same-CPC (6-SM group) pairs, and cross-CPC pairs. A statistically significant difference (> 3 MAD) is the first evidence of a hierarchical fabric; none means a single-stage crossbar at the resolution of `%clock64`.

### 2.5 DRAM channel distance — `14__gmem_latency_vs_controller_distance`
- Fine-grained p-chase on a ≥ 1 GB buffer with L2 bypass (`ld.global.cv` or working set ≫ L2 and random order), classify per-line latency, and correlate with (a) partition label from §2.2, (b) address bits 8–20 (channel interleave candidates). Row-buffer-conflict pairs (GPUHammer method: alternate between two lines, look for a ≥ 25 ns step) identify same-bank lines; this yields the channel/bank hash structure needed to say whether an SM is "closer" to a controller. Prior expectation: partition term dominates; controller term, if any, is small [INF].

### 2.6 Statistics and reporting
- Per-pair medians with MAD; ≥ 1000 samples for matrices, ≥ 100 per line for maps; report clock, SKU, driver.
- Always keep raw per-sample CSV rows so the notebooks can re-cluster.
- Jitter reference: 0.09 cycles median temporal jitter over 6 h at full load for an SM×region signature (RTX 5090) [LIT] — i.e. differences of a few cycles are real if sample counts are adequate.
