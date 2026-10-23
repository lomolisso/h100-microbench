# Prior-work reference numbers (H100-class GPUs)

Purpose: every number below was measured on real Hopper silicon by a published microbenchmarking study. They are the **targets to reproduce** in our experiments. When one of our notebooks reproduces a number within its stated tolerance, the notebook must cite the row below ("matches Lühnen et al. 2024, Fig. 4a").

Rules for using these numbers:

1. Record the GPU variant (H100 SXM5 = 132 SMs, H100 PCIe = 114 SMs, H800 PCIe = 114 SMs, H200 = 132 SMs) and, where known, the SM clock. Cycle counts are comparable across variants; TB/s numbers are not (clock and SM count differ).
2. "Cycles" below are SM clock cycles measured with `clock()`/`clock64()` (per-SM counter) unless stated otherwise.
3. Papers disagree with each other by 5–15 % on identical quantities (different measurement loops, compilers, clocks). Our tolerance for "reproduced" is ±10 % for latency, ±15 % for bandwidth unless an experiment file says otherwise.

## P1 — Luo et al., "Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis", ACM TOCS 2026 (extends IPDPS 2024 "Benchmarking and Dissecting the Nvidia Hopper GPU Architecture", arXiv:2402.13499). GPU: **H800 PCIe** (114 SMs, 1755 MHz, 50 MB L2, HBM2e 2039 GB/s), CUDA 12.6. Code: https://github.com/HPMLL/NVIDIA-Hopper-Benchmark

| Quantity | Value | Method |
|---|---|---|
| L1 hit latency | 32.0 cycles | random-stride p-chase, 64 B chunks, 1 thread |
| Shared memory latency (direct `ld.shared`) | 29.0 cycles | uniform 4 B stride p-chase |
| Local SMEM via DSMEM interface (`mapa` + generic `ld`, own rank) | 33 cycles | same |
| L2 latency (traditional p-chase, working set < ½ L2) | 264.5 cycles | random stride |
| L2 latency (working set between ½ L2 and full L2) | up to 502 cycles | random stride |
| Global (DRAM) latency | 656 cycles | random stride, working set ≫ L2 |
| Fine-grained p-chase (32 B stride) latency classes, K-means centres: L2 **near hit** / **far hit** / **near miss** / **far miss** | **258.0 / 414.1 / 555.5 / 743.7** cycles (SD 14.9 / 22.6 / 24.1 / 66.9) | 8 MB array → 2 classes (near/far hit); 40 MB array → near hit, near miss, far miss |
| Observation | traditional-p-chase global latency ≈ (near miss + far miss)/2 | — |
| L1 throughput (FP32 / FP32.v4) | 125.8 / 127.4 B/clk/SM | 1 block × 1024 threads |
| SMEM throughput (FP32 / FP32.v4) | 124.1 / 127.9 B/clk/SM | 1 block × 1024 threads |
| L2 throughput (FP32 / FP32.v4) | 4472.3 / 3942.4 B/clk (chip-wide) → ≈ 7.8 TB/s at 1755 MHz | 2×SMs blocks, data pre-loaded with `.cg` |
| Global memory throughput | 1861.5 GB/s (91 % of 2039 GB/s theoretical) | float4, 5 reads + 1 write per thread, 4×SMs blocks |
| L2 : DRAM throughput ratio | 4.23× | — |
| TMA latency (1D `cp.async.bulk` 16 B–, incl. mbarrier init/expect/wait) | ≈ regular-load latency + ~170 cycles; **no L1 inflection** (TMA served by L2) | random stride p-chase through TMA |
| TMA throughput, non-tensor / 2D / 3D tensor, large boxes, ≥228 blocks | > 1800 GB/s (≈ DRAM peak) | 4 GB streamed, box sizes 1–16 KB |
| TMA 1D-tensor max box | 256 elements → 2 KB (int64), lower throughput | — |
| TMA box shape effect (16 KB, float): large innermost (x) dim best; large y/z worse | 256×16×1 ≈ 64×64×1 > 16×256×1 > 4×4×256 | Fig. 5 |
| DSMEM remote latency, cluster size 2 | **181 cycles** (≈ 30 % lower than L2 hit) | uniform-stride p-chase, 1 thread/block, blocks on distinct SMs |
| DSMEM remote latency, cluster size 2…16 | **184 – 213 cycles** | same |
| SM→SM via global memory (store + load) | 1110 cycles | — |
| DSMEM local access via DSMEM interface throughput | 205 GB/s per SM (80 % of 1755 MHz × 128 B = 225 GB/s) | — |
| SM-to-SM aggregate throughput, ring pattern, cluster 2 / 4 | **3.28 TB/s / 2.78 TB/s** (chip-wide, 114 SMs) | 50 160 blocks, 1024 thr, 16 KB smem, 2 blocks/SM |
| Pattern ranking | ring ≈ pair ≫ broadcast (broadcast degrades strongly with cluster size) | Fig. 10 |
| Scheduling policy | LoadBalancing slightly > Default ≈ Spread; LoadBalancing may co-locate cluster blocks on one SM | Fig. 10 |
| Block size / ILP | block 64 under-utilises; ILP helps only for small blocks | Fig. 11 |

## P2 — Lühnen, Marschner, Lal, "Benchmarking Thread Block Cluster", HPEC 2024. GPU: **H100 PCIe** (114 SMs, 50 MB L2, 228 KB smem), CUDA 12.3.

| Quantity | Value | Method |
|---|---|---|
| `barrier.cluster.arrive` latency, 1 thread/block, cluster 1–8 | **≈ 1050 cycles** (median) | `clock()` around single inline-asm instruction |
| same, cluster 16 | ≈ 1200 cycles | — |
| `barrier.cluster.wait`, cluster 1–2 | **45 cycles**; up to 87 cycles for cluster > 4 | — |
| `cluster.sync()` (= arrive + wait) | up to ≈ **1300 cycles** median | — |
| `__syncthreads()` (comparable block dims) | 14 cycles | — |
| `grid.sync()` (cooperative groups) | ≈ 2200 cycles | — |
| Cluster launch overhead vs. non-cluster, same grid | cluster 2: ≈ 0 %; 4 and 8: ≤ 5 %; 16: up to **22 %** (grid larger than one wave) | cudaEvent timing, each block sleeps 100 000 cycles |
| SM utilisation at max occupancy | cluster 16 → 18 SMs idle; cluster 8 → 4 idle; cluster 4 → SMs 110–113 idle; cluster 2 → 100 % | `%smid` readout |
| GPC capacity inferred from cluster-16 placement (H100 PCIe) | 6 GPCs can host a 16-block cluster; one GPC hosts ≤ 14 blocks; two hosts ≤ 2 (per ClusterSim's reading of Fig. 3) | — |
| Placement invariants | blocks of one cluster never share an SM (Default policy); rank→SM assignment is stable across launches | — |
| Producer→consumer transfer, 32-thread blocks | **push** (producer `st` into consumer's smem) up to 2.3× faster than via global for 8 KiB; **pull** slower than global for ≤128 B | Fig. 5 |
| DSMEM store cost vs. datatype, 256 KiB, 32 threads | uint8 154.11 µs, uint16 77.06 µs, uint32 38.59 µs, uint64 19.36 µs → **cost per store instruction is width-independent** | Table II |
| 15-to-1 push (2 MiB each, cluster 16) | sequential: 308 µs per writer; simultaneous: 308–608 µs per writer (congestion); total 582–608 µs depending on receiver rank (±4 %) | `%globaltimer` (32 ns tick, cross-SM consistent) |

## P3 — Lühnen, Behera, Tripathy, Lal, "ClusterSim: Modeling Thread Block Clusters in Hopper GPUs", IISWC 2025. GPU: **H100 SXM** (132 SMs, 1620 MHz in their config), CUDA 12.8. Code: https://github.com/Tim453/ClusterSim

| Quantity | Value | Method |
|---|---|---|
| DSMEM latency matrix (sender × receiver), cluster 8 | **187 – 218 cycles**, "fairly uniform" (heatmap, Fig. 2a) | pointer chase, 1 thread/block, each SM reads from every other |
| Local SMEM latency | ≈ 40 cycles (simulator 46 incl. loop overhead) | same loop |
| Sequential one-reader bandwidth vs. threads: 128 / 256 / 512 / 1024 thr | 4.5–4.8 / ~9.5 / 19.5 / **21.1–21.3 B/cycle** per reader SM (saturates ~21.2 B/cycle ≈ 34 GB/s @1620 MHz) | parallel copy kernel |
| Pairwise (disjoint pairs, concurrent) | 16.1–17 B/cycle per SM | — |
| Ring (each reads next) | ≈ 13 B/cycle per SM ("up to 8 B/cycle below sequential") | — |
| Broadcast (N readers of one SM) | per-reader BW ≈ 21.2/N; 7 readers → 3.0–3.7 B/cycle (outliers 5.2) | — |
| Local SMEM bandwidth while being read remotely | drops from 118.5–120.3 to 71.3–102.7 B/cycle | — |
| All-to-all with N groups of 1024/N threads | equals broadcast with 1024/N threads | — |
| Topology conclusion | crossbar-like with fair arbitration; not ring/mesh (uniform latency), not fully connected (all-to-all degrades) | — |
| Simulator calibration constants | arrive 610 cycles, wait base 60 cycles, DSMEM fixed 198 cycles | — |

## P4 — Li, Huang, Ogras, "CREDIT: Cost-guided Reduction-reuse with Efficient DSMEM Inter-CTA Tiling", arXiv:2609.01864 (2026). GPU: **H100 SXM**, CUDA 13.0 (also RTX 5090). Code: https://github.com/zhengxiongli08/CREDIT

| Quantity | H100 | RTX 5090 | Method |
|---|---|---|---|
| Local smem load latency | 30 cycles | 34 | dependent loads, 2 CTAs on different SMs, median of 15 runs |
| Remote (DSMEM) load latency | **191 cycles** (6.4× local) | 216 | — |
| Cluster sync (`cluster.sync()`) | **851 cycles** | 404 | repeated syncs |
| Local read rate | 20.89 B/cycle/thread-block config (their "rate") | 20.89 | many independent reads |
| Remote read rate | 4.77 B/cycle (4.4–5.4× below local) | 3.84 | — |
| Local store rate | 27.67 B/cycle | 33.01 | — |
| Remote store rate | **21.37 B/cycle** | 2.14 | — |

(Interpretation: on H100, remote stores are ~4.5× faster than remote loads per issuing block; the 21.4 B/cycle remote-store rate coincides with the ~21 B/cycle per-SM SM-to-SM limit seen by P3 and P5.)

## P5 — Yu et al., "FlashGPU-sim: Enabling GPU Modeling for Modern Architectures and AI Workloads", arXiv:2609.15311 (2026). TMA/mbarrier numbers on RTX 5090 (2580 MHz); DSM numbers on **H200** (identical GH100 die to H100 SXM, 132 SMs). Cycle counts from `ncu`.

| Quantity | Value | GPU | Method |
|---|---|---|---|
| Same-warp `cp.async.bulk` issue gap | **44 cycles** per instruction (T(N)=1+44N, N=1..24) | 5090 | unrolled batches, waits outside timed region |
| 4 TMA transactions serial vs pipelined vs 4-subpartition parallel | 1669 → 410 cycles completion; issue span 248 → 58 cycles | 5090 | — |
| TMA front-end admission | **16 active transactions** per SM; 17th backpressures | 5090 | issue-time jump |
| TMA per-SM read bandwidth | **32 B/cycle** (ncu `l1tex__m_xbar2l1tex_read_bytes_mem_global_op*`); completion-time staircase of +49 cycles every 48 sectors (1536 B) | 5090 | single transaction size sweep 128 B–16 KB |
| Single-SM TMA DRAM-path sustained | 30.816 B/cycle; DRAM round trip 845.7 cycles → ≥ 814 outstanding 32 B requests | 5090 | 8 KiB transfers, 1 GiB working set |
| mbarrier `test_wait` / `try_wait` (no hint) predicate latency | **42 cycles** (SASS `SYNCS.PHASECHK.TRANS64`) | 5090 | — |
| `try_wait` with suspendTimeHint, barrier complete | 150 cycles regardless of hint (adds `NANOSLEEP.SYNCS`) | 5090 | — |
| TMA OOB handling | writes: no extra traffic; reads: filtered before DRAM; H100 filters part on-chip, 5090 forwards all OOB reads to L2 | 5090/H100 | ncu |
| Intra-GPC fabric model (from NVIDIA patent US 12,248,788 B2) | GPC = several CPCs; CPC = 3 TPCs (2 SMs each) + GPCMMU + GPCARB; two switch planes GX0/GX1; 4 links × 32 B/cycle per 6-SM CPC → **≈ 21.3 B/cycle/SM** | — | — |
| DSMEM remote load latency (mean) | **193.4 cycles** | H200 | — |
| Increment of remote load over local load | 156.4 cycles | H200 | — |
| One-way SM-to-SM latency | **78.2 cycles** | H200 | derived (half round trip) |
| Dependent remote-load round trip | 220.0 cycles | H200 | — |
| Remote **store visibility** latency (store → observed by owner) | **625.2 cycles** | H200 | flag polling |
| Remote load latency under concurrent disjoint pairs | 216.5 cycles | H200 | — |
| Remote load BW, unidirectional / bidirectional | **19.88 / 30.70 B/cycle** | H200 | — |
| Remote store BW, unidirectional / bidirectional | 18.95 / 36.31 B/cycle | H200 | — |
| TMA peer copy (smem→remote smem) BW, uni / bi | 21.06 / 41.37 B/cycle | H200 | — |
| Load + TMA co-direction / counter-direction | 21.38 / 28.80 B/cycle | H200 | — |
| Two SMs loading from each other simultaneously | per-direction **read** BW drops 23 %; store / TMA-put drop only 4–5 % | H200 | — |
| Same-direction two streams | saturate ≈ 21 B/cycle; opposite-direction streams 36–39 % higher → per-SM *send* budget | H200 | — |
| Unused neighbour capacity | not redistributed: one SM alone still ≈ 21 B/cycle | H200 | — |
| TMA peer-copy aggregate scaling 2→16 SMs | near-linear | H200 | — |
| H100 FA-3 profiling note | SM clock locked to 1.5 GHz for reproducible ncu cycles; ncu flags `--cache-control all --pipeline-boost-state stable` | H100 | — |

## Consolidated H100 cheat-sheet (medians across sources)

| Metric | Expected on H100 SXM5 | Sources |
|---|---|---|
| SMEM/L1 hit latency | 29–33 cycles | P1, P3, P4 |
| DSMEM remote load latency (dependent) | 180–220 cycles (≈ 6–7× local) | P1, P3, P4, P5 |
| L2 hit latency, near / far partition | ≈ 258 / 414 cycles | P1 |
| DRAM latency, near / far | ≈ 555 / 744 cycles (avg ≈ 650) | P1 |
| TMA load latency | ≈ L2/DRAM latency + ~170 cycles | P1 |
| `barrier.cluster.arrive` / `.wait` / `cluster.sync()` | ≈ 600–1050 / 45–90 / 850–1300 cycles | P2, P3, P4 |
| `__syncthreads()` | ≈ 14–25 cycles | P2 |
| mbarrier phase check | ≈ 42 cycles | P5 (5090 — verify on H100) |
| TMA issue gap (same warp) | ≈ 44 cycles; ≤ 16 in flight per SM | P5 (5090 — verify on H100) |
| SM-to-SM per-SM injection limit | ≈ 21 B/cycle (≈ 32 B/cycle × 4 links / 6 SMs); ≈ 36–41 B/cycle bidirectional | P3, P4, P5 |
| SM-to-SM aggregate, 114 SMs, ring | 2.8–3.3 TB/s | P1 |
| Local SMEM bandwidth | ≈ 128 B/cycle/SM | P1, P3 |
| TMA per-SM read bandwidth | 32 B/cycle | P5 |
| L2 bandwidth (chip) | ≈ 4000–4500 B/cycle ≈ 7–8 TB/s | P1 |
| DRAM bandwidth | ≈ 90 % of 3.35 TB/s ≈ 3.0 TB/s (SXM5) | P1 (PCIe: 1.86 TB/s) |
