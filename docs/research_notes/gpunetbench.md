# Jin et al., "Uncovering Real GPU NoC Characteristics: Implications on Interconnect Architecture" (MICRO-57, 2024) and its artifact GPUNetBench — full reading

Status: paper read in full (14 pages, PDF from the UBC author page); artifact read in full at commit `731dd53df51bb243acf3b3a81fdb7a76a4407832` of https://github.com/chrirocca/GPUNetBench (2026-04-14, GPL-3.0). The artifact is **not vendored** in this repository: its licence is GPL-3.0 and our kernels are written from our specs. Agents may clone it *outside* the repo to run a benchmark side by side; nothing from it is copied into `experiments/`. This note is supporting material; `docs/context/` takes precedence (numbers for sanity checks live only in `docs/context/08` P6).

Why this note exists: GPUNetBench measures, on V100/A100/H100, almost exactly the SM→L2-slice and SM→SM quantities that experiments 13, 14, 15 (and parts of 09–11) are about. §3 maps every artifact benchmark onto our experiment and states what to adopt; §4 lists what not to inherit; §5 records how they identify an L2 slice and an issuing SM, which is the input to OI-1.

## 1. The paper

### 1.1 Setup (Table I) and what was actually measured
| | V100 | A100 | H100 (Table I) | H100 actually measured |
|---|---|---|---|---|
| SMs / TPCs / GPCs / max SMs per GPC | 80 / 40 / 6 / 14 | 108 / 54 / 7 / 16 | 132 / 66 / 8 / 18 | **114 / 57 / 7 / 18 (PCIe)**: the artifact's hard-coded maps (`PER_SLICE/main.cu`, `SPEEDUP/main.cu`, `BISECTION/main.cu`) and its README table list an H100 PCIe with 7 GPCs of 18/16/16/16/16/16/16 SMs |
| L2 size / slices / memory partitions | 6 MB / 32 / 4 | 40 MB / 80 / 10 | 50 MB / 80 / 10 | 50 MB / 80 |
| Memory bandwidth / controllers / max clock | 0.9 TB/s / 8 / 1.38 GHz | 2 TB/s / 10 / 1.41 GHz | 3.35 TB/s / 10 / 1.755 GHz | 2 TB/s HBM2e (README), 1.755 GHz |

"MP" in the paper = memory partition = one memory controller with its L2 slices (8 on A100/H100, 4 on V100); "partition" (quoted) = one of the two halves of the A100/H100 L2 ("left/right"). Our vocabulary: *partition* = L2 half; their MP = our FBPA/controller group.

### 1.2 Methodology (Sec. II.C)
- **Latency (Algorithm 1).** One thread of one warp per launch; `D[]` pre-loaded into L2; for every slice *s* and core *n*: `start = clock(); __ldcg(&D[M[s]]); end = clock()`. Compiled with `-Xptxas -dlcm=cg` so that L1 is bypassed. The round trip includes SM, NoC and L2; differences between (SM, slice) pairs are attributed to the NoC. **The load is not part of a dependent chain** (one timed load; the compiler may hoist the second `clock()` ahead of the load's completion unless the loaded value is consumed first — the artifact consumes it through a `volatile` accumulator). Source SM selected by `%smid` filter; destination slice by address, with `M[s]` = addresses known to map to slice *s* (§1.5).
- **Bandwidth (Algorithm 2).** All threads of many CTAs, sequential strided reads of addresses that all map to the target slice, L1 bypassed, L2 warmed; bandwidth = bytes / time from `cudaEvent`s or from the profiler (`l1tex__m_xbar2l1tex_read_bytes.sum.per_second`). Measured per slice, per MP and chip-wide ("L2 fabric bandwidth", which is *interconnect* bandwidth to/from the L2, not the L2 array's own bandwidth — footnote 2).
- **Slice identification.** V100: `nvprof` in non-aggregated mode (per-slice `l2_subp*` events; `MP/run.sh` still carries the commented `nvprof --aggregate-mode off -e all` line). A100/H100 (**footnote 1**): "The profiler (and the performance counters) used for V100 is no longer supported on A100 or H100. As a result, a more manual approach was taken where two kernels were used and one kernel always accessing a single address (single slice) while the address accessed by the other kernel was varied. When 'contention' occurs, we assume that the same L2 slice is being accessed." → on H100 they had **no per-slice counters either** and resolved only **6 slices** (`PER_SLICE/L2_slices_H100.csv`).
- **Footnote 5 (H100):** "because of local caching in H100, the L2 slice ID for the same address can vary among the GPCs" — i.e. they saw the copy mechanism that our 04/08 explained (a read leaves a copy in the reader's partition), without resolving it.

### 1.3 Observations and implications (verbatim gist)
1. Latency from the SMs to individual L2 slices through the GPU NoC is non-uniform. (V100: 175–248 cycles from one SM to the 32 slices, average 212; Jia et al. reported 193.)
2. The average L2 latency from a GPC is similar across GPCs; the variation *within* a GPC differs across GPCs (V100 GPC0 μ 213 σ 13.9; GPC2 μ 209 σ 7.5; GPC4 spread 71 cycles ≈ 33 %).
3. The non-uniform latency is determined by the physical location of the SM within the GPC and of the slice within the memory partition (V100: sorted slice order identical for all SMs; nearest SM/slice pair 180, farthest 217 cycles).
4. Pearson correlation of per-SM latency signatures reveals SM placement (V100: GPC0&1 and GPC4&5 similar, GPC2&3 central).
5. Larger GPUs with multiple "partitions" (A100, H100) add latency non-uniformity; H100 has another core hierarchy level between TPC and GPC, which they call **CPC** (groups of 4–6 SMs = 2–3 TPCs with nearly identical latency signatures; H100 Pearson heatmap, Fig. 6c). SM-to-SM (DSMEM) latency by CPC pair (Fig. 7b): **196** cycles inside CPC0, ≈ 201 (1,0), ≈ 201 (1,1), ≈ 203 (2,0), ≈ 206 (2,1), **213** inside CPC2 (read from the figure except 196/213 which are in the text).
6. Recent GPUs with multiple partitions add uniformity that depends on the L2 cache policy: on A100, GPCs on the far partition see ≈ 400 cycles vs ≈ 212 near (Fig. 8b); on **H100 the hit latency is ≈ 300–340 cycles and almost uniform across GPCs** (Fig. 8c; attributed to NVIDIA's "L2 … caches data for memory accesses from SMs in GPCs directly connected to the partition"), while the **L2 miss penalty is bimodal by GPC, ≈ 250–270 vs ≈ 440–470 cycles** (Fig. 8f; A100/V100 miss penalty flat ≈ 230–260, Fig. 8d/e). Values other than those in the text are read from the figures.
7. On-chip aggregate L2 fabric bandwidth exceeds the off-chip memory bandwidth: 2.4–3.5× (Fig. 9a: ≈ 2.2 vs 0.8 TB/s V100, ≈ 4.8 vs 1.7 A100, ≈ 7.0 vs 2.0 H100; memory reaches 85–90 % of peak).
8. Latency to slices is non-uniform but **bandwidth to the slices is (mostly) uniform**: V100 one SM → one slice 34 GB/s (σ 0.147 GB/s over all SM×slice pairs); one GPC → one slice 85 GB/s (σ 0.06); **≥ 4 SMs saturate one slice**.
9. GPUs are organised hierarchically with **input speedup** into the NoC (Fig. 10/11; H100 values from `SPEEDUP/results/H100`, read / write): TPC 1.95 / 2.00, CPC 5.48 / 4.57, GPC_l (local: TPCs of one GPC) 7.75 / 8.97, GPC_g (global: all SMs of the GPC sending to all slices) 9.1 / 12.38. Full read bandwidth needs a GPC speedup of 7 / 8 / 9 (V100 / A100 / H100); H100 reaches ≈ 85 %. CPC: full bandwidth needs 6; reads get it, writes only 4.6 (slight degradation when all 6 SMs write).
10. A100/H100 have more on-chip bandwidth and speedup but non-uniform bandwidth across partitions: A100 slice bandwidth near 39.5 GB/s vs far 26 GB/s (Fig. 12), bimodal 28/40 GB/s over SMs (Fig. 13a); **H100: single peak ≈ 40–45 GB/s** (Fig. 13b, local caching). A100 single-slice bandwidth saturates with ≈ 8 SMs (≈ 200 GB/s); the far partition is up to 28 % lower with 1–2 SMs (Fig. 14).
11. Load-balancing kernels across SMs matters more than across slices (V100, Fig. 15): 28 SMs of 2 GPCs ("contiguous") streaming to one MP lose 62 % vs 28 SMs spread over 6 GPCs; 14 contiguous SMs gain 218 % when their traffic is spread over 4 MPs instead of 1 — GPC output speedup is partly *spatial*.
12. Memory traffic is load-balanced across channels by address hashing, so NoC traffic is balanced too (Rodinia bfs/gaussian per-slice traffic, Fig. 16).

Implications: (1) NoC characterisation reveals core/slice placement for timing side channels; (2) non-uniform latency shifts, but does not fundamentally change, timing attacks (AES unique-cache-line slope shifts per SM; RSA square kernel varies 1.7× across A100 partitions, 12 % within one); (3) random thread-block scheduling exploits the non-uniformity as a defence; (4) the NoC does not bottleneck memory or L2 bandwidth on real GPUs (85–90 % of peak) — simulators that assume a reply-network bottleneck are wrong; (5) the bandwidth hierarchy (interface bandwidth BW_NoC-MEM, not bisection) must be provisioned; (6) multi-hop topologies (2-D mesh) cannot give uniform bandwidth without age-based arbitration — a hierarchical crossbar is the right baseline.

### 1.4 What the paper does *not* contain
- No address→slice function, no slice ids tied to physical positions, no per-pair tables (only figures).
- No zero-load dependent-chain latency (every number is a single timed `__ldcg`).
- No distinction between home and copy on H100 (footnote 5 only).
- No TPC-pair effect on DSMEM latency (only CPC pairs).

### 1.5 How they identify the issuing SM and the target slice (input to OI-1)
| | Method | Where in the artifact |
|---|---|---|
| Issuing SM | `grid(numSM, 2) × 1024` threads (2 CTAs/SM so that every SM holds a block); only the block whose `%smid == target` runs the loop; others exit (or, in GPC mode, SMs whose id is in a hard-coded per-GPC list) | `PER_SLICE/main.cu` kernels, `SPEEDUP/main.cu`, `BISECTION/main.cu` |
| Target slice, V100 | `nvprof --aggregate-mode off -e all` per-slice events | `MP/run.sh` (commented), paper Sec. II.C |
| Target slice, A100/H100 | two concurrently running kernels; contention oracle (footnote 1); result = 32 addresses (1 KiB units, inside ≈ 2 MiB) per identified slice, 80 rows for A100, **6 rows for H100** | `PER_SLICE/L2_slices_{A100,H100}.csv`; the oracle kernels themselves are not in the repository |
| Partition of an address (A100/H100) | address lists per partition (64 / 164 rows × 32) — derived from latency/bandwidth, no code in the repo | `BISECTION/{A100,H100}-{0,1}.csv` |
| Profiler use on H100 | ncu with `l1tex__m_xbar2l1tex_read_bytes.sum.per_second`, `l1tex__m_l1tex2xbar_write_bytes.sum.per_second`, `gpu__time_duration.avg`, `dram__bytes_{read,write}.sum.per_second` — SM-side and chip-wide aggregates only; **no `lts__` instances, no CUPTI** | every `run.sh` |

Conclusion for OI-1: nobody has had per-slice counters on A100/H100 through the public tools; the paper's own fallback is a timing-contention oracle that resolved 6 of 80 slices. Our busiest-unit (`.max`) rollup oracle is an exact hardware count and strictly more capable; their oracle is the natural profiler-free cross-check (it proves a counter-defined slice is also a bandwidth domain). Decision recorded in `AGENTS.md` OI-1 and `docs/context/06` §2.2.

## 2. The artifact, benchmark by benchmark
Common to all: `nvcc -Xptxas -dlcm=cg -O3` (global loads bypass L1), `-arch=sm_70/80/90` (not `sm_90a`), one `main.cu` per benchmark, `run.sh` sweeps, results as `.log` text parsed by `parse_logs.sh`/`plot*.py`. SM ids are read with `mov.u32 %0, %smid`. No clock locking; GB/s uses `prop.clockRate` (the *max* clock) when derived from `clock64()`.

| Benchmark | What it measures | Kernel design (as implemented) | Knobs / outputs |
|---|---|---|---|
| **AGGREGATE** | STREAM-like aggregate read bandwidth of L2 (`sizeMultiple = 1` × L2 size, `loopCount` 1000) and HBM (`sizeMultiple` 1000, one loop) | grid `numSM × CTA` blocks of `32 × WARP` threads; grid-stride `__ldcg` of `unsigned long long`; `volatile` sinks; cudaEvent timing per launch; mean over `ITERATION` launches | `./BW <CTA> <WARP> <ITERATION> <loopCount> <sizeMultiple>`; sweep CTA 1–32 × WARP 1–32; 2-D colour map of GB/s |
| **BISECTION** | L2-to-L2 partition-link ("bisection") bandwidth: SMs of chosen GPCs read only lines homed in one partition (`-DPARTITION=0/1`) | grid `(numSM, ctaspersm) × 1024`; selected SMs by hard-coded GPC lists; each warp reads one address from the per-partition CSV (`index = lane·8 + addr·256` words, i.e. 32 B per lane within a 1 KiB block), 2 reads × ITERATION; cudaEvent; GB/s = reads × 4 B / time | `./BISECTION <CTAs per SM> <GPC ids…>`; `run.sh` grows the GPC set 1→4 (H100: GPCs 0,2,3,4 = one partition) |
| **HBM_LAT-BW** | HBM loaded latency distribution and throughput vs injection rate; every element read once (buffer = total HBM − 4 GB); optional random per-thread/per-warp delay before each load | grid `numSM × CTAs_per_SM`, threads 1…1024; 5 iterations, **one `clock()`-timed plain load per thread at iteration 4** stored to smem then global; stream (`tid·16 + …`) or random start; throughput from ncu `dram__bytes_read.sum.per_second` + `gpu__time_duration.avg` | `./HBM_LAT-BW -t <threads> -c <CTAs/SM> -o {a,d,b}`; logs `<threads>_<ctas>_<pattern>.log` (per-thread samples), `<pattern>_<ctas>.log` (averages) |
| **L2_LAT-BW** | same for L2 (buffer 45 MiB on H100 = `USABLE_BYTES`, warmed by a kernel), patterns stream / strided / random, random-delay variants (`RANDOM_DELAY_STEPS` add instructions, per thread or per warp) | as above with `__ldcg`; **`cudaSetDevice(1)` hard-coded**; warm-up kernel `k` reads the buffer once; ncu `l1tex__m_xbar2l1tex_read_bytes.sum.per_second` | `./L2_LAT-BW -t -c -o`; `run.sh <ncu\|normal> <pattern> <delay 0/1> <thread\|warp> <steps>`; `lat-bw.py` plots throughput (log₂ x) vs mean latency or kernel time; `hist.py` latency histograms |
| **MP** | V100 only (nvprof): bandwidth when SMs of chosen GPCs (or 28 interleaved SMs "with no TPC contention") access slices of 1–4 MPs, contiguous vs interleaved (Fig. 15) | non-coalesced accesses to slice address rows from `L2_slices_4.csv`; `mpMapping[5][8]` = slice rows per MP | `./MP <GPCs> <SMmax> <MPnum> <slicesPerMP> <MP ids…>` |
| **PER_SLICE** | bandwidth from one SM (direct mode) or from up to `SMmax` SMs of chosen GPCs (GPC mode) to **one** L2 slice (Fig. 9b/c, 12, 13) | grid `(numSM, 2) × 1024`; selected SMs loop `ITERATION` = 10 000 × (32 warps × 2 loads of `d[warp·8 + slice[lane]·256]`) — each lane hits a different 1 KiB block of the slice's address list, each warp a different 32 B sector; cudaEvent; GB/s = `1024·4·8·ITERATION·SMmax·CTAperSM / t` (assumes 8 B per thread-iteration) | `make ARCH=h100 MODE=direct`; `./SLICE <SMid> <slice_row>`; `run.sh` loops all SMs × all slices; `plot.py` histogram of per-(SM,slice) bandwidth |
| **SM2SM/ONE2ONE** | DSMEM bandwidth or per-request latency between every (dest rank, src rank) pair of a 16-CTA cluster (one cluster per launch) | `__cluster_dims__(16)`, 1024 threads, dynamic smem = max opt-in (227 KiB); dest rank fills its smem and records `%smid`; src rank streams the whole remote buffer `ITERATION` times with `ILP_FACTOR` = 8 independent `int` loads per step (`STRIDE` for bank conflicts); **BW mode:** `clock64()` around the whole loop, GB/s via `prop.clockRate`; **latency mode:** `clock()` around each ILP group, mean per thread → "Avg Latency" = loaded latency of 8 outstanding loads, not zero-load | `./SM2SM <dest> <src> <numClusters> <blockSize>`; `run.sh` all 240 pairs × {latency, bw}; `parse_logs.sh` + `plot.py` histogram |
| **SM2SM/ALL2ALL** | DSMEM traffic patterns inside one 16-cluster: `randperm` (fixed random partner), `roundrobin` (rank+1), `uniform` (new random partner every iteration), `all2all` (threads split into 15 groups, one per peer; `READ_ALL2ALL_FULL` reads the full buffer of each peer instead of 1/15) vs block size 32–1024 | same timing modes as ONE2ONE; per-block lines "Src SM / Dest SM / cycles / GB/s"; aggregate per cluster | `run.sh` patterns × modes × block sizes |
| **SM2SM/SM2SM+L2** | interference between DSMEM traffic and L2 traffic inside one GPC: each rank gets a role (0 idle, 1 L2 stream over an L2-resident buffer, 2 DSMEM stream to peers of the same CPC — or of another CPC with `-DDSM_DEST_OTHER_CPC`); ILP of each branch swept 1–8 with the other fixed at 8; both branches start after one `cluster.sync()` and are sized to run about equally long | CPC rank map of their 16-SM GPC1: CPC0 = ranks {4,5,10,11} (SMs 30,31,72,73), CPC1 = {2,3,8,9,14,15} (SMs 16,17,58,59,98,99), CPC2 = {0,1,6,7,12,13} (SMs 2,3,44,45,86,87); `GPC0_MAPPING.png` draws it (titled "GPC 0", it is the code's GPC1) | `run.sh [default\|custom] [--dsmdest 1] [--readall 1]`; results per role/ILP in `results/` |
| **SMEM_LAT-BW** | local shared-memory latency/bandwidth vs threads (32–1024), ILP (1–8) and stride (1–32 words, bank conflicts); whole opt-in smem read `ITERATION` times | one block; timing modes as above | `./SMEM <blockSize> <numBlocks>` |
| **SPEEDUP** | NoC input speedup: bandwidth of 1…N SMs of one GPC (or of one CPC with `-DUSE_H100cpc`) streaming reads (`__ldcg`, 2 loads per element) or writes over an L2-sized buffer; `SORT_GPC` sorts even ids first so that TPC mates are added last | grid `(numSM, CTA) × 32·WARP`; selected SMs stride the buffer by `GPCsize·blockDim`; cudaEvent; GB/s normalised by `GPCsize`; `results/{V100,A100,H100}` hold the read/write speedups plotted in Fig. 10 | `./SPEEDUP <CTA> <WARP> <ITER> <GPC list> <SMmax>`; `run.sh` sweeps SMmax 1–14 (V100) |

### 2.1 Hard-coded H100 PCIe topology (their card; not ours)
```
gpc0 = {0,1,14,15,28,29,42,43,56,57,70,71,84,85,110,111,112,113}  (18 SMs, partition 0)
gpc1 = {2,3,16,17,30,31,44,45,58,59,72,73,86,87,98,99}            (partition 1)
gpc2 = {4,5,18,19,32,33,46,47,60,61,74,75,88,89,100,101}          (partition 0)
gpc3 = {6,7,20,21,34,35,48,49,62,63,76,77,90,91,102,103}          (partition 0)
gpc4 = {8,9,22,23,36,37,50,51,64,65,78,79,92,93,104,105}          (partition 0)
gpc5 = {10,11,24,25,38,39,52,53,66,67,80,81,94,95,106,107}        (partition 1)
gpc6 = {12,13,26,27,40,41,54,55,68,69,82,83,96,97,108,109}        (partition 1)
```
Regularities worth testing on our card (03 found the analogous `TPC t → t mod 8` for t < 32):
- `smid = 14·t + 2·g + {0,1}` for TPC index t = 0…6 (round-robin TPC enumeration over the 7 GPCs), remaining TPCs 49–56 filled into GPCs 0–6 in order.
- Inside their GPC1, the three CPCs are **CPC = (TPC index within the GPC) mod 3**: CPC2 = TPCs {0,3,6}, CPC1 = {1,4,7}, CPC0 = {2,5,(8)}. With our 03 map (TPC index within GPC = rank of the TPC in ascending smid order) this gives a concrete CPC hypothesis for 10 and 15.
- Partition membership of GPCs (4 + 3 on their 7-GPC card; 4 + 4 on ours, 08).

## 3. Mapping onto our experiments — what to adopt
Rule: our specs stay binding; the items below are *added* to them (this PR) so that the implementing agent writes the same measurement the paper reports, in our harness, with our timers, clock lock, CSV schema and placement control.

| GPUNetBench | Our experiment | Adopt (now in the spec) | Keep ours (why) |
|---|---|---|---|
| SMEM_LAT-BW | 05 K1 (done) | — (05 measured 128 B/cycle/SM, 0 conflicts; stride sweep is 12's bank-conflict study) | timers, locked clock |
| AGGREGATE | 05 K3/K4 (done) | — (7.7 TB/s L2 read, 3.68 TB/s HBM) | — |
| BISECTION | 05 link mode (done: 1.11 TB/s per direction) | their address-list-per-partition selection = our home labels from 08's hash | our labels are exact (hash), theirs empirical |
| HBM_LAT-BW | **14 K5** (new) | loaded latency vs injection: all SMs, 2 CTAs/SM, threads/CTA 1→1024, every element read once, one timed load per thread at a fixed iteration, stream and random, per-warp random-delay variant; throughput from ncu `dram__bytes_read.sum.per_second` | timed load made dependent (`read_clock64_after`), `%clock64`, clock lock |
| L2_LAT-BW | **15 K7** (new) | same design on an L2-resident 16 MiB buffer; latency–throughput curve (log₂ x) and histograms per injection rate | footprint from `10` §3, home-controlled lines |
| PER_SLICE, MP | **15 K6** (new) | one SM → one slice, one GPC → one slice, N = 1…18 SMs → one slice, for every slice and from both partitions; contiguous (one GPC) vs distributed (one SM per GPC) senders; per-(SM, slice) bandwidth histogram | slice addresses from 13's map (not an empirical 6-row list) |
| SPEEDUP | **15 K5** (new) | input speedup read and write for TPC / CPC / GPC_l / GPC_g: 1…18 SMs of one GPC added TPC-sorted and GPC-sorted; `__ldcg` streaming reads and plain stores over an L2-resident local-home buffer; speedup = BW(N)/BW(1); also with `l1tex__m_xbar2l1tex_read_bytes.sum.per_second` | SM subsets from 03's map and 10's CPC map instead of hard-coded lists |
| SM2SM/ONE2ONE | 09 K1, **10 §4** | the (dest rank, src rank) sweep at 1024 threads / ILP 8 as a **loaded** per-request latency variant (secondary metric next to our zero-load chase); CPC-pair analysis of the matrix | zero-load dependent chase stays the primary latency |
| SM2SM/ALL2ALL | **11 patterns** | `randperm`, `uniform_random` (new partner every iteration), `all2all_full` added to our pattern list; block-size sweep 32–1024 already present | affine payload fits, `%globaltimer` makespan |
| SM2SM/SM2SM+L2 | **11 K5** (new) | DSMEM stream and L2 stream in the same GPC with per-rank roles, destinations in the same vs another CPC, synchronised start, equal durations; question: do DSMEM and L2 traffic share the GPC crossbar/CPC port? | — |
| Pearson-correlation placement (Obs 4) | **15 analysis** | correlation heatmap of per-SM slice signatures as an independent GPC/CPC clustering (cross-validates 03 and 10) | — |

## 4. Defects and limitations of the artifact (do not inherit)
1. `L2_LAT-BW/main.cu`: `cudaSetDevice(1)` and `cudaGetDeviceProperties(&devProp, 1)` hard-coded.
2. GB/s from `clock64()` uses `prop.clockRate` (max clock), not the running clock; no clock lock anywhere.
3. "Latency" modes time an ILP group of 8 independent loads with `clock()` and report the mean per thread: a loaded latency under 8 outstanding requests × up to 32 warps, not a zero-load dependent-chain latency. Algorithm 1's single timed load is not chained either.
4. `BISECTION/main.cu` expects 164 (H100) / 64 (A100) CSV rows; the files have 154 / 61 lines → the program aborts unless edited. `AGGREGATE`: `unsigned targetSize = sizeMultiple * l2CacheSize` overflows at `sizeMultiple = 1000` on a 50 MB L2.
5. Every SM/GPC/CPC/partition table is their H100 PCIe's; the H100 slice list has 6 rows; nothing is valid for another card.
6. Compiled for `sm_90`, not `sm_90a`; no SASS checks; PER_SLICE's bandwidth formula assumes 8 B per thread-iteration while the kernel loads 4 B twice (a factor that cancels in relative comparisons).
7. Profiler runs use ncu defaults (`--clock-control base`, `--cache-control all`, kernel replay), so warmed-L2 experiments under ncu are flushed between passes unless the kernel re-warms.

## 5. Numbers kept for sanity checks
All numeric values above are duplicated in `docs/context/08__prior_work_reference_numbers.md` **P6**, with the chip caveat (H100 PCIe, 114 SMs, 1755 MHz, 50 MB, 80 slices). Notebooks never cite them.
