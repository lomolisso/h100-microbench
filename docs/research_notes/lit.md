# Literature notes: reverse-engineering the physical layout of NVIDIA datacenter GPUs (GH100 focus)

Compiled 2026-10-23. Scope: SM->GPC/TPC mapping, L2 slice/partition NUCA, address->L2-slice/partition hashing,
DRAM channel/bank mapping, NoC/crossbar topology, intra-GPC SM-to-SM (DSMEM) fabric.

**Provenance caveat.** In this session only github.com was fetchable; arxiv.org, patents.google.com, USPTO, Justia,
chipsandcheese.com, IEEE/ACM, UBC and TUHH hosts were all blocked by the egress proxy. Every number below therefore
comes from (a) search-engine extracts of the primary PDF/HTML, (b) GitHub-hosted code/READMEs/issues, or (c)
secondary summaries. Confidence tags: **[P]** = quoted from the primary text via search extract; **[S]** = secondary
summary (blog/aggregator/forum), verify before citing; **[?]** = not confirmed, flagged as a gap.

---

## 0. Quick reference: GH100 / H100 physical facts (NVIDIA whitepaper + HC34)

| Item | Full GH100 die | H100 SXM5 | H100 PCIe / H800 PCIe | Source |
|---|---|---|---|---|
| GPCs x TPCs x SMs | 8 x 9 x 2 = 144 SMs, 72 TPCs | 8 GPCs, 66 TPCs, 132 SMs | 7 GPCs, 57 TPCs, 114 SMs | Hopper whitepaper v1.02 [P] |
| L2 | 60 MB | 50 MB (2 partitions x 25 MB) | 50 MB | whitepaper [P]; 2x25 MB split: Chips&Cheese / Hazy Research [S] |
| L2 slices | - | 80 (Jin et al. Table) | 80 | Jin MICRO'24 spec table [P] |
| HBM | 6 stacks, 12 x 512-bit MCs, 6144-bit | 5 HBM3 stacks, 10 x 512-bit MCs, 5120-bit, 3.35 TB/s | 5 HBM2e stacks, 10 MCs, 2.0 TB/s | whitepaper [P]; PCGH spec [S] |
| Boost clock | - | 1980 MHz | 1755 MHz | NVIDIA specs |
| Die | 814 mm^2, 80 B transistors, TSMC 4N | | | whitepaper [P] |
| Graphics-capable TPCs | | 2 | 2 | whitepaper [P] |
| Cluster size | | 8 portable, 16 non-portable (`cudaFuncAttributeNonPortableClusterSizeAllowed`) | same | CUDA docs |
| L2 topology statement | "partitioned crossbar structure"; each partition caches data for SMs in GPCs "directly connected to the partition"; "SMs in a given GPC have a closer path" | | | whitepaper [P]; cudacourseh100 deck [S] |
| DSMEM claim | "~7x" faster block-to-block exchange vs global memory; "dedicated SM-to-SM network for SMs in a GPC" | | | whitepaper / dev blog / HC34 Choquette slides [P] |

Useful arithmetic: 80 L2 slices / 10 active MCs = 8 slices per 512-bit MC = 16 slices per HBM stack (HBM3 stack =
16 x 64-bit channels, each split into 2 pseudo-channels). 50 MB / 80 slices = 640 KB per slice (A100: 40 MB/80 = 512 KB).

---

## 1. Jin, Rocca, Kim, Kasan, Rhu, Bakhoda, Aamodt, Kim — "Uncovering Real GPU NoC Characteristics: Implications on Interconnect Architecture", MICRO 2024

- Citation: Z. Jin, C. Rocca, J. Kim, H. Kasan, M. Rhu, A. Bakhoda, T. M. Aamodt, J. Kim. MICRO-57, Austin TX, Nov 2024,
  pp. 885-898. DOI 10.1109/MICRO61859.2024.00070 (acceptance 113/497).
- URLs: PDF https://people.ece.ubc.ca/aamodt/publications/papers/realgpu-noc.micro2024.pdf ;
  IEEE https://ieeexplore.ieee.org/document/10764573/ ; KAIST record
  https://pure.kaist.ac.kr/en/publications/uncovering-real-gpu-noc-characteristics-implications-on-interconn/ ;
  Chinese walkthrough https://blog.benx.dev/posts/paper-gpu-noc/ (blocked here).
- GPUs: **V100 (80 SMs, 32 L2 slices), A100 (108 SMs, 80 L2 slices), H100 (132 SMs, 80 L2 slices)** — the paper's spec
  table also lists TPC/GPC/memory-controller counts [P]. (An X post by Ogawa Tadashi summarises: "V100, A100, H100 ...
  12 Observations, 6 Implications" [S].)
- Methodology [P, from search extracts of the PDF]:
  - *Latency*: one thread in one warp issues loads so there is no coalescing/contention; loads **bypass L1 with `__ldcg()`**
    (ld.global.cg); timing with the per-SM cycle counter; the **source SM is selected via `%smid`** (launch enough blocks to
    cover all SMs, only the block whose smid matches runs the probe); the **destination L2 slice is selected by choosing the
    address, with the address->slice mapping taken from the profiler** (i.e. NCU per-slice `lts__*` instance counters —
    see Section 9 for how to do this). Result = full SM x slice round-trip latency matrix.
  - *Bandwidth*: two concurrently running kernels; one kernel always hammers a single address (one slice) while the address
    used by the other kernel is varied, to see whether SM->slice bandwidth depends on placement.
- Headline findings [P]: (1) on-chip latency is **non-uniform by physical placement of SM and memory partition/slice, up to
  ~70% difference**; (2) **bandwidth from cores to memory partitions is approximately uniform**; (3) GPUs built from multiple
  "partitions" (A100, H100) show different latency *and* bandwidth when communicating across partitions; the L40 follow-up
  paper describes Jin's data as "a discrete step across the A100/H100 dual-partition boundary and an approximately uniform
  per-core average" [S]; (4) implications: NoC latency non-uniformity enables timing side channels that reveal the SM an
  attacker/victim runs on; NoC simulators must model the real (non-ideal) topology.
- Per-pair cycle numbers, slice-cluster groupings, and the "memory-side vs SM-side" discussion: **[?] not recoverable
  through search extracts** — read Secs. IV-V of the PDF. DGNA (Sec. 3) reports that on A100 the `__ldcg`-based numbers
  cluster near **~350 and ~500 cycles** vs ~200 cycles for a block resident in the local L2 partition, i.e. the `.cg` path may
  be measuring remote-partition / DRAM paths [P-DGNA].
- Physical-position method: the paper's own method is "latency to each slice as a signature per SM"; it is the direct
  ancestor of the L40 (Sec. 4) and attestation (Sec. 5) fingerprinting papers.

## 2. Fan et al. — "Dissecting How Die Scaling Breaks GPU Fine-grained Scheduling" (arXiv 2609.24270, Sep 2026; targets ASPLOS'27)

- URLs: https://arxiv.org/abs/2609.24270 , HTML https://arxiv.org/html/2609.24270v1 ; summary issue
  https://github.com/AI-Infra-Team/awesome-papers/issues/232 [fetched].
- GPUs: **H100 PCIe (GH100, 114 SMs, 7/8 GPCs, 2 L2 partitions), H200 (28 cards sampled), B200**. Background table lists
  GH100: 144 full SMs, 114 SKU SMs (PCIe), 7/8 GPCs, 2 L2 partitions [P].
- Compute-topology method ("S1/E1") [P]:
  - *Cluster probing*: `cudaLaunchKernelEx` with cluster sizes 3..8; log each cluster's constituent `%smid` set; since all
    CTAs of a cluster are guaranteed to be in one GPC, co-occurrence gives GPC membership for the "stable" SMs.
  - *L2-bandwidth probing* places the remaining SMs (those that never co-schedule in large clusters). Note: "SM IDs 124-131
    (4 TPCs) never appear at cluster sizes > 2" on the sampled card.
  - Total < 1 min/GPU; **100% agreement with nvdebug** (Bakita's kernel module, Sec. 8).
  - Finding: logical SM IDs split into a group that follows an **architecture-specific predefined SM->GPC mapping** and a
    group whose physical GPC depends on that chip's floorsweeping (fixed per chip, differs across chips). On H200, 23/28 cards
    have **six 16-SM GPCs + two 18-SM GPCs**; the rest differ. H100 PCIe: 7 GPCs, 6 with >=16 SMs and 1 with >=14 SMs.
    Worst-case MIG example: four 18-SM GPCs each losing 2 SMs to floorsweeping.
- Memory-topology method ("S2") [P]:
  - A **single-SM probe** separates local vs remote L2/HBM latency; then **flip physical-address bits one at a time** and watch
    whether the line's partition changes -> recovers the **XOR partition hash**. Result: all three GPUs use an XOR hash,
    **lowest participating bit = 12 (4 KB partition granularity)**; **H100 PCIe: 14 bits participate; H200 & B200: 16 bits**.
    Takes ~10 s/GPU (dominated by measuring the HBM latency distribution). They also describe an **invalidate-then-reload**
    microbenchmark to discriminate remote-L2 access models, and note that remote lines are **replicated in the local partition**
    so NUMA GPUs expose only **~65% of physical L2 capacity**.
  - They state prior work found a *separate, non-XOR hash* for **L2-slice and HBM-channel** selection when counts are not powers
    of two (consistent with accel-sim's IPOLY+MODULO, Sec. 7).
- Latency numbers [P via issue summary]: H200 HBM local **~490** / remote **~655** cycles (+34%); H200 L2 **~309 / ~466**;
  B200 HBM **~552 / ~920** (+67%); B200 L2 **~364 / ~725**. "Remote accesses increase HBM latency by up to 67% and nearly
  double L2 latency"; topology-oblivious allocation -> up to **1.33x** performance variation.
- Open question [?]: how they obtain physical addresses on an unprivileged path (likely large contiguous allocations with 2 MB
  pages so that bits < 21 are virtual==physical); check Sec. 4.

## 3. DGNA — "Dissecting GPU NUMA Architecture through Microbenchmarking and Data Analysis" (arXiv 2607.19922, Jul 2026; ACM DOI 10.1145/3841173)

- URLs: https://arxiv.org/abs/2607.19922 ; https://dl.acm.org/doi/pdf/10.1145/3841173 .
- GPUs: **A100 and H100** (SXM variants implied; clock not extracted [?]).
- Method [P]: pointer-chase style latency microbenchmarks that *do not* rely on vendor-specific cache-control loads (critique
  of `__ldcg` in Jin et al.); a **Gaussian mixture model** automatically removes outliers and separates latency modes; data
  analysis yields SM->partition affinity and address interleaving.
- Numbers [P]:
  - L2 local / remote: **A100 218 / 379 cycles; H100 295 / 470 cycles** (remote penalty ~161 A100, ~175 H100).
  - DRAM local / remote penalty: ~161 cycles (A100), **~174 cycles (H100)**.
  - **H100 has sub-NUMA nodes inside each L2 partition: +~32 cycles** (one summary reports two local peaks at **280 and 312**
    cycles); A100 has no sub-NUMA structure (attributed to H100's larger 50 MB L2).
  - Topology: **2 GPCs per sub-NUMA node, 4 GPCs per L2 NUMA node, on both A100 and H100**. H100 balances SMs across the two
    L2 nodes better than A100 but is imbalanced at sub-NUMA level.
  - **First-touch**: the L2 node that homes a line is the one local to the SM that first touched it; "home SMs read/write their
    home node directly, remote SMs buffer and update both nodes" (replication).
  - Interleaving: a secondary summary cites **8 KB and 64 KB interleaving patterns** for sub-NUMA/NUMA nodes [S] — conflicts with
    the 4 KB XOR granularity in Sec. 2; a reviewer notes DGNA's DRAM-interleave formula is an empirical fit, not a measured spec.

## 4. Alpay & Basaran — "Non-Uniform L2 Cache Latency Across the Streaming Multiprocessors of an NVIDIA L40" (arXiv 2606.22588, Jun 2026)

- URLs: https://arxiv.org/abs/2606.22588 ; HTML https://arxiv.org/html/2606.22588v1 .
- GPU: **L40 (AD102, 142 SMs, 96 MiB L2)**; two physically identical L40s.
- Method [P]: a **turn-serialized, `%smid`-resolved pointer-chase probe** maps L2-hit latency for all 142 SMs in one launch;
  the chain is a **random permutation over a 2 MiB footprint** that touches the whole slice ensemble, so per-SM latency is the
  mean distance to all slices; **256 slice probes** give the per-slice dimension; line-tag and prefetch controls.
- Numbers [P]: hit latency is **222-339 cycles (52% spread)** rather than a constant ~279; additive model
  **L = mu + a(sm) + b(slice)**, **R^2 = 0.87 (0.98 with one rank-1 term)**; SM term spans **57.2 cycles**, slice term
  **39.5 cycles**; SM term is **two-fold symmetric** (two halves of 72 SMs correlated r = 0.999, matching AD102's GPC layout);
  independent access patterns agree per SM at r = 1.000; map unchanged after 1 h at full load; **self-localization 92%
  (abstract) / 99.2% (conclusion)**; two identical devices separated with 100% accuracy; latency-aware placement cuts makespan
  up to 11%.
- Relevance: the exact statistical treatment (additive two-factor model + low-rank residual) is the template for an H100
  SM x slice matrix.

## 5. Alpay & Alpay — "Unprivileged Topology Certificates for Cloud GPU Attestation" (arXiv 2606.24934, Jun 2026)

- URL: https://arxiv.org/abs/2606.24934 . GPUs: RTX 5090, **H200**, **B200**.
- Method [P]: CUDA probe builds an **SM x memory-region latency matrix**; a reducer packs it into a SHA-256 certificate.
- Numbers [P]: **H200 two-way L2 split recovered via cache-bypassing HBM sweeps**; **B200: two-die partition 74/74 SMs with a
  30-cycle (15.5 ns => ~1.94 GHz) cross-die penalty**; RTX 5090 6-h full-load run: median temporal jitter **0.09 cycles**;
  leave-one-out die classification 100%.

## 6. Luo, Fan, Li, Du, Liu, Wang, Chu — "Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis" (arXiv 2501.12084; extended from "Benchmarking and Dissecting the Nvidia Hopper GPU Architecture", IPDPS 2024, arXiv 2402.13499; TOCS version per user)

- URLs: https://arxiv.org/abs/2501.12084 ; https://arxiv.org/abs/2402.13499 .
- GPUs: **H800 PCIe** (primary; boost 1755 MHz), A100, RTX 4090 (H100 for some tests).
- L2 partition method [P]: fine-grained P-chase (Mei & Chu lineage) with random strides, 64 B chunks; they state uniform-stride
  P-chase "fails to capture the mapping between global memory and these partitions"; distinct latency groups attributed to
  dual-partitioned L2 and varying SM-partition distance. (User's note of 32 B stride + K-means into 4 classes: K-means not
  confirmed by extracts [?].)
- **Table 4 (H800 PCIe, cycles)**: **near hit 258, far hit 414, near miss 555, far miss 744**; global-memory latency ~
  (near miss + far miss)/2 [P]. A100: two L2 levels **~202.8 and ~408** cycles with inflections at half and full L2 size;
  A100 global 566 cycles; RTX 4090 L2 ~284.8 [P].
- DSMEM [P]: direct shared-memory load **29 cycles**; own shared memory via the DSM interface (`mapa` + `ld.shared::cluster`)
  **33 cycles**; **cluster size 2: SM-to-SM 181 cycles (30% below L2)**; **cluster sizes 2..16: 184-213 cycles**; global
  store+load transfer **1110 cycles -> DSM is 6.13x faster** (vs NVIDIA's "7x"). Three copy patterns: **ring** (block R adds
  its registers into block (R+1) mod CS), **pair**, **broadcast**; broadcast throughput "decreases significantly" with cluster
  size, ring/pair minimal change. Three scheduling policies: default, Spread, LoadBalancing. They record `%smid` per block to
  confirm distinct SMs and examine SM-ID allocation within a cluster (details in Sec. 7.2 — not extracted [?]).
- DSMEM throughput (2402.13499, ring-based copy, aggregate over the GPU) [P]: **~3.27 TB/s at CS=2, 2.65 TB/s at CS=4**
  (one summary says 2.78), decreasing further for 8/16 (values not extracted [?]); larger blocks + more ILP => higher
  throughput; local SMEM ~128 B/clk/SM.

## 7. Other Hopper/Blackwell memory-system microbenchmarks

- **Jarmusch, Graddon, Chandrasekaran — "Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks"** (arXiv 2507.10789;
  RTX 5080 vs **H100 PCIe**): H100 **L2 hit ~273 cycles baseline; ~508 cycles when both partitions are saturated (footprints
  31-45 MB)**; Blackwell GB203 monolithic L2 ~358 baseline [P/S]. Companion IPDPS'26 B200 paper arXiv 2512.02189.
- **Chips and Cheese, "Nvidia's H100: Funny L2, and Tons of Bandwidth"** (H100 PCIe; https://chipsandcheese.com/p/nvidias-h100-funny-l2-and-tons-of-bandwidth):
  L2 split into two halves (later article: **2 x 25 MB**); any SM reaches all 50 MB but the **far half takes "nearly twice as
  long"**, roughly RX 6900 XT VRAM latency; **L2 read BW > 5.5 TB/s (near region)**; VRAM latency ~330 ns on the PCIe card,
  < 300 ns on GH200's H100 (higher clocks); clock sagged to **1395 MHz** under L2-read load (suspected power limit) [P].
  Secondary (Spheron) claims: 50 MB aggregate BW **3.8 TB/s vs A100 4.5 TB/s**; "L2 ~200-263 cycles, HBM ~290-480" [S, unverified].
- **RRZE gpu-benches** (https://github.com/RRZE-HPC/gpu-benches, fetched): A100 "2 x 20 MB", H100 "2 x 30 MB" segmented L2
  (plot drops at ~20 / ~25 MB) producing an intermediate latency plateau when data comes from the far section; cache BW
  ~128 B/cycle/SM; L2 BW "about 5.x TB/s"; method = single-warp pointer chase over randomly ordered buffer, each line hit once
  per pass.
- **"Concurrency Response of Plain Global Loads on the NVIDIA H100"** (arXiv 2608.15764; 3 H100 SXM5 dies): attained LDG
  bandwidth peaks at ~2 loads/thread in flight then falls ~35% by K=8 while L2 sector traffic rises; cp.async 2.1-2.9x better
  at high depth [P]. Context for bandwidth probes: use cp.async.bulk / TMA, not plain loads.
- **vibe-megakernel issue #29** (https://github.com/kamahori/vibe-megakernel/issues/29, fetched; **B200**, 1.96 GHz; ncu
  locks 1.15 GHz): **DSMEM hop ~89 ns even under full HBM streaming**; L2 load latency flat **~295 cycles (~150 ns)** for every
  (SM, line); global ping-pong hop 412-455 ns idle, 444-492 ns with all 148 SMs streaming; BW vs in-flight bytes/SM: 16 KB ->
  3 TB/s, 32 KB -> 4.5, 64 KB -> 5.4, 96 KB -> 5.8 TB/s; max co-resident clusters 15x8, 33x4, 7x16 [P-issue].
- **NVIDIA forum, TMA multicast thread** (https://forums.developer.nvidia.com/t/implementing-h100-tma-multicast-.../355253):
  non-multicast `cp.async.bulk` tops at **~38 B/cycle/SM on 132-SM H100 (~5 KB/cycle GPU-wide from L2)**; **4x cluster multicast
  ~70 B/cycle/SM**, then "limited by intra-GPC bandwidth", **no gain 4x -> 8x**; another poster measured multicast at only
  ~2 TB/s (benchmarking pitfalls) [S].

## 8. SM-to-SM / DSMEM characterizations (beyond Luhnen HPEC'24, ClusterSim IISWC'25, Luo TOCS)

- **ClusterFusion** (Luo et al., NeurIPS 2025; arXiv 2508.18850; code https://github.com/xinhao-luo/ClusterFusion): H100,
  cluster up to 16; primitives ClusterReduce / ClusterGather over DSMEM; Fig. 5 sweeps cluster size 1-16 for SM-to-SM latency,
  bandwidth and number of active SMs; one extract: **avg DSMEM ~190 cycles at CS=2 vs > 470 cycles global** [S]; 1.61x mean
  end-to-end decode speedup. ClusterFusion++ arXiv 2604.23553 extends to full decoder blocks.
- **FlashFuser** (Huang, Zhou, Leng et al., HPCA 2026; arXiv 2512.12949): compiler with DSM-communication primitive, dataflow
  analyzer, cost model; H100: DRAM traffic -58%, up to 4.1x kernel, 1.24x e2e [S]. No fabric numbers extracted [?].
- **CREDIT** (Li, Huang, Ogras; arXiv 2609.01864; code https://github.com/zhengxiongli08/CREDIT, fetched): primitive
  microbenchmarks (`src/primitive_benchmark.cu`, work-free cluster overhead calibration `src/cluster_control.cu`) on **RTX 5090
  and H100**: **remote DSMEM load ~6.4x the latency of a local SMEM read and 4.4-5.4x lower throughput**; cluster-sync cost
  modeled; speedups 1.466x (5090) / 1.318x (H100) [P/S]. Result bundles in `results/` hold raw numbers.
- **Shan & Araya-Polo**, "Evaluation of Programming Models and Performance for Stencil Computation on Current GPU
  Architectures" (arXiv 2404.04441; GH200 main platform): adding clusters/DSMEM to CUDA stencils gave **no significant
  speedup**; minimizing inter-block exchange mattered more [P].
- **Luhnen, Marschner, Lal — "Benchmarking Thread Block Cluster"**, HPEC 2024 (TUHH PDF): per-instruction latencies incl. a
  `barrier.cluster.wait` cycles chart vs cluster dims; up to **2.1x** for SMEM-constrained apps via TBC [P]. (User already has.)
- **gpgpu-sim-dsm** (https://github.com/Divyansh9778/gpgpu-sim-dsm, fetched): ClusterSim-derived DSM model, one network per
  GPC in its own SM2SM clock domain; defaults **171-cycle latency, 88 bits/cycle** per link (from ClusterSim), *not* validated
  on hardware; crossbar/ring/ideal options.
- NVIDIA official numbers: whitepaper/dev-blog "**~7x**" DSMEM vs global; HC34 Choquette: "dedicated SM-to-SM network for
  direct low-latency access without going through L2". No official bandwidth figure found [?].
- No published measurement found for **DSMEM atomics (`atom.shared::cluster`/`red.async`), `st.async`, or
  `cp.reduce.async.bulk` latency** [?]; CREDIT notes remote atomics supported.

## 9. NVIDIA patents (Hopper CGA / DSMEM / multicast / TMA family, all priority 2022-03-10)

Full texts could not be fetched; items below are from search extracts of Google Patents / Justia / USPTO text.

- **US12248788B2 "Distributed shared memory"** (app. US20230289189A1, filed 2022-03-10, granted 2025-03-11; inventors
  Prabhakar, Hirota, Krashinsky, Long, Pharris, Dash, Tuckey, Duluk Jr., Shah, Choquette...; CN116774914A).
  https://patents.google.com/patent/US12248788B2/en
  - DSMEM = shared memory **physically located in each SM**; CTAs of a **GPC-level CGA** address each other's SMEM; direct
    **SM-to-SM reads, writes and atomics**; CGA co-residency guaranteed [P].
  - Routing text in the family: "a first SM transmits a message **via a MMU to a crossbar that interconnects all the SMs in the
    GPC**"; "memory access requests to the L2 cache are communicated **through the same crossbar interconnect**" (i.e. the GPC
    crossbar fronts both the SM2SM path and the GPC->L2 path) [P, from US20230289242A1/US20230315655A1 extracts].
  - "MULTI_CAST mode may be used on GPC_CGAs that want to take advantage of the new multicast hardware in the SM and generic
    network interface controller (**GNIC**), for example the Tensor Memory Access Unit (TMA)" [P].
  - **[?] Not confirmed from extracts**: CPC, TPCARB, GPCMMU, GX0/GX1 switch planes, remote-mbarrier-arrive delivery, ack/
    ordering, behavior when the target CTA has exited. The term **"GPCARB"** does appear in the CGA application: "a new
    low-latency **SM2SM communication network in GPCARB**" (US20230289215A1) [P]. Read Figs. and detailed description of
    US12248788B2 directly for the fabric diagram.
- **US20230289215A1 / US12333311B2 "Cooperative Group Arrays"** (granted 2025-06-17):
  https://patents.google.com/patent/US20230289215A1/en — GPC_CGA supports SM-to-SM communication, CGA linear memory in
  global memory, and a **hardware CGA barrier**; **all-or-nothing launch** ("GPU hardware will not permit any CTAs in a CGA to
  launch unless there is room on the GPU for all the CTAs"); CGA may specify its hardware domain (GPC / GPU / multi-GPU);
  hardware maintains a running-CTA count per CGA; each CGA gets at least one hardware barrier [P]. GPC-level scheduler
  dispatch logic not extracted [?].
- **US12020035B2 / US12608212B2 "Programmatically controlled data multicasting across multiple compute engines"**
  (granted 2024-06-25 and 2026-04-21): https://patents.google.com/patent/US12020035B2/en — one thread requests data on behalf
  of other threads of its CGA; **tracking circuitry between multicast requests and memory**; "increase effective L2 bandwidth
  by multicast of **response data from one L2 read to multiple CTAs on multiple SMs**"; data "**travels through at least part of
  the crossbar NoC as a single packet before being distributed**" (replication happens at/after the GPC-side crossbar) [P].
  Related **US20230289304A1 / US12141082B2** gives the TMA multicast load format with a **16-bit mask of receiving CTA IDs** [S].
- **US20230289242A1 / US12536056B2 "Hardware accelerated synchronization with asynchronous transaction support"**
  (granted 2026-01-27): transaction barrier (= `mbarrier` with tx-count) synchronizing threads and async transactions from
  hardware data-movement units (TMA, copy engines), DSMEM exchanges and programmatic multicast; receives arrival updates
  **from CTAs of the same CGA running on different SMs**; "transaction arrival may be tracked by a **zero-balanced counter
  without a pre-armed expectation count**"; barrier lives in cache/SMEM with a hardware sync circuit [P].
- **US20230289292A1 / US12141082B2 / US12499052 "Method and apparatus for efficient access to multidimensional data
  structures..."** (TMA): tensor descriptor + tensor parameters, out-of-bounds detection, "memory access request processing
  pipeline in the tensor memory access unit", high-level pseudo code [P]. High-level only.
- **US20230315655A1 "Fast data synchronization in processors and memory"**: reiterates GPC MMU -> crossbar path [P].

## 10. Address -> L2 slice / partition hashing (what is actually known)

- **XOR partition hash (2 partitions)**: Fan et al. (Sec. 2): bit 12 lowest; 14 bits on H100 PCIe, 16 on H200/B200 [P].
  DGNA: 8 KB / 64 KB interleave patterns [S]. Both agree on partition-level NUMA at >= 4 KB granularity.
- **Slice hash**: no public reverse-engineering of the NVIDIA L2 *slice* (80-way) or set hash found. The NDSS'26 GPU-TLB paper
  states "GPU caches are physically addressed and their set hash functions are highly non-linear and remain unknown despite
  reverse-engineering attempts" [P]. TLB set/slice hashes were recovered for A100/A30 (Zhang et al., "TunneLs for
  Bootlegging", CCS'23) — same methodology (eviction-set + timing) is the template [P].
- **Invalidate+Compare** (Zhang et al., USENIX Sec'24; https://github.com/0x5ec1ab/invalidate-compare): uses the `discard`
  cache-maintenance instruction to reverse-engineer Ampere/Ada cache properties and build eviction sets timer-free; results in
  `cache-policy/` folder, not extracted [?].
- **Accel-Sim/GPGPU-Sim** (https://github.com/accel-sim/gpgpu-sim_distribution/blob/dev/src/gpgpu-sim/hashing.cc, fetched):
  `ipoly_hash_function` implements Rau's IPOLY for 16/32/64 banks (polys 5/37/67), e.g. 64 banks:
  n0 = a18^a17^a16^a15^a12^a10^a6^a5^a0^b0 ... n5 = a17^a16^a15^a14^a11^a9^a5^a4^b5 (a = higher address bits, b = bank
  index bits); `bitwise_hash_function` = index ^ (higher_bits & (N-1)); `ipolymodulo_hash_function` = IPOLY on group index then
  modulo for non-power-of-two bank counts. Accel-Sim 2.0 release notes: `-gpgpu_memory_partition_indexing 6` = **IPOLY+MODULO
  for non-power-of-two bank/sub-partition counts (H200)**. The ISCA'20 paper admits IPOLY "does not exactly match the hardware's
  indexing function" [P]. SM90 H100 config file path in accel-sim-framework returned 404 (moved) [?].
- Mei & Chu, "Dissecting GPU Memory Hierarchy through Microbenchmarking", TPDS 28(1):72-86, 2017 (arXiv 1509.02308):
  origin of **fine-grained P-chase** (per-element latency instead of averages) used by all later L2-partition work; Fermi/
  Kepler/Maxwell; no slice hash.
- Jia et al., "Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking" (arXiv 1804.06826) and T4 (arXiv
  1903.07486): V100 L2 6144 KiB, 16-way, **~193 cycles** average; treated as unified (no partition analysis).

## 11. DRAM channel / bank mapping on GPUs

- **GPUHammer** (Lin, ..., Saileshwar; USENIX Security 2025; arXiv 2507.08166; https://github.com/sith-lab/gpuhammer, fetched):
  **RTX A6000 (GDDR6)**. Physical addresses are not exposed even to privileged users, so they map **virtual offsets** -> (bank,
  row) using **row-buffer-conflict timing** (DRAMA-like): step 1 identify tRC from the hit/conflict latency gap (README:
  "tRC is ~43", samples 327-360 — units unclear), step 2 `conf_set --threshold 27 --step 256 --it 15` (threshold **25-30 ns**
  works on A6000), step 3 `row_set` (same-row groups), optional `bank_set --max 5`. Findings: **contiguous 256 B chunks map to
  one row+bank, successive 256 B chunks go to different banks**; A6000 GDDR6 **row size 2 KB, 64K rows/bank**; row-set
  recovery ~4 h/bank at 256 B granularity; bank function hypothesized to be a complex many-bit XOR; mapping stable for a 47 GB
  of 48 GB allocation; **tREFI 1407 ns**; A100 (HBM2e) and RTX 3080 showed no flips [P]. Follow-ups: GPUThor (CCS'26, arXiv
  2609.16546), GDDRHammer/GeForge, GPUBreach (arXiv 2605.03812).
- **No H100/HBM3 channel- or bank-mapping reverse engineering found** [?]. Closest: Fan et al.'s statement that a separate
  non-XOR hash selects L2 slice and HBM channel (non-power-of-two counts) [P]; accel-sim IPOLY+MODULO; DGNA's interleave fit.
  CPU-side black-box tools adaptable to GPUs: **Knock-Knock** (arXiv 2509.19568, platform-agnostic DRAM address-mapping RE) and
  **Sudoku** (arXiv 2506.15918, decomposing DRAM address mapping).

## 12. Floorplan / physical-distance inference

- **Die shots**: no public annotated GH100 die shot located (Locuza published only a leak-based block diagram in Mar 2022;
  TechInsights sells a paywalled H100 floorplan report with block-size table). HC34 Choquette slides and the whitepaper give the
  logical floorplan: GPCs in two rows, L2 partitions in the middle band, HBM PHYs on the left/right edges, NVLink along the
  bottom [S]. Treat exact GPC->partition adjacency as unknown; infer it from DGNA's "4 GPCs per L2 node, 2 GPCs per sub-node".
- **SemiAnalysis "Dissecting Nvidia Blackwell"** (Mar 2026, paywalled; summarised by Wallstreetcn): cluster-size `%smid` method
  to group TPCs into GPCs; some TPCs are never co-scheduled with others; **SM-pair L2-access latency matrix shows a > 300-cycle
  step** between the two dies; inferred TPC distribution die A 10/10/10/9, die B 9/9/9/(5+3); identical logical configs can have
  different physical layouts across cards [S].
- **CPU techniques transferable to GPUs**:
  - *Lord of the Ring(s)* (Paccagnella et al., S&P'21, arXiv 2103.03443): per core x LLC-slice load latency, **100,000 samples
    per pair**, latency monotone in ring distance; recovers logical ring order and address->slice mapping [P].
  - DATE'22 (Intel mesh): reconstruct unknown core map from traffic with dimension-order (Y-then-X) routing; per-chip maps
    differ; needed MSR access [P].
  - ARCS'24 (Ampere Altra CMN): derive mesh positions of cores/HN-F/memory controllers from latency [P].
  - Caveat (AnandTech Graviton2): "core-to-core" is a 3-point core->home-slice->core latency because lines are hashed to
    slices — must fix the home slice (control the address) before interpreting a heatmap as distance [P]. Same applies to
    SM x SM via L2; DSMEM latency is a true 2-point measurement and avoids this.

## 13. Tools for SM/GPC identification and per-slice counters

- **nvdebug** (Bakita, UNC; http://rtsrv.cs.unc.edu/cgit/cgit.cgi/nvdebug.git/about/): Linux kernel module exposing sysfs
  GPC-disable mask (bit 0 = GPC 0), on-chip TPC count per GPC, per-GPC TPC-disable masks -> authoritative **TPC->GPC
  correspondence**; used by Fan et al. as ground truth (100% agreement). **libsmctrl** masks index TPCs after driver
  renumbering, so a TPC->GPC map is needed for GPC-aligned partitions. Bakita & Anderson, RTAS'23 / ECRTS'25 (`nvtaskset`):
  "SM IDs are programmed by the driver and effectively random"; GPC-boundary partitions cost 6-15% capacity on H100 [P].
- **MIG**: 1g.10gb = **14 SMs** per `nvidia-smi mig -lgip` on H100 (1/7 of SMs); each GPC has a fixed SM composition but which
  GPCs back a profile varies by chip (Fan et al.) [P/S].
- **Nsight Compute per-slice counters**: `--print-metric-instances values` prints the GPU aggregate followed by every instance
  value for instanced metrics; `lts__t_sectors*` ("lts" = L2 slice tag stage) is instanced per L2 slice; `device__attribute_l2s_count`
  returns the slice count (expect 80 on H100). Example: `ncu --metrics lts__t_sectors_op_read.sum --print-metric-instances values ./probe`.
  Jin et al. obtained their address->slice map "from the profiler" this way [P].
- **PTX**: `%smid`, `%nsmid`, `%clusterid`, `%cluster_ctarank`, `mapa`, `ld/st.shared::cluster`, `barrier.cluster`, `%clock`/`%globaltimer`.

---

## 14. Consolidated H100-class numbers from the literature

| Metric | Value | Source | GPU variant | Clock / notes |
|---|---|---|---|---|
| L2 near hit | **258 cyc** | Luo et al. Table 4 [P] | H800 PCIe | ~1755 MHz boost; fine-grained P-chase |
| L2 far hit | **414 cyc** | Luo [P] | H800 PCIe | |
| L2 near miss (HBM via near partition) | **555 cyc** | Luo [P] | H800 PCIe | |
| L2 far miss | **744 cyc** | Luo [P] | H800 PCIe | global ~ (555+744)/2 |
| L2 local / remote node | **295 / 470 cyc** | DGNA [P] | H100 (SXM?) | GMM-filtered P-chase; remote penalty ~175 |
| L2 local sub-NUMA peaks | **280 / 312 cyc (+32)** | DGNA [S] | H100 | 2 GPCs per sub-node, 4 per node |
| DRAM remote penalty | **~174 cyc** | DGNA [P] | H100 | |
| L2 hit baseline / both partitions saturated | **273 / 508 cyc** | Jarmusch et al. 2507.10789 [P/S] | H100 PCIe | 31-45 MB footprint |
| L2 hit (NoC-only probe) spread | **up to ~70%** across SM/slice pairs | Jin MICRO'24 [P] | H100 (132 SM, 80 slices) | `__ldcg`, single thread; per-pair values in PDF |
| Far-partition latency | "nearly 2x" near; ~RX 6900 XT VRAM latency | Chips&Cheese [P] | H100 PCIe | clock sagged to 1395 MHz |
| VRAM latency | ~330 ns (PCIe), < 300 ns (GH200) | Chips&Cheese [P] | H100 PCIe / GH200 | |
| L2 read bandwidth (near) | **> 5.5 TB/s** | Chips&Cheese [P] | H100 PCIe | aggregate 50 MB: 3.8 TB/s [S, unverified]; A100 4.5 TB/s [S] |
| L2 bandwidth | "~5.x TB/s"; per-SM cache BW ~128 B/cyc | RRZE gpu-benches [P] | H100 PCIe | |
| SM->partition bandwidth | approximately uniform across SMs | Jin [P] | V100/A100/H100 | |
| L2 segmentation | 2 x 25 MB (NVIDIA/C&C); RRZE lists 2 x 30 MB with drop at ~25 MB | [S] | H100 | |
| Partition hash | XOR, lowest bit 12 (4 KB), **14 bits** | Fan et al. [P] | H100 PCIe | H200/B200: 16 bits |
| Effective L2 capacity | ~65% of physical (remote-line replication) | Fan et al. [P] | H100/H200/B200 | |
| GPC composition | 6 GPCs >= 16 SMs + 1 GPC >= 14 SMs | Fan et al. [P] | H100 PCIe (114 SM) | H200: 6x16 + 2x18 on 23/28 cards |
| SMEM direct / via DSM interface | **29 / 33 cyc** | Luo [P] | H800 | |
| DSMEM SM-to-SM, CS=2 | **181 cyc** (30% < L2) | Luo [P] | H800 PCIe | ~103 ns @1755 MHz |
| DSMEM SM-to-SM, CS=2..16 | **184-213 cyc** | Luo [P] | H800 | |
| DSMEM avg CS=2 vs global | ~190 cyc vs > 470 cyc | ClusterFusion [S] | H100 | |
| Global store+load exchange | **1110 cyc**; DSM 6.13x faster | Luo [P] | H800 | NVIDIA claims ~7x |
| DSMEM aggregate throughput (ring copy) | **3.27 TB/s @CS=2, 2.65 TB/s @CS=4**, lower for 8/16 | Luo IPDPS'24 [P] | H800 | per-SM ~ 3.27e12/132/1.755e9 ~ 14 B/cyc/SM |
| DSMEM remote vs local SMEM | ~6.4x latency, 4.4-5.4x lower throughput | CREDIT [P/S] | H100 & RTX 5090 | |
| ClusterSim model defaults | 171 cyc, 88 bits/cyc per link | gpgpu-sim-dsm [P] | (simulator) | not HW-validated |
| TMA bulk copy from L2 | ~38 B/cyc/SM (~5 KB/cyc chip-wide) | NVIDIA forum [S] | H100 SXM 132 SM | |
| TMA 4x multicast | ~70 B/cyc/SM, no gain at 8x ("intra-GPC bandwidth" limit) | NVIDIA forum [S] | H100 | |
| HBM3 | 3.35 TB/s, 5 stacks, 10 x 512-bit MCs | whitepaper [P] | H100 SXM5 | |
| Cluster co-residency | 8 portable / 16 non-portable | CUDA docs [P] | H100 | B200 example: 15x8, 33x4, 7x16 |
| Fingerprint stability (ref.) | 0.09 cyc jitter over 6 h; 100% die ID | Alpay attestation [P] | RTX 5090 / B200 | B200 cross-die 30 cyc |

---

## 15. Recommended methodology (synthesised from the above)

### 15.1 Assign every `%smid` to a GPC (and TPC)
1. **Cluster co-scheduling probe** (Fan et al.; SemiAnalysis; Luo): compile with `-arch=sm_90a`; launch with
   `cudaLaunchKernelEx` + `cudaLaunchAttributeClusterDimension` for cluster sizes **16 (non-portable), 8, 4, 3, 2**; each CTA
   writes `(%clusterid, %cluster_ctarank, %smid)`; launch enough clusters to fill the GPU (use
   `cudaOccupancyMaxActiveClusters`), repeat hundreds of launches with varying resource footprints (SMEM/regs) so that
   different SM subsets are chosen. Union-find over "same clusterid" edges yields GPC membership for SMs that ever
   co-schedule. Cluster size 16 forces 16 SMs of one GPC -> directly reveals >=16-SM GPCs; sizes 3/5/7 catch odd TPC counts.
2. **TPC pairing**: SMs sharing a TPC are the pair that always co-occur in cluster size 2 placements and share the
   L1.5/texture path; Fan et al. observed "SM IDs 124-131 (4 TPCs) never appear at cluster sizes > 2" — such SMs belong to
   GPCs that cannot host larger clusters (floorswept-down GPCs) and must be placed by step 3.
3. **L2-bandwidth / latency probing** (Fan et al. "L2-bandwidth probing"; L40 paper): SMs in the same GPC share one GPC->L2
   crossbar port; running a bandwidth-saturating kernel on a known GPC and measuring the slowdown of the unknown SM, or
   comparing its slice-latency signature vector, places the leftovers.
4. **Ground truth**: `nvdebug` sysfs (GPC mask, per-GPC TPC masks) if you have root; `nvidia-smi mig` profile SM counts
   (1g.10gb = 14 SMs) as a sanity check; expect 8 GPCs on SXM5 (132 SMs -> e.g. six 16-SM + two 18-SM, or similar), 7 on PCIe.
5. Record per-card: logical SM ID -> (GPC, TPC, SM-in-TPC); Fan et al. report a "predefined" part of the mapping that is
   constant across cards plus a floorsweep-dependent remainder, so repeat on each card.

### 15.2 Map global addresses to L2 partition (and, as far as possible, slice)
1. Allocate one large buffer (>= 2 GB, `cudaMalloc`; 2 MB pages so low 21 bits are physical). Pin a single-thread probe to a
   known SM (step 15.1). For each **32 B sector** (or 128 B line) in a 4-16 MB window: fine-grained P-chase with the line
   *first touched by the probe SM* (DGNA first-touch) and then re-read; `__ldcg`/`ld.global.cg` to bypass L1 (or
   `discard`/`invalidate` + reload as in Fan et al.). Collect >= 100 samples per line.
2. **Cluster latencies**: 1-D K-means / GMM with k = 2 (near/far) for hits, or k = 4 when misses are included (Luo:
   258/414/555/744 on H800). Luo's approach: run the chain once (misses) and again (hits) to label both dimensions.
3. **Recover the partition hash**: for a fixed probe SM, flip address bits 12..(log2 size) one at a time against a base
   address and record the near/far label (Fan et al.): a bit "participates" if the label flips for most bases; verify
   linearity by checking that the label of a^b equals XOR of labels (XOR hash => linear over GF(2)); solve the resulting
   GF(2) system for the hash vector (expect 14 participating bits on H100 PCIe, bit 12 lowest; check SXM5).
4. **Sub-partition / slice**: DGNA's +32-cycle sub-NUMA level (2 GPCs per sub-node) should appear as a 3rd/4th latency class
   if you probe from SMs in different sub-nodes; use `ncu --metrics lts__t_sectors_op_read.sum --print-metric-instances values`
   on a kernel that touches exactly one line to find which of the **80 slice instances** incremented — this gives the exact
   address->slice map per line without any timing (Jin et al.'s approach). Fit slice id = f(address bits) by trying XOR
   combinations; expect a non-linear/modulo component (80 = 16 x 5 is not a power of two; accel-sim uses IPOLY+MODULO).
5. Keep a dictionary of "pure" addresses: for each slice s, a set of lines that map to s (needed for the SM x slice matrix).

### 15.3 Build the SM x slice (or SM x partition) latency matrix
- For each SM i (pinned via `%smid`, one thread, others exit) and each slice s (addresses from 15.2.5): warm the line into L2
  from SM i, then P-chase a short chain of lines all homed on s (hit regime), timing with `%clock` (SM-local, drift-free)
  and cross-checking with `%globaltimer`; 100-1000 samples per pair, take median + MAD (Lord-of-the-Rings used 100k samples
  per pair; L40 paper found 0.09-cycle jitter so 1000 suffices). Lock clocks (`nvidia-smi -lgc`) and note the clock; C&C saw
  power-limit throttling to 1395 MHz under L2 load — keep probes single-threaded.
- Statistical treatment (L40 paper): fit additive model L(i,s) = mu + a(i) + b(s) (+ one rank-1 term); report R^2; the SM
  effect a(i) should cluster into GPC groups (two-fold symmetry on AD102; expect 2 partitions x 2 sub-nodes on GH100); the
  slice effect b(s) should split into two partitions with ~150-175-cycle offset (DGNA) and sub-node offsets of ~32 cycles.
  Hierarchical clustering of the row vectors of L yields GPC groups independent of 15.1 — use as cross-validation.
- Bandwidth matrix: Jin's two-kernel contention method, or TMA `cp.async.bulk` streams (plain loads saturate at ~2 in-flight
  loads/thread per arXiv 2608.15764); expect ~uniform SM->partition bandwidth except across the partition boundary.

### 15.4 Build the SM x SM DSMEM latency/bandwidth matrix
- Launch clusters of size 16 (non-portable) and 8 so that every SM pair (i, j) within a GPC co-occurs in some cluster; CTA
  rank r reads/writes rank r' SMEM via `mapa` + `ld.shared::cluster` / `st.shared::cluster` in a dependent chain (Luo: 29 cyc
  local, 33 via DSM interface locally, 181 cyc remote at CS=2, 184-213 at CS=2..16). Time with `%clock` on the issuing SM;
  record both `%smid`s; 1000 samples per pair; also time `barrier.cluster.arrive/wait` and `red.async`/`atom.shared::cluster`
  (no published numbers — fill the gap) and a `st.async` + mbarrier complete-tx ping-pong (2-point, avoids the 3-point L2 trap).
- Bandwidth: ring (R -> R+1) and pair patterns with ILP (Luo); broadcast pattern to find the per-GPC fabric limit (forum data
  suggests ~70 B/cyc/SM at 4x multicast = GPC-port limit); sweep CS = 2,4,8,16 and bytes in flight.
- Analysis: within a GPC, test whether latency depends on (TPC adjacency, same-TPC pair, |i-j| in physical order); a flat
  matrix (Luo's narrow 184-213 range, B200 "~89 ns" flat) means a single-stage crossbar; structure (two latency levels) would
  indicate a hierarchical GX-plane fabric as hinted by the DSMEM patent. Combine with the SM x slice matrix: SMs that share
  a GPC must share the slice-signature cluster; disagreement flags mis-assigned SMs.

### 15.5 Reporting
- Always report GPU SKU (SXM5 vs PCIe vs H800), driver/CUDA version, locked SM and memory clocks, cycles and ns, sample
  counts and dispersion, and whether lines were first-touched by the probe SM (DGNA) — these explain most discrepancies in the
  table above (e.g. Luo 258 vs DGNA 295 vs Jarmusch 273 for a "near" L2 hit).

---

## 16. Gaps left open (could not be resolved from accessible sources)
- Jin et al.'s per-pair latency values and the exact partition-boundary numbers for H100 (PDF blocked).
- Full text of US12248788B2 (CPC/TPCARB/GPCARB/GPCMMU/GX0-GX1, mbarrier-arrive delivery, ordering/acks).
- Luo Sec. 7.2 SM-ID allocation within clusters; DSMEM throughput at CS = 8/16.
- Any HBM3 channel/bank mapping for H100; any public annotated GH100 die shot; any DSMEM-atomics / `cp.reduce.async.bulk`
  latency measurements.
