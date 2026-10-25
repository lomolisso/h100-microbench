# H100 (GH100) architecture — what matters for this project

Scope: the planar baseline. Only facts needed to design, place, and interpret the microbenchmarks. Each fact carries a provenance tag: **[NV]** official NVIDIA document (whitepaper "NVIDIA H100 Tensor Core GPU Architecture", Hopper Tuning Guide, Programming Guide, driver headers), **[PAT]** NVIDIA patent text, **[LIT]** measured in published microbenchmark papers (see `08__prior_work_reference_numbers.md`), **[INF]** inference — must be confirmed by our own experiments. Items marked **[INF]** are exactly the things the microbenchmarks exist to settle.

## 1. Chip-level organisation

| Item | Full GH100 die | H100 SXM5 | H100 PCIe (and H800 PCIe) | Tag |
|---|---|---|---|---|
| GPCs | 8 | 8 (some GPCs have TPCs disabled) | 7 or 8 active | NV |
| TPCs per GPC / SMs per TPC | 9 / 2 → 18 SMs per GPC, 72 TPCs, 144 SMs | 66 TPCs, **132 SMs** | 57 TPCs, **114 SMs** | NV |
| SM count reported by `cudaDeviceProp.multiProcessorCount` | — | 132 | 114 | NV |
| L2 cache | 60 MB | **50 MB**, "partitioned crossbar structure" | 50 MB | NV |
| L2 partitions | — | 2 (≈25 MB each) | 2 | LIT (Luo; Chips&Cheese; Fan et al.) |
| L2 slices ("lts" units in Nsight Compute) | 96 (12 FBPs × 8) | 80 (8 slices per 512-bit memory controller, 640 KB per slice) | 80 | LIT (Jin et al.) for 50 MB SKUs; **our H100 NVL reports 96 (`device__attribute_l2s_count`) with 94 active (`lts__*.avg = .sum/94`, 6016-bit bus = 94 × 64 bit)** [MEAS 08] |
| HBM stacks / memory controllers | 6 stacks / 12 × 512-bit | 5 × HBM3 / 10 × 512-bit = 5120-bit | 5 × HBM2e / 10 × 512-bit | NV |
| DRAM bandwidth (peak) | — | **3.35 TB/s** | 2.0 TB/s | NV |
| Boost / typical SM clock | — | 1980 MHz boost (1755 MHz is used in the whitepaper's throughput tables) | 1755 MHz | third-party tables; **read from `nvidia-smi`** |
| Unified L1 / texture / shared memory per SM | — | **256 KB**; shared-memory carveouts 0, 8, 16, 32, 64, 100, 132, 164, 196, **228 KB** | same | NV |
| Max shared memory per block | — | 227 KB (1 KB reserved by the driver per block) | same | NV |
| Shared-memory banks | — | 32 banks × 4 B, 128 B/cycle/SM | same | NV |
| Registers | — | 64 K × 32-bit per SM; 255 per thread | same | NV |
| Max resident threads / warps / blocks per SM | — | 2048 / 64 / 32 | same | NV |
| MIG | — | up to 7 instances (1g.10gb = 14 SMs → 1 GPC with 14 enabled SMs); each instance owns its own L2 banks, crossbar ports, memory controllers | — | NV (instance ↔ GPC equivalence: INF) |
| Thread block cluster size | — | 8 portable, **16 non-portable** (opt-in) | same | NV |

Consequences for experiments:
- 132 SMs ≠ 8 × 16. GPC sizes are heterogeneous after floor-sweeping. Reported co-scheduling domains on an H100 SXM: `18,18,16,16,16,16,16,8,4,2,2` (ClusterSim `scheduler.cu`, one card) and "six 16-SM GPCs + two 18-SM GPCs" on 23/28 H200 cards (Fan et al. 2026). On H100 PCIe: 6 GPCs with ≥16 SMs and 1 GPC with ≥14 SMs. **Every card must be mapped individually** (`docs/experiments/03__sm_gpc_topology.md`).
- Logical `%smid` is assigned by the driver and is "effectively random" with respect to physical position (Bakita & Anderson); part of the smid→GPC map is fixed per architecture, part depends on the chip's floor-sweeping (Fan et al.). Never assume `%smid / 18` is the GPC.
- 16-block clusters can only be hosted by GPCs with ≥16 free SMs; on the ClusterSim card 18 SMs were left idle with cluster size 16 (Lühnen 2024: 18 idle on PCIe).

## 2. SM internals relevant to memory experiments [NV]

- 4 sub-partitions (SMSP) per SM, each with its own warp scheduler, register file slice, tensor core, and LSU issue; one **TMA unit per SM** shared by the 4 sub-partitions [NV blog; FlashGPU-sim: 44-cycle same-warp TMA issue gap, 16 admitted transactions per SM — LIT, measured on RTX 5090].
- The LSU/L1TEX pipeline services generic-proxy `ld/st` to global (through L1), shared (`LDS/STS`), and **distributed shared memory** (`LD/ST` with a `.shared::cluster` address, routed to the SM-to-SM network instead of the local SRAM).
- The async proxy (TMA, `cp.async.bulk*`, `st.async`, `red.async`) bypasses L1 for global data: TMA loads are served from L2 (Luo et al.: TMA latency curve has no L1 inflection point) and TMA writes go to L2.
- Shared memory and L1 are the same SRAM (carveout). Local SMEM latency ≈ 29–33 cycles, 128 B/cycle/SM throughput [LIT].

## 3. GPC internals and the SM-to-SM network

From NVIDIA patent US 12,248,788 B2 ("Distributed shared memory") and the CGA patent family (US 2023/0289215 A1, US 12,333,311 B2) [PAT], as summarised by FlashGPU-sim and the H200 DSM blog:

- A GPC is built from **CPCs** (compute processing clusters); each CPC contains **3 TPCs = 6 SMs**, a TPC arbiter (TPCARB) per TPC, and shares a **GPCMMU** (address translation, including the hash that picks the L2 slice) and a **GPCARB** (GPC arbiter) [PAT/INF — the CPC/GX details are from the FlashGPU-sim reading of the patent figures and are not confirmed verbatim].
- SM-to-SM traffic ("SM2SM") and GPC→L2 traffic enter the **same GPC crossbar** through the GPCMMU; the patent says a "new low-latency SM2SM communication network in GPCARB" carries DSMEM loads/stores/atomics, remote mbarrier arrivals, and shared→shared TMA copies **without touching L2** [PAT].
- Two parallel switch planes **GX0/GX1** are described; firmware may disable one plane at reduced bandwidth [PAT via FlashGPU-sim — INF].
- Per-CPC ingress to the switch: **4 lanes × 32 B/cycle shared by 6 SMs → 21.3 B/cycle/SM** when all six are active; measured one-way per-SM remote-load/store/TMA-put throughput is **19.9–21.3 B/cycle on H200** and the limit is *not* work-conserving (an SM alone still gets ≈21 B/cycle) [LIT, blog + FlashGPU-sim]. Whether this is a hard per-SM shaper or a TDM slot assignment is **[INF]**.
- Reads cost a request flit plus 128 B response quanta; writes carry payload forward and get coalesced ACKs back. Consequently bidirectional loads lose ≈23 % per direction while stores/TMA puts lose only 4–5 % [LIT].
- Measured DSMEM remote-load latency is flat across SM pairs inside a GPC (184–218 cycles, ±8 %), which rules out ring/mesh and points to a (possibly two-stage) crossbar [LIT: ClusterSim, Luo]. One-way latency ≈ 78 cycles (FlashGPU-sim, H200). Whether same-TPC or same-CPC pairs are measurably faster is **[INF]** — this is the sensitivity analysis the 3D-stacking study needs (`docs/experiments/10__dsmem_latency_heatmap.md`).
- Jin et al. (MICRO 2024, H100 PCIe) found the CPC in the data: groups of 2–3 TPCs with identical L2-latency signatures (Pearson correlation) and an SM-to-SM load latency that rises from 196 cycles inside one CPC to 213 cycles between the two outer CPCs; in their 16-SM GPC the grouping is CPC = (TPC index within the GPC) mod 3 [LIT, `08` P6] — the concrete hypothesis `10` tests.
- Multicast: a TMA global→shared load with `.multicast::cluster` returns **one L2 read response that is replicated at/after the crossbar** to every CTA in the 16-bit mask [PAT US 12,020,035 B2]; forum measurements show ≈38 B/cycle/SM without multicast and ≈70 B/cycle/SM with 4× multicast, with no gain from 4×→8× ("intra-GPC bandwidth" limit) [LIT, unverified].

## 4. L2 cache and memory side

- Two L2 partitions, each "directly connected" to a subset of GPCs. **On this device** [MEAS 04/05/08]: partition 0 = GPCs {G0, G4, G5, G6}, partition 1 = {G1, G2, G3, G7}, each split into two sub-nodes; a line held only at its home costs 251 cycles from its own sub-node, +29 from the other sub-node of the partition, +162 from the other partition (417/443); misses 505 near / 667 far. Other chips [LIT]: near/far hit 258–295 / 414–470 cycles, DRAM 555 / 744 (H800).
- **Home is a function of the address, not of first touch** [MEAS 08]: partition = XOR of physical address bits 12, 13, 15, 17, 19 (⊕ 21, 22, 24), constant per 4 KiB. **Any read or write from partition P leaves a copy in P** (partner sub-node of the home's sub-node) that serves every SM of P near until a write from the other partition invalidates it; this is what prior work called "first-touch homing" and "replica" (DGNA, Fan et al.) and why only ≈ half of the L2 is unique capacity for one SM (`06` §1.1). Slice selection inside a partition (48 slices on this device; non-power-of-two) is unknown; accel-sim models it as IPOLY-modulo. **Nsight Compute on this setup exports no per-slice counter values**; the slice of a line is identified with the busiest-unit (`.max`) rollup oracle of `docs/experiments/13__l2_slice_mapping.md`.
- L2 line 128 B, sector 32 B; L2 throughput measured ≈ 4000–4500 B/cycle chip-wide (7–8 TB/s) [LIT]; DRAM ≈ 90 % of peak achievable with float4 streaming or TMA [LIT].
- Each L2 slice talks to its memory controller; 8 slices per 512-bit HBM3 controller (16 × 64-bit channels per stack, 2 pseudo-channels each) [LIT/INF]; on this NVL 12 FBPs × 8 = 96 slices of which 94 are active, matching the 6016-bit bus [MEAS 08, INF; tested in `13` Step 4]. The DRAM channel/bank hash of H100 has not been published [LIT survey].

## 5. Clocks and timers [NV]

- SM clock domain drives `%clock64`; L2/crossbar clock on GH100 is believed to be a separate domain (accel-sim models icnt/L2 at 2× core; FlashGPU-sim notes a possible hidden clock crossing for DSMEM) — **[INF]**; measure the ratio `lts__cycles_elapsed.avg.per_second / sm__cycles_elapsed.avg.per_second` with ncu in `01__timers_and_harness`.
- `%globaltimer` is a global nanosecond timer, consistent across SMs; observed update granularity ≈ 32 ns [LIT: Lühnen] — characterise in `01__timers_and_harness`.
- H100 throttles under heavy L2/HBM load (Chips&Cheese saw 1395 MHz on PCIe); clocks must be locked and recorded (see `09__measurement_conventions.md`).

## 6. Five (actually nine) memory-access paths of a single H100

The user's list "(1) SM→local SMEM, (2) SM→remote SMEM, (3) SM→L2, (4) SM→GMEM" is refined below; the complete per-instruction taxonomy with completion mechanisms is in `02__memory_access_taxonomy.md`.

| # | Path | Generic proxy (SM LSU) | Async proxy (TMA / bulk engine) |
|---|---|---|---|
| 1 | SM → own SMEM | `ld/st.shared::cta` | `cp.async.bulk` global→shared (dest is own smem) |
| 2 | SM → remote SMEM (same GPC, same cluster) | `ld/st/atom.shared::cluster` via `mapa` | `cp.async.bulk` shared::cta→shared::cluster, `cp.reduce.async.bulk` →shared::cluster, `st.async`, `red.async` |
| 3 | SM → L2 (hit) | `ld.global.cg` / `ld.global.ca` after warm-up | TMA load with L2-resident working set |
| 4 | SM → HBM (L2 miss) | `ld.global` with ≫L2 working set / `ld.global.cv` | TMA load with ≫L2 working set |
| 5 | SM → L1 (hit) | `ld.global.ca`, ≤ carveout | — (TMA bypasses L1) |
| 6 | L2/GMEM → *N* SMEMs (multicast) | — | `cp.async.bulk{.tensor}...multicast::cluster` |
| 7 | SMEM → L2/GMEM with reduction | `red.global`/`atom.global` (per thread) | `cp.reduce.async.bulk{.tensor}` shared→global |
| 8 | SMEM → remote SMEM with reduction | `atom/red.shared::cluster` | `cp.reduce.async.bulk` shared→shared::cluster, `red.async` |
| 9 | Prefetch into L2 | `prefetch.global.L2` | `cp.async.bulk.prefetch{.tensor}.L2` |

Hardware-supported collectives (the only ones in scope): multicast load (one source, many SMEM destinations, path 6); bulk reduce into one SMEM or into L2 (paths 7–8); remote `red.async` with mbarrier completion (path 8). Everything else (AllReduce, AllGather, ReduceScatter, AlltoAll) is a software composition of paths 2, 6, 8 and is treated in `docs/experiments/17__dsmem_collectives.md`.
