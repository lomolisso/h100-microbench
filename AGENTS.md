# H100 microbenchmarking for 3D-stacked GPGPU research — index for the planning and coding agents

## Mission
Characterise, on a real H100, the micro-architectural quantities that a 3D-stacking study of the Hopper GPC/DSMEM/L2 hierarchy needs: latency and bandwidth of **every** hardware memory-access path (generic proxy and async proxy/TMA), the cost of **every** synchronisation primitive, the **SM-to-SM network** behaviour versus cluster size and topology, point-to-point latency **heatmaps** across the whole hierarchy (SM×SM, SM×L2-slice, SM×partition, SM×memory-controller) from which physical distance and interconnect structure are inferred, and the performance of **hardware-supported intra-cluster collectives**. Everything is written in CUDA C++ / inline PTX, writes CSV, and is explained in one self-contained `.ipynb` per experiment. The executed notebooks (figures embedded) are the published result; raw data stays on this machine. Published prior-work numbers (`docs/context/08`) are used by the agents only to sanity-check our measurements. The notebooks report our own results and never cite other work.

## Experiment status (a new conversation starts here; see CLAUDE.md "Starting a new conversation")
Status values:
- `todo`: not started.
- `in progress`: being implemented in the current conversation.
- `in review`: committed locally; waiting for the user's review of the notebook; not pushed.
- `done`: approved by the user and pushed.

Update the row **before** the experiment's commit. "Key results" records what later experiments consume (numbers, file names).

| NN | Experiment | Status | Key results for later experiments |
|---|---|---|---|
| — | Environment, device configuration, SM-clock selection | done | `docs/environment/README.md`, `docs/context/10__device_config.md` (SM clock locked at 1635 MHz) |
| 01 | timers_and_harness | done | Harness: `experiments/harness/{harness.cuh,host.hpp,Makefile.inc}` (functional test: `01/src/harness_selftest.cu`). Timer read issue cost: `%clock64`/`%clock`/`%globaltimer` 2 cycles, timer pair 4 cycles. `%globaltimer`: 32 ns tick; offsets between SMs ≤ 1 tick; flag observation across SMs 224–448 ns. Dependent `ld.shared` latency **23 cycles** (30 at unroll 1). Loop overhead at the default (unroll 16, iters 4096): fixed region 66–192 cycles ⇒ ≤ 0.05 cycles/access; use the slope between two `iters` (`results/01__timers__loop_overhead.csv`). ptxas fuses `add.u32` chains and deletes unused timer reads: use `mad.lo` / sinks. **L2/SM = 0.954** (0.951–0.956; L2 ≈ 1556 MHz, DRAM 2619 MHz) at the 1635 MHz lock. Real SM clock under the lock: 1622–1635 MHz, varying per launch and per **8 clock groups (18/18/18/18/16/16/16/12 SMs; SM→GPC candidate for 03, `k2all` run in `results/01__timers__globaltimer_resolution.csv`)**. Empty-kernel launches are host-bound: ≈ 7–8 µs/launch; event pair alone 3.9 µs (`results/01__timers__event_floor.csv`) |
| 02 | cluster_launch_and_occupancy | done | Cluster launch overhead = wave quantisation only: per-wave time and empty-kernel launch (4.2–4.3 µs) identical for CS 0–16; back-to-back gap 1.06 µs (no cluster) vs 1.12–1.15 µs (clusters). Overhead at 2112 CTAs: CS 1/2 ≈ 0 %, CS 4/8 +12 %, CS 16 +19 % (1 CTA/SM) / +24 % (2 CTAs/SM); +90 % just above one cluster wave. Max active clusters (1 CTA/SM): CS 2 → 66, CS 4 → 30, CS 8 → 15, CS 16 → 7 (idle SMs 0/12/12/20); 2 CTAs/SM: 132/62/30/14 (LoadBalancing 132/66/30/15 by co-locating). `cudaOccupancyMaxActiveClusters` = ncu `launch__cluster_max_active` = measured first wave in every launch. **GPCs (SM→GPC candidate for 03)**: 8 GPCs of 18/18/18/18/16/16/16/12 SMs = 01's clock groups; G0 = {0,1,16,17,32,33,48,49,124,125,130,131}, G1 = {2,3,18,19,34,35,50,51,64,65,78,79,92,93,106,107} … (TPC t = SMs 2t,2t+1; table in the notebook). **SMs 124–131 (TPCs 62–65; in G0 ×2, G4, G5) never join clusters ≥ 4** → cluster-capable GPC sizes 18/18/16/16/16/16/16/8. First wave: ranks 2k,2k+1 share a TPC (100 %), rank→SM identical across launches; later waves fill freed SMs (28–86 % stable). Default never co-locates and ncu reports it as Spread. Placement: `results/02__cluster_launch__placement.csv` (every CTA of 1470 launches; local only) |
| 03 | sm_gpc_topology | done | **Map: `experiments/03__sm_gpc_topology/results/03__sm_gpc_topology__map.csv`** (local only; latest run_id; columns smid, gpc, gpc_size, hosts_cs16, tpc_mate_hyp, partition/subnode empty for 08, cosched_domain, cosched_domain_size, max_cs_observed, clock_domain). **8 GPCs G0–G7 = 12/16/16/16/18/18/18/18 SMs** (labels by smallest smid, same as 02). G0 = {0,1,16,17,32,33,48,49,124,125,130,131}; TPC t = SMs 2t,2t+1, TPC→GPC: t<32 → t mod 8; 32–59 → 1+(t−32) mod 7; 60→G6, 61→G7, 62→G0, 63→G4, 64→G5, 65→G0. Three signals agree: cluster co-membership (3000 launches, CS 16/8/4/3/2, no cluster ever crossed a GPC), per-SM clock domains (identical within a GPC to 0.002 MHz), ncu `gpc__cgas_launched` sum/min/max/avg = map prediction in 10/10 configs. **SMs 124–131 (TPCs 62–65) only ever pair with smid^1** (never in clusters ≥ 3) → cluster-capable sizes 18/18/16/16/16/16/16/8 = 124 SMs; **7 GPCs (G1–G7, 112 SMs) host CS 16**; G0 max 1×CS8 / 2×CS4 at 1 CTA/SM. TPC mates (2t,2t+1): 100 % of one-wave CS-2 clusters and rank pairs (2r,2r+1) — hypothesis for 10/15. **Per-GPC SM clock under the lock differs by up to 12.5 MHz (0.76 %) within one launch** (1622–1635 MHz): convert cycles→ns per GPC when sub-percent matters (`results/03__sm_gpc_topology__clock_domains.csv`). MIG profile list (MIG off): 1g.12gb = 16 SMs × 7. Card serial 1793225009411 |
| 04 | memory_hierarchy_latency | done | **Summary: `experiments/04__memory_hierarchy_latency/results/04__latency__summary.csv`** (local only; latest run_id; levels smem, smem_dsmem_if, l1, l2_near, l2_far, dram_near, dram_far, dram_slow, l2_chase, dram_chase). SM cycles at 1635 MHz: `ld.shared` **23**, own smem via DSMEM path (mapa + `ld.shared::cluster`, incl. 1 MOV/hop) **32.75**, L1 hit **32**, L2 near **270** / far **426**, HBM near **505** / far **667** (+ ~6.5 % slow tail ~856), plain chase L2 **269**, HBM **614** (618 at 2 GiB). Per-access (K3) values have the 3-cycle timer pair subtracted. **L2 partitions = GPC groups {G0,G4,G5,G6} | {G1,G2,G3,G7}** (preliminary `partition` label for 08); home partition by address (50/50 of a random buffer), and writing or reading a line leaves a copy in the accessor's partition (2nd read always near) → plain chases see only near hits, far hits need first-touch control (K3 method: flush, whole-line touch by SM T, probe). **Sub-nodes**: {G0,G6},{G4,G5} | {G1,G2},{G3,G7}, +23 cycles across sub-nodes. Single-SM chase leaves the near plateau at ≈ 29 MiB (½ L2), plateau 478 at 40–50 MiB, HBM from ≈ 85 MiB. L1 = 224 KiB (driver carveout 32 KiB without smem); `.cv` does not bypass L2; `.ca`→`LDG.E.64.STRONG.SM`, `.cg`→`.STRONG.GPU`, `.cv`→`.STRONG.SYS`. Each HBM miss = 2 L2 lookups + ~1 partition-link sector. ncu needs `--replay-mode application` for warm-cache counters (kernel replay's save/restore wipes L2). **Chain generator for 06: `experiments/04__memory_hierarchy_latency/src/pchase_chain.hpp`** (Sattolo, kChainSeed, 128 B lines). Harness: `claim_target_sm` / `hb::SmClaim` pin a one-warp block to an SM without reserving smem |
| 05 | memory_hierarchy_bandwidth | done | **Summary: `experiments/05__memory_hierarchy_bandwidth/results/05__bandwidth__summary.csv`** (local only; latest run_id; levels smem_ld/st, l1_ld, l1_st_all_sms, l2_{ld,st}_{mixed,local,remote}, l2_mix_mixed, l2_ld_max, l2_link_one_way/both_ways, l2_ld_one_gpc, l2_ld_first_pass_{near,far}, dram_{read,write,mix,mix11}, dram_read_per_sm). At 1635 MHz: **SMEM = L1 = 128.0 B/cycle/SM** (ld/st, 4 or 16 B; 27.5 TB/s for 132 SMs; 0 bank conflicts, 1 wavefront per 128 B); stores leave an SM at ≤ **32 B/cycle**. **L2 read 7.7 TB/s** (16 MiB, all SMs; local data 7.8–8.0 up to 64 MiB; ≈ 36 B/cycle/SM chip-wide, 81 B/cycle for one SM alone), **L2 store 4.0 TB/s**, 50/50 mix 6.9. **Per GPC ≤ 1.16 TB/s** (16 or 18 SMs; GPC↔L2 port). **L2 partition link 1.11 TB/s per direction (2.22 both ways)**: remote stores (never replicated) and first reads of far-homed lines both saturate there; far first reads ≈ 10.6 vs near 22 B/cycle/SM (66 readers). Repeated reads are always near (copies; no ld cache op avoids it: .cv = .cg, .cs/.lu lose residency, .cs may hit L1). Line homes: 50/50, constant per 4 KiB, alternate within 8 KiB, all 8 GPCs agree (in-run timing labels, `src/bw_l2.cu` label_homes). Capacity knees: local 64–72 MiB, mixed ~52, remote 40–44 MiB; beyond, remote streams at 2.35 TB/s. Small footprints (1–4 MiB, many SMs on the same lines) reach 19 TB/s by request merging before the L2 tags (only 35 % of sectors looked up) — not L2 data bandwidth. **HBM read 3.68 / write 3.64 / 5R1W 3.52 / 1R1W 3.40 TB/s** (93/92/89/86 % of 3.94); one SM reads ≈ 99 GB/s (61 B/cycle) with 64 KiB in flight, 90 % at 67 SMs; writes 30 B/cycle/SM, 90 % at 113 SMs; half of HBM traffic crosses the partition link. ncu: the profiler adds ≈ 1.1 M L2 write + 0.37 M link sectors per profiled kernel (empty-kernel baseline); never trust first-pass hit rates under ncu. Harness lessons: ptxas hoists loop-invariant `LDG.STRONG.SM/GPU` and plain `LDS` out of timed loops (add `(iter & zero)` to the address or use `.volatile` shared) and merges repeated identical global loads; 64-bit shared atomicAdd from every thread costs tens of µs per block (warp-reduce first). `CleanL2` (bw_common.hpp) = flush + clean read: no dirty write-backs in timed kernels |
| 06 | tma_latency_and_issue | done | **Summary: `experiments/06__tma_latency_and_issue/results/06__tma__summary.csv`** (local only; latest run_id). Dependent 16-B TMA chase (expect_tx + `cp.async.bulk` + try_wait loop + ld.shared per hop): **L2 323 / HBM 671 cycles** vs `ld.global.cg` 271 / 618 → **TMA − generic ≈ 52 cycles** at both levels (tensor 1-D/2-D +15 more; `cp.async.cg` +35); no L1 plateau (TMA never hits L1). Wait costs: try_wait/test_wait predicate on a completed phase **≈ 30 cycles** (44 incl. selp+mad chain glue), with suspendTimeHint ≈ 60; try_wait on an incomplete phase returns false after **≈ 7770 cycles (4.75 µs)**, hint value (0/100/1000 ns) not used by the first TRYWAIT; test_wait loop period ≈ 50 cycles (+18 cycles/hop). `arrive.expect_tx` 8 cycles back to back, ≈ 40 dependent (for 07). One copy: **T ≈ 340 + bytes/107 B/cycle** (flat ≈ 330–347 up to 1 KiB; 16 B from HBM ≈ 705). Issue (handing a command over; data arrives later): **48 cycles/copy per warp** (any size/source), each warp's first copy +≈ 55 cycles of address setup; first commands of different warps accepted ≈ 37 cycles apart, then 2–4 warps ≈ 22/13 cycles per copy aggregate; issue throttles to the **128 B/cycle** TMA processing rate (= ncu rated peak, `l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.avg.peak_sustained`); **no 16-transaction admission limit**: ≥ 52 copies (2 KiB, 4 warps, HBM) outstanding without stall. TMA units of 8 SMs (incl. TPC mates) independent. Box cost ≈ fixed + max(rows, bytes/128) cycles (one box row per cycle; 16-KiB box from L2: 455 cycles with ≥ 256-B rows, 1340 with 16-B rows); SWIZZLE_128B costs no load time but pads each box row to 128 B in smem (64-B rows → 2×, 16-B rows → 8× span; read-side benefit and 32B/64B modes → 16 P12); tensor-map first use in a launch +35 cycles, removed by `prefetch.tensormap`. `cp.async.bulk.prefetch.L2` two hops ahead: HBM chase 671 → 513. Anomaly: in HBM, hint/test_wait loops intermittently 45 cycles/hop faster than try_wait (sticky across launches). PTX lessons: `cuTensorMapEncodeTiled` rejects globalStrides = nullptr even for rank 1; with N a template, unrolled bulk issues use the uniform datapath (runtime guards add ≈ 20 instr/copy); time TMA loads with `read_clock64_after(coords)` when coordinates come from memory. Harness: tensor/hint/test_wait/cache-hint/cp.async wrappers added to `harness.cuh` |
| 07 | synchronization_primitives | in review | **Summary: `experiments/07__synchronization_primitives/results/07__sync__summary.csv`** (local only; latest run_id; level `<kernel>/<primitive>/<variant>`, config = cs/threads; cycles per op with the 2-cycle timer pair subtracted; propagation rows `…/one_way_ns`, `…/round_trip_cycles`). At 1635 MHz: **every release = one GPU-scope fence**: `MEMBAR.ALL.GPU; ERRBAR; CGAERRBAR` costs **264–284 cycles if no store since the last fence, ≈ 520 after any store (however old)**, 407 / 776 at 32 warps; all `.cluster` fences/releases/acquires compile to the GPU-scope MEMBAR (+ `CCTL.IVALL`), there is no cheaper cluster fence. **Cluster barrier** (CS 1–16 identical): arrive.release 284 (≤ 128 thr/CTA; 335 / 575 at 512 / 1024), behind a pending store 520–550, arrive.relaxed 22, wait with all arrived 23, `cluster.sync()` 328–360 (≤ 128 thr; 640–700 at 1024); signal propagation 64 ns relaxed / 224 ns release (CS 2–16, any placement); split phase hides no fence (split = arrive + work + wait). **mbarrier**: init 32, arrive 29, arrive.expect_tx 38, arrive_drop 29, expect_tx / complete_tx issue 0–4 (≈ 7 back to back), test/try_wait on a completed phase 38/39 (43 per dependent poll), incomplete try_wait returns false after 7765 cycles (4.75 µs), suspendTimeHint 0–10000 ns changes nothing (→ `NANOSLEEP.SYNCS`), wake-up after completion 60–190; 32 warps arriving on one barrier ≈ 1 warp-arrive / 2 cycles; remote arrive release 269 / relaxed 0 issue, one-way 250 / 84 ns. **CS-2 hand-off round trips** (rank 0 cycles; one-way ns): remote relaxed mbarrier arrive 517 (128), **st.async / red.async + mbarrier 565 (160; data visible, no sender fence) → cheapest correct data hand-off**, 2× cluster.sync 712, st + ld.volatile poll 754, pull poll 512, release remote arrive 953, st.release/ld.acquire, red.release, atom.exch 1302–1333 (384–416), relaxed + 2 fences 1824, **global flag 1964 / 2130 (576 / 672 ns; per L2 home page)**; TPC mates vs other TPC vs SMs 124/125 within ≈ 55. **CTA**: `__syncthreads` 9 / 13 / 41 / 85 (1 / 4 / 16 / 32 warps), named `bar.sync 1,N` ≈ 33 for one warp, `bar.arrive` 9; `__syncwarp` and `fence.mbarrier_init` → NOP; `fence.*.cta` 13–46; `fence.proxy.async` 144–287, `.shared::cta` 8–33, `.global` 0; `grid.sync()` 1400–1424 for 1–132 blocks (+256 with stores pending). Stalls (ncu per PC, `results/07__sync__stalls.csv` + `07__sync__ncu_pcs.csv`): `UCGABAR_WAIT` → barrier, release fence → membar (sampled at the `ERRBAR`), try_wait loop → branch_resolving / long_sb, hinted try_wait → sleeping (sampled at the lanes' `WARPSYNC`). **Harness lessons**: `bar.sync` = `BAR.SYNC.DEFER_BLOCKING`: a clock read right after it can precede the barrier's completion by the whole arrival skew → make t0 / t1 depend on a shared load after the barrier (`bar_sync0_then_lds`, `07/src/sync_common.hpp`); issue an untimed fence before timing any release operation (store history shifts it by ≈ 256); a lone CS-2 cluster lands on SMs 124/125, CS 4/8 in GPC 0, CS 16 in GPC 1. Harness: named-barrier, non-aligned cluster-barrier, mbarrier (state-returning arrive, expect_tx, complete_tx, relaxed remote arrive), `st.async`, `red.async`, DSMEM flag and atomic wrappers added to `harness.cuh` |
| 08 | l2_partitions_and_first_touch | todo | |
| 09 | dsmem_latency_vs_cluster_size | todo | |
| 10 | dsmem_latency_heatmap | todo | |
| 11 | dsmem_bandwidth_vs_cluster_size | todo | |
| 12 | dsmem_atomics_and_async_stores | todo | |
| 13 | l2_slice_mapping | todo | |
| 14 | gmem_latency_vs_controller_distance | todo | |
| 15 | noc_latency_heatmaps | todo | |
| 16 | async_proxy_paths_bandwidth | todo | |
| 17 | dsmem_collectives | todo | |
| 18 | access_type_summary | todo | |

Experiments run strictly in numerical order (01 → 18). The order respects the dependency graph below.

## Repository layout
```
AGENTS.md                      ← this index (CLAUDE.md imports it and adds the Claude Code workflow)
CLAUDE.md                      ← Claude Code project instructions
Makefile                       ← builds every experiments/*/Makefile in order
requirements.txt               ← Python packages for the notebooks
docs/
  context/                     ← precise background; read before planning
    00__h100_architecture.md             chip/GPC/SM/L2/HBM facts, 9 access paths, what is known vs. inferred
    01__clusters_and_dsmem.md            cluster programming model, host/device APIs, PTX, placement facts, kernel skeleton
    02__memory_access_taxonomy.md        the master table: 26 access forms × proxy × completion mechanism; hardware collectives
    03__tma_and_bulk_async_copies.md     TMA unit, every sm_90 bulk form, tensor-map constraints, canonical patterns
    04__mbarrier.md                      object/phases/tx-count, all ops incl. remote forms, templates
    05__synchronization_primitives.md    decision table, memory-model rules, PTX templates, protocol recipes
    06__l2_nuca_and_physical_topology.md L2 partitions/slices/hash, SM→GPC mapping, methodology for heatmaps and hash recovery
    07__profiling_tools.md               Nsight Compute flags and verified gh100 metric names (DSMEM = `dshared`, TMA = `tma`), SASS map, nsys, clocks
    08__prior_work_reference_numbers.md  every H100-silicon number from the 5 reference papers + blog, with tolerances
    09__measurement_conventions.md       timers, clock locking, placement control, cache control, statistics, CSV schema, notebook layout
    10__device_config.md                 THE device (H100 NVL) and the single fixed run configuration (locked clock, limits); prior-work chips side by side
  experiments/                 ← one self-contained experiment SPEC per file (kernel design, sweeps, CSV columns, figures, agent-only sanity checks, ncu metrics)
    01__timers_and_harness.md … 18__access_type_summary.md
  research_notes/              ← raw research notes the context files were distilled from (PTX ISA, CUDA API, Nsight, repos, literature)
experiments/                   ← IMPLEMENTATIONS, one directory per spec, same NN__name
  harness/                     ← shared device/host helpers produced by experiment 01 (harness.cuh, host.hpp, Makefile.inc)
  01__timers_and_harness/      {Makefile, src/, results/, results/sass/, 01__timers_and_harness.ipynb}
  02__cluster_launch_and_occupancy/
  03__sm_gpc_topology/
  04__memory_hierarchy_latency/
  05__memory_hierarchy_bandwidth/
  06__tma_latency_and_issue/
  07__synchronization_primitives/
  08__l2_partitions_and_first_touch/
  09__dsmem_latency_vs_cluster_size/
  10__dsmem_latency_heatmap/
  11__dsmem_bandwidth_vs_cluster_size/
  12__dsmem_atomics_and_async_stores/
  13__l2_slice_mapping/
  14__gmem_latency_vs_controller_distance/
  15__noc_latency_heatmaps/
  16__async_proxy_paths_bandwidth/
  17__dsmem_collectives/
  18__access_type_summary/     (roll-up notebook only, no kernels)
```
Every `experiments/<NN__name>/` contains: `Makefile` (sets `EXP`, `BINS`, includes `../harness/Makefile.inc`, defines `run`), `src/*.cu` + runner scripts, `results/*.csv`, `results/sass/*.sass`, and `<NN__name>.ipynb` (executed, figures embedded). SASS dumps and the executed notebook are committed. Raw data (`results/*.csv`, logs, ncu/nsys exports), binaries and `.ncu-rep`/`.nsys-rep` files are not: raw data stays in the main checkout on this machine (rules in `docs/context/09` §6).

## Dependency order (what must exist before an experiment can run)
```
01 timers/harness ─┬─► 02 cluster launch ─► 03 SM→GPC map ─┬─► 09 DSMEM latency ─► 10 DSMEM heatmap
                   │                                       ├─► 11 DSMEM bandwidth ─► 17 collectives
                   │                                       ├─► 12 DSMEM atomics/async stores
                   ├─► 04 latency ladder ─► 08 L2 partitions ─► 13 slice map ─► 15 NoC heatmaps
                   │                                       └─► 14 DRAM/controller distance
                   ├─► 05 bandwidth ladder ─► 16 async-proxy bandwidth
                   ├─► 06 TMA latency/issue ─► 16
                   └─► 07 synchronization primitives ─► 09, 12, 16, 17
05 bandwidth (L2 home labelling, partition-link mode) ─► 12 global atomics near/far
04, 05, 06, 09, 11, 12, 16 (+07) ─► 18 summary
```
Groups that are independent of each other once 01–03 exist: {04, 05, 06, 07}, {09, 10, 11, 12}, {08, 13, 14, 15}, {16, 17}; 18 last. This is dependency information only: experiments are run one at a time, in numerical order (see "Experiment status").

## User-requested deliverables and where they are produced
| Request (project brief) | Experiment(s) |
|---|---|
| Avg latency (cycles) and bandwidth (TB/s GPU-wide) for all memory-access types | 04, 05, 06, 09, 11, 12, 16 → table/figures in 18 |
| Latency and throughput of **atomics on every path**: local SMEM, own SMEM via the DSMEM window, remote DSMEM, global at L2 near / far partition, HBM, host memory; contention and hot spots | 12 (+ bulk reductions in 16) → atomic ladders in 18 |
| SM-to-SM latency histogram vs cluster size with GMEM latency dotted | 09 (fig. 1), repeated in 18 |
| SM-to-SM bandwidth histogram vs cluster size with GMEM bandwidth dotted | 11 (fig. 1), repeated in 18 |
| Point-to-point latency heatmaps across the whole hierarchy; physical-distance sensitivity | 10 (SM×SM DSMEM), 15 (SM×slice, SM×partition, inferred interconnect), 14 (SM×controller) |
| Which physical SM issues / receives a DSMEM access | `%smid` recorded per rank everywhere; 03 maps smid→GPC; 10 bins by (smid_src, smid_dst) |
| L2 NUCA / slices / banks / hashing functions; other SRAM arrays | 08 (partitions, first-touch), 13 (slice map + hash), 05/12 (SMEM bank behaviour via conflict counters) |
| GMEM latency vs distance to the memory controller | 14 |
| How things are interconnected (hierarchical crossbars?) | 10 §analysis, 15 §analysis (hierarchical model), 11 (fabric arbitration), 16 (multicast fan-out) |
| Synchronization primitives: what/when/cost | docs/context 05 (templates) + experiment 07 (costs) |
| Hardware collectives only; tailored algorithms per topology | docs/context 02 §3 + experiment 17 |
| Check our numbers against published ones | agents only, during result analysis: docs/context 08 + each spec's "Sanity checks"; never in the notebooks |

## Rules for the implementing agents (non-negotiable)
1. Run every experiment under the configuration in `docs/context/10__device_config.md` (deviations only in a spec section "Device-config deviations"). Follow `docs/context/09__measurement_conventions.md` for timers, clock locking, placement control, cache control, repetitions, CSV schema and notebook layout. Mandatory CSV columns come first in every file; every raw per-sample row is kept (only data that would reach GiB scale is stored as exact reductions, `09` §6).
2. Use the exact PTX forms in `docs/context/02–05` (all validated with ptxas 12.9 for sm_90a). Compile with `-arch=sm_90a -O3 -lineinfo` (the shared `experiments/harness/Makefile.inc` does this). Dump SASS of every binary to `results/sass/` and check that the timed loop contains the intended instructions. Explain each kernel in a docstring at the top of its `.cu` file.
3. Record `%smid` (and cluster ids/ranks) for every block that produces a number; assert 1 CTA/SM where the spec says so.
4. Use the Nsight Compute metric names in `docs/context/07__profiling_tools.md` (verified for gh100: DSMEM traffic is `*_mem_dshared*`, TMA is `*_op_tma_*`, L2 fabric is `lts__t_sectors_srcunit_ltcfabric`, L2→L1 fills are `l1tex__m_xbar2l1tex_read_bytes_mem_lg_op_ld`); run ncu with `--clock-control none --cache-control <as specified> --csv --page raw --print-units base --print-metric-name name`; verify unconfirmed items listed in `docs/context/07` §8 on the device first (`ncu --query-metrics`).
5. Every notebook follows `docs/context/09` §8: goal → background → experimental setup → results/figures → concise insights. It is self-contained: no references to our markdown docs, agents or workflow; no other works or their numbers; no pass/fail. Figures use matplotlib; cycles on the left axis, ns on a secondary axis where helpful; GPU-wide references drawn as dotted lines.
6. Report what was measured honestly, including anomalies and partial runs; keep every CSV on this machine (never deleted, never committed). Sanity checks against expectations guide the investigation, never the data.
7. Environment, GPU selection and safety rules: `docs/environment/README.md` (run everything through `tools/run`; GPU 2 default, GPU 3 explicit only). The device here is an H100 NVL (detect and record), CUDA ≥ 12.4 (PTX ≥ 8.4; `.shared::cta` bulk destinations need PTX 8.6), Nsight Compute ≥ 2024, root for `nvidia-smi -lgc` (if unavailable, record clocks continuously and filter).

## Known unknowns the experiments must settle (summarised from the context files)
- Exact GPC composition and smid→GPC map of this card (03); whether TPC/CPC structure is visible in DSMEM latency (10).
- Whether the per-SM SM-to-SM injection cap (≈ 21 B/cycle) is a hard shaper and how it is shared under contention (11); true multicast gain and where fan-out happens (16).
- First-touch homing / replication in L2 and the effective capacity (08); partition hash bits and the slice hash or lookup table (13); any memory-controller distance term (14).
- Costs of DSMEM atomics, `st.async`, `red.async`, bulk reduce-puts — unpublished (12, 16).
- Whether global atomics execute in the line's home L2 partition (always far for half the lines, sharing the partition link like stores) or near the issuer like cached reads; atomic-unit parallelism and same-address merging in the L2 (12).
- Which completion/signalling mechanism is cheapest for inter-CTA hand-off (07, 09) and the resulting best collective algorithms per cluster size (17).
- Clock-domain ratio L2/SM, `%globaltimer` granularity, polling quantisation (01, 06, 07).
