# H100 microbenchmarking for 3D-stacked GPGPU research — index for the planning and coding agents

## Mission
Characterise, on a real H100, the micro-architectural quantities that a 3D-stacking study of the Hopper GPC/DSMEM/L2 hierarchy needs: latency and bandwidth of **every** hardware memory-access path (generic proxy and async proxy/TMA), the cost of **every** synchronisation primitive, the **SM-to-SM network** behaviour versus cluster size and topology, point-to-point latency **heatmaps** across the whole hierarchy (SM×SM, SM×L2-slice, SM×partition, SM×memory-controller) from which physical distance and interconnect structure are inferred, and the performance of **hardware-supported intra-cluster collectives**. Everything is written in CUDA C++ / inline PTX, writes CSV, and is explained in one `.ipynb` per experiment. Prior-work numbers are reproduced and cited where they exist.

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
  experiments/                 ← one self-contained experiment SPEC per file (kernel design, sweeps, CSV columns, figures, pass criteria, ncu metrics)
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
Every `experiments/<NN__name>/` contains: `Makefile` (sets `EXP`, `BINS`, includes `../harness/Makefile.inc`, defines `run`), `src/*.cu` + runner scripts, `results/*.csv`, `results/sass/*.sass`, and `<NN__name>.ipynb` (executed, figures embedded). CSV results and SASS dumps are committed; binaries and `.ncu-rep` files are not.

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
04, 05, 06, 09, 11, 12, 16 (+07) ─► 18 summary
```
Independent groups that can be implemented in parallel once 01–03 exist: {04, 05, 06, 07}, {09, 10, 11, 12}, {08, 13, 14, 15}, {16, 17}; 18 last.

## User-requested deliverables and where they are produced
| Request (project brief) | Experiment(s) |
|---|---|
| Avg latency (cycles) and bandwidth (TB/s GPU-wide) for all memory-access types | 04, 05, 06, 09, 11, 12, 16 → table/figures in 18 |
| SM-to-SM latency histogram vs cluster size with GMEM latency dotted | 09 (fig. 1), repeated in 18 |
| SM-to-SM bandwidth histogram vs cluster size with GMEM bandwidth dotted | 11 (fig. 1), repeated in 18 |
| Point-to-point latency heatmaps across the whole hierarchy; physical-distance sensitivity | 10 (SM×SM DSMEM), 15 (SM×slice, SM×partition, inferred interconnect), 14 (SM×controller) |
| Which physical SM issues / receives a DSMEM access | `%smid` recorded per rank everywhere; 03 maps smid→GPC; 10 bins by (smid_src, smid_dst) |
| L2 NUCA / slices / banks / hashing functions; other SRAM arrays | 08 (partitions, first-touch), 13 (slice map + hash), 05/12 (SMEM bank behaviour via conflict counters) |
| GMEM latency vs distance to the memory controller | 14 |
| How things are interconnected (hierarchical crossbars?) | 10 §analysis, 15 §analysis (hierarchical model), 11 (fabric arbitration), 16 (multicast fan-out) |
| Synchronization primitives: what/when/cost | docs/context 05 (templates) + experiment 07 (costs) |
| Hardware collectives only; tailored algorithms per topology | docs/context 02 §3 + experiment 17 |
| Reproduce paper numbers and point it out | docs/context 08 tolerances; every notebook §5 "Reproduction of prior work"; 18 flags |

## Rules for the implementing agents (non-negotiable)
1. Follow `docs/context/09__measurement_conventions.md` for timers, clock locking, placement control, cache control, repetitions, CSV schema and notebook layout. Mandatory CSV columns come first in every file; raw per-sample rows are kept.
2. Use the exact PTX forms in `docs/context/02–05` (all validated with ptxas 12.9 for sm_90a). Compile with `-arch=sm_90a -O3 -lineinfo` (the shared `experiments/harness/Makefile.inc` does this). Dump SASS of every binary to `results/sass/` and show the timed loop in the notebook.
3. Record `%smid` (and cluster ids/ranks) for every block that produces a number; assert 1 CTA/SM where the spec says so.
4. Use the Nsight Compute metric names in `docs/context/07__profiling_tools.md` (verified for gh100: DSMEM traffic is `*_mem_dshared*`, TMA is `*_op_tma_*`, L2 fabric is `lts__t_sectors_srcunit_ltcfabric`, L2→L1 fills are `l1tex__m_xbar2l1tex_read_bytes_mem_lg_op_ld`); run ncu with `--clock-control none --cache-control <as specified> --csv --page raw --print-units base --print-metric-name name`; verify unconfirmed items listed in `docs/context/07` §8 on the device first (`ncu --query-metrics`).
5. Every notebook: goal, kernel + SASS excerpt, environment table, figures listed in the spec, reproduction table against `docs/context/08`, short interpretation. Figures use matplotlib; cycles on the left axis, ns on a secondary axis where helpful; GPU-wide references drawn as dotted lines.
6. Report failures honestly: if a pass criterion is not met, the notebook says so and the CSV is still kept.
7. Environment, GPU selection and safety rules: `docs/environment/README.md` (run everything through `tools/run`; GPU 2 default, GPU 3 explicit only). The device here is an H100 NVL (detect and record), CUDA ≥ 12.4 (PTX ≥ 8.4; `.shared::cta` bulk destinations need PTX 8.6), Nsight Compute ≥ 2024, root for `nvidia-smi -lgc` (if unavailable, record clocks continuously and filter).

## Known unknowns the experiments must settle (summarised from the context files)
- Exact GPC composition and smid→GPC map of this card (03); whether TPC/CPC structure is visible in DSMEM latency (10).
- Whether the per-SM SM-to-SM injection cap (≈ 21 B/cycle) is a hard shaper and how it is shared under contention (11); true multicast gain and where fan-out happens (16).
- First-touch homing / replication in L2 and the effective capacity (08); partition hash bits and the slice hash or lookup table (13); any memory-controller distance term (14).
- Costs of DSMEM atomics, `st.async`, `red.async`, bulk reduce-puts — unpublished (12, 16).
- Which completion/signalling mechanism is cheapest for inter-CTA hand-off (07, 09) and the resulting best collective algorithms per cluster size (17).
- Clock-domain ratio L2/SM, `%globaltimer` granularity, polling quantisation (01, 06, 07).
