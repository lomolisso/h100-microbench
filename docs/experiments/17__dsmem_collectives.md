# 17 — Intra-cluster collectives on the hardware primitives: AllReduce, AllGather, ReduceScatter, AlltoAll (and Broadcast)

> **Device configuration:** runs under [`docs/context/10__device_config.md`](../context/10__device_config.md): H100 NVL (GPU 2), 132 SMs, 60 MiB L2, HBM3 3.94 TB/s, SM clock locked at 1635 MHz, power limit 400 W, ECC on, CUDA 12.9.1 `sm_90a`. Reference numbers quoted in this spec come from **other chips and configurations** (H800 PCIe, H100 PCIe, H100 SXM, H200, RTX 5090; see `10` §4). They are kept unchanged for the agent's sanity checks only, never cited in the notebook and not expectations for this device.

> **Method note (OI-1, decided):** Nsight Compute on this setup exports no per-instance counter values. Per-rank injection/ejection is accounted **in the kernel** (`bytes_injected` per CTA, exact for every algorithm) and hot spots are read from the `.max/.avg/.min` rollups of the DSMEM link metrics; no per-SM ncu instances are needed.

## Goal
Measure the latency (small messages) and bandwidth (large messages) of the four collectives inside one thread block cluster for cluster sizes 2–16, implemented only with hardware-supported data-movement primitives (multicast TMA, bulk puts, bulk reduce-puts, `st.async`/`red.async`, remote atomics) and the cheapest correct completion mechanism (per-destination mbarrier). Decompose each into data-movement time and synchronisation time, compare algorithms on the same fabric, and produce the per-primitive cost model the 3D-stacked-topology study will re-evaluate.

## Background
`docs/context/02__memory_access_taxonomy.md` §3 (hardware collectives), `docs/context/05__synchronization_primitives.md` §4 (protocols), `docs/context/01__clusters_and_dsmem.md`. Fabric facts to exploit: per-SM injection cap ≈ 21 B/cycle not work-conserving; bidirectional stores lose only 4–5 %; reads lose 23 %; broadcast reads collapse as 1/N; TMA multicast fan-out happens after one L2 read; DSMEM reduce ops are integer-only (`u32/s32/u64 add`, `min/max`, bitwise) — fp32 AllReduce must use integer-encoded or generic-proxy paths. Prior application-level data: 15-to-1 push congestion 308→608 µs per writer [P2]; CREDIT replicated all-gather protocol (P(P−1) remote stores, 2 barriers per stage) [P4]; ClusterFusion ClusterReduce/ClusterGather [lit].

## Setup
Cluster of CS ∈ {2, 4, 8, 16}, 1 CTA per SM, message size per CTA M ∈ {256 B, 1 KB, 4 KB, 16 KB, 64 KB, 128 KB} (AllReduce vector length M/4 u32 or f32), 1024 threads (also 256), 100 repetitions, `%clock64` per CTA from the first instruction of the collective to the completion wait returning, plus `%globaltimer` for the cluster-wide makespan; results verified against a CPU reference every repetition (error → row dropped and flagged). One cluster on an idle chip for per-cluster numbers; then `cudaOccupancyMaxActiveClusters` concurrent clusters for GPU-wide aggregate (TB/s of useful payload and collectives/s).

## Algorithms (each a separate kernel; the completion mechanism is always a destination-side mbarrier pre-armed with `expect_tx` of the exact byte total, unless stated)

### Broadcast (1 → N)
B1 TMA multicast from global (`cp.async.bulk{.tensor}...multicast::cluster`, source staged in L2); B2 root bulk-puts to each peer (N−1 `cp.async.bulk.shared::cluster.shared::cta`); B3 root `st.async` loop (1024 threads × v4); B4 peers pull with `ld.shared::cluster` after one `cluster.sync()` (read-based); B5 global round trip (root `st.global` + `__threadfence` + flag; peers `ld.global`) as the non-DSMEM baseline.

### AllGather (each CTA's M bytes → everyone)
AG1 direct: every CTA bulk-puts its block to the N−1 peers (N(N−1) puts, one mbarrier per destination expecting (N−1)·M); AG2 `st.async` version; AG3 ring (N−1 steps, each CTA forwards the block it just received to the next rank; per-step mbarrier phases); AG4 pull-based (`cluster.sync()` then each CTA reads N−1 blocks with `ld.shared::cluster.v4`); AG5 via global memory (each writes to a global buffer, `cluster.sync()`, each reads all with `ld.global`).

### ReduceScatter (each CTA ends with the reduced 1/N slice)
RS1 direct reduce-put: every CTA issues N−1 `cp.reduce.async.bulk ... shared::cluster.add.u32` of the slice owned by each peer into the peer's accumulation buffer (destination pre-initialised with its own slice; mbarrier expects (N−1)·M/N bytes); RS2 `red.async.add.u32` per element (1024 threads); RS3 generic `red.shared::cluster.add` + `cluster.sync()`; RS4 ring reduce-scatter with bulk puts and local adds (N−1 steps); RS5 fp32 version of RS1 is impossible in hardware → RS5 = puts of raw fp32 slices + local fp32 add after mbarrier (the "put-then-reduce-locally" pattern); RS6 via global `red.global.add.f32`/`cp.reduce.async.bulk.global.add.f32` + `cluster.sync()`.

### AllReduce (everyone ends with the full reduced vector)
AR1 = RS1 + AG1 (reduce-scatter then all-gather, 2 phases, 2 mbarrier phases); AR2 replicated: every CTA reduce-puts its *whole* vector into every peer (N(N−1) reduce-puts of M bytes, one phase; CREDIT-style but with hardware reduction); AR3 root-based: all reduce-put into rank 0, rank 0 multicasts result (B2 or, via a global bounce, B1); AR4 ring (RS4 + AG3); AR5 fp32 via RS5-style put-and-add + AG1; AR6 via global memory (`cp.reduce.async.bulk.global.add` + `cluster.sync()` + each reads back with TMA G→S, or `red.global` + `__threadfence`).

### AlltoAll (each CTA sends a distinct M/N block to every peer)
A2A1 direct bulk puts (N(N−1) puts of M/N); A2A2 `st.async`; A2A3 pull after `cluster.sync()`; A2A4 via global; A2A5 staggered schedule (CTA r sends to rank (r+k) mod N at step k to avoid the all-to-one hotspot) vs naive order — tests the fabric's arbitration.

## Measured quantities
Per (algorithm, CS, M): makespan cycles (median, p5, p95); data-movement estimate = bytes injected per SM / per-SM cap from `11`; sync overhead = makespan − max(data-movement, latency floor); useful bandwidth = N·M / makespan (B/cycle and TB/s when all clusters run); for small M the latency floor vs the primitive latencies from `07`/`09`.

## CSV columns
`17__collectives__runs.csv`: `collective, algorithm, cs, msg_bytes_per_cta, threads, cluster_id, rank, smid, rep, cycles_local, makespan_ns, bytes_injected, verified`.
`17__collectives__summary.csv`: `collective, algorithm, cs, msg_bytes_per_cta, median_cycles, p5, p95, useful_bpc_per_sm, tbps_gpu_wide, model_data_cycles, sync_overhead_cycles`.

## Figures
1. For each collective: makespan vs M (log-log), one line per algorithm, panels per CS; dotted line = via-global algorithm.
2. Latency floor (M = 256 B) per algorithm and CS vs the `07` primitive costs.
3. Useful bandwidth vs CS at M = 64 KB; dotted lines = per-SM cap × N and DRAM TB/s.
4. Stacked bars: data vs synchronisation share per algorithm (CS 8, M = 16 KB).
5. Staggered vs naive AlltoAll and multicast vs put Broadcast speed-ups.

## Sanity checks (agent-only)
Expected ranges for judging plausibility during result analysis. They are never presented as pass/fail and never quoted in the notebook. If a value falls outside, investigate (placement, clocks, SASS, cache state, ncu replay) and re-run. A value that survives is a finding about this device; describe it on its own terms.
Every algorithm verified bit-exact (integer) / within fp tolerance; AG1 at CS 8, M = 64 KB achieves ≥ 70 % of N × per-SM put cap; B1 multicast ≥ 2× B2 for CS ≥ 4; AR1 beats AR6 (global) for M ≥ 4 KB; the notebook ranks algorithms per (collective, CS, M) and states the crossover sizes.

## NVIDIA tools
ncu DSMEM/TMA metric set from `11` and `16` as rollups: `l1tex__m_l1tex2xbar_write_bytes_mem_dshared.{sum,max,avg,min}` and `l1tex__m_xbar2l1tex_read_bytes_mem_dshared.{sum,max,avg,min}` (the `.max/.avg` ratio over the active SMs is the hot-spot factor; `.sum` must equal the in-kernel `bytes_injected` total; single-rank-active launches when one rank's own counter is needed), `l1tex__m_l1tex2xbar_throughput_pipe_lsu_mem_dshared.avg.pct_of_peak_sustained_elapsed`, `lts__t_sectors_srcunit_tex.sum` (must be ≈ 0 for DSMEM-only algorithms), `gpc__cgas_active.avg`; `nsys` for the GPU-wide concurrent-cluster runs.
