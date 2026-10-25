# NVIDIA tooling for the microbenchmarks: Nsight Compute, Nsight Systems, binary utilities, clocks

Metric names verified offline against Nsight Compute **2026.2.1** (`ncu --query-metrics --chip gh100`; every metric below marked ✔ exists for compute capability 9.0 with the quoted NVIDIA description); **the device runs Nsight Compute 2025.2.1** (`10` §1), whose behaviour (no per-instance values, `--print-metric-name name` rejected with `--page raw`) is recorded in §8 and against `ptxas 12.9` + `nvdisasm 13.4` for SASS. Items marked ⚠ must be confirmed on the actual H100 (listed again in §8).

## 1. Nsight Compute CLI essentials

```bash
# Canonical counter run (clocks locked by us, caches left warm, warm-ups skipped)
ncu --clock-control none --cache-control none \
    -k regex:^bench_ --launch-skip 2 --launch-count 5 \
    --metrics <comma-separated fully-suffixed metrics> \
    --csv --page raw --print-units base \
    --log-file ncu.log ./bench ARGS > results/<exp>__ncu_<variant>.csv
# (2025.2.1 rejects --print-metric-name name with --page raw; --print-metric-instances adds nothing here, see §8)

# Full report (for the GUI) and later CSV extraction
ncu --set full -o rep ./bench ARGS
ncu -i rep.ncu-rep --csv --page raw --print-units base > rep.csv
```

| Flag | Meaning / rule for this project |
|---|---|
| `--metrics a.sum,b.avg.per_second` | always pass fully-suffixed names (deterministic CSV columns); `regex:`, `group:<name>`, `breakdown:<throughput>` expand lists |
| `--section X` / `--set basic|detailed|full` | sections: `MemoryWorkloadAnalysis{,_Chart,_Tables}`, `InstructionStats`, `SourceCounters`, `WarpStateStats`, `LaunchStats`, `Occupancy`, `SpeedOfLight`; `--list-sections`, `--list-metrics` |
| `--csv --page raw --print-units base --print-metric-name name` | one row per profiled launch, one column per metric, base units, raw metric names as headers |
| `--print-metric-instances values\|details` | documented to print per-instance values (per SM, per L2 slice, per memory controller, per PC/opcode). **On this setup (ncu 2025.2.1) hardware-unit counters have 0 instances: only `.sum/.avg/.min/.max` exist** (§8, AGENTS.md OI-1). Per-unit information is obtained with the rollups instead: `.max == .sum` proves that a single unit served everything; `.max` with known, distinct repetition counts identifies which unit (the busiest-unit oracle of `13`); `.max/.avg` is the hotspot ratio; per-SM values come from in-kernel accounting or one-SM-active launches. `details` still works for source-level metrics (`sass__*`). There is no `.per_instance` suffix. |
| `--clock-control none|base|boost|force-boost|reset` | **Default is `boost` since 2026.1 (was `base` ≤ 2025.4).** `base` locks GPC+memory clocks to rated base (≈1590 MHz on SXM5 — 20 % below boost); `none` leaves our `nvidia-smi -lgc` lock in place. Project rule: lock with `nvidia-smi -lgc` and run ncu with `--clock-control none`; always record `sm__cycles_elapsed.avg.per_second`. Thermal/power throttling overrides every setting. |
| `--cache-control all|none` | `all` (default) flushes all GPU caches before every replay pass → use for cold-cache / DRAM experiments; `none` keeps caches warm → use for L2-hit and warm-up-dependent experiments (then keep metric sets single-pass: check `profiler__replayer_passes`). |
| `--replay-mode kernel|application|range|app-range` | kernel replay saves/restores device memory between passes (fine for our kernels). `range` with `cudaProfilerStart/Stop` (prefer `cuProfilerStart/Stop`) profiles a whole range of launches without serialisation — use for concurrent-kernel contention experiments (the Jin et al. contention oracle of `13` Step 6 and `15` K4 runs the two kernels on two streams; profile the pair as one range or run it without ncu). |
| `-k regex:^name`, `--kernel-id ::name:N`, `-s N`, `-c N`, `--kernel-name-base function` | kernel filtering; `-s` counts only matching launches |
| `--nvtx --nvtx-include "range/"` | NVTX filtering (push/pop ranges) |
| `--warp-sampling-interval 0..31` (2^(5+v) cycles), `--section WarpStateStats,SourceCounters` | stall-reason sampling |
| `--query-metrics [--chip gh100] [--query-metrics-mode base|suffix|all] [--query-metrics-collection profiling|launch|warpsampling|source|device|stats|pmsampling]` | metric discovery; `launch__*` (launch collection) and `smsp__pcsamp_*` (warpsampling) do not appear in the default list |
| `--pipeline-boost-state stable` | default; keep |

Metric grammar: `unit__(subunit)_(pipestage)_(interface)_quantity_(qualifiers).(rollup).(submetric)` with rollups `sum|avg|min|max` across unit instances and sub-metrics `.per_second`, `.per_cycle_active`, `.per_cycle_elapsed`, `.pct_of_peak_sustained_{active,elapsed}`, `.peak_sustained`; `.pct`, `.ratio`. Pipe stages: `l1tex__t` tag, `l1tex__m` miss, `l1tex__d` data; `lts__t` tag (what clients requested, hits + misses), `lts__d` data banks/fills. Interfaces: `xbar2l1tex` (into the SM), `l1tex2xbar` (out of the SM), `lts2xbar`, `xbar2lts`, `ltcfabric2lts`. Units: `sm`, `smsp`, `tpc`, `gpc`, `l1tex` (one per SM), `lts` (L2 slice; count = `device__attribute_l2s_count` = 96 here, `.avg` divides by 94 ⚠ `08`), `ltcfabric` ("communication fabric for the L2 cache partitions"), `lrc` (L2 request coalescer — serves multicast requests and compression), `fbpa` (memory controller), `dram`, `gpu`. `sass_` in a name = SW-patched instruction counters (extra pass). "sector" = 32 B, "wavefront" = one data-stage cycle's worth of work, "request" = one instruction's command.

## 2. Metric map per memory path (all ✔ on gh100)

### Shared memory (local)
`l1tex__data_pipe_lsu_wavefronts_mem_shared_op_{ld,st}.sum` (cycles of data-stage work), `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st}.sum`, `smsp__inst_executed_op_shared_{ld,st}.sum`, `sm__sass_data_bytes_mem_shared_op_{ld,st}.sum`, `l1tex__throughput.avg.pct_of_peak_sustained_elapsed`, `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` (smem latency stalls). Table: `group:memory__shared_table`.

### Distributed shared memory — qualifier is `dshared`
- Instruction counts (SASS-patched): `sm__sass_inst_executed_op_dshared.sum` and `_op_dshared_{ld,st,atom,redas,stas,tma_st,tma_red}.sum` (generic `LD.E/ST.E/ATOM.E` to a peer's smem; `REDAS` = `red.async`; `STAS` = `st.async`; TMA/bulk copies whose destination is `shared::cluster`).
- Request/sector/byte counts at the SM: `l1tex__t_{requests,sectors,bytes}_pipe_lsu_mem_dshared(_op_{ld,st,atom,redas}).sum`, `l1tex__average_t_sectors_per_request_pipe_lsu_mem_dshared.ratio` (coalescing), `l1tex__t_set_conflicts_pipe_lsu_mem_dshared.sum`.
- **SM-to-SM link traffic**: outbound `l1tex__m_l1tex2xbar_write_bytes_mem_dshared(_op_{ld,st,atom,redas,tma_st,tma_red}).sum`, inbound `l1tex__m_xbar2l1tex_read_bytes_mem_dshared(_op_…).sum`; utilisation `l1tex__m_{l1tex2xbar,xbar2l1tex}_throughput_pipe_lsu_mem_dshared.avg.pct_of_peak_sustained_elapsed` (the rated peak of these links — read `.peak_sustained` — is the hardware's own statement of the per-SM SM-to-SM bandwidth ⚠); `l1tex__m_l1tex2xbar_req_cycles_active_mem_dshared{,_loopback}.sum` (loopback = own-SM accesses through the DSMEM path). ⚠ which side carries payload bytes for *loads* (`_op_ld` on the write side counts request packets) must be checked with a pure-load kernel.
- Cluster launch facts: `launch__cluster_size`, `launch__cluster_dim_{x,y,z}`, `launch__cluster_max_active`, `launch__cluster_max_potential_size`, `launch__cluster_scheduling_policy`, `launch__occupancy_cluster_pct`, `gpc__cgas_launched.sum`, `gpc__cgas_active.avg`.
- DSMEM traffic does **not** appear in `lts__*` (it never reaches L2) — use that as a check that an experiment is not accidentally going through global memory.

### L1 / global through L1
`l1tex__t_sectors_pipe_lsu_mem_global_op_{ld,st}.sum`, `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`, `l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_{hit,miss}.sum`, `l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct`, L2→L1 fill bytes `l1tex__m_xbar2l1tex_read_bytes_mem_lg_op_ld.sum` (**there is no `_mem_global_op_ld` variant on gh100**), SM→L2 `l1tex__m_l1tex2xbar_write_bytes_mem_lg_op_st.sum`, `smsp__inst_executed_op_global_{ld,st,red}.sum`, `sm__sass_inst_executed_op_ldgsts{,_cache_access,_cache_bypass}.sum` (cp.async), stall `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio`.

### L2 (per slice = `lts` instance)
`lts__t_sectors.sum`, `lts__t_sectors_op_{read,write,atom,red}.sum`, `lts__t_sector_hit_rate.pct`, `lts__t_sectors_srcunit_tex_op_read.sum` (requests from SMs), `lts__t_sectors_srcunit_tex_op_read_lookup_{hit,miss}.sum`, `lts__t_requests_srcunit_tex.sum`, **`lts__t_sectors_srcunit_ltcfabric.sum`** (sectors that crossed the fabric between the two partitions; variants `_op_read/_op_write`, `_lookup_hit/_miss`, `_aperture_device`), `lts__ltcfabric2lts_cycles_{active,stalled}.sum`, `lts__t_sectors_aperture_{device,sysmem,peer}.sum`, `lts__d_sectors_fill_device.sum` (true DRAM fills), `lts__t_bytes.sum.per_second`, `lts__throughput.avg.pct_of_peak_sustained_elapsed`, `lts__t_sectors_evict_{first,last,normal}.sum`, `lts__cycles_elapsed.avg.per_second` (L2 clock ⚠), `lts__t_sectors.max`/`.avg` (slice imbalance without instances), `device__attribute_l2s_count`, `device__attribute_num_l2s_per_fbp`, `device__attribute_fbp_count`. Table: `group:memory__l2_cache_table`. Per-slice address mapping: no instances here → the `.max` oracle on `lts__t_sectors_srcunit_tex_op_read` (`13` Steps 1–2; `docs/context/06` §2.2).

### DRAM / HBM
`dram__bytes_{read,write}.sum(.per_second)`, `dram__throughput.avg.pct_of_peak_sustained_elapsed`, `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`, `dram__cycles_elapsed.avg.per_second` (memory clock), `fbpa__dram_{read,write}_{bytes,sectors}.{sum,max,avg,min}` (per-controller imbalance via `.max/.avg`; controller identity via the `.max` oracle, `13` Step 4; instances not exported here), `lts__d_sectors_fill_device.sum`.

### TMA / bulk async (qualifier `tma`; bulk copies are counted as TMA)
`sm__sass_inst_executed_op_tma_ld.sum` ("UTMALDG, UBLKCP.S.G"), `_op_tma_st.sum` ("UTMASTG, UBLKCP.G.S"), `_op_tma_red.sum`, `smsp__inst_executed_op_tma_{ld,st}.sum` (hardware counter, no patching), `sm__pipe_tma_cycles_active.avg`, `l1tex__tmain_requests.sum` (instructions sent to the TMA unit), **`l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum`** (TMA load bytes from L2; `.avg.per_cycle_active` → B/cycle/SM, expected ≤ 32 ⚠; `.avg.peak_sustained` reveals the rated per-SM TMA read peak), `_tma_ld_dest_self.sum` / `_tma_ld_dest_multicast.sum` (unicast vs multicast), `l1tex__m_l1tex2xbar_read_requests_mem_global_op_tma_ld_dest_multicast{,_size_32B/64B/96B/128B,_replayed}.sum`, `l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_{st,red}.sum`, and the DSMEM-destination variants `l1tex__m_{l1tex2xbar_write,xbar2l1tex_read}_bytes_mem_dshared_op_tma_{st,red}.sum`. No metric is named `utma*` or `bulk*`.

### Timing, clocks, launch provenance
`gpu__time_duration.sum` (ns), `sm__cycles_elapsed.avg` / `.max`, `sm__cycles_active.avg`, `smsp__cycles_active.avg`, `gpc__cycles_elapsed.max`, `sm__cycles_elapsed.avg.per_second` (**actual SM clock during the run — mandatory CSV column**), `lts__cycles_elapsed.avg.per_second`, `dram__cycles_elapsed.avg.per_second`, `launch__{grid_size,block_size,sm_count,waves_per_multiprocessor,shared_mem_per_block_dynamic,registers_per_thread,occupancy_limit_*}`, `profiler__replayer_passes`.

### Stalls and instruction mix
Hardware-averaged: `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio` and `smsp__warp_issue_stalled_<reason>_per_warp_active.pct` with reasons `barrier, membar, long_scoreboard, short_scoreboard, sleeping, wait, mio_throttle, lg_throttle, math_pipe_throttle, imc_miss, no_instruction, not_selected, selected, drain, dispatch_stall, branch_resolving, misc` (`--section WarpStateStats`). Sampled per PC: `smsp__pcsamp_warps_issue_stalled_<reason>{,_not_issued}` (`--section SourceCounters`). SASS opcode histogram: `sass__inst_executed_per_opcode` with `--print-metric-instances details` (`--section InstructionStats`). ⚠ which reason absorbs `UCGABAR_WAIT` and `SYNCS.PHASECHK` spins (expect `barrier`/`misc`/`branch_resolving`).

## 3. PTX → SASS cheat-sheet (sm_90a, verified by assembling with ptxas and disassembling)

| PTX | SASS |
|---|---|
| `barrier.cluster.arrive{.release}.aligned` / `barrier.cluster.wait{.acquire}.aligned` | `UCGABAR_ARV` / `UCGABAR_WAIT` (release/acquire add `MEMBAR.ALL.GPU; ERRBAR; CGAERRBAR` before arrive and `CCTL.IVALL` after wait; `.relaxed` arrive omits them) |
| `mbarrier.init` / `.arrive.expect_tx` / `.try_wait.parity` / `.test_wait` | `SYNCS.EXCH.64` / `SYNCS.ARRIVE.TRANS64` / `SYNCS.PHASECHK.TRANS64.TRYWAIT` / `SYNCS.PHASECHK.TRANS64` (try_wait with a suspend hint adds `NANOSLEEP.SYNCS`) |
| `fence.mbarrier_init.release.cluster` | `MEMBAR.ALL.CTA` (observed) |
| `fence.proxy.async.shared::cta` / `.global` | `FENCE.VIEW.ASYNC.S` / `FENCE.VIEW.ASYNC.G` |
| `fence.acq_rel.cluster`, `membar.gl` | `MEMBAR.ALL.GPU; ERRBAR; CGAERRBAR; CCTL.IVALL` |
| `cp.async.bulk.tensor.Nd.shared::cluster.global...` / `...global.shared::cta.bulk_group` | `UTMALDG.ND` (preceded by `ELECT`) / `UTMASTG.ND`; reduce-store `UTMAREDG`; prefetch `UTMAPF.L2.ND`; `prefetch.tensormap` → `UTMACCTL.PF` |
| `cp.async.bulk.commit_group` / `wait_group{.read} N` | `UTMACMDFLUSH` / `DEPBAR.LE SB0, N` (+ `CCTL.IVALL` for the non-`.read` wait) |
| `cp.async.bulk` G→S / S→G / S→S(cluster) | `UBLKCP.S.G` / `UBLKCP.G.S` / `UBLKCP.S.S`; `cp.reduce.async.bulk` S→G → `UBLKRED.G.S.<OP>`; prefetch → `UBLKPF.L2` |
| `ld/st/atom.shared::cluster` (or generic pointer from `mapa`) | generic `LD.E` / `ST.E` / `ATOM.E.<op>.STRONG.GPU` with a cluster-window address (vs `LDS`/`STS`/`ATOMS` for `.shared::cta`) |
| `mapa.shared::cluster.u32` | folded into uniform-datapath arithmetic (`ULEA`/`UPRMT` on `SR_CgaCtaId`/`SR_SWINHI`); no `MAPA` opcode observed ⚠ |
| `st.async` / `red.async` | `STAS` / `REDAS.<OP>` |
| `cp.async.{cg,ca}` / `commit_group` / `wait_group` | `LDGSTS.E{.BYPASS.128}` / `LDGDEPBAR` / `DEPBAR.LE SB0, N` |
| `%clock64` / `%clock` / `%globaltimer` | `CS2R R, SR_CLOCKLO` / `CS2R.32`/`S2UR SR_CLOCKLO` / `CS2R R, SR_GLOBALTIMERLO` |
| `%cluster_ctarank`, `%smid` | `S2UR/S2R SR_CgaCtaId`, `S2R SR_VIRTUALSMID` (verified in 01's SASS) |
| `ld.global.cg` / `.ca` / `.cv` / `ld.volatile` | `LDG.E.STRONG.GPU` / `LDG.E` / `LDG.E.STRONG.SYS` / `LDG.E.STRONG.SYS` |
| `discard.global.L2` / `prefetch.global.L2{::evict_last}` / `.L1` | `CCTL.E.RML2` / `CCTL.E.PF2` (`PML2`) / `CCTL.E.PF1` |
| `bar.sync 0` / `bar.arrive 1,64` / `elect.sync` / `nanosleep` | `BAR.SYNC.DEFER_BLOCKING 0x0` / `BAR.ARV 0x1,0x40` / `ELECT` / `NANOSLEEP` |

Every benchmark binary's SASS must be dumped (`cuobjdump -sass -fun <kernel> bench > results/sass/<kernel>.sass`) and the timed loop checked (the notebook shows an excerpt only where it explains a key result), so that one can see e.g. `LD.E` (DSMEM) vs `LDS` (local), the dependent register chain, and `CS2R SR_CLOCKLO` placement.

## 4. Compilation flags
`nvcc -arch=sm_90a -O3 -lineinfo -Xptxas -v` (`sm_90a` is required for `.multicast::cluster` TMA forms, `wgmma`, `setmaxnreg`; everything else assembles for `sm_90`). Add `-Xptxas -warn-spills`. Keep PTX with `--keep` for the notebooks. Hand-written PTX can be validated without nvcc using `ptxas -arch=sm_90a -O3 file.ptx -o file.cubin` then `nvdisasm -c file.cubin`.

## 5. Nsight Systems (launch gaps, concurrency, timelines — no counters)
`nsys profile -t cuda,nvtx --stats=true -o run ./bench` (optionally `--capture-range=cudaProfilerApi --capture-range-end=stop`); `nsys stats --report cuda_gpu_kern_sum --format csv --output . run.nsys-rep` (also `cuda_api_sum`, `nvtx_sum`; `nsys stats --help-reports` for exact ids ⚠). Use nsys, not ncu, for `02__cluster_launch_and_occupancy` wall-clock gaps, because ncu serialises, flushes and replays.

## 6. Clocks
- Read: `nvidia-smi -q -d CLOCK`; `nvidia-smi --query-gpu=clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,clocks_event_reasons.active --format=csv -l 1` (log during long runs).
- Lock (root): `nvidia-smi -lgc <MHz>,<MHz>` / reset `-rgc`; memory `-lmc` may be refused on Hopper → `--lock-memory-clocks-deferred` / `-lmcd` (HBM clock is effectively fixed) ⚠. `nvidia-smi -ac` is deprecated.
- In-kernel: `%clock64` per SM (not comparable across SMs), `%globaltimer` ns global (32 ns granularity, verified in 01).
- `cudaDeviceProp.clockRate` is the max clock, not the running one. Use `sm__cycles_elapsed.avg.per_second` (ncu) or NVML `nvmlDeviceGetClockInfo(NVML_CLOCK_SM)` for the running clock.
- Third-party spec values (verify on device): SXM5 base 1590 / boost 1980 MHz, PCIe base 1095 / boost 1755 MHz; `nvidia-smi` reports HBM3 as 2619 MHz on SXM5.

## 7. Energy (optional)
NVML `nvmlDeviceGetTotalEnergyConsumption` (mJ, cumulative; integrate over ≥ 100 ms of back-to-back launches) and `nvmlDeviceGetFieldValues(NVML_FI_DEV_POWER_INSTANT)`.

## 8. Must be verified on the H100 before results are trusted
1. Actual locked clock values under `-lgc` and under ncu `--clock-control base|boost`; whether `-lmc` is accepted.
2. `lts__cycles_elapsed.avg.per_second` vs `sm__cycles_elapsed.avg.per_second` (separate L2 clock domain?).
3. `device__attribute_l2s_count` (80?) and slice-id ↔ partition grouping from `--print-metric-instances details`. (Settled by `08`: 96 reported, 94 active, no instances; `13` Step 0 re-checks the export once under kernel vs application replay, `--set full` + `ncu_report`, unsuffixed metric names, and for `fbpa__`/`l1tex__`/`sm__`; the result is appended below.)
4. Exact metrics behind the Hopper memory chart's DSMEM/TMA links: `ncu --section MemoryWorkloadAnalysis_Chart --list-metrics ./bench`.
5. Which side (`l1tex2xbar_write` vs `xbar2l1tex_read`) carries payload for DSMEM loads; `loopback` semantics.
6. Stall reason that absorbs `UCGABAR_WAIT` / `SYNCS.PHASECHK` spins.
7. `l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.avg.peak_sustained` (rated TMA read peak per SM) and `l1tex__m_l1tex2xbar_throughput_pipe_lsu_mem_dshared.avg.peak_sustained` (rated SM-to-SM link peak).
8. `%globaltimer` granularity; whether `MAPA` is ever emitted with a dynamic rank; `nsys` report ids; NVML energy update period.

Settled on this H100 NVL by `01__timers_and_harness` (2026-10-23):
- Item 1, clocks: under `-lgc 1635,1635`, NVML reports 1635 MHz, but `sm__cycles_elapsed.avg.per_second` reads 1627–1635 MHz. With ncu `--clock-control base`: SM 1075 MHz, L2 1215 MHz. Our lock is still in place after such a profile.
- Item 2, L2 clock: a separate domain. L2 runs at ≈1556 MHz under our lock (L2/SM 0.951–0.956) and at 1215 MHz under ncu base (L2/SM 1.13). DRAM stays at 2619 MHz.
- Item 8, `%globaltimer`: increments in exact 32 ns steps. Timer offsets between SMs are ≤ 1 tick.
- ncu 2025.2.1 rejects `--print-metric-name name` together with `--page raw` ("only supported for the details page"). The raw page's column headers are already metric names, so omit that flag for raw exports.
- With `--print-metric-instances values`, `gpc__cycles_elapsed.sum` / `sm__cycles_elapsed.sum` are exported as totals (one value), not per instance.

Settled by `08__l2_partitions_and_first_touch` (2026-10-24):
- Item 3:
  - `device__attribute_l2s_count` = 96 (`fbp_count` 12 × `num_l2s_per_fbp` 8), but every `lts__*.avg` = `.sum / 94` (likely 2 slices disabled).
  - **No per-slice values exist**: the report holds 0 instances for `lts__*` metrics, so `--print-metric-instances` cannot give the slice-id ↔ partition grouping.
  - **See AGENTS.md OI-1 (decided):** slice/FBPA identity comes from the busiest-unit `.max` oracle (`13`), cross-checked by Jin et al.'s timing-contention oracle and by `08`'s timing classes; per-SM values from in-kernel accounting; `08` requested only rollup metric names under application replay, so `13` Step 0 re-checks the instance export under other collection modes before the oracle runs.
- Under application replay with a long metric list, the first counter pass ran the profiled kernels 2.6–2.8× slower (counts unaffected).
- Only one `lts__` counter fits in one replay pass (any two `lts__` metrics need 4 passes; `dram__` and `gpu__time_duration` fit alongside). Single-pass profiles run the kernel at its unprofiled speed. The profiler's own traffic depends on the grid and grows with the kernel's work, and the state it leaves disturbs later kernels' cache contents: baselines must be all-exit launches of the same grid in the same session, and cache-state-sensitive counters need one profiled kernel per binary run (`--launch-skip k --launch-count 1`; `08` `ncu_fabric1`, `05` `ncufp`).
- `lts__t_sector_hit_rate.pct` can exceed 100 %; use `lts__t_sectors_srcunit_tex_op_read_lookup_{hit,miss}`.
