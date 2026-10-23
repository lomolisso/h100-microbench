# 01 — Timers, loop overhead, clocks, and the shared harness

> **Device configuration:** runs under [`docs/context/10__device_config.md`](../context/10__device_config.md): H100 NVL (GPU 2), 132 SMs, 60 MiB L2, HBM3 3.94 TB/s, SM clock locked at 1635 MHz, power limit 400 W, ECC on, CUDA 12.9.1 `sm_90a`. Reference numbers quoted in this spec come from **other chips and configurations** (H800 PCIe, H100 PCIe, H100 SXM, H200, RTX 5090; see `10` §4). They are kept unchanged for the agent's sanity checks only, never cited in the notebook and not expectations for this device.

## Goal
Establish the measurement floor every later experiment subtracts or relies on: (1) cost and resolution of `%clock64`, `%globaltimer`, `cudaEvent`; (2) the empty-loop overhead of the dependent-chain pattern; (3) the ratio between SM clock and L2 clock as seen by Nsight Compute; (4) a reusable host/device harness (`harness/`) that all other experiments include.

## Background
`docs/context/09__measurement_conventions.md` (timers, clocks), `docs/context/07__profiling_tools.md` (ncu flags). Prior numbers: `%globaltimer` updates every ≈32 ns on H100 [P2]; `clock64()` read cost is a few cycles; FlashGPU-sim observed mbarrier polling quantisation of ≈147 cycles — unrelated to timers but shows why per-access timing must be validated.

## Deliverables
1. `experiments/harness/harness.cuh` — device helpers: `read_clock64()`, `read_globaltimer()`, `read_timers_pair()` (both in one `asm volatile` block), `read_smid()`, `read_cluster_regs()`, `mapa_u32()`, `smem_u32()`, `cluster_arrive/wait()`, mbarrier wrappers (init/arrive_expect_tx/try_wait_parity loop), TMA 1D bulk wrappers, `fence_proxy_async_shared_cta()`, and a `timed_region` macro that reads `%clock64` and `%globaltimer` before/after with a fake data dependency on the last loaded value (`add.u64` of the loaded value into the timer read, as in the H200 DSM blog) so the last load has retired.
2. `experiments/harness/host.hpp` — `DeviceInfo` (name, SM count, clocks from `cudaDeviceGetAttribute` + `nvmlDeviceGetClockInfo`, driver/CUDA versions, L2 size), `ClusterLauncher` (wraps `cudaLaunchKernelEx` with cluster dims + policy, asserts distinct `%smid` per rank when requested), `CsvWriter` (mandatory columns first; appends; never overwrites; crash-safe per `docs/context/09` §6: header only on empty file, flock while appending, fflush per launch, fsync per configuration, `__runs.csv` started/complete registry), `lock_clocks_note()` (reads `nvidia-smi -q -d CLOCK` output and the current `clocks_event_reasons` and records them), `flush_l2()` (streams a 128 MB buffer), `warmup()`.
3. `results/01__timers__timer_cost.csv`, `results/01__timers__globaltimer_resolution.csv`, `results/01__timers__loop_overhead.csv`, `results/01__timers__clock_domains.csv`, `01__timers_and_harness.ipynb`.

## Kernels and procedure

### K1 — timer read cost
One warp, one block. Unrolled 1024× back-to-back `%clock64` reads into registers; also 1024× `%globaltimer` reads; also `clock()` (32-bit `%clock`). Report cycles per read = (last − first)/1023. Record per-lane values (lane 0 only matters; verify lanes agree).

### K2 — `%globaltimer` resolution
One thread spins reading `%globaltimer` 1 M times, storing every value change (new value, `%clock64` at the change) into a global buffer. Report the distribution of deltas between consecutive distinct values (expected ≈32 ns ≈ 50–60 SM cycles at 1.6–1.98 GHz) and the SM-cycle count between changes, which yields the SM clock in MHz independently of `nvidia-smi` (`sm_mhz_from_globaltimer`). Repeat on 8 different SMs (launch 132 blocks, keep 8 chosen `%smid`) to confirm cross-SM consistency: all blocks record the `%globaltimer` at a `cluster`-free global flag release (block 0 writes flag, others spin and record arrival time) → skew histogram.

### K3 — dependent-chain loop overhead
The canonical p-chase loop:
```cpp
// ITERS dependent iterations, pointer in register p
for (int i = 0; i < ITERS; ++i) { asm volatile("ld.shared.u32 %0, [%1];" : "=r"(p) : "r"(p) : "memory"); }
```
versus the same loop with the load replaced by `mov.u32 %0, %1;` (register move) and with `add.u32`. ITERS ∈ {256, 1024, 4096, 16384}; unroll factors 1, 4, 16. Report cycles/iteration for the mov version (= loop overhead, expected 1–2 cycles when unrolled) and (ld.shared − mov) (= smem latency, expected ≈ 29–30; this doubles as a sanity check against [P1,P4]). The chosen unroll/ITERS pair becomes the project default: **unroll 16, ITERS 4096**, unless the notebook finds a better one.

### K4 — clock domains via ncu
Any trivial kernel (e.g. K3) profiled with
`ncu --clock-control base --metrics sm__cycles_elapsed.avg.per_second,lts__cycles_elapsed.avg.per_second,dram__cycles_elapsed.avg.per_second,gpc__cycles_elapsed.avg.per_second,gpu__time_duration.sum --csv --page raw --print-units base`
and again with `--clock-control none` after `nvidia-smi -lgc`. Report the three clocks and the ratio L2/SM. This ratio converts L2-side cycle counts in later ncu outputs.

### K5 — cudaEvent floor
Empty kernel `<<<1,32>>>` timed with events, 1000 launches; and `cudaEventElapsedTime` around 1000 back-to-back launches / 1000 → per-launch overhead (expected 2–5 µs). Used by `02__cluster_launch_and_occupancy`.

## Measured quantities and CSV columns
- `timer_cost.csv`: `timer ∈ {clock64, clock32, globaltimer}, reads, cycles_per_read, smid`.
- `globaltimer_resolution.csv`: `smid, delta_ns, delta_cycles` (one row per observed change) and `skew_ns` rows for K2b.
- `loop_overhead.csv`: `variant ∈ {mov, add, ld_shared}, iters, unroll, cycles_per_iter, smid`.
- `clock_domains.csv`: `clock_control ∈ {base, none}, sm_hz, lts_hz, dram_hz, gpc_hz, lts_over_sm`.
- `event_floor.csv`: `launches, us_per_launch`.

## Figures (notebook)
1. Histogram of `%globaltimer` deltas (ns) with the implied SM MHz annotated.
2. Bar chart: cycles/iteration for mov/add/ld.shared × unroll.
3. Table of clocks from ncu vs `nvidia-smi` vs globaltimer-derived.

## Sanity checks (agent-only)
Expected ranges for judging plausibility during result analysis. They are never presented as pass/fail and never quoted in the notebook. If a value falls outside, investigate (placement, clocks, SASS, cache state, ncu replay) and re-run. A value that survives is a finding about this device; describe it on its own terms.
- `ld.shared − mov` ∈ [27, 35] cycles (reproduces P1/P4 local SMEM latency).
- `%globaltimer` granularity ≈ 32 ns (P2); if it differs, update `docs/context/09__measurement_conventions.md` §1.
- Cross-SM skew of `%globaltimer` < 1 tick.

## NVIDIA tools
`ncu` (K4), `nvidia-smi -q -d CLOCK`, `nvidia-smi -lgc`/`-rgc`, `cuobjdump -sass` on the harness test binary to confirm `CS2R R, SR_CLOCKLO` / `S2UR`/`S2R SR_GLOBALTIMERLO` are emitted inside the timed region and that the ld.shared chain is `LDS` with a true dependency.
