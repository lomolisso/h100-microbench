# Measurement conventions shared by all microbenchmarks

Every experiment spec in `docs/experiments/` (implemented under `experiments/`) follows these conventions unless its own file overrides a rule explicitly. The planning agent must treat this file as the contract between experiments, the CSV outputs, and the notebooks.

## 1. Timers and what each one may be used for

| Timer | What it counts | Scope | Resolution | Use for |
|---|---|---|---|---|
| `clock64()` / PTX `%clock64` (SASS `CS2R R, SR_CLOCKLO/HI`) | SM cycles of the SM executing the thread | per-SM; **not** synchronised across SMs | 1 cycle; read cost ≈ 2–10 cycles | all intra-SM latency measurements (p-chase, instruction latency), bandwidth in B/cycle per SM |
| `%globaltimer` (PTX `mov.u64 %0, %globaltimer;`) | nanoseconds, global | consistent across all SMs | updates every ~32 ns on Hopper (empirical — verify in `01__timers_and_harness`) | cross-SM event ordering (producer/consumer, store-visibility latency), long phases (> 10 µs) |
| `cudaEventElapsedTime` | wall time between host-enqueued events | whole GPU | ~0.5 µs | kernel-level throughput (TB/s GPU-wide), launch overhead |
| Nsight Compute `gpu__time_duration.sum`, `sm__cycles_elapsed.avg`, `smsp__cycles_active` | hardware counters, kernel scope | whole GPU | exact | authoritative cycles for bandwidth; traffic counters |

Rules:
- **Latency in cycles** is always taken from `clock64()` deltas inside the kernel, divided by the number of dependent operations in the timed loop, after subtracting the measured loop overhead (an identical loop with the dependent memory op replaced by a register move; measure once in `01__timers_and_harness` and store it in `results/overhead.csv`).
- **Bandwidth in TB/s (GPU-wide)** = bytes moved / `cudaEventElapsedTime` of the kernel, with enough blocks to cover every SM, reported together with the ncu-measured `sm__cycles_elapsed.avg.per_second` (actual clock) so the notebook can also express B/cycle.
- **Bandwidth in B/cycle/SM** = bytes moved by one SM / `clock64()` delta of that SM's block.
- Convert between the two only with the clock actually measured during the run (never with the datasheet boost clock).

## 2. Clock control

- Lock clocks before every run: SM clock to a fixed value (`nvidia-smi -lgc <MHz>,<MHz>`; use the **base** clock reported by `nvidia-smi -q -d CLOCK` → "Default Applications Clocks" — on H100 SXM5 this is 1590 MHz? **record the value that `nvidia-smi` actually reports**, do not hard-code) and memory clock (`nvidia-smi -lmc`). Record both in every CSV row (`sm_clock_mhz`, `mem_clock_mhz`).
- When profiling with ncu, pass `--clock-control none` so that our `nvidia-smi -lgc` lock stays in force (ncu's own default is `boost` since 2026.1 and was `base` before; never rely on the default). For runs where the kernel's own `clock64()` numbers are the deliverable, run **without** ncu, with `nvidia-smi` locks, and separately with ncu for counters.
- Report `cudaDeviceProp.clockRate` and the measured `sm__cycles_elapsed.avg.per_second` in the notebook header.

## 3. Placement control (which SM / which GPC)

- Every kernel records `%smid` (and `%clusterid`, `%cluster_ctarank` when launched as a cluster) for each block into the output buffer. CSVs contain the physical SM id of **every** block that produced a number.
- To pin "one block per SM": launch exactly `multiProcessorCount` blocks with enough dynamic shared memory (e.g. 200 KB) or threads (1024) that only one block fits per SM, then verify in the kernel that all `%smid` values are distinct (host asserts; if not, the run is discarded and repeated).
- To obtain a cluster on a chosen set of SMs we cannot choose directly; instead launch many clusters, read `%smid` per rank, and **bin results by (smid_src, smid_dst)** rather than by rank. The SM→GPC map is produced by `03__sm_gpc_topology` and consumed by all later heatmaps.
- Use `cudaLaunchKernelEx` with explicit `cudaLaunchAttributeClusterDimension` for all cluster launches; always also set `cudaLaunchAttributeClusterSchedulingPolicyPreference` explicitly (`Default` unless the experiment sweeps it) and record it.

## 4. Cache state control

- L2-resident data: pre-touch the working set with a separate warm-up kernel, keep working set ≤ 20 MB (well under ½ of 50 MB so that it fits in one partition's share) unless the experiment is about L2 capacity.
- DRAM-resident data: working set ≥ 256 MB with random (shuffled) chain order so that neither L2 nor its prefetch helps; flush L2 between repetitions by streaming a 128 MB scratch buffer (or ncu `--cache-control all`).
- L1 bypass: use `ld.global.cg` (or `__ldcg`) when L2 is the target; L1-resident: `ld.global.ca` with working set ≤ 128 KB per SM.
- Disable the persisting-L2 window unless the experiment uses it (`cudaCtxResetPersistingL2Cache` at start).

## 5. Statistical treatment

- Each (configuration) point: ≥ 5 independent kernel launches after 2 warm-up launches; inside a kernel, latency loops iterate ≥ 4096 dependent accesses (p-chase) or ≥ 1024 instruction repetitions.
- CSV rows store **raw per-launch (or per-block) values**, never only aggregates. Notebooks compute median, mean, std, p5/p95, and plot median with p5–p95 error bars.
- For latency-class discovery (near/far L2, DRAM channel distance) use per-access samples (one row per access with `addr_offset`, `smid`, `cycles`), then cluster with K-means (k chosen by elbow, state SD/CV per cluster as in Luo et al.).
- Discard a launch if the measured `%smid` set differs from the intended placement, or if `nvidia-smi` reported a clock-throttle event (read `clocks_event_reasons` before/after).

## 6. CSV conventions

- One CSV per experiment under `experiments/<NN__name>/results/`, named `<NN__name>__<variant>.csv`. UTF-8, header row, comma separated, no index column.
- Mandatory columns (first, in this order): `gpu_name, sm_count, sm_clock_mhz, mem_clock_mhz, cuda_version, driver_version, run_id, launch_idx`.
- Then the experiment-specific columns defined in the experiment file. Units go in the column name: `_cycles`, `_ns`, `_bytes`, `_bpc` (bytes per cycle), `_tbps`, `_pct`.
- ncu outputs: exported with `ncu --csv --page raw --print-units base` into `<NN__name>__ncu_<variant>.csv` (raw metric names as columns), joined in the notebook on `launch_idx`/`kernel name`.
- Never overwrite a results file: append `run_id` (ISO timestamp) and let the notebook pick the latest run or all runs.
- **Crash-safe writing** (a run can be killed at any moment; only rows already on disk survive):
  - Write rows **directly** into `experiments/<NN__name>/results/` (NFS). Never write results to `/tmp`: inside the container it is RAM and disappears with the container. Never stage results elsewhere and copy them later.
  - Open in append mode; write the header only if the file is new or empty. Never truncate, rewrite or sort an existing results file.
  - Emit one complete row per `write()`. `fflush` after every launch, and `fsync` at the end of every configuration, so a kill loses at most the configuration in progress.
  - Hold an exclusive `flock` on the CSV while appending, because two runs (e.g. GPU 2 and GPU 3) may append to the same file.
  - Run registry `results/<NN__name>__runs.csv` (`run_id, started_iso, finished_iso, status, gpu_uuid, cmdline`). Append a `status=started` row before the first sample and a `status=complete` row after the last flush. A `run_id` without a `complete` row is **partial**. Notebooks label it as partial and never mix it silently with complete runs.

## 7. Kernel hygiene

- All timed PTX in `asm volatile` with `"memory"` clobber; dependent chains carry the loaded value into the next address (`p = *(T**)p` pattern) so the compiler cannot hoist/parallelise.
- Keep the timed region free of `printf`, division, and host-visible writes; store results to a per-block slot of a global buffer **after** the second `clock64()` read.
- One warp (or one thread) for latency; full occupancy for bandwidth. State the exact block/grid dims in every experiment.
- Compile with `-arch=sm_90a -O3 -lineinfo`; keep a `cuobjdump -sass` dump of every benchmark binary in `results/sass/` and have the notebook show the SASS of the timed loop (so reviewers can see that the dependent chain and the intended instruction, e.g. `UBLKCP`, `UTMALDG`, `LD.E` for DSMEM, `SYNCS.*`, are present).

## 8. Notebook structure (one `.ipynb` per experiment)

1. Title + one-paragraph goal + link to the experiment markdown.
2. "What the kernel does" — the timed loop in CUDA/PTX and the SASS excerpt.
3. Environment table (from the CSV mandatory columns).
4. Figures (matplotlib; one figure per quantity listed in the experiment file's "Figures" section; axes labelled with units; GPU-wide reference lines drawn as dotted lines where the experiment file says so).
5. "Reproduction of prior work" — a table comparing our medians with `docs/context/08__prior_work_reference_numbers.md` rows and a ✔/✘ at the stated tolerance.
6. "Interpretation" — 5–10 sentences max, including anomalies.
