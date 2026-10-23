# h100-microbench — instructions for Claude Code

@AGENTS.md

## What this repository is
Specifications (`docs/`) and implementations (`experiments/`) of 18 H100 microbenchmarks that characterise Hopper's memory hierarchy, DSMEM / SM-to-SM network, TMA, mbarriers and synchronisation primitives for a 3D-stacked-GPGPU study. Each experiment = CUDA C++/PTX kernels → CSV results → one self-contained Jupyter notebook explaining and analysing our own measurements.

## Starting a new conversation
One conversation = one experiment, run strictly one after another. A new conversation needs no further prompt:
1. Open the **Experiment status** table in `AGENTS.md`. Work on the first row whose status is not `done`:
   - `in review`: the previous experiment is committed but not yet approved. Ask the user for their review, apply the requested changes, and push once they approve (see "What an experiment is", step 6).
   - `todo` or `in progress`: this is the experiment to implement. Its dependencies (AGENTS.md "Dependency order") must be `done`; if they are not, tell the user.
2. Work in **plan mode** first (enter it if the conversation did not start in it). Read the files listed below, then present a plan: kernels, sweeps, runs, expected duration on the GPU, and the notebook outline. After the user approves, carry it out in auto mode.
3. Read, in this order: `AGENTS.md`, `docs/experiments/NN__name.md` (the spec), `docs/context/09__measurement_conventions.md`, `docs/context/10__device_config.md` (the device and the fixed run configuration), `docs/environment/README.md`, then the context files the spec cites. Also read the "Key results" column of earlier rows: later experiments consume them (e.g. 01's loop overhead, 03's SM→GPC map).

## What an experiment is (in this order)
1. **Code.** `experiments/NN__name/Makefile` (set `EXP`, `BINS`, `include ../harness/Makefile.inc`, define `run`), kernels in `src/*.cu`, runner scripts in `src/` (sweeps, `ncu` invocations, clock checks), results in `results/` (CSV + `results/sass/*.sass`). The spec is binding: kernels, sweeps, CSV columns, figures, ncu metrics. Do not invent a different methodology. If the spec is impossible on this device, say so in the notebook and in the commit message. Build with `tools/run --no-gpu -- make -C experiments/NN__name`.
2. **Run on the H100.** `tools/run --profile --lock-sm-mhz 1635 -- make -C experiments/NN__name run`: warm-ups, the fixed configuration of `docs/context/10`, CSVs with the mandatory columns written crash-safely into `results/` (`09` §6), and the ncu command lines given in the spec.
3. **Understand the results.** Before writing anything, sanity-check the numbers against the spec's "Sanity checks" section and `docs/context/08__prior_work_reference_numbers.md`, using the cross-chip comparison rules of `10` §4. Prior-work numbers are used **here and only here**. If something looks off, investigate before accepting it: SASS of the timed loop, `%smid` placement, clocks and throttle reasons, cache state, ncu replay passes. Re-run if needed. A value that survives investigation is a finding about this device, not an error.
4. **Notebook.** Write `experiments/NN__name/NN__name.ipynb` following `09` §8, then execute it end-to-end (`tools/run --no-gpu -- jupyter nbconvert --execute --to notebook --inplace …`) so the figures are stored in the file.
5. **Update the status table and commit.** In `AGENTS.md` → "Experiment status", set the row to `in review` and fill in "Key results" (the numbers and files later experiments need). Commit everything in one commit: code, CSVs, SASS dumps, executed notebook and status update. Message: `NN__name: <what was measured>; <main finding>`. Never commit binaries or `.ncu-rep`/`.nsys-rep` files. **Do not push.**
6. **Push only after the user's review.** Once the user has reviewed the notebook and explicitly approved it: apply any fixes, set the row to `done`, commit, and push. Never force-push. Work on the branch that is checked out; create, switch or merge branches only when asked.

## Subagents and tools
- Spawning subagents to parallelise work *within* the experiment is encouraged: independent kernel variants, CSV analysis, figure code, notebook sections, documentation lookups. Every subagent uses **Claude Opus 5.5 (`model: opus`) with effort `high`**.
- Only one GPU job at a time. Subagents never start GPU runs in parallel. `tools/run` refuses while the GPU lock is held: wait and retry, and never switch to GPU 3 on your own.
- Web search is allowed when needed (PTX ISA, CUDA, Nsight documentation). The curated `docs/context/` files take precedence over the web and over `docs/research_notes/`.

## Conventions enforced here
- PTX: use the exact instruction forms in `docs/context/02–05` (validated with ptxas 12.9 for sm_90a). Use inline `asm volatile` with `"memory"` clobbers for timed regions and `-arch=sm_90a -O3 -lineinfo`.
- **Kernel documentation lives in the code.** Every benchmark `.cu` file starts with a docstring block comment stating:
  - what is measured;
  - how each kernel does it: timed region, dependency chain, placement, parameters;
  - what is written to the CSV.

  Add inline comments where the code is not self-explanatory. Be precise, concise and understandable.
- Timers: `%clock64` for intra-SM cycles, `%globaltimer` for cross-SM ordering, cudaEvents for GPU-wide bandwidth. Subtract the loop overhead measured in 01 (`docs/context/09`).
- Placement: record `%smid` (+ cluster id/rank) for every block that produces a number. Force 1 CTA/SM with 200 KB dynamic smem when the spec says so. Bin by physical SM, not by rank.
- Nsight Compute: metric names from `docs/context/07__profiling_tools.md`. Always pass `--clock-control none` (our own lock) and `--csv --page raw --print-units base --print-metric-name name`. Verify the "unconfirmed" items of `docs/context/07 §8` on the device before trusting them.
- CSV: mandatory columns first (`gpu_name, sm_count, sm_clock_mhz, mem_clock_mhz, cuda_version, driver_version, run_id, launch_idx`). Keep raw per-sample rows and never overwrite: append with a new `run_id`. Write straight into `results/` (never `/tmp`), flush per launch, and register runs in `__runs.csv` so killed runs show up as partial (`docs/context/09` §6).
- **Notebooks are the public result.** They must read as standalone technical reports (`09` §8):
  - never mention `docs/`, the specs, `CLAUDE.md`/`AGENTS.md`, agents, Claude, or how the work was organised;
  - never cite, plot or tabulate other works or their numbers;
  - no pass/fail anywhere.
- Honesty: report what was measured, including anomalies, discarded launches (with counts) and partial runs. Never tune parameters or drop data to approach expected values.

## Working style
- Sequential experiments, one conversation each (see above); no parallel experiments, so no extra branches or worktrees are needed.
- The research notes in `docs/research_notes/` (raw PTX ISA, CUDA API, Nsight, repo-mining and literature notes) are supporting material. The curated `docs/context/` files take precedence when they disagree.
- Environment: follow `docs/environment/README.md` (the only place environment instructions live):
  - Edit, run agents and use git directly in this NFS repo.
  - Run every build, benchmark and notebook through `tools/run` (`--no-gpu` for builds and notebooks; GPU 2 by default, `--gpu 3` only when the user says so, never other GPUs).
  - Write results straight into `experiments/<NN>/results/`.
  - Never `docker run` by hand, never `docker … prune`, never touch other users' processes.
