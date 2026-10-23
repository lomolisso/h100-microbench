# h100-microbench — instructions for Claude Code

@AGENTS.md

## What this repository is
Specifications (`docs/`) and implementations (`experiments/`) of 18 H100 microbenchmarks that characterise Hopper's memory hierarchy, DSMEM / SM-to-SM network, TMA, mbarriers and synchronisation primitives for a 3D-stacked-GPGPU study. Each experiment = CUDA C++/PTX kernels → CSV results → one Jupyter notebook with figures and a comparison against prior work.

## How to implement an experiment `NN__name`
1. Read, in this order: `AGENTS.md`, `docs/experiments/NN__name.md` (the spec), `docs/context/09__measurement_conventions.md`, then the context files the spec cites. The spec is binding: kernels, sweeps, CSV columns, figures, pass criteria, ncu metrics. Do not invent a different methodology; if the spec is impossible on the device, say so in the notebook and in the commit message.
2. Check that the experiments it depends on (AGENTS.md "Dependency order") have results in `experiments/<dep>/results/`; `01__timers_and_harness` must exist first because every experiment includes `experiments/harness/harness.cuh` and `host.hpp`.
3. Create `experiments/NN__name/Makefile` (set `EXP`, `BINS`, `include ../harness/Makefile.inc`, define `run`), kernels in `src/`, runner scripts in `src/` (bash/python; sweeps, `ncu` invocations, clock locking/recording), results in `results/` (CSV + `results/sass/*.sass`), notebook `experiments/NN__name/NN__name.ipynb`.
4. Build with `make -C experiments/NN__name`; run with `make -C experiments/NN__name run` (the `run` target must lock clocks via the harness, run warm-ups, write CSVs with the mandatory columns, and run the `ncu` command lines given in the spec).
5. Execute the notebook end-to-end (`jupyter nbconvert --execute --to notebook --inplace`) so figures are stored in the committed `.ipynb`.
6. Commit per experiment: `NN__name: <what was measured>, <pass/fail summary>`; keep CSVs and SASS dumps in the commit; never commit binaries or `.ncu-rep` files.

## Conventions enforced here
- PTX: use the exact instruction forms in `docs/context/02–05` (validated with ptxas 12.9 for sm_90a); inline `asm volatile` with `"memory"` clobbers for timed regions; `-arch=sm_90a -O3 -lineinfo`.
- Timers: `%clock64` for intra-SM cycles, `%globaltimer` for cross-SM ordering, cudaEvents for GPU-wide bandwidth; subtract the loop overhead measured in 01 (`docs/context/09`).
- Placement: record `%smid` (+ cluster id/rank) for every block that produces a number; force 1 CTA/SM with 200 KB dynamic smem when the spec says so; bin by physical SM, not by rank.
- Nsight Compute: metric names from `docs/context/07__profiling_tools.md`; always `--clock-control none` (own `nvidia-smi -lgc` lock), `--csv --page raw --print-units base --print-metric-name name`; verify the "unconfirmed" items of `docs/context/07 §8` on the device before trusting them.
- CSV: mandatory columns first (`gpu_name, sm_count, sm_clock_mhz, mem_clock_mhz, cuda_version, driver_version, run_id, launch_idx`), raw per-sample rows kept, never overwrite (append with a new `run_id`).
- Notebook sections: goal · kernel + SASS excerpt · environment table · figures (as listed in the spec, matplotlib, dotted GPU-wide references) · reproduction table against `docs/context/08__prior_work_reference_numbers.md` · interpretation.
- Honesty: a failed pass criterion is reported as failed; partial runs are labelled partial.

## Working style
- Prefer plan mode before starting an experiment group; the groups that can proceed in parallel are listed in AGENTS.md.
- One experiment per branch/PR when several agents work concurrently; results directories never conflict because they are per-experiment.
- The research notes in `docs/research_notes/` (raw PTX ISA, CUDA API, Nsight, repo-mining and literature notes) are supporting material; the curated `docs/context/` files take precedence when they disagree.
- No environment-setup instructions are to be added anywhere; the target is an H100 with CUDA ≥ 12.4 and Nsight Compute ≥ 2024 already installed.
