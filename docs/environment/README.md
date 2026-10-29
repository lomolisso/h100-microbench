# Development and execution environment (nishil-4xh100-01)

Binding for every human and coding agent working in this repo. Since 2026-10-29 the server-wide tooling lives in
`~/h100/` (manual: **`~/h100/H100.md`**): the `h100` command, the per-user FCFS GPU queue, the target-GPU switch
(`h100-target`) and the single toolchain image shared with `~/work/fa2026/iacoma/h100-dsmem-collectives`. This page
keeps only what is specific to this repository. The former in-repo `tools/{run,gpu,env}`, `docker/` and
`requirements.lock` were removed in that change (git history has them; `requirements.txt` still lists what the
notebooks import).

## Workflow

```
edit / agents / git ─┐
                     ▼
   ~/work/fa2026/iacoma/h100-microbench   (NFS, authoritative)
        ▲  │  h100 run: container mounts $HOME at the same path, runs as you,
        │  │  sees only the GPU the queue gives it, deleted when the command ends
        │  ▼
   build (bin/, gitignored) → run benchmark → CSV written into experiments/<NN>/results/
                     ▼
   notebook executed (h100 run --no-gpu) → commit code + notebook (raw results stay local) → push after the user's review
```

There is **no copy step**. The binary is not staged anywhere and the results are not copied back. The container reads the repo and writes the results in place over NFS. The container's `/tmp` is RAM and is discarded.

| What | Where |
|---|---|
| Source, executed notebooks, SASS dumps (authoritative) | this repo on NFS; commit per experiment, push after the user's review |
| Raw results (`experiments/*/results/*.csv`, logs, ncu/nsys exports) | main checkout on NFS **only**; gitignored, never pushed or deleted; rules in `docs/context/09` §6 |
| Editing, Claude Code / Codex, git | same directory, on the host |
| Build outputs (`experiments/*/bin/`), `.ncu-rep`/`.nsys-rep` | in-tree, gitignored |
| Toolchain | the shared image `h100:cuda12.9.1-<hash>` (`~/h100/docker/`): CUDA 12.9.1 (nvcc 12.9.86), Nsight Compute 2025.2.1, Nsight Systems 2025.1.3, Python 3.12 with the notebook stack, plus clang-20, NVBit, CUTLASS, PyTorch. Experiments never build their own image. Rows 01–12 ran in `h100-microbench:cuda12.9.1-26d4f436f8c5`, which had the same CUDA/ncu/nsys versions |

## Commands

```bash
h100-target                     # show the target GPU and the queue (setting it is the user's decision)
make gpu-status                 # = h100 status: all GPUs, processes with owners, locks, our containers, the queue
make gpu-check [GPU=2]          # = h100 check: preflight of one GPU (default: the target)

h100 run --no-gpu -- make -C experiments/01__timers_and_harness                       # build (no GPU, runs at once)
h100 run --gpu 3 --profile --lock-sm-mhz 1635 -- make -C experiments/NN run          # queued GPU run, pinned to the experiment's card
h100 submit --gpu 3 --profile --lock-sm-mhz 1635 -- make -C experiments/NN run       # same, returns the job id (h100 wait / h100 log -f)
h100 run --no-gpu -- jupyter nbconvert --execute --to notebook --inplace experiments/NN/NN.ipynb
h100 shell                      # interactive shell without GPU
h100 queue                      # queue state, freeze reason, jobs
```

- `--profile` is needed for `ncu` (the driver has `RmProfilingAdminOnly=1`) and for the clock lock; `--lock-sm-mhz 1635` is the fixed configuration of `docs/context/10` §2.
- **Pin every GPU run of an experiment to the card its plan names** (`--gpu N`). Results that depend on the card (L2 partition labels, slice/hash maps, clock domains) are not portable between GPUs 2 and 3. A pinned job never runs on another card: if the user switches the target, the queue freezes until the card is the target again.
- The queue runs one GPU job at a time, first come first served, and **freezes** when anyone else uses the GPU or a job of ours is ended from outside; the interrupted job stays at the head and reruns after the user unfreezes it. Agents never unfreeze, freeze, switch the target or cancel jobs on their own: report `h100 queue`'s reason and wait. Benchmarks must therefore stay restartable: append-only CSVs with a new `run_id` and the run registry of `docs/context/09` §6, so an interrupted attempt shows up as a partial run.

## GPU history of this repository
- Authorized cards: GPU 2 (`GPU-428e544c-5649-1dde-a6bf-95de7485dd6a`, PCI `C3:00.0`, serial 1793225009411) and GPU 3 (`GPU-1241dcbc-33ea-80bf-5f87-37f7868c3126`, PCI `DA:00.0`, serial 1793225008254). Others are refused by `h100`.
- Experiments 01–08 and the clock selection (`clock_selection/`, 2026-10-23) ran on **GPU 2**. On 2026-10-25 the work moved to **GPU 3**: the GPU-3 bring-up row of `AGENTS.md` re-checked it (same 1635 MHz lock, SM→GPC map, cluster placement and occupancy; 08's L2 labels not re-measured). Experiments 09 onwards ran on GPU 3. 12 found GPU 3's L2 partitions to be GPCs {0,5,6,7} | {1,2,3,4} (G4 and G7 swapped compared with GPU 2).
- Until 2026-10-29 the default GPU was hard-coded (`DEFAULT_GPU` in the removed `tools/common.sh`). Since then there is no default: the user sets the target, and experiments pin their runs.

## Host facts
The full list is in `~/h100/H100.md` §7. The facts that shaped this repo's conventions:
- `$HOME` is NFS: results are written straight into `experiments/<NN>/results/` from inside the container. Never write results to `/tmp` (RAM, lost when the job ends).
- `import numpy` spins one OpenBLAS thread per core (256); the image caps BLAS/OpenMP threads at 8.
- The device is an **H100 NVL** (132 SMs, 60 MiB L2 reported by `cudaDeviceProp`, max SM clock 1785 MHz, HBM 2619 MHz), not SXM5. Record it, and compare with SXM5 prior-work numbers accordingly.
- A DCGM exporter runs on the node; if ncu reports the counters busy, that is the likely cause.

## Recovery after scratch loss
Nothing of ours is stored only on scratch. Run `h100 image build`, then rebuild with `make`. Source and notebooks are on NFS and GitHub; raw results only on NFS (main checkout).
