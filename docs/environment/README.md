# Development and execution environment (nishil-4xh100-01)

Binding for every human and coding agent working in this repo. Findings date: 2026-10-23.

## Workflow

```
edit / agents / git ─┐
                     ▼
   ~/work/fa2026/iacoma/h100-microbench   (NFS, authoritative)
        ▲  │  tools/run: container mounts $HOME at the same path, runs as you,
        │  │  sees only the selected GPU, removed (--rm) when the command ends
        │  ▼
   build (bin/, gitignored) → run benchmark → CSV written into experiments/<NN>/results/
                     ▼
   notebook executed (tools/run --no-gpu) → commit + push to GitHub
```

There is **no copy step**. The binary is not staged anywhere and the results are not copied back. The container reads the repo and writes the results in place over NFS. The only things on scratch are Docker's image and each container's small writable layer. The container's `/tmp` is RAM (tmpfs).

| What | Where |
|---|---|
| Source, notebooks, CSV results, SASS dumps (authoritative) | this repo on NFS; commit and push per experiment |
| Editing, Claude Code / Codex, git | same directory, on the host |
| Build outputs (`experiments/*/bin/`), `.ncu-rep`/`.nsys-rep` | in-tree, gitignored |
| Toolchain | **one** Docker image for all experiments (experiments never build their own); `tools/env build` removes our previous versions after a successful build. Image `h100-microbench:cuda12.9.1-<hash>`: nvcc 12.9.86, Nsight Compute 2025.2.1, Nsight Systems, gdb, Python 3.12 + `requirements.lock` |

NFS adds only milliseconds per git or grep operation for this repo, and nothing to compile time (CPU-bound). It is slow for trees of thousands of small files, so dependencies live in the image, not on NFS.

## Commands

```bash
make gpu-status                 # all GPUs, compute processes with user/container, locks, our containers
make gpu-check [GPU=3]          # preflight only
tools/env info                  # image + container-store free space
tools/env build                 # build the image (refuses below 40 GB free on the container store)
tools/env verify [--gpu N] [--profile]   # toolchain + 32-thread probe + 1-metric ncu

tools/run --no-gpu -- make -C experiments/01__timers_and_harness       # build (no GPU)
tools/run -- make -C experiments/01__timers_and_harness run            # run on GPU 2
tools/run --gpu 3 -- ...                                               # GPU 3 only when named explicitly
tools/run --profile --lock-sm-mhz 1635 -- make -C experiments/NN run   # ncu counters + the project's locked clock (docs/context/10)
tools/clocktest/select_clock.sh [--gpu N]  # re-run the clock selection (only if power limit/driver/card change)
tools/run --no-gpu -- jupyter nbconvert --execute --to notebook --inplace experiments/NN/NN.ipynb
tools/run [--no-gpu]            # interactive shell
```

## Container lifecycle
- What survives a kill (by us, by Ctrl-C, or by anyone with sudo/docker rights): every row already written to `results/` on NFS. What is lost: data still in the benchmark's memory and anything in the container's `/tmp` (RAM). Benchmarks therefore follow the crash-safe CSV rules in `docs/context/09` §6.
- `docker run --rm --init`, in the foreground: the container exists only while the command runs and is deleted when it exits. `--init` reaps child processes inside it.
- If `tools/run` is interrupted (Ctrl-C) or terminated, its exit trap runs `docker stop` on its own container. In `--profile` mode the entrypoint forwards the SIGTERM and resets a clock lock before exiting. Only `kill -9` of `tools/run` can leave a container behind. `make gpu-status` lists any container labelled `h100mb.owner=$USER`.
- The GPU lock is held by `tools/run` for exactly the lifetime of its container.

## GPU safety

- Authorized: **GPU 2 (default)** and **GPU 3 (explicit `--gpu 3`)**, pinned in `tools/common.sh` by UUID and PCI id. Every other index is refused.
- `tools/run` (GPU mode) refuses to launch when any of these holds for the target GPU:
  - index, UUID or PCI id don't match the pinned values;
  - any compute process is on it (owner and container are reported);
  - memory used > 1 GiB, or utilization > 5 % in a 2 s sample;
  - a non-infrastructure container was given the GPU (`NVIDIA_VISIBLE_DEVICES`/`--gpus` containing the index, UUID or `all`);
  - the cooperative lock is held.

  With `H100MB_REQUIRE_RESERVATION=1` it also refuses if no window in `~/.config/h100-microbench/reservations.tsv` covers now.
- Every run (GPU or not) refuses to start a container when the container store has < 2 GB free. Building refuses below 40 GB.
- Cooperative lock: `flock` on `/tmp/h100-gpu-locks/gpu<N>.lock` (a few bytes). It only coordinates processes that use it, i.e. our tools and agents. It is **not** exclusive ownership.
- Container isolation: `--gpus device=<UUID>` and `NVIDIA_VISIBLE_DEVICES=<UUID>`. The image defaults to `void`, because the host's default Docker runtime is `nvidia` and would otherwise expose all four GPUs. The entrypoint aborts unless it sees exactly the expected UUID (or nothing for `--no-gpu`). Never `docker run` the image by hand.
- Not checkable from this account: Kubernetes allocations. The node's nvidia-device-plugin advertises GPUs 2 and 3.
- Never kill, renice or touch other users' processes, containers, locks or allocations, and never `docker system prune`/`image prune` (the daemon is shared).

## Host facts that shaped this

- `/home/$USER` is NFSv3 (autofs, root-squashed, mode 0700). The root Docker daemon can `stat` `$HOME` but nothing inside it, so a subdirectory bind mount fails with `permission denied`. `tools/run` mounts all of `$HOME` and runs as your uid. Docker's runtime sets the working directory as root, which cannot enter the home, so `tools/run` never passes `-w`. It changes into `$PWD` as your uid inside the container. In `--profile` mode that happens after the entrypoint has dropped from root.
- `/scratch` (= `/srv/local/scratch`, local XFS 21 TB, shared by group `cs_nishil`, no quota) was **100 % full** on 2026-10-23. Docker's image store `/var/lib/containerd` is a symlink onto that volume, so the image cannot be built until space is freed.
- Profiling: `RmProfilingAdminOnly=1` means a plain user gets `ncu` → `ERR_NVGPUCTRPERM`. Clock locking (`nvidia-smi -lgc`) needs root. `--profile` runs the container as root with `CAP_SYS_ADMIN`. The entrypoint locks the clock (if asked), resets it on exit, and runs the command as your uid with `CAP_SYS_ADMIN` kept as an ambient capability. A DCGM exporter runs on the node; if ncu reports the counters busy, that is the likely cause.
- The device is an **H100 NVL** (132 SMs, 60 MiB L2 reported by `cudaDeviceProp`, max SM clock 1785 MHz, HBM 2619 MHz), not SXM5. Record it, and compare with SXM5 prior-work numbers accordingly.
- `import numpy` spins one OpenBLAS thread per core (256). The image caps BLAS/OpenMP threads at 8.

## Recovery after scratch loss
Nothing of ours is stored only on scratch. Run `tools/env build`, then rebuild with `make`. Source and results are on NFS and GitHub.
