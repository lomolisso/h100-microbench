# Device configuration: the one hardware/software configuration every experiment runs under

Binding for experiments 01–17 (18 only aggregates). An experiment may deviate only in a section titled **"Device-config deviations"** in its own spec, stating what changes and why. Measurement choices that a spec makes deliberately (ncu `--cache-control`, 01-K4's `--clock-control base` comparison, cache flushing) are not deviations.

**Why one configuration:** 18 combines rows from 04, 05, 06, 09, 11, 12 and 16 into one table. 01's loop overhead and L2/SM clock ratio are subtracted or applied everywhere. Cycle counts that cross into L2/HBM depend on the SM-clock : L2-clock ratio. A single fixed configuration keeps all of these comparable.

## 1. Device (measured on this card, 2026-10-23)

| Item | Value | Source |
|---|---|---|
| GPU | **NVIDIA H100 NVL** (GH100, PCIe form factor), VBIOS 96.00.AF.00.02 | `nvidia-smi` |
| Card used | **GPU 3 since 2026-10-25**: `GPU-1241dcbc-33ea-80bf-5f87-37f7868c3126`, PCI `00000000:DA:00.0`, serial 1793225008254 (H100 NVL, 400 W, ECC on, MIG off, checked 2026-10-25). Experiments 01–08 and the clock selection of §2 ran on GPU 2 (`GPU-428e544c-5649-1dde-a6bf-95de7485dd6a`, PCI `00000000:C3:00.0`, serial 1793225009411), now used only when the user selects it (`h100-target`, or a run pinned with `h100 run --gpu 2`). GPU-3 bring-up (2026-10-25, `AGENTS.md`): clock selection, SM→GPC map (03) and cluster placement/occupancy (02) re-measured on GPU 3 and identical to GPU 2; 08's L2 partition/sub-node labels were not re-measured on GPU 3 (empty in its map run) | `~/h100/lib/common.sh`, `nvidia-smi` |
| Compute capability / SASS target | 9.0 / `sm_90a` | `cudaDeviceProp`, ncu |
| SMs / TPCs | **132** / 66 (`launch__tpc_count`) | `cudaDeviceProp`, ncu |
| L2 cache | **60 MiB** (62 914 560 B) as reported (`cudaDeviceProp.l2CacheSize`, ncu `device__attribute_l2_cache_size`). Max persisting L2: 37.5 MiB. Partition/slice structure: settled by 08/13 | measured |
| Shared memory | 228 KiB per SM (233 472 B), 227 KiB opt-in per block | `cudaDeviceProp`, ncu |
| Memory | 94 GB HBM3, 6016-bit bus, 2619 MHz (supported: 2619, 1593) → theoretical peak 2 × 2619 MHz × 6016 bit / 8 = **3.94 TB/s** | `nvidia-smi`, ncu |
| SM clock range | 345–1785 MHz in 15 MHz steps; unlocked boost under light load ≈ 1778 MHz (ncu) | `nvidia-smi -q -d SUPPORTED_CLOCKS` |
| L2 clock | ≈ 1556 MHz, independent of the SM lock: 1555 MHz with the SM at 1778 MHz, 1558 MHz with the SM locked at 1635 MHz → **L2/SM ≈ 0.95 at our lock** (01-K4 measures it precisely) | ncu `lts__cycles_elapsed.avg.per_second` |
| Power limit | **400 W** (default 310 W, min 200 W, max 400 W). Set by the machine's administrators, not by us | `nvidia-smi -q -d POWER` |
| ECC / MIG / persistence / compute mode | enabled / disabled / enabled / Default | `nvidia-smi` |
| Host link | PCIe Gen5 x16 | `nvidia-smi` |
| Driver / toolkit | 580.126.20 (CUDA 13.0 driver API) / CUDA 12.9.1 (nvcc 12.9.86), Nsight Compute 2025.2.1, Nsight Systems 2025.1.3, image `h100-microbench:cuda12.9.1-<hash>` until 2026-10-29, then the shared image `h100:cuda12.9.1-<hash>` (same CUDA, ncu and nsys versions; `~/h100/H100.md`) | `h100 image verify` |

## 2. Fixed run configuration (chosen for this device)

| Setting | Value | Rationale |
|---|---|---|
| SM clock | **locked at 1635 MHz** (`h100 run --profile --lock-sm-mhz 1635`); selected on GPU 2 (2026-10-23) and confirmed on GPU 3 (2026-10-25: same outcome — 1785…1665 MHz all ran at 1635 MHz under load, 1635 passed with no throttle reason, HBM copy 3.38 TB/s, L2 streaming 6.95 TB/s, ≤ 364 W, ≤ 56 °C GPU / 72 °C HBM; one throttle sample each at the 1755 and 1725 MHz candidates) | Highest clock this card sustains under the suite's heaviest load (full-chip HBM copy 3.38 TB/s, L2 streaming 6.91 TB/s). Selected by `tools/clocktest/select_clock.sh` (now `h100 clocktest`) on 2026-10-23 (data: `docs/environment/clock_selection/`). Every lock from 1785 to 1665 MHz ran at exactly 1635 MHz under load, with no throttle reason flagged, about 357 W of 400 W, and ≤ 51 °C GPU / 68 °C HBM. 1635 MHz is the card's own ceiling under sustained load, so locking there gives the same clock for light and heavy kernels (a light probe kernel runs at 1630 MHz under the lock, vs 1778 MHz unlocked). Not copied from prior work, whose chips and limits differ (§4) |
| Memory clock | 2619 MHz (default; not locked, never changed) | the only full-speed HBM3 state; recorded and checked per run |
| Power limit | 400 W as found; never changed by us | the clock selection is valid only at this limit: if it differs, refuse to run and re-run the selection |
| ECC, MIG, compute mode | as in §1; never changed | ECC changes DRAM bandwidth and needs a reset; MIG would change SM/L2 topology |
| Compiler | `nvcc -arch=sm_90a -O3 -lineinfo -std=c++17` (harness `Makefile.inc`) | as in AGENTS.md |
| ncu | `--clock-control none` (keeps our lock), other flags per spec | `docs/context/07` |

**Every run checks and records** (harness `DeviceInfo`/`lock_clocks_note()`, CSV mandatory columns): GPU UUID, locked SM clock (`sm_clock_mhz`), memory clock, power limit, ECC and MIG mode, driver and CUDA versions, and `clocks_event_reasons` before and after. A run whose clocks dropped below the lock or whose throttle reasons fired during the timed region is discarded (`09` §5). A run with a different power limit, ECC or MIG mode refuses to start.

## 3. Device constants used by the specs
Use these instead of other chips' values when a spec needs this device's capacity or peak, e.g. reference lines, footprint ranges or "% of peak":

| Constant | Value |
|---|---|
| SM count | 132 |
| L2 size (reported) | 60 MiB; per-partition share from 08 |
| DRAM theoretical peak | 3.94 TB/s |
| SM clock for B/cycle ↔ TB/s conversions | the clock *measured* during the run (`sm__cycles_elapsed.avg.per_second`), nominally 1635 MHz |

## 4. Prior work was measured on other chips and configurations
The numbers in `08__prior_work_reference_numbers.md` and in the experiment specs' "References" lines are **kept unchanged**. They give the agents intuition for sanity-checking our results (never cited in the notebooks), but they come from different silicon:

| Source | Chip | SMs | SM clock | L2 | Memory | Toolkit |
|---|---|---|---|---|---|---|
| P1 Luo et al. | H800 PCIe | 114 | 1755 MHz | 50 MB | HBM2e, 2039 GB/s | CUDA 12.6 |
| P2 Lühnen et al. 2024 | H100 PCIe | 114 | not recorded in 08 | 50 MB | HBM2e | CUDA 12.3 |
| P3 ClusterSim | H100 SXM | 132 | 1620 MHz | 50 MB | HBM3, 3.35 TB/s | CUDA 12.8 |
| P4 CREDIT | H100 SXM (+ RTX 5090) | 132 | not recorded in 08 | 50 MB | HBM3 | CUDA 13.0 |
| P5 FlashGPU-sim | RTX 5090 (TMA/mbarrier, 2580 MHz); H200 (DSMEM) | 170 / 132 | 2580 / not recorded in 08 | — / 50 MB | GDDR7 / HBM3e | ncu cycles |
| P6 Jin et al. (MICRO 2024) | H100 PCIe (Table I lists the SXM spec; the artifact maps a 114-SM PCIe card), also V100/A100 | 114 | 1755 MHz max, not locked | 50 MB, 80 slices | HBM2e, 2 TB/s | nvprof/ncu, `-dlcm=cg` |
| FA-3 profiling note | H100 | — | locked 1.5 GHz | — | — | — |
| Cheat-sheet "Expected on H100 SXM5" | H100 SXM5 (medians of the above) | 132 | 1590 base / 1980 boost | 50 MB | 3.35 TB/s | — |
| **This project** | **H100 NVL** | **132** | **locked 1635 MHz** | **60 MiB, 96 slices (94 active)** | **HBM3, 3.94 TB/s** | **CUDA 12.9.1, Nsight Compute 2025.2.1** |

How to compare:
- **Cycles inside one SM** (SMEM, register, instruction, mbarrier and DSMEM latency): comparable across GH100 variants, with an SM-count caveat for chip-wide numbers.
- **Cycles that cross into L2/HBM:** depend on the SM:L2 clock ratio of each setup. Expect offsets.
- **TB/s and B/cycle chip-wide:** scale with SM count, clock and memory type. Compare per SM per cycle, and state the scaling.
- **RTX 5090 numbers** (P5 TMA/mbarrier) are a different architecture: intuition only.

These comparisons happen during result analysis only. Notebooks describe this device's configuration (§1–2) and report our own numbers.
