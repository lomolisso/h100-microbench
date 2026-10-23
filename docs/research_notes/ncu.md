# NVIDIA profiling tooling for H100 (GH100, cc 9.0) microbenchmarks — reference note

Audience: engineers measuring latency/bandwidth of SMEM, DSMEM (SM-to-SM), L1, L2, HBM and TMA on an H100 and dumping results to CSV.
Scope: Nsight Compute CLI (`ncu`), Nsight Systems (`nsys`), binary tools, on-device timers/clocks. No install/setup.

## 0. Provenance and how things were verified

`docs.nvidia.com` was not reachable from the sandbox that produced this note, so the text below was taken from NVIDIA's own copies of the same documents and binaries:

* **Nsight Compute 2026.2.1.0 (build 38283040)** — the official `nsight-compute` package from NVIDIA's conda channel (`conda.anaconda.org/nvidia`, file `nsight-compute-2026.2.1.5-h597d895_0.conda`). It bundles `docs/ProfilingGuide/index.html`, `docs/NsightComputeCli/index.html`, `docs/ReleaseNotes/index.html` (same content as https://docs.nvidia.com/nsight-compute/ProfilingGuide/, https://docs.nvidia.com/nsight-compute/NsightComputeCli/, https://docs.nvidia.com/nsight-compute/ReleaseNotes/), the `sections/*.section` files, and the `ncu` binary.
* **Metric existence for cc 9.0** was checked offline with that `ncu`: `ncu --query-metrics --chip gh100` (5,642 base metrics), `ncu --query-metrics-mode all --chip gh100` (494k fully-suffixed names), and `--query-metrics-collection {launch,warpsampling,pmsampling,source,device,stats,groups} --chip gh100`. "Confirmed (gh100)" below means the name appears in that output; the quoted description is NVIDIA's.
* **SASS mnemonics** were verified by assembling hand-written PTX with `ptxas` 12.9.86 (`-arch=sm_90a`, from the `nvidia-cuda-nvcc-cu12` PyPI wheel) and disassembling with `nvdisasm`/`cuobjdump` 13.4.92 (`nvidia-cuda-nvdisasm`, `nvidia-cuda-cuobjdump` PyPI wheels). The official opcode list is https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#hopper-instruction-set.
* Web-search summaries of docs.nvidia.com pages were used for `cudaEventElapsedTime`, PTX special registers, nvidia-smi and nsys; those are marked "(web summary)".

---

## 1. Nsight Compute CLI (`ncu`)

Docs: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html (options) and https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html (semantics).

### 1.1 Flags you will actually use (exact text from `ncu --help`, 2026.2.1)

| Flag | Semantics (verbatim or near-verbatim) | Notes for microbenchmarks |
|---|---|---|
| `--metrics <list>` | "Specify all metrics to be profiled, separated by comma." Prefixes: `regex:<expr>` ("Expands to all metrics that partially match the expression. Enclose the regular expression in ^...$ to force a full match."), `group:<name>` ("Lists all metrics of the metric group with that name. See section files for valid group names."), `breakdown:<metric>` ("Expands to the input metrics of the high-level throughput metric"). "If a metric requires a suffix to be valid, and no prefix is used this option automatically expands the name to all available first-level sub-metrics. Metrics that cannot be collected will generate an error." | Pass fully-suffixed names (`x.sum`, `x.avg.per_second`) to keep CSV columns deterministic. `--disable-extra-suffixes`: "Disables the collection of extra suffixes (avg, min, max, sum). Only collect what is explicity specified." |
| `--section <id>` | "Collect the section by providing section identifier ... `<section identifier>` exact match; `regex:<expression>`. If option is not specified, the default section set is collected. Section metrics that cannot be collected will usually generate a warning only." | Useful ids: `MemoryWorkloadAnalysis`, `MemoryWorkloadAnalysis_Chart`, `MemoryWorkloadAnalysis_Tables`, `InstructionStats`, `SourceCounters`, `WarpStateStats`, `SchedulerStats`, `LaunchStats`, `Occupancy`, `SpeedOfLight`, `PmSampling`, `PmSampling_WarpStates` (`ncu --list-sections`). |
| `--set <id>` | "Identifier of section set to collect. If not specified, the basic set is collected." Sets (`--list-sets`): `basic` (LaunchStats, Occupancy, SpeedOfLight, WorkloadDistribution; ~213 metrics), `detailed` (~1071), `full` (all sections incl. the three MemoryWorkloadAnalysis sections, InstructionStats, SourceCounters, WarpStateStats, PmSampling…; ~7381 metrics), `nvlink`, `pmsampling`, `roofline`. | `--set full` on H100 means dozens of replay passes; for latency kernels prefer explicit `--metrics`. |
| `--csv` | "Use comma-separated values in the output. Implies --print-units base by default." | CSV goes to stdout; use `--log-file` to redirect tool messages, or `ncu --import x.ncu-rep --csv --page raw > x.csv`. |
| `--page {details,raw,source,session}` | "details: sections and rules; raw: all collected metrics; source: source code; session: session and device attributes". Default `details`. "Note that if --page is not used but --export is, no results will be printed to the console." | `--page raw --csv` prints one row per kernel with one column per metric (plus launch__/device__ attributes). |
| `--print-units {auto,base}` | "auto: Scale metrics to fitting magnitude; base: Show metrics with their base unit" (SI scaling by 1000, prefixes K/M/G and n/u/m). | Always `base` for CSV (implied by `--csv`). |
| `--print-fp` | "Show all numeric metrics as floating point numbers." | |
| `--print-metric-name {label,name,label-name}` | Column shows metric label (default), raw name, or both. | Use `name` so CSV headers are the metric identifiers. |
| `--print-details {header,body,all}` | "Select which part of a section should be shown in the details page output: header (default) / body / all (Show all metrics from the section)". | Older docs call this `--details-all`. |
| `--print-metric-instances {none,values,details}` | "none (default): Only show aggregate value; values: Show aggregate followed by all instance values; details: Show aggregate value, followed by correlation IDs and instance values". Details page: "Some metrics will show multiple values, separated by ';' ... Those are instanced metrics, which have one value per represented instance. An instance can be a streaming multiprocessor, an assembly source line, etc." Release notes: "The raw page csv output now includes metric instance values when these enabled for printing." | This is the per-SM / per-L2-slice / per-PC breakdown switch. See §2.9. |
| `--print-summary {none,per-gpu,per-kernel,per-nvtx}` | Summary output mode. | `per-kernel` aggregates repeated launches. |
| `-k`/`--kernel-name <name>` | "`<kernel name>` exact match; `regex:<expression>`". `--kernel-name-base {function,demangled,mangled}` (default `function`). | |
| `--kernel-id <ctx>:<stream>:[name-op:]<name>:<invocation>` | "Skip entries that shouldn't be matched, e.g. use `::foobar:2` to match the second invocation of 'foobar' in any context or stream. Use `:7:regex:^foo:` to match any kernel in stream 7 beginning with 'foo'." | |
| `-s`/`--launch-skip N` | "Set the number of kernel launches to skip before starting to profile. The count is incremented for launches that match the kernel filters only." (`--launch-skip-before-match` counts all launches.) | Skip warm-up launches. |
| `-c`/`--launch-count N` | "Limit the number of collected profile results. The count is only incremented for launches that match the kernel filters." | |
| `--kill {0,1}` | "Terminate the target application when the requested --launch-count was profiled." | |
| `--replay-mode {kernel,application,range,app-range}` | `kernel` (default): "Replay individual kernel launches transparently during the execution of the application." `application`: "Relaunch the entire application multiple times. Requires deterministic program execution." `range`: "Replay ranges of kernel launches and API calls during the execution of the application." `app-range`: "Profile ranges without API capture by relaunching the entire application multiple times." | See §1.3. |
| `--cache-control {all,none}` | `all` (default): "All GPU caches are flushed before each kernel replay iteration during profiling ... most reproducible metric results." `none`: "No GPU caches are flushed during profiling. This can improve performance and better replicates the application behavior if only a single kernel replay pass is necessary ... some metric results will vary depending on prior GPU work, and between replay iterations." | For L2-hit / warm-cache experiments use `none` + warm-up launches + single-pass metric sets. |
| `--clock-control {base,boost,force-boost,none,reset}` | CLI reference: `base`: "GPC and memory clocks are locked to their respective base (rated tdp) frequency during profiling." `boost`: "GPC and memory clocks are locked to their respective turbo boost frequency ... If setting boost clock is not supported by GPU or driver, falls back to locking clocks to 'base' if possible." `force-boost`: "... Supports Ampere+ on NVIDIA kernel mode driver 560 and Turing+ on driver 580 or newer." `none`: "No GPC or memory frequencies are changed during profiling." `reset`: "Reset GPC and memory clocks for all or the selected devices and exit. Use if a previous, killed execution of ncu left the GPU clocks in a locked state." **Default: `boost` since 2026.1** ("The default value of --clock-control is now boost."); **`base` in 2025.4.x and earlier** (boost/force-boost options were added in 2025.4). | **Important:** with `base` the SM runs at the rated base clock (H100 SXM5 ≈1590 MHz vs 1980 MHz boost), so ns-based numbers drop while cycle-based numbers (`*_cycles_*`) stay comparable. Pick one mode and record `sm__cycles_elapsed.avg.per_second` in every run. HBM clock cannot be changed on HBM GPUs (NVIDIA forum answer, web summary), so `base` changes the compute:bandwidth ratio. To use your own `nvidia-smi -lgc` lock, pass `--clock-control none`. "Thermal throttling directed by the hardware or driver cannot be controlled by the tool and always overrides any selected options." |
| `--pipeline-boost-state {stable,dynamic}` | "Control the Tensor Core boost state. Setting stable Tensor Core boosting is recommended ..." (default `stable`). | |
| `--query-metrics` | "Query available metrics for devices on the system. Use --devices and --chips to filter ... By default, metrics reported by this option require a suffix to be collected." | `ncu --query-metrics --chip gh100` works **without a GPU**. |
| `--query-metrics-mode {base,suffix,all}` | "base (default): base names; suffix: suffix names for metrics. Use --metrics to specify the base metrics to query; all: full names". | `ncu --query-metrics-mode suffix --metrics lts__t_sectors --chip gh100` |
| `--query-metrics-collection {device,groups,launch,numa,nvlink,occupancy,pmsampling,profiling,source,stats,warpsampling}` | "Set which metric collection kind to query." `warpsampling` = "metrics available for periodic sampling of the warp program counter and warp scheduler state" (the `smsp__pcsamp_*` family lives here, **not** in the default `profiling` list). | |
| `--list-chips` / `--chips gh100` | Chips (2026.2.1): ad102…, ga100…, gb100, gb102, gb10b, gb110, gb112, gb202…, **gh100**, tu102… | |
| `--target-processes {application-only,all}` | Default **`all`** ("profile the application and its child processes"); `--target-processes-filter` for names/regex. | |
| `-o`/`--export <file>` | "Set the output file for writing the profile results. If not set, a temporary file will be used which is removed afterwards." `-f` to overwrite. | Produces `<file>.ncu-rep`. |
| `-i`/`--import <file>` | "Set the input file for reading profile results." "When using this flag, most other options are not available, except for certain result filtering options." `--import` + `--export` = filtered re-export. | Pattern: profile once with `-o`, then `ncu -i rep.ncu-rep --csv --page raw --print-units base --print-metric-name name > out.csv`. |
| `--nvtx`, `--nvtx-include <expr>`, `--nvtx-exclude <expr>`, `--range-filter` | "NVTX filtering requires --nvtx option." Examples from the doc: `ncu --nvtx --nvtx-include "A_range/"` (push/pop range), `"Domain-A@A_range/"`, `"[A_range"` (bottom of stack), `"A_range]"` (top), `"A_range/*/B range"`, `"regex:Domain[A-Z]@Range[0-9]/"`, start/end form `--nvtx-include "A_range"`. "NVTX filtering honors cudaProfilerStart() and cudaProfilerStop()." `--range-filter <yes/no>:<start/stop instances>:<NVTX instances>` e.g. `--range-filter :2:3 --nvtx-include A/`. `--print-nvtx-rename kernel` renames kernels by enclosing push/pop range. | |
| `--profile-from-start {on,off}` / `--disable-profiler-start-stop` | "Set if application should be profiled from its start." / "When set, cu(da)ProfilerStart/Stop APIs are ignored." | `--profile-from-start off` + `cudaProfilerStart()` around the measured launches. |
| `--graph-profiling {node,graph}` | "node (default): Profile individual kernel nodes; graph: Profile entire graphs". | |
| `--warp-sampling-interval [0..31|auto]` | "Actual frequency is 2 ^ (5 + value) cycles." `--warp-sampling-max-passes` (default 5), `--warp-sampling-buffer-size` (default 33554432). | For stall-reason experiments. |
| `--pm-sampling-interval`, `--pm-sampling-buffer-size`, `--pm-sampling-max-passes`, `--warp-samples-per-interval`, `--disable-pm-warp-sampling` | PM sampling (time-series counters). | |
| `--import-source {0,1}`, `--source-folders`, `--print-source {sass,ptx,cuda,cuda,sass}`, `--resolve-source-file` | Source correlation (needs `-lineinfo`). "Metrics specified with --metrics and specified section file with --section are correlated." | |
| `--log-file <file|stdout|stderr>` | Redirect tool output. | |
| `--check-exit-code`, `--forward-signals`, `--null-stdin` | Launch conveniences. | |

Not a real flag: `--print-metric-instances` takes `values`/`details`, **not** `.per_instance` suffixes (no such suffix exists).

### 1.2 Minimal CSV recipes

```bash
# 1) explicit metrics, one CSV row per profiled launch
ncu --clock-control none --cache-control none \
    -k regex:^bench_ -s 5 -c 20 \
    --metrics gpu__time_duration.sum,sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second,\
lts__t_sectors_srcunit_tex_op_read.sum,dram__bytes_read.sum,dram__bytes_write.sum \
    --csv --page raw --print-units base --print-metric-name name \
    --log-file ncu.log ./bench > results.csv

# 2) full report then CSV extraction (keeps a .ncu-rep for the UI)
ncu --set full -o rep ./bench
ncu -i rep.ncu-rep --csv --page raw --print-units base --print-metric-name name --print-metric-instances values > rep.csv

# 3) range replay around cudaProfilerStart/Stop (concurrent kernels, whole-range counters)
ncu --replay-mode range --profile-from-start off --metrics <...> --csv --page raw ./bench
```

### 1.3 Replay modes (Profiling Guide §2.2.3)

* **Kernel Replay** (default): "all metrics requested for a specific kernel instance ... are grouped into one or more passes. For the first pass, all GPU memory that can be accessed by the kernel is saved. After the first pass, the subset of memory that is written by the kernel is determined. Before each pass (except the first one), this subset is restored." "If NVIDIA Nsight Compute determines that only a single replay pass is necessary ..., no save-and-restore is performed at all."
* **Application Replay**: "the complete application is run multiple times, so that in each run one of those passes can be collected per kernel." Requires deterministic kernel sequence; "application replay also allows to disable Cache Control". Matching via `--app-replay-match {name,grid,all}` and `--app-replay-mode {strict,balanced,relaxed}`.
* **Range Replay**: "captures and replays complete ranges of CUDA API calls and kernel launches ... Metrics are then not associated with individual kernels but with the entire range. This allows the tool to execute kernels without serialization." Ranges are defined by `cu(da)ProfilerStart`/`cu(da)ProfilerStop` ("This is the default") or `--nvtx --nvtx-include <expr>`. "prefer the CUDA driver API calls cuProfilerStart/Stop. Internally, NVIDIA Nsight Compute only intercepts the CUDA driver API variants and the CUDA runtime API may not trigger these if no CUDA context is active on the calling thread." Only a subset of driver APIs may appear inside the range (`cuLaunchKernel`, `cuLaunchCooperativeKernel`, `cuMemcpy*`, `cuMemset*`, `cuMemAlloc`, stream/event mgmt …; graph APIs, VMM, stream-ordered allocator **not** supported). `--range-replay-options enable-greedy-sync,disable-host-restore,disable-host-save,disable-dependent-kernel-detection`.
* **Application Range Replay** (`app-range`): ranges without API capture, application re-run per pass; "instruction-level SASS metrics do not include profiling data for JIT-compiled kernels".
* For range results, "launch__* metrics are instanced for each launch".

### 1.4 Metric naming grammar (Profiling Guide §2.3.2 "Metrics Structure")

```
Unit-Level Counter : unit__(subunit?)_(pipestage?)_quantity_(qualifiers?)
Interface Counter  : unit__(subunit?)_(pipestage?)_(interface)_quantity_(qualifiers?)   # interface = sender2receiver, e.g. xbar2l1tex, l1tex2xbar, lts2xbar
Unit Metric        : (counter_name).(rollup_metric)            # rollup = sum | avg | min | max  ("aggregate across all instances of a HW unit (e.g., all SMs, all SMSPs, all L2 slices, etc.)")
Sub-Metric         : (counter_name).(rollup_metric).(submetric)
```

Counter sub-metrics (verbatim list): `.peak_sustained`, `.peak_sustained_active`, `.peak_sustained_active.per_second`, `.peak_sustained_elapsed`, `.peak_sustained_elapsed.per_second`, `.per_second` ("the number of operations per second"), `.per_cycle_active` ("per unit active cycle"), `.per_cycle_elapsed` ("per unit elapsed cycle"), `.pct_of_peak_sustained_active`, `.pct_of_peak_sustained_elapsed`. Deprecated/removed: `.peak_burst`, `.pct_of_peak_burst_*`, `*_region`, `*_frame`, `.per_cycle_in_region/frame`.
Ratios: `.pct`, `.ratio`, `.max_rate`. Throughputs: `.pct_of_peak_sustained_active`, `.pct_of_peak_sustained_elapsed`; "You can collect `breakdown:<throughput-metric>` to collect a throughput's breakdown metrics." "Percentages of sustained rate can occasionally exceed 100% in edge cases."
`sass_` in a name = "SASS metrics collected with SW-patching" (extra pass, instruction instrumentation).
Cycle metrics: "`unit__cycles_elapsed`: The number of cycles within a range. The cycles' DimUnits are specific to the unit's clock domain. `unit__cycles_active`: ... where the unit was processing data. `unit__cycles_stalled` ... `unit__cycles_idle`". Interface: `unit__(interface)_active` / `_stalled`.
Pipe-stage letters in `l1tex`/`lts` names: "l1tex__t refers to its Tag stage. l1tex__m refers to its Miss stage. l1tex__d refers to its Data stage." "lts__t refers to its Tag stage. lts__m refers to its Miss stage. lts__d refers to its Data stage." So `lts__t_sectors*` = sector lookups at the L2 tag stage (hits+misses, what clients requested), `lts__d_sectors*` = data-bank accesses/fills (`lts__d_sectors_fill_device` = "# of sectors filled from device memory").

Hardware units (Profiling Guide §2.3.4, verbatim excerpts; counts of gh100 base metrics per prefix from the query in parentheses):

| Unit | Meaning |
|---|---|
| `sm` (291) | Streaming Multiprocessor. |
| `smsp` (450) | "Each SM is partitioned into four processing blocks, called SM sub partitions ... A sub partition manages a fixed size pool of warps." |
| `tpc` (23) | "Thread Processing Clusters are units in the GPC. They contain one or more SM, Texture and L1 units, the Instruction Cache (ICC) and the Indexed Constant Cache (IDC)." |
| `gpc` (8) | "contains SM, Texture and L1 in the form of TPC(s). It is replicated several times across a chip." Includes `gpc__cgas_launched/active` (cluster = CGA). |
| `l1tex` (447) | "The Level 1 (L1)/Texture Cache is located within the GPC. It can be used as directed-mapped shared memory and/or store global, local and texture data in its cache portion." |
| `lts` (2167) | "A Level 2 (L2) Cache Slice is a sub-partition of the Level 2 cache." |
| `ltc` | "The Level 2 cache." (no `ltc__` counters on gh100; appears as `srcunit_ltcfabric` qualifier) |
| `ltcfabric` | "The LTC fabric is the communication fabric for the L2 cache partitions." |
| `lrc` (39) | "L2 Request Coalescer ... coalesce read requests before forwarding them to the L2 cache. It also serves programmatic multicast requests from the SM and supports compression for writes." (H100 uses `lrc__*ilc*` for L2 compression metrics in the chart section, filter CC_90.) |
| `dram` (13) | "Device (main) memory". `fbpa` (13): "FrameBuffer Partition is a memory controller which sits between the level 2 cache (LTC) and the DRAM." `dramc__*` are "DEPRECATED: use dram__…". |
| `gpu` (24) | "The entire Graphics Processing Unit." (`gpu__time_duration`, `gpu__cycles_elapsed`, `gpu__dram_throughput`, `gpu__compute_memory_*_throughput`). |
| `sys` (4) | "Logical grouping of several units." |
| `gcc`, `idc`, `imc`, `icc`, `dcc` | Constant caches (GPC L1.5, indexed, immediate, instruction, data). |
| `fe`, `gr`, `pcie`, `ctc`, `nvlrx`/`nvltx` | Front end, graphics engine, PCIe, C2C, NVLink. |

Subunits/qualifiers: `aperture_device` ("Memory interface to local device memory (dram)"), `aperture_peer`, `aperture_sysmem`, `global`, `local`, `lg` ("Local/Global memory"), `shared`, `surface`, `texture`, `lsu` ("Load/Store unit"), `lsuin`, `mio`, `mioc`, `texin`, `xbar` ("The Crossbar (XBAR) is responsible for carrying packets from a given source unit to a specific destination unit"), `workid` (cluster launch control). **`dshared` is the qualifier for distributed shared memory** (it is not in the decoder table, but every `mem_dshared` metric description reads "distributed shared").
Pipelines: `lsu` ("issues load, store, atomic, and reduction instructions to the L1TEX unit for global, local, and shared memory. It also issues special register reads (S2R), shuffles, and CTA-level arrive/wait barrier instructions"), `tma` ("Tensor Memory Accelerator ... efficient data transfer mechanisms between global and shared memories with the ability to understand and traverse multidimensional data layouts"), `adu`, `alu`, `cbu`, `fma`, `fp64`, `tensor`, `tex`, `uniform`, `xu`.
Quantities: `request` ("A command into a HW unit ... Each request accesses one or more sectors"), `sector` ("Aligned 32 byte-chunk ... An L1 or L2 cache line is four sectors, i.e. 128 bytes"), `tag`, `wavefront` ("the maximum unit that can pass through that pipeline stage per cycle ... Wavefronts therefore represent the number of cycles required to process the requests"). "On SM 7.0 (Volta) and newer architectures, each shared memory instruction generates exactly one request."

---

## 2. Metrics per memory level on GH100

Legend: **C** = confirmed present for `--chip gh100` in `ncu --query-metrics` (NVIDIA description quoted). Use with `.sum` (totals), `.avg` (per-unit-instance), `.sum.per_second` (bandwidth), `.avg.pct_of_peak_sustained_elapsed` (utilisation), `.avg.per_cycle_active`.

### 2.1 Shared memory (SMEM)

| Metric (base name) | NVIDIA description | Unit | Experiment |
|---|---|---|---|
| `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld` **C** | "# of shared memory wavefronts processed by Data-Stage for LDS, LD, 3D …" | wavefront | SMEM load throughput / bank-conflict serialisation (cycles ≈ wavefronts) |
| `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st` **C** | "… for STS, ST, 3D …" | wavefront | SMEM store |
| `l1tex__data_pipe_lsu_wavefronts_mem_shared` **C** (+ `_op_atom`, `_op_gmma`, `_op_ipa`, `_op_misc`, `_cmd_read`, `_cmd_write`) | all shared wavefronts | wavefront | includes `op_gmma` (wgmma operand reads) and `op_ldgsts` paths |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld` **C** | "# of shared memory data bank conflicts generated by LDS, LD, 3D …" | — | bank-conflict sweep |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st` **C**, `..._op_atom`, `..._op_gmma`, `..._op_ldgsts`, `..._op_misc`, `l1tex__data_bank_conflicts_pipe_lsu_mem_shared` **C** | | | |
| `smsp__inst_executed_op_shared_ld` **C** / `_st` **C** / `_atom` / `_stsm` | "# of warp instructions executed: LDS" / "STS" | inst | normalise per instruction |
| `smsp__inst_executed_op_ldsm` **C**, `sm__sass_inst_executed_op_ldsm` **C** | "# of warp instructions executed: LDSM" | inst | ldmatrix |
| `sm__sass_data_bytes_mem_shared` **C** (+ `_op_ld`, `_op_st`, `_op_atom`, `_op_ldsm`, `_op_stsm`, `_op_gmma`, `_op_ldgsts`, `_op_ldgsts_cache_access/_bypass`) | "# of shared memory bytes required for LDS, LD, STS, ST, ATOMS, ATOM, …" | byte | SMEM bytes (SASS-patched; extra pass) |
| `sm__sass_inst_executed_op_shared_ld` **C** / `_st` **C** / `_shared` / `_shared_gmma` | "# of warp instructions executed: LDS, LD" | inst | |
| `l1tex__throughput` **C** (`.avg.pct_of_peak_sustained_elapsed`) | "L1TEX Throughput" (max of constituents; `breakdown:l1tex__throughput`) | % | SoL |
| `l1tex__lsu_writeback_active_mem_shared` **C** | writeback cycles | cycle | |
| Source-level: `memory_l1_wavefronts_shared`, `memory_l1_wavefronts_shared_ideal`, `derived__memory_l1_conflicts_shared_nway` ("Average N-way conflict in L1 per shared memory instruction … memory_l1_wavefronts_shared / inst_executed"), `derived__memory_l1_wavefronts_shared_excessive` **C** (`--section SourceCounters`, instanced per PC) | | | per-instruction conflict degree |
| `sass__inst_executed_shared_loads` / `_stores` **C** (source collection) | | inst | |

Table group: `--metrics group:memory__shared_table` ("Group of metrics for the shared memory workload analysis table"), also `--section MemoryWorkloadAnalysis_Tables`.

### 2.2 Distributed shared memory (DSMEM, cluster SM-to-SM)

NCU's qualifier is **`mem_dshared`** ("distributed shared"); the instruction-level family is **`op_dshared`**. All of the following are **C** (gh100):

| Metric | NVIDIA description | Unit | Experiment |
|---|---|---|---|
| `sm__sass_inst_executed_op_dshared` / `smsp__sass_inst_executed_op_dshared` | "# of warp instructions executed, accessing distributed shared: ATOM, …" | inst | DSMEM instruction count (SASS-patched) |
| `sm__sass_inst_executed_op_dshared_ld`, `_st`, `_atom`, `_redas`, `_stas`, `_tma_st`, `_tma_red` (and `smsp__` variants) | "… accessing distributed shared: LD" / "ST" / ATOM / REDAS / STAS / TMA store / TMA reduction | inst | split by primitive (generic `LD.E/ST.E` to peer SMEM, `STAS`/`REDAS` = st.async/red.async, `UTMASTG`/`UBLKCP` to `shared::cluster`) |
| `l1tex__t_requests_pipe_lsu_mem_dshared` (+`_op_ld`, `_op_st`, `_op_atom`, `_op_redas`) | "# of requests sent to T-Stage for distributed shared operations" | request | |
| `l1tex__t_sectors_pipe_lsu_mem_dshared` (+`_op_ld/_st/_atom/_redas`, `_lookup_miss`) | "# of sectors requested for distributed shared" | sector | DSMEM bytes = sectors×32 |
| `l1tex__t_bytes_pipe_lsu_mem_dshared` (+op variants) | "# of bytes requested for distributed shared" | byte | |
| `l1tex__t_set_accesses_pipe_lsu_mem_dshared`, `l1tex__t_set_conflicts_pipe_lsu_mem_dshared` (+op) | set accesses/conflicts | — | |
| `l1tex__t_output_wavefronts_pipe_lsu_mem_dshared` (+op) | wavefronts | wavefront | |
| `l1tex__average_t_sectors_per_request_pipe_lsu_mem_dshared` (+op) | sectors/request ratio | ratio | coalescing of DSMEM accesses (Hopper tuning guide: align to 32 B segments) |
| `l1tex__m_l1tex2xbar_write_bytes_mem_dshared` (+`_op_ld`, `_op_st`, `_op_atom`, `_op_redas`, `_op_tma_st`, `_op_tma_red`), `l1tex__m_l1tex2xbar_write_sectors_mem_dshared*` | "# of bytes written from L1tex M-Stage for distributed shared …" — **this is the SM→XBAR (outbound SM-to-SM) traffic** (note: DSMEM *loads* also count here: "_op_ld … for distributed shared loads" = request packets going out) | byte | SM-to-SM network bandwidth |
| `l1tex__m_xbar2l1tex_read_bytes_mem_dshared` (+`_op_ld`, `_op_st`, `_op_atom`, `_op_redas`, `_op_tma_st`, `_op_tma_red`), `l1tex__m_xbar2l1tex_read_sectors_mem_dshared*` | "# of bytes read into L1TEX M-Stage for distributed shared …" — **XBAR→SM (inbound)** | byte | SM-to-SM network bandwidth |
| `l1tex__m_l1tex2xbar_throughput_pipe_lsu_mem_dshared`, `l1tex__m_xbar2l1tex_throughput_pipe_lsu_mem_dshared` | "L1TEX M-Stage to XBAR throughput for LSU pipe distributed shared" / "XBAR to M-Stage …" | % of peak | utilisation of the SM-to-SM links |
| `l1tex__m_l1tex2xbar_req_cycles_active_mem_dshared`, `..._mem_dshared_loopback` | request-interface active cycles (loopback = own SM) | cycle | separate self-SM vs remote-SM traffic |
| Launch/cluster: `launch__cluster_size`, `launch__cluster_dim_x/y/z`, `launch__cluster_max_active` ("Maximum number of clusters that can co-exist on the target device"), `launch__cluster_max_potential_size`, `launch__cluster_scheduling_policy`, `launch__occupancy_cluster_pct`, `launch__occupancy_cluster_gpu_pct`, `launch__occupancy_per_cluster_size`, `launch__preferred_cluster_size` (all **C** in `--query-metrics-collection launch`) ; `gpc__cgas_launched` ("# of CGAs launched"), `gpc__cgas_active` ("cumulative # of CGAs in flight") **C** | | | cluster placement / occupancy |

Not found for gh100: any `*_op_shared_cluster*` or `*dsmem*` names (the naming is `dshared`). DSMEM traffic goes through the L1TEX M-stage ⇄ XBAR interfaces, so it is **not** visible in `lts__*` (unless you misroute to global).

### 2.3 L1 / TEX (global/local through L1)

| Metric | NVIDIA description | Unit | Experiment |
|---|---|---|---|
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld` **C** / `_st` **C** | "# of sectors requested for global loads" / stores | sector | L1 traffic (bytes = ×32) |
| `l1tex__t_requests_pipe_lsu_mem_global_op_ld` **C** / `_st` | "# of requests sent to T-Stage for global loads" | request | sectors/request = coalescing |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit` **C** / `_lookup_miss` **C** | "# of sector requests to T-Stage with tag-hit and data-hit for global loads" / "that missed" | sector | L1 hit/miss |
| `l1tex__t_sector_hit_rate` **C** (`.pct`) ; `l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate` **C** (`.pct`) | "# of sector hits per sector"; "for global loads" | % | L1 hit rate ("Not to be confused with Tag Hit Rate") |
| `l1tex__t_sectors` (sum of all), `l1tex__t_requests_*`, `l1tex__t_bytes_*` | | | |
| `l1tex__m_xbar2l1tex_read_bytes` **C** | "# of bytes read into L1TEX M-Stage" (all sources: L2 returns + DSMEM) | byte | L2→SM return bandwidth |
| `l1tex__m_xbar2l1tex_read_bytes_mem_lg_op_ld` **C** | "# of bytes read from L2 into L1TEX M-Stage for local/global loads" (**there is no `_mem_global_op_ld` variant on gh100; use `mem_lg`**) | byte | L1-miss fill bandwidth |
| `l1tex__m_xbar2l1tex_read_bytes_mem_global_op_atom` **C**, `..._mem_global_op_ldgsts_cache_access` **C** ("for LDGSTS.ACCESS"), `..._mem_global_op_tma_ld` **C**, `..._mem_surface_*`, `..._mem_texture`, `..._pipe_lsu`, `..._pipe_tex`, `..._pipe_tma` **C**, `..._realtime` | per-source breakdown of XBAR→L1 bytes | byte | |
| `l1tex__m_l1tex2xbar_write_bytes` **C** | "# of bytes written from L1tex M-Stage" | byte | SM→L2 write/atomic/request bandwidth |
| `l1tex__m_l1tex2xbar_write_bytes_mem_lg_op_st` **C** ("to L2 for local/global stores"), `..._mem_global_op_atom`, `..._mem_global_op_red`, `..._mem_global_op_tma_st` **C**, `..._mem_global_op_tma_red` **C**, `..._pipe_lsu`, `..._pipe_tma` **C** | | byte | |
| `l1tex__throughput` **C**, `l1tex__cycles_active` **C**, `l1tex__cycles_elapsed` **C** | | | |
| `smsp__inst_executed_op_global_ld/_st/_red` **C**, `sm__sass_inst_executed_op_global_ld` ("LDG, LD") **C**, `sm__sass_inst_executed_op_ldgsts` / `_cache_access` / `_cache_bypass` **C**, `smsp__inst_executed_op_ldgsts` **C** | | inst | cp.async paths |
| `sm__sass_inst_executed_op_memory_8b/16b/32b/64b/128b` **C** | by access width | inst | |
| Source-level: `memory_l1_tag_requests_global`, `memory_l2_theoretical_sectors_global(_ideal)`, `derived__memory_l2_theoretical_sectors_global_excessive`, `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio` (table "Average Bytes Per Sector For Global Loads") | | | |
| Stall link: `smsp__pcsamp_warps_issue_stalled_long_scoreboard` = "waiting for a scoreboard dependency on a L1TEX (local, global, surface, texture) operation" | | | L1/L2/DRAM latency attribution |

Table group: `group:memory__first_level_cache_table`.

### 2.4 L2 (`lts` = L2 slice; `t` = tag stage, `d` = data stage)

| Metric | NVIDIA description | Unit | Experiment |
|---|---|---|---|
| `lts__t_sectors` **C** | "# of LTS sectors" (all clients) | sector | L2 traffic |
| `lts__t_sectors_op_read` **C** / `_op_write` **C** / `_op_atom` / `_op_red` / `_op_membar` | "# of LTS sectors for reads" … | sector | |
| `lts__t_sector_hit_rate` **C** (`.pct`) | "proportion of L2 sector lookups that hit" | % | L2 hit rate |
| `lts__t_sector_op_read_hit_rate` **C**, `lts__t_sector_op_write_hit_rate` | "# of sector hits for _op_read per sector access for _op_read" | % | |
| `lts__t_sectors_srcunit_tex` **C**, `..._op_read` **C**, `..._op_write` **C** | "# of LTS sectors from unit L1TEX (for reads/writes)" — "srcunit_tex" = traffic from the SMs' L1TEX | sector | SM-originated L2 traffic |
| `lts__t_sectors_srcunit_tex_op_read_lookup_hit` **C** / `_lookup_miss` **C** | "# of LTS sectors from unit L1TEX for reads that hit/missed" | sector | L2 hit/miss for SM loads |
| `lts__t_requests_srcunit_tex` **C** (+ op variants) | "# of LTS requests from unit L1TEX" | request | sectors/request |
| `lts__t_sectors_srcunit_l1` **C**, `lts__t_requests_srcunit_l1`, `lts__t_sectors_srcunit_gcc` | other source units present on gh100 (`l1`, `gcc`, `ltcfabric`, `tex`) | | |
| `lts__t_sectors_srcunit_ltcfabric` **C** | "# of LTS sectors from LTC Fabric" — **traffic crossing the fabric between the two L2 partitions** (H100 has a split L2; the L2 table has an "L2 Fabric Total" row "only shown for kernel launches on CUDA devices with L2 fabric"). Qualified variants exist: `_op_read/_op_write/_op_membar`, `_aperture_device/_peer/_sysmem`, `_evict_first/last/normal(_demote)`, `_lookup_hit/_lookup_miss`. `lts__t_requests_srcunit_ltcfabric` **C**. | sector | near vs far L2 partition experiments |
| `lts__ltcfabric2lts_cycles_active`, `lts__ltcfabric2lts_cycles_stalled`, `lts__ltcfabric2ltc_cycles_active` **C** | fabric interface cycles | cycle | fabric saturation |
| `lts__t_sectors_aperture_device` **C** / `_aperture_sysmem` **C** / `_aperture_peer` **C** | "# of LTS sectors accessing device memory (vidmem)" / "system memory (sysmem)" / "peer memory (peermem)" (+ `_lookup_hit/_miss`, op, evict variants) | sector | which backing memory |
| `lts__d_sectors_fill_device` **C** / `lts__d_sectors_fill_sysmem` **C** | "# of sectors filled from device memory" / system memory (data-stage fills = L2 misses actually fetched) | sector | DRAM-fill count |
| `lts__d_sectors` **C** | "# of sectors accessed in data banks" | sector | |
| `lts__t_bytes` **C** (`.sum.per_second`) | "# of bytes requested" | byte | L2 bandwidth (table column "Throughput") |
| `lts__throughput` **C** (`.avg.pct_of_peak_sustained_elapsed`) | "LTS throughput" | % | L2 SoL |
| `lts__t_sectors_evict_first/_last/_normal` **C** (+`_lookup_hit/_miss`) | eviction-policy breakdown (`ld.global.L2::evict_last`, `createpolicy`) | sector | persisting-L2 experiments; table `group:memory__l2_cache_evict_policy_table` |
| `lts__lts2xbar_cycles_active` **C**, `lts__xbar2lts_cycles_active` **C` | "# of cycles where interface from LTS to XBAR was active" / "from XBAR to LTS" | cycle | |
| `lts__cycles_elapsed` **C**, `lts__cycles_active` **C** | "# of cycles elapsed on LTS" | cycle | L2 clock (see §2.7) |
| `lts__t_sectors_lookup_miss`, `lts__t_sectors_lookup_miss_data_promoted` (header "L2 Sector Misses from Promotion"), `lrc__ilc_input_sectors`, `lrc__average_ilc_input_sector_success_rate` (L2 compression on CC_90 per chart section file) | | | |
| `device__attribute_l2s_count` **C** (device collection) | "Total number of Level 2 cache slices." ; `device__attribute_num_l2s_per_fbp` "Number of Level 2 cache slices per frame buffer partition." | — | **number of `lts` instances on your H100**: `ncu --query-metrics-collection device` on device, or read it from `--page raw`. |

Table group: `group:memory__l2_cache_table`.

### 2.5 DRAM / HBM

| Metric | NVIDIA description | Unit | Experiment |
|---|---|---|---|
| `dram__bytes_read` **C** / `dram__bytes_write` **C** / `dram__bytes` **C** | "# of bytes read from DRAM" / written / accessed | byte | HBM traffic; `.sum.per_second` = achieved HBM bandwidth (chart label "DRAM Bandwidth" = `dram__bytes.sum.per_second`) |
| `dram__sectors_read` / `_write` / `dram__sectors` **C** | sectors | sector | |
| `dram__throughput` **C** (`.avg.pct_of_peak_sustained_elapsed`) ; `gpu__dram_throughput` **C** (used by the Tables section as "DRAM Throughput" on CC_89–100) | "DRAM throughput" | % | HBM SoL |
| `dram__cycles_elapsed` **C** | "# of elapsed DRAM memory clock cycles" | cycle | `dram__cycles_elapsed.avg.per_second` = DRAM clock domain frequency |
| `dram__cycles_active` **C** (+`_read`, `_write`) | "# of cycles where DRAM was active" | cycle | |
| `fbpa__dram_read_bytes` / `fbpa__dram_write_bytes` / `fbpa__dram_sectors` / `fbpa__dram_read_sectors` / `fbpa__dram_write_sectors` **C** | "# of DRAM read bytes" (per memory controller) | byte | per-FBPA imbalance (`--print-metric-instances values`) |
| `fbpa__cycles_elapsed`, `fbpa__dram_cycles_elapsed`, `fbpa__throughput`, `fbpa__dram_read_throughput`, `fbpa__dram_write_throughput` **C** | | | |

Table group: `group:memory__dram_table`.

### 2.6 TMA (Tensor Memory Accelerator) and bulk-async copies

All **C** for gh100:

| Metric | NVIDIA description | Unit | Experiment |
|---|---|---|---|
| `sm__sass_inst_executed_op_tma_ld` / `smsp__sass_inst_executed_op_tma_ld` | "# of warp instructions executed: UTMALDG, UBLKCP.S.G" | inst | counts both tensor-map loads and non-tensor bulk copies G→S |
| `sm__sass_inst_executed_op_tma_st` | "# of warp instructions executed: UTMASTG, UBLKCP.G.S" | inst | |
| `sm__sass_inst_executed_op_tma_red`, `sm__sass_inst_executed_op_tma` | TMA reductions / all TMA | inst | |
| `smsp__inst_executed_op_tma_ld` / `_st` (+ `_pred_on_any`, `_pred_off_all`), `smsp__inst_executed_pipe_tma`, `sm__inst_executed_pipe_tma` | "# of warp instructions executed by tma pipe" (HW counter, no SASS patching) | inst | |
| `sm__pipe_tma_cycles_active`, `smsp__pipe_tma_cycles_active` | "# of cycles where TMA pipe was active" | cycle | TMA issue-side occupancy |
| `l1tex__tmain_requests`, `smsp__l1tex_tmain_requests` | "# of TMA instructions sent to TMA" | inst | |
| `l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld` | "# of bytes read from L2 into L1TEX M-Stage for global TMA loads" | byte | **TMA load bandwidth** (this is the `l1tex__m_xbar2l1tex_read_bytes_mem_global_op_*` family FlashGPU-sim cites for the per-SM TMA read limit; measure `.avg.per_cycle_active` / `.max.per_cycle_elapsed` to get B/clk/SM) |
| `l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld_dest_self` / `_dest_multicast` | "… for global TMA self" / "global TMA multicast" | byte | `.multicast::cluster` vs unicast |
| `l1tex__m_l1tex2xbar_read_requests_mem_global_op_tma_ld_dest_multicast` (+`_size_32B/64B/96B/128B`, `_replayed`), `l1tex__average_m_xbar2l1tex_read_bytes_mem_global_op_tma_ld_dest_multicast_data_valid` | multicast request stats | request | multicast efficiency |
| `l1tex__m_xbar2l1tex_read_bytes_pipe_tma`, `l1tex__m_xbar2l1tex_read_sectors_pipe_tma` | "# of bytes read into L1TEX M-Stage for TMA pipe" | byte | |
| `l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_st` / `_tma_red`, `l1tex__m_l1tex2xbar_write_bytes_pipe_tma`, `..._write_sectors_*` | "# of bytes written from L1tex M-Stage for global TMA stores/reductions" | byte | TMA store bandwidth |
| `l1tex__m_l1tex2xbar_req_cycles_active_op_tma` | TMA request-interface active cycles | cycle | |
| `l1tex__m_xbar2l1tex_read_bytes_mem_dshared_op_tma_st` / `_tma_red`, `l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_tma_st` / `_tma_red`, `sm__sass_inst_executed_op_dshared_tma_st` / `_tma_red` | TMA/bulk copies whose destination is **another CTA's SMEM in the cluster** (`cp.async.bulk.shared::cluster.shared::cta`, `cp.reduce.async.bulk` to `shared::cluster`) | byte/inst | DSMEM via TMA engine |
| `sm__sass_data_bytes_mem_shared_op_ldgsts(_cache_access/_cache_bypass)`, `l1tex__m_xbar2l1tex_read_bytes_mem_global_op_ldgsts_cache_access`, `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts` | the Ampere-style `cp.async` (LDGSTS) path, for comparison | | |

Nothing named `utma*`/`bulk*` exists in the metric set; TMA = `tma`, bulk copies are folded into `op_tma_ld/st`. The Profiling Guide pipeline entry: "**tma** — Tensor Memory Accelerator. (Tensor Memory Access Unit) Provides efficient data transfer mechanisms between global and shared memories with the ability to understand and traverse multidimensional data layouts."

### 2.7 Timing and clocks

| Metric | NVIDIA description | Unit | Use |
|---|---|---|---|
| `gpu__time_duration` **C** (`.sum`) | "equals to gpu__time_duration_measured_user if collectable, otherwise equals to gpu__time_duration_measured_wallclock" ; `gpu__time_duration_measured_user` = "context-aware time duration in nanoseconds", `gpu__time_duration_measured_wallclock` = "wall-clock duration in nanoseconds"; `gpu__time_active`, `gpu__time_start`, `gpu__time_end` | ns | kernel duration |
| `sm__cycles_elapsed` **C** (`.avg`, `.max`) | "# of cycles elapsed on SM" | cycle | kernel duration in SM clocks; `.avg.per_second` = **SM clock (Hz) actually used during the run** — record it in every CSV |
| `sm__cycles_active` **C** / `smsp__cycles_active` **C** / `smsp__cycles_elapsed` | "# of cycles with at least one warp in flight" | cycle | active vs elapsed; per-SM via instances |
| `gpc__cycles_elapsed` **C** (`.max`) | "# of cycles elapsed on GPC" | cycle | used by SoL as the elapsed-cycle base |
| `gpu__cycles_elapsed` / `gpu__cycles_active` **C** | | cycle | |
| `lts__cycles_elapsed` **C** (`.avg.per_second`) | "# of cycles elapsed on LTS" — cycles are "in the unit's clock domain" | cycle | **L2 clock**; on GA100 the LTS domain differs from the SM clock — compare `lts__cycles_elapsed.avg.per_second` with `sm__cycles_elapsed.avg.per_second` on your H100 to see whether they match (to verify, §9) |
| `dram__cycles_elapsed` **C** (`.avg.per_second`) | "# of elapsed DRAM memory clock cycles" | cycle | DRAM clock domain |
| `l1tex__cycles_elapsed`, `fbpa__cycles_elapsed`, `fbpa__dram_cycles_elapsed` **C** | | cycle | |
| `sm__throughput` **C** | "SM throughput assuming ideal load balancing across SMSPs" | % | |
| `launch__grid_size`, `launch__block_size`, `launch__sm_count` ("Number of SMs utilized in the launch"), `launch__waves_per_multiprocessor`, `launch__shared_mem_per_block_dynamic/_static/_allocated/_driver`, `launch__shared_mem_config_size`, `launch__registers_per_thread`, `launch__occupancy_limit_*`, `launch__func_cache_config` (all **C**, `launch` collection; "collected per kernel launch, and do not require an additional replay pass") | | | provenance columns in CSV |
| `profiler__replayer_passes`, `profiler__replayer_passes_type_warmup`, `profiler__replayer_bytes_mem_backed_up.*` (`stats` collection) | passes used | pass | sanity-check single-pass runs |

### 2.8 Stall reasons / warp-state sampling / instruction counts

* Sampled (statistical) stall reasons — `--query-metrics-collection warpsampling`, instanced "mapping from the function address (uint64) to the number of samples (uint64)": `smsp__pcsamp_warps_issue_stalled_{barrier, branch_resolving, dispatch_stall, drain, imc_miss, lg_throttle, long_scoreboard, math_pipe_throttle, membar, mio_throttle, misc, no_instructions, not_selected, selected, short_scoreboard, sleeping, tex_throttle, wait, warpgroup_arrive}` and the `_not_issued` twins ("Warp sampling by default now focuses on the Not Issued (_not_issued) variants", 2025.4). `smsp__pcsamp_sample_count`, `smsp__pcsamp_interval_cycles`, `smsp__pcsamp_buffer_overflow` (stats). Groups: `group:smsp__pcsamp_warp_stall_reasons`, `group:smsp__pcsamp_warp_stall_reasons_not_issued`, `group:smsp__warpidsamp_warp_stall_reasons` (per warp ID). Collected by `--section SourceCounters` / `WarpStateStats`; interval via `--warp-sampling-interval`.
  Definitions (Profiling Guide §2.4.7): `barrier` "waiting for sibling warps at a CTA barrier"; `membar` "waiting on a memory barrier"; `long_scoreboard` "scoreboard dependency on a L1TEX (local, global, surface, texture) operation"; `short_scoreboard` "scoreboard dependency on a MIO (memory input/output) operation (not to L1TEX) ... typically memory operations to shared memory"; `sleeping` "all threads in the warp being in the blocked, yielded, or sleep state ... NANOSLEEP"; `wait` "fixed latency execution dependency"; `warpgroup_arrive` "waiting on a WARPGROUP.ARRIVES or WARPGROUP.WAIT instruction"; `lg_throttle` "L1 instruction queue for local and global (LG) memory operations ... not full"; `mio_throttle` "MIO instruction queue"; `selected` "Warp was selected by the micro scheduler and issued an instruction". Note: cluster barrier (`UCGABAR_WAIT`) and mbarrier `SYNCS.PHASECHK` waits have no dedicated reason — expect them under `barrier`/`long_scoreboard`/`short_scoreboard`/`misc` (verify, §9).
* Averaged (HW-counter) stall metrics, no sampling, `--section WarpStateStats`: `smsp__average_warps_issue_stalled_<reason>_per_issue_active` **C** (`.ratio`; "average # of warps resident per issue cycle, waiting for a scoreboard dependency …"), `smsp__warp_issue_stalled_<reason>_per_warp_active` **C** (`.pct`), `smsp__warps_issue_stalled_<reason>` **C** (cumulative). Reasons: `barrier`, `membar`, `long_scoreboard`, `short_scoreboard`, `sleeping`, `wait`, `mio_throttle`, `lg_throttle`, `math_pipe_throttle`, `imc_miss`, `no_instruction`, `not_selected`, `selected`, `drain`, `dispatch_stall`, `branch_resolving`, `misc`, … (query `regex:^smsp__average_warps_issue_stalled_.*_per_issue_active$`). Also `smsp__average_warp_latency_per_inst_issued`, `smsp__warps_eligible`, `smsp__issue_active`.
* Instruction counts: `smsp__inst_executed` **C** / `sm__inst_executed` ("# of warp instructions executed", HW), `smsp__sass_inst_executed` **C** (SASS-patched), `smsp__thread_inst_executed`, per-opcode families `smsp__inst_executed_op_{branch, generic_ld/st/atom, global_ld/st/red, ldc, ldgsts, ldsm, local_ld/st, redas, shared_ld/st/atom/stsm, stas, surface_*, texture, tma_ld, tma_st}` (+`_pred_on_any`/`_pred_off_all`), `sm__sass_inst_executed_op_*` (see §2.1–2.6).
* SASS opcode breakdown (source collection, instanced "mapping from the SASS opcode or opcode category (string) to the number of executions"): `sass__inst_executed_per_opcode` **C**, `sass__inst_executed_per_opcode_category`, `sass__inst_executed_per_opcode_pipeline`, `sass__inst_executed_per_opcode_with_modifier_all` / `_selective`, `sass__thread_inst_executed_true_per_opcode`. Collect with `--section InstructionStats` (or `--metrics sass__inst_executed_per_opcode`) and print with `--print-metric-instances details` to get the opcode→count map in CSV.

### 2.9 Per-SM / per-slice / per-PC values

* Instanced metrics: "An instance can be a streaming multiprocessor, an assembly source line, etc." `--print-metric-instances values` prints `aggregate; v0; v1; …`; `details` adds correlation IDs (SM id, L2 slice id, PC/function address, opcode string). CSV: "The raw page csv output now includes metric instance values when these enabled for printing."
* Use `.sum` + instances for per-unit totals; `.avg/.min/.max` give the spread across instances without instance printing (e.g. `lts__t_sectors.max` vs `.avg` = slice imbalance; `sm__cycles_active.max` vs `.avg` = SM imbalance).
* Per-SM shared-memory wavefronts: `l1tex__data_pipe_lsu_wavefronts_mem_shared.sum` with instances (one `l1tex` per SM on GH100). Per-L2-slice: `lts__t_sectors_*` instances (count = `device__attribute_l2s_count`). Per-FBPA: `fbpa__dram_read_bytes`.
* `--print-details all` only affects the details page (prints body metrics of sections); it is not the per-instance switch.

---

## 3. Nsight Systems (`nsys`) — brief

Docs: https://docs.nvidia.com/nsight-systems/UserGuide/index.html, https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html (web summaries; confirm flag spelling with `nsys profile --help` / `nsys stats --help-reports`).

* `nsys profile -t cuda,nvtx --stats=true -o run ./bench` — `-t/--trace` "default set is cuda, opengl, nvtx, and osrt"; `--stats=true` prints the summary tables after collection. Restrict to a window: `--capture-range=cudaProfilerApi --capture-range-end=stop` (profile only between `cudaProfilerStart()`/`cudaProfilerStop()`). GPU HW metrics sampling: `--gpu-metrics-device=all` (Turing+, root/admin), `--gpu-metrics-frequency`.
* `nsys stats --report cuda_gpu_kern_sum --format csv --output . run.nsys-rep` → `run_cuda_gpu_kern_sum.csv`; other reports: `cuda_api_sum`, `cuda_gpu_mem_size_sum`, `cuda_api_gpu_sum` (combination of the three), `nvtx_sum`, `cuda_gpu_kern_sum:base` (summarise by base kernel name). "Time" column is a share of the summed Total Time, not wall-clock. Also `nsys export --type sqlite` for custom queries.
* When to use nsys instead of ncu: launch-overhead / back-to-back kernel gaps, stream concurrency, CPU→GPU latency, CUDA-graph timing, anything where ncu's serialisation, cache flush, clock lock and replay would distort the measurement. nsys records start/end timestamps per kernel with no replay (duration resolution ~ns timestamps, but no HW counters except the optional GPU-metrics sampler). ncu for counters; nsys for timelines.

---

## 4. Binary tools and SASS for the Hopper primitives

Docs: https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html (cuobjdump, nvdisasm, Hopper Instruction Set table in §"Instruction Set Reference").

### 4.1 Tools and flags (from `--help` of cuobjdump/nvdisasm 13.4.92, ptxas 12.9.86)

* `cuobjdump -sass` (`--dump-sass`), `-ptx`, `-elf`, `-res-usage` (`--dump-resource-usage`: registers/smem/cmem per kernel), `-fun <name>` (`--function`), `-arch sm_90a` (`--gpu-architecture`), `-lelf/-lptx`, `-xelf/-xptx` (extract cubin/ptx from a fatbin), `-all`, `-sort`.
* `nvdisasm` on a cubin: `-c` (`--print-code`, only sections with code), `-g` (`--print-line-info`, needs `-lineinfo`), `-gi` (inline line info), `-gp` (PTX line info), `-hex` (`--print-instruction-encoding`, flag only, no argument), `-ndf` (`--no-dataflow`), `-plr` (`--print-life-ranges`), `-lrm <mode>` (`--life-range-mode`), `-cfg`/`-bbcfg` (control-flow graph, DOT), `-poff`, `-raw`, `-sf` (`--separate-functions`), `-b SMxy` (`--binary`, raw instruction stream). Note: `nvdisasm` 13.x prints "Disassembling Std Elf to Old format … switching to Old format" on 12.x cubins; harmless.
* `nvcc`: `-lineinfo` (`--generate-line-info`; also a ptxas flag), `-gencode arch=compute_90a,code=sm_90a` (arch-specific `sm_90a` is required for wgmma/setmaxnreg; TMA/mbarrier/cluster/DSMEM assemble for plain `sm_90` too), `-Xptxas -v` / `--ptxas-options=-v` ("Enable verbose mode which prints code generation statistics": registers, smem, spills), `-Xptxas -O3` (`--opt-level <N>`, "Default value: 3"), `-Xptxas -warn-spills` / `-warn-lmem-usage`, `-maxrregcount`, `-Xptxas -regUsageLevel <0..10>`, `--keep` / `-ptx` to inspect PTX, `-cubin` + `cuobjdump -sass`. `ptxas -arch=sm_90a -O3 -v k.ptx -o k.cubin` assembles hand-written PTX without nvcc (how the table below was produced).
* Keeping timing code in order: `asm volatile("..." ::: "memory")` (volatile stops deletion/hoisting of the asm; the `"memory"` clobber stops the compiler reordering memory ops across it — it does **not** stop the hardware; combine with `__syncwarp()`/fences and consume the loaded value in a data dependency). `cuda::ptx::*` (CCCL `<cuda/ptx>`) and `cuda::barrier`, `cuda::memcpy_async`, `cuda::device::experimental::cp_async_bulk*` wrap the same PTX with correct `memory` clobbers; inline `asm volatile` is what you reach for when you need exact instruction sequences (`%clock64`, `mbarrier.try_wait` loops, `mapa`).

### 4.2 PTX → SASS on sm_90a (verified by assembling + disassembling; ✔ = seen in output)

| Primitive (PTX) | SASS (sm_90a) | Status |
|---|---|---|
| `barrier.cluster.arrive[.release][.aligned]` / `barrier.cluster.wait[.acquire][.aligned]` | `UCGABAR_ARV` / `UCGABAR_WAIT` (release/acquire variants additionally emit `MEMBAR.ALL.GPU; ERRBAR; CGAERRBAR` before arrive and `CCTL.IVALL` after wait) | ✔ |
| `mbarrier.init.shared::cta.b64` | `SYNCS.EXCH.64 URZ, [addr], UR` | ✔ |
| `mbarrier.arrive.expect_tx.shared::cta.b64` | `SYNCS.ARRIVE.TRANS64 RZ, [addr], R` | ✔ |
| `mbarrier.try_wait.parity.shared::cta.b64` | `SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [addr], RZ` | ✔ |
| `fence.mbarrier_init.release.cluster` | `MEMBAR.ALL.CTA` (in this test) | ✔ |
| `fence.proxy.async.shared::cta` / `fence.proxy.async.global` | `FENCE.VIEW.ASYNC.S` / `FENCE.VIEW.ASYNC.G` | ✔ |
| `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes` | `UTMALDG.2D [URsmem], [URtmap]` (preceded by `ELECT` so one thread issues) | ✔ |
| `cp.async.bulk.tensor.2d.global.shared::cta.bulk_group` | `UTMASTG.2D` | ✔ |
| `cp.async.bulk.commit_group` / `cp.async.bulk.wait_group[.read] N` | `UTMACMDFLUSH` / `DEPBAR.LE SB0, N` (+ `CCTL.IVALL` after the non-`.read` wait) | ✔ |
| `prefetch.tensormap [p]` | `UTMACCTL.PF [UR]` | ✔ |
| `cp.async.bulk.prefetch.tensor.2d.L2.global` | `UTMAPF.L2.2D` | ✔ |
| `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes` (1-D bulk copy) | `UBLKCP.S.G [smem], [gmem], URsize` | ✔ |
| `cp.async.bulk.global.shared::cta.bulk_group` | `UBLKCP.G.S` | ✔ |
| `cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes` (SMEM→peer SMEM) | `UBLKCP.S.S` | ✔ |
| `cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32` | `UBLKRED.G.S.ADD` | ✔ |
| `cp.async.bulk.prefetch.L2.global` | `UBLKPF.L2` | ✔ |
| `mapa.shared::cluster.u32 r, addr, rank` | folded into uniform-datapath arithmetic on `SR_CgaCtaId`/`SR_SWINHI` (`ULEA`/`UPRMT`/`PRMT`) when the rank is compile-time or uniform; no `MAPA` opcode was emitted in any test (ncu's opcode table contains no `MAPA` either) | ✔ (absence) |
| `ld.shared::cluster.u32` / `st.shared::cluster.u32` / `atom.shared::cluster.add.u32` | generic `LD.E R, [RZ.U32+UR]` / `ST.E` / `ATOM.E.ADD.STRONG.GPU` (DSMEM uses the generic path with a cluster-window address) vs. `LDS`/`STS` for `ld.shared`/`st.shared` | ✔ |
| `st.async...shared::cluster.mbarrier::complete_tx::bytes` / `red.async...shared::cluster...` | `STAS [R.64], R` / `REDAS.ADD [R.64], R` (ncu descriptions: "Asynchronous Store to Distributed Shared Memory With Explicit Synchronization", "Asynchronous Reduction on Distributed Shared Memory With Explicit Synchronization") | ✔ |
| `fence.acq_rel.cluster` / `membar.gl` / `fence.sc.gpu` / `membar.cta` | `MEMBAR.ALL.GPU; ERRBAR; CGAERRBAR; CCTL.IVALL` / same / `MEMBAR.SC.GPU; ERRBAR; CGAERRBAR; CCTL.IVALL` / `MEMBAR.SC.CTA` | ✔ |
| `cp.async.cg.shared.global [s],[g],16` / `cp.async.ca... 4` ; `cp.async.commit_group` ; `cp.async.wait_group 0` | `LDGSTS.E.BYPASS.128` / `LDGSTS.E` ; `LDGDEPBAR` ; `DEPBAR.LE SB0, 0x0` | ✔ |
| `nanosleep.u32 100` | `NANOSLEEP 0x64` | ✔ |
| `mov.u64 r, %clock64` | `CS2R R, SR_CLOCKLO` (64-bit read) ; `mov.u32 r, %clock` → `CS2R.32 R, SR_CLOCKLO` or `S2UR UR, SR_CLOCKLO` | ✔ |
| `mov.u64 r, %globaltimer` | `CS2R R, SR_GLOBALTIMERLO` | ✔ |
| `%cluster_ctarank` | `S2UR UR, SR_CgaCtaId` / `S2R R, SR_CgaCtaId` | ✔ |
| `ld.global.cg` / `ld.global.ca` / `ld.relaxed.gpu` / `ld.acquire.gpu` / `ld.volatile` | `LDG.E.STRONG.GPU` / `LDG.E` / `LDG.E.STRONG.GPU` / `LDG.E.STRONG.GPU`+fence / `LDG.E.STRONG.SYS` | ✔ |
| `discard.global.L2` / `prefetch.global.L2` / `prefetch.global.L1` / `prefetch.global.L2::evict_last` | `CCTL.E.RML2` / `CCTL.E.PF2` / `CCTL.E.PF1` / `CCTL.E.PML2` | ✔ |
| `bar.sync 0` / `bar.arrive 1,64` | `BAR.SYNC.DEFER_BLOCKING 0x0` / `BAR.ARV 0x1, 0x40` | ✔ |
| `elect.sync` | `ELECT P, URZ, PT` | ✔ |
| `ldmatrix` | `LDSM` (emitted name; `@!PT LDS RZ,[RZ]` dummies appear around smem fences) | ✔ |
| `setmaxnreg` / `wgmma.*` / `wgmma.fence` | `USETMAXREG` / `HGMMA`, `IGMMA`, `QGMMA`, `BGMMA` / `WARPGROUP.ARRIVE`, `WARPGROUP.WAIT` | present in ncu's Hopper opcode table (not assembled here) |

Hopper-added opcodes as listed in ncu 2026.2.1's internal table (same list as the CUDA Binary Utilities "Hopper Instruction Set" section): `FENCE LDGMC STSM STAS SYNCS REDAS UCGABAR_ARV UCGABAR_WAIT ULEPC USETMAXREG BGMMA HGMMA IGMMA QGMMA WARPGROUP WARPGROUPSET UBLKCP UBLKPF UBLKRED UTMACCTL UTMACMDFLUSH UTMALDG UTMAPF UTMAREDG UTMASTG ACQBULK CGAERRBAR ENDCOLLECTIVE PREEXIT` (plus inherited `LDGSTS`, `LDGDEPBAR`, `NANOSLEEP`, `CS2R`, `MEMBAR`, `CCTL`, `ERRBAR`, `ELECT`, `DEPBAR`, `BAR`, `LDS/STS/LD/ST/LDG/STG`, `ATOMS/ATOM/ATOMG/RED/REDG`, `S2R/S2UR`). `MAPA`, `S2UR` and `UTMAREDG` were not exercised; `UTMAREDG` is the TMA reduction-store (`cp.reduce.async.bulk.tensor`).

Cross-check against accel-sim's `gpu-simulator/ISA_Def/hopper_opcode.h` (local clone): present there — `UCGABAR_ARV`, `UCGABAR_WAIT`, `SYNCS`, `UTMALDG`, `UTMASTG`, `UTMACCTL`, `UTMAPF`, `UTMACMDFLUSH`, `UTMAREDG`, `UBLKCP`, `UBLKRED`, `UBLKPF`, `CGAERRBAR`, `STAS`, `LDGSTS`, `NANOSLEEP`, `CS2R`, `S2UR`, `MEMBAR`, `FENCE`, `CCTL`, `ERRBAR`, `ELECT`, `DEPBAR`, `LDGDEPBAR`, `HGMMA`, `USETMAXREG`, `WARPGROUP`, `ACQBULK`, `PREEXIT`; **absent** there — `MAPA` (consistent with ptxas never emitting it), `REDAS` and `ENDCOLLECTIVE` (both are real sm_90 opcodes: `REDAS` was emitted by ptxas above and both are in ncu's table, so accel-sim's list is incomplete, not the other way round).

---

## 5. On-device timers and clocks

* `clock64()` / `%clock64` → `CS2R SR_CLOCKLO`: per-SM cycle counter at the SM clock. NVIDIA forum (web summary): "the numerical value returned by clock64() is not intended to be comparable from one SM to another" — use only for intra-SM deltas; read twice with a data dependency and subtract. The 32-bit `clock()`/`%clock` wraps every 2^32 cycles (~2.2 s at 1.98 GHz).
* `%globaltimer` (PTX ISA 3.1+, sm_35+) → `CS2R SR_GLOBALTIMERLO`: 64-bit **nanosecond** global timer, consistent across SMs; its update granularity is coarser than 1 ns (commonly observed ~32 ns to 1 µs steps; measure on your device). Use for cross-SM / cluster-wide ordering, not for sub-100 ns latencies.
* `cudaEventElapsedTime`: CUDA Runtime API — "Computes the elapsed time between two events (in milliseconds with a resolution of around 0.5 microseconds)." Warnings: events recorded in non-NULL streams may include asynchronous launch/record latency; for sub-µs kernels use many iterations or in-kernel timers. https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
* Host-side `cudaDeviceProp::clockRate` (kHz) reports the **boost/max** SM clock, not the running clock; the running clock during an ncu run is `sm__cycles_elapsed.avg.per_second`; outside ncu use `nvidia-smi --query-gpu=clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem --format=csv -l 1` (fields `clocks.sm`/`clocks.mem` confirmed via web summary; `clocks.max.*` to verify on device) or NVML `nvmlDeviceGetClockInfo(NVML_CLOCK_SM|NVML_CLOCK_MEM)`.
* `nvidia-smi` clock locking (web summaries of NVIDIA blog/forums): `nvidia-smi -q -d CLOCK` (reads current/applications/default/max clocks); `nvidia-smi -lgc <min>,<max>` = `--lock-gpu-clocks` (MHz; e.g. `-lgc 1980,1980` or `-lgc 1590`), `-rgc` = `--reset-gpu-clocks`; `-lmc <min>,<max>` = `--lock-memory-clocks`, `-rmc` = `--reset-memory-clocks`; `-lmcd`/`--lock-memory-clocks-deferred` ("applied the next time GPU is initialized"; one guide states Hopper needs the deferred form for memory clocks). Requires root; "Not all NVIDIA GPUs support the --lock-gpu-clocks and/or --lock-memory-clocks options"; locking above the rated max is refused. Combine with `ncu --clock-control none` so ncu does not override your lock.
* H100 clock figures (third-party spec databases, web summaries — verify with `nvidia-smi -q -d CLOCK`): **H100 SXM5**: base 1590 MHz, boost 1980 MHz, HBM3 5120-bit, 3.35 TB/s; nvidia-smi max memory clock commonly reported as **2619 MHz** (Crusoe docs; TechPowerUp instead lists 1313 MHz — different convention). **H100 PCIe**: base 1095 MHz, boost 1755 MHz (one vendor lists 1065/1650), HBM2e 1593 MHz, 2.0 TB/s. ncu `--clock-control base` on SXM5 therefore means ~1590 MHz SM clock (≈20% lower than boost); report every measurement with the measured `sm__cycles_elapsed.avg.per_second`.

---

## 6. Memory Workload Analysis chart on Hopper

Profiling Guide §2.10 (2026.2.1 text): logical units (green) "Kernel, Global, Local, Texture, Surface, Shared, Load Global Store Shared"; physical units (blue) "L1/TEX Cache, Shared Memory, L2 Cache, L2 Compression, System Memory, Device Memory, Peer Memory"; "Depending on the exact GPU architecture, the exact set of shown units can vary, as not all GPUs have all units." Links Kernel→logical unit = instructions (`Inst`), logical→physical = requests (`Req`), colour = % of peak; ports = shared data ports of a unit (`Port` markers). Collect with `--set full`, `--section MemoryWorkloadAnalysis_Chart` or `--metrics group:memory__chart`.

What is and is not documented for Hopper:
* The section file `MemoryWorkloadAnalysis_Chart.section` declares the chart as a single `MemoryChart` body item whose metric list is resolved by the tool per architecture (the file itself only lists header metrics such as `lts__t_sectors.sum`, `lts__t_sectors_srcunit_tex.avg.pct_of_peak_sustained_elapsed`, `lts__t_sectors_srcunit_tex_lookup_miss.sum`, `dram__bytes.sum.per_second`, `pcie__read/write_bytes.sum.per_second`, and on CC_90 `lrc__average_ilc_input_sector_success_rate.pct` for L2 compression). The DSMEM/TMA nodes are therefore **not enumerated in any text file**; they come from the `mem_dshared` / `op_tma_*` / `pipe_tma` counters in §2.2/§2.6 (the only metrics that can feed such links). Release notes only state "Added support for the Hopper GPU architecture" (2022.3) and later "The memory chart shows a heatmap for link and port utilization".
* The L2 table documents the Hopper-relevant row: "L2 Fabric Total — Total for all operations across the L2 fabric connecting the two L2 partitions. This row is only shown for kernel launches on CUDA devices with L2 fabric." (metrics `lts__t_sectors_srcunit_ltcfabric*`).
* To see exactly which metrics the Hopper chart collects, run on the H100: `ncu --section MemoryWorkloadAnalysis_Chart --list-metrics ./bench` (or `--metrics group:memory__chart --list-metrics`), then export `--page raw --csv`; hover in the UI shows the metric name behind each link ("For most table entries, you can hover over it to see the underlying metric name and description").

---

## 7. CUPTI / NVML for energy (brief)

* NVML: `nvmlDeviceGetTotalEnergyConsumption(device, unsigned long long *energy)` returns cumulative energy in **millijoules** since driver load (Volta+); per-kernel energy = Δ around many back-to-back launches (counter update granularity is coarse — tens of ms — so integrate over ≥100 ms of repeated launches). `nvmlDeviceGetPowerUsage` returns milliwatts, "On Ampere (except GA100) or newer GPUs, the API returns power averaged over 1 sec interval"; `nvmlDeviceGetFieldValues` with `NVML_FI_DEV_POWER_INSTANT` for instantaneous power. A forum report notes the energy counter can disagree with integrated power on some boards, so cross-check both. (NVML reference: https://docs.nvidia.com/deploy/nvml-api/ — web summaries only.)
* CUPTI: the same PerfWorks metrics as ncu are available programmatically through the CUPTI Profiling API / Range Profiling API (`libnvperf_host.so` holds the metric database; metric names identical to §2), useful for in-process CSV logging without ncu's replay machinery (single-pass metric sets only, otherwise you implement replay yourself).

---

## 8. Experiment → metric quick map

| Experiment | Core metrics (fully suffixed) | ncu options |
|---|---|---|
| SMEM latency (pointer chase) | `gpu__time_duration.sum`, `sm__cycles_elapsed.max`, `smsp__inst_executed_op_shared_ld.sum`, `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum`, `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | `--clock-control none` + own `-lgc`; `-s`/`-c` |
| SMEM bandwidth / bank conflicts | `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_{ld,st}.sum`, `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st}.sum`, `sm__sass_data_bytes_mem_shared_op_{ld,st}.sum`, `l1tex__throughput.avg.pct_of_peak_sustained_elapsed`, `derived__memory_l1_conflicts_shared_nway` | `--section SourceCounters` for per-PC |
| DSMEM latency/bandwidth | `sm__sass_inst_executed_op_dshared_{ld,st,atom,redas,stas}.sum`, `l1tex__t_sectors_pipe_lsu_mem_dshared_op_{ld,st}.sum`, `l1tex__m_l1tex2xbar_write_bytes_mem_dshared.sum(.per_second)`, `l1tex__m_xbar2l1tex_read_bytes_mem_dshared.sum(.per_second)`, `l1tex__m_{l1tex2xbar,xbar2l1tex}_throughput_pipe_lsu_mem_dshared.avg.pct_of_peak_sustained_elapsed`, `launch__cluster_size`, `gpc__cgas_launched.sum` | `--print-metric-instances values` for per-SM |
| Cluster barrier cost | `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio`, `..._membar_...`, `..._short_scoreboard_...`, `sass__inst_executed_per_opcode` (UCGABAR_ARV/WAIT, SYNCS.*, MEMBAR, CGAERRBAR counts), `smsp__pcsamp_warps_issue_stalled_*` | `--section WarpStateStats,InstructionStats,SourceCounters` |
| L1 hit/miss, L1→L2 | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`, `..._lookup_hit.sum`, `..._lookup_miss.sum`, `l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct`, `l1tex__m_xbar2l1tex_read_bytes_mem_lg_op_ld.sum`, `lts__t_sectors_srcunit_tex_op_read.sum` | `--cache-control none` for warm; `all` for cold |
| L2 hit/miss, near/far partition | `lts__t_sectors_srcunit_tex_op_read_lookup_{hit,miss}.sum`, `lts__t_sector_hit_rate.pct`, `lts__t_sectors_srcunit_ltcfabric.sum`, `lts__t_requests_srcunit_ltcfabric.sum`, `lts__d_sectors_fill_device.sum`, `lts__t_bytes.sum.per_second`, `lts__throughput.avg.pct_of_peak_sustained_elapsed`, `lts__cycles_elapsed.avg.per_second`, `lts__t_sectors.max`/`.avg` (+instances) | `--print-metric-instances details` for per-slice |
| HBM bandwidth | `dram__bytes_read.sum.per_second`, `dram__bytes_write.sum.per_second`, `dram__throughput.avg.pct_of_peak_sustained_elapsed`, `dram__cycles_elapsed.avg.per_second`, `fbpa__dram_read_bytes.sum` (+instances) | |
| TMA load/store/multicast | `sm__sass_inst_executed_op_tma_{ld,st,red}.sum`, `l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum(.per_second)`, `..._dest_self.sum`, `..._dest_multicast.sum`, `l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.avg.per_cycle_active` (B/clk/SM), `l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_st.sum`, `sm__pipe_tma_cycles_active.avg`, `l1tex__tmain_requests.sum`, vs. `sm__sass_data_bytes_mem_shared_op_ldgsts.sum` for cp.async | |
| Launch overhead / back-to-back gaps | nsys `cuda_gpu_kern_sum`, `cuda_api_sum`; in-kernel `%globaltimer` | not ncu |

---

## 9. Unconfirmed items — verify on the H100 with `ncu --query-metrics` / a test run

1. **Clock lock values**: what `--clock-control base` and `boost` actually lock to on your SKU (read `sm__cycles_elapsed.avg.per_second`, `dram__cycles_elapsed.avg.per_second` from the report) and whether `nvidia-smi -lmc`/`-lmcd` is accepted on H100 (HBM clock is reported fixed on HBM parts). H100 SXM5 1590/1980 MHz, PCIe 1095/1755 MHz, HBM 2619 MHz (SXM) / 1593 MHz (PCIe) are from third-party spec pages.
2. **Whether `lts__cycles_elapsed.avg.per_second` equals the SM clock on GH100** (separate LTC/XBAR clock domain on GA100); needed to convert L2 cycle counters to ns.
3. **Number of `lts` instances** (`device__attribute_l2s_count`, `device__attribute_num_l2s_per_fbp`) and which slice IDs belong to which partition — only observable via `--print-metric-instances details` on `lts__t_sectors.sum` (instance correlation IDs) and a known-address sweep. Also whether HW counters such as `lts__t_sectors.sum` and `l1tex__*` expose per-unit instance values in CSV (the docs say instances can be "a streaming multiprocessor"; confirm for lts/fbpa).
4. **Exact metric set behind the Hopper memory chart's DSMEM and TMA links/ports** (`ncu --section MemoryWorkloadAnalysis_Chart --list-metrics ./bench`; the UI tooltip gives the metric per link). Chart node labels for DSMEM/TMA are not documented in text.
5. **Semantics of `l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_ld`** ("bytes written … for distributed shared loads" — presumably request/return packets for loads) vs `l1tex__m_xbar2l1tex_read_bytes_mem_dshared_op_ld`: run a pure-load DSMEM kernel and check which side carries the payload bytes; also confirm `..._mem_dshared_loopback` counts intra-SM accesses only.
6. **Which stall reason absorbs cluster-barrier (`UCGABAR_WAIT`) and mbarrier `SYNCS.PHASECHK.TRYWAIT` spin loops** (`barrier`? `misc`? `branch_resolving`/`wait` from the retry loop?) — run a barrier-only kernel with `--section WarpStateStats,SourceCounters --print-metric-instances details`.
7. **`l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.*.per_cycle_active` as the "32 B/clk per SM TMA read" limit**: confirm the peak via `.avg.peak_sustained` / `.avg.pct_of_peak_sustained_elapsed` on device (the `peak_sustained` sub-metric exposes NCU's rated peak for the counter).
8. **`%globaltimer` update granularity on H100** (measure the minimum non-zero delta) and `cudaEventElapsedTime` jitter under locked clocks.
9. **MAPA**: the opcode never appeared (address translation was folded into `ULEA/UPRMT` on `SR_CgaCtaId`); check with a fully dynamic rank whether `MAPA` is ever emitted, so you know what to count in `sass__inst_executed_per_opcode`.
10. `--query-metrics-collection pmsampling` on device for H100 PM-sampling metric names (1,355 lines for gh100 offline; time-series counters for bandwidth-over-time plots) and whether `group:smsp__pmwarpsamp_warp_stall_reasons` is collectable ("These metrics can not yet be collected from the command line").
11. `nsys` flag spellings on the installed version (`--gpu-metrics-device` vs `--gpu-metrics-devices`, `--capture-range-end`), and `nsys stats --help-reports` for exact report ids.
12. `nvmlDeviceGetTotalEnergyConsumption` update period and agreement with integrated `nvmlDeviceGetPowerUsage` on H100.
