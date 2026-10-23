# 18 — Roll-up: latency (cycles) and bandwidth (TB/s GPU-wide, B/cycle/SM) for every memory-access type

> **Device configuration:** runs under [`docs/context/10__device_config.md`](../context/10__device_config.md): H100 NVL (GPU 2), 132 SMs, 60 MiB L2, HBM3 3.94 TB/s, SM clock locked at 1635 MHz, power limit 400 W, ECC on, CUDA 12.9.1 `sm_90a`. Reference numbers quoted in this spec come from **other chips and configurations** (H800 PCIe, H100 PCIe, H100 SXM, H200, RTX 5090; see `10` §4). They are kept unchanged for the agent's sanity checks only, never cited in the notebook and not expectations for this device. Aggregate only rows produced under this configuration; flag any other row.

## Goal
No new kernels. One notebook that reads the results CSVs of `04, 05, 06, 09, 11, 12, 16` (and `07` for completion costs) and produces the master table and figures requested in the project brief: for every access type in `docs/context/02__memory_access_taxonomy.md` §2, the median dependent latency, the per-SM bandwidth, the GPU-wide bandwidth, the completion mechanism and its fixed overhead, and the SASS opcode. (The comparison against `docs/context/08__prior_work_reference_numbers.md` is an agent-only sanity check; see below.)

## Inputs (file → rows of the taxonomy)
- `04__latency__summary.csv` → rows 1, 3 (local), 23 (L1/L2/HBM latency by class).
- `05__bandwidth__*.csv` → rows 1–2, 23–24 bandwidth.
- `06__tma__chase.csv`, `06__tma__completion.csv` → rows 10, 13, 21 latency + completion overhead.
- `09__dsmem_latency__chase.csv`, `09__dsmem_latency__visibility.csv` → rows 3–6 latency/visibility.
- `11__dsmem_bw__fits.csv` → rows 4–5, 15 bandwidth.
- `12__dsmem_atomics__*.csv` → rows 6–9.
- `16__async__fits.csv` → rows 10–21 bandwidth and intercepts.
- `07__sync__primitives.csv` → completion-wait costs.
- `01__timers__clock_domains.csv`, mandatory columns → clock used for cycles↔ns and B/cycle↔TB/s.

## Table columns
`row, path, instruction, proxy, completion, latency_cycles_median, latency_ns, bpc_per_sm, tbps_gpu_wide, completion_overhead_cycles, sass_opcode`.

## Figures
1. **Latency ladder**: horizontal bar chart of median latency (cycles) for SMEM, SMEM-via-DSMEM-interface, DSMEM remote (CS 2 … 16), L1, L2 near, L2 far, HBM near, HBM far, TMA L2, TMA HBM, remote atomic, st.async visibility, plain remote-store visibility, cluster.sync, mbarrier wait.
2. **Bandwidth ladder**: GPU-wide TB/s for SMEM (aggregate), L1, L2, HBM, DSMEM ring ld/st/TMA put (CS 2…16), TMA G→S, multicast ×4/×8/×16 (delivered), S→G, reduce-store, reduce-put; dotted lines = this device's DRAM peak (3.94 TB/s) and the `05` L2 bandwidth.
3. **The two user-requested histograms** (from `09` and `11`): SM-to-SM latency per cluster size with GMEM latency dotted; SM-to-SM bandwidth per cluster size with GMEM bandwidth dotted.
4. Completion-overhead bars: mbarrier (TMA load), bulk group (TMA store), st.async, cluster.sync, flag — in cycles, next to the pure transfer time of a 16 KB payload.

## Sanity checks (agent-only)
Expected ranges for judging plausibility during result analysis. They are never presented as pass/fail and never quoted in the notebook. If a value falls outside, investigate (placement, clocks, SASS, cache state, ncu replay) and re-run. A value that survives is a finding about this device; describe it on its own terms.
Every taxonomy row has either a measured value or an explicit "not measurable because …" note.
- Check every row against `docs/context/08` with the bands of its rule 3, and investigate outliers before writing the notebook.
- The old bandwidth-ladder reference line was the SXM5 DRAM peak, 3.35 TB/s; ours is 3.94 TB/s.
