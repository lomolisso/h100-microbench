# 11 — SM-to-SM network bandwidth vs cluster size and access pattern (generic proxy and TMA put)

> **Device configuration:** runs under [`docs/context/10__device_config.md`](../context/10__device_config.md): H100 NVL (GPU 2), 132 SMs, 60 MiB L2, HBM3 3.94 TB/s, SM clock locked at 1635 MHz, power limit 400 W, ECC on, CUDA 12.9.1 `sm_90a`. Reference numbers quoted in this spec come from **other chips and configurations** (H800 PCIe, H100 PCIe, H100 SXM, H200, RTX 5090; see `10` §4). They are kept unchanged as intuition and reproduction targets, not as expectations for this device.

## Goal
Per-SM injection bandwidth (B/cycle) and GPU-wide aggregate bandwidth (TB/s) of DSMEM traffic for loads, stores and TMA shared→shared copies, as a function of cluster size (2–16), access pattern (pair, ring, broadcast, all-to-all, packed pairs), direction mix (uni/bidirectional, mixed read+write), threads/ILP, and policy. Figure: histogram/bar of aggregate TB/s per cluster size with the GMEM bandwidth from `05` as a dotted line (user's requested figure).

## Background
References: per-SM saturation ≈ 21.1–21.3 B/cycle (sequential, 1024 threads) [P3]; one-way 19.9 (ld) / 18.9 (st) / 21.3 (TMA put) B/cycle, bidirectional 15.4 / 18.1 / 20.3 per direction; mixed opposite-direction read+store 29.3–29.7 aggregate; TMA ring 2→16 SMs scales linearly (≈ 20.3–21.2 B/cycle/SM); 128 B TMA commands only 2.35 B/cycle (≈54.5-cycle command cost) [blog/P5]; ring/pair ≈ 3.28 TB/s at CS 2 and 2.78 at CS 4 chip-wide (114 SMs), broadcast collapses [P1]; broadcast per-reader BW ≈ 21.2/N [P3]; local SMEM bandwidth drops to 71–103 B/cycle while being read remotely [P3]; model: 4 lanes × 32 B / 6 SMs = 21.33 B/cycle hard per-SM cap, not work-conserving [blog].

## Kernels
All with 1 CTA per SM (200 KB dynamic smem: 96 KB source tile + 96 KB destination tile + barriers), 1024 threads (sweep 128–1024), payload per active SM swept 16, 32, 48, 64, 80, 96 KB with every 16 B address touched once (no reuse), ≥ 30 samples per point; time per CTA with `%clock64` + `%globaltimer` pair (dependent read-after-last-load) and whole-launch with cudaEvents.

### K1 — remote load stream
Each thread: `ld.shared::cluster.v4.u32` ×ILP (ILP = 4 default) over its slice of the peer's 96 KB tile, XOR into registers, store checksum after the end timestamp. Peer determined by pattern.

### K2 — remote store stream
`st.shared::cluster.v4.u32` of register data into the peer's destination tile; completion = `cluster.sync()` at the end (fence included); also `st.async` variant with a remote mbarrier expecting the full byte count (hardware completion, no polling).

### K3 — TMA put stream
`cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes` in chunks of {128 B, 1 KB, 4 KB, 16 KB}, one remote mbarrier per tile (`mbarrier.arrive.expect_tx` once with the total bytes, issued by the *destination* CTA on its own barrier before `cluster.sync()`), issuer = one elected thread; completion by the destination polling `mbarrier.try_wait.parity.acquire.cluster`. Also `cp.reduce.async.bulk ... .shared::cluster.add.u32` (reduce-put) with the same structure.

### K4 — local pressure control
Source SM runs additional warps doing local `ld.shared` while K1 runs, to show whether the destination/source SRAM port limits remote bandwidth (blog found no effect).

### Patterns (parameterised)
`pair_uni` (0→1), `pair_bi` (0↔1), `ring(+k)` (r → (r+k) mod CS), `packed_pairs` (groups of 2 with stride), `broadcast` (all → 0; and 0 → all for stores/TMA = the software broadcast), `all_to_all` (threads of each CTA split into CS groups, group g targets rank g), `mixed_same_dir` (0 reads 1 while 1 stores into 0 = same physical direction), `mixed_opposite` (0 reads 1 while 0 stores into 1).

### Sweeps
CS ∈ {2, 4, 8, 16}; threads ∈ {128, 256, 512, 1024}; ILP ∈ {1, 2, 4, 8}; policy ∈ {Default, Spread, LoadBalancing}; (a) single cluster on an idle chip (per-SM numbers), (b) full chip: `cudaOccupancyMaxActiveClusters` clusters simultaneously (GPU-wide TB/s), (c) 50 160-block launch as in [P1] for direct comparison.

## Analysis
- Per-SM B/cycle from the affine fit `T(bytes) = α + β·bytes` over the payload sweep (1/β), with α reported as the fixed cost (cluster.sync + barrier set-up); residual RMSE must be < 2 %.
- GPU-wide TB/s = Σ bytes / event time; also B/cycle/SM × active SMs × clock as a cross-check.
- Broadcast: per-reader BW vs N readers; fit 1/N.
- Compare the per-SM one-way cap with 21.33 B/cycle and with the ncu rated peak of `l1tex__m_l1tex2xbar_throughput_pipe_lsu_mem_dshared.avg.peak_sustained`.

## CSV columns
`11__dsmem_bw__samples.csv`: `kernel, pattern, cs, policy, threads, ilp, chunk_bytes, payload_bytes, cluster_id, rank, smid, peer_smid, sample, cycles, ns, bytes_moved, launch_mode ∈ {single, full, p1_style}`.
`11__dsmem_bw__fits.csv`: `kernel, pattern, cs, policy, threads, ilp, chunk_bytes, bpc_per_sm, alpha_cycles, rmse_pct, tbps_gpu_wide, active_sms`.

## Figures
1. **Bar/histogram: aggregate TB/s vs CS** for ring (loads), ring (stores), ring (TMA put), with dotted lines for DRAM TB/s and L2 TB/s from `05`.
2. Per-SM B/cycle vs threads × ILP (K1), lines per CS.
3. Pattern comparison at CS 8: pair_uni / pair_bi / ring / broadcast / all_to_all / mixed_same / mixed_opposite, for ld / st / TMA.
4. Broadcast per-reader BW vs number of readers with the 21.2/N curve.
5. TMA chunk-size curve (128 B → 16 KB) with the implied per-command cost.
6. Local SMEM bandwidth of the victim SM while being read (K4) vs idle.

## Pass criteria
One-way remote load/store/TMA put ∈ [18, 23] B/cycle/SM; ring TMA scales ≥ 0.9× linearly to CS 16; broadcast per-reader ≈ 21/N (±20 %); P1's 3.28 TB/s (CS 2 ring, 114 SMs) reproduced within ±15 % after scaling by SM count and clock.

## NVIDIA tools
`ncu --metrics l1tex__m_l1tex2xbar_write_bytes_mem_dshared.sum.per_second,l1tex__m_xbar2l1tex_read_bytes_mem_dshared.sum.per_second,l1tex__m_l1tex2xbar_throughput_pipe_lsu_mem_dshared.avg.pct_of_peak_sustained_elapsed,l1tex__m_xbar2l1tex_throughput_pipe_lsu_mem_dshared.avg.pct_of_peak_sustained_elapsed,sm__sass_inst_executed_op_dshared_tma_st.sum,l1tex__m_xbar2l1tex_read_bytes_mem_dshared_op_tma_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum --print-metric-instances values --clock-control none --cache-control none` on each kernel/pattern at CS 8 (per-SM instance values give injection/ejection bytes per SM → directly the asymmetry of broadcast).
