# 16 — Bandwidth and completion overhead of every async-proxy path: TMA/bulk G→S, G→R, multicast, S→R put, reduce-put, S→G store, reduce-store, tensor vs non-tensor

> **Device configuration:** runs under [`docs/context/10__device_config.md`](../context/10__device_config.md): H100 NVL (GPU 2), 132 SMs, 60 MiB L2, HBM3 3.94 TB/s, SM clock locked at 1635 MHz, power limit 400 W, ECC on, CUDA 12.9.1 `sm_90a`. Reference numbers quoted in this spec come from **other chips and configurations** (H800 PCIe, H100 PCIe, H100 SXM, H200, RTX 5090; see `10` §4). They are kept unchanged for the agent's sanity checks only, never cited in the notebook and not expectations for this device.

## Goal
Per-SM B/cycle and GPU-wide TB/s for rows 10–21 of `docs/context/02__memory_access_taxonomy.md`, with the fixed completion overhead of each mechanism (mbarrier vs bulk group) separated from the streaming rate; the multicast fan-out gain; and the cost of hardware reductions (bulk reduce into DSMEM and into L2) versus plain copies. Together with `05` (generic) and `11` (DSMEM generic), this completes the "every access type" bandwidth table.

## Background
`docs/context/03__tma_and_bulk_async_copies.md` §3–4. References: TMA G→S ≈ 1.8 TB/s on H800 PCIe for ≥ 2 KB boxes (= DRAM peak) [P1]; per-SM 32 B/cycle read path [P5]; ≈ 38 B/cycle/SM from L2 and ≈ 70 B/cycle/SM with 4× multicast, no gain at 8× [forum, unverified]; TMA put S→R ≈ 21 B/cycle/SM, scales linearly 2→16 SMs [blog]; 1-D tensor box limit 2 KB [P1]; large innermost box dims best [P1].

## Common harness
1 CTA per SM (200 KB dyn. smem), elected issuer thread, multi-stage ring of smem buffers (stages ∈ {1, 2, 4, 8}) with one mbarrier per stage; consumer warps do nothing but wait and release stages; payload per command ∈ {128 B, 1 KB, 4 KB, 16 KB, 32 KB}; bytes per SM swept 1–64 MB (affine fit → rate and intercept); time with `%clock64` per SM and cudaEvents GPU-wide; active-SM sweep {1, 2, 4, 8, 16, 33, 66, 132} for the saturation curve; sources: L2-resident (16 MB working set, warmed) and HBM (≥ 1 GB streaming, random 2 MB chunks per SM).

## Kernels / paths
- **P1 G→S bulk** (row 10): `cp.async.bulk.shared::cluster.global` into own smem; also PTX 8.6 `.shared::cta` form; `.L2::cache_hint` evict_first/evict_last policies; compare with `cp.async.cg` (LDGSTS, 16 B/thread, 1024 threads) and with plain `ld.global.cg.v4` (from `05`).
- **P2 G→S tensor** (row 13): 1-D box (≤ 2 KB), 2-D (64×64 f32 = 16 KB), 3-D boxes {16,16,16}, {256,16,1}, {16,256,1}, {4,4,256}; swizzle NONE vs 128B (box inner 32 floats); `l2Promotion` NONE/128B/256B; `.im2col` on a 4-D NHWC tensor (one shape).
- **P3 G→R** (row 11): same as P1 but destination/mbarrier in a peer CTA (CS 2); does the fabric hop cost bandwidth?
- **P4 multicast** (rows 12, 14): CS ∈ {2, 4, 8, 16}, one issuer loads one tile with `ctaMask` = all ranks (and masks of 2, 4, 8 of 16); every destination `expect_tx`; compare delivered bytes per SM per cycle vs N independent unicast loads of the same tile; also "rotating issuer" (each CTA multicasts 1/N of the tiles) — the real-world pattern; measure L2 read traffic (`lts__t_sectors_srcunit_tex_op_read`) to prove the single-read fan-out, and `l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld_dest_multicast`.
- **P5 S→R put** (row 15) and **P6 reduce-put** (row 16): from `11` K3 (TMA put) plus `cp.reduce.async.bulk ... shared::cluster .add.u32/.add.u64/.max.s32/.and.b32`, ring and all-to-one patterns, CS 2–16; cost of reduction vs copy at the destination SM (local smem bandwidth while receiving a reduce-put, via `l1tex__data_pipe_lsu_wavefronts_mem_shared`).
- **P7 S→G store** (row 17) and **P9 tensor store** (row 19): bulk group completion; `wait_group.read` vs `wait_group`; stages; GPU-wide write TB/s vs `st.global.v4` from `05`.
- **P8 reduce-store** (row 18) and **P10 tensor reduce-store** (row 20): `.add.f32`, `.add.u32`, `.max.s32`, `.add.noftz.bf16`; destination L2-resident (16 MB) vs HBM; compare with `red.global.add` from 1024 threads; measure `lts__t_sectors_op_red.sum` to show the reduction executes at L2.
- **P11 prefetch** (row 21): `cp.async.bulk.prefetch.L2` / `.tensor` of the next tile ahead of P1/P2 loads from HBM: achieved rate and latency hiding vs no prefetch.

## CSV columns
`16__async__samples.csv`: `path, variant, source ∈ {l2, hbm}, dest ∈ {own, peer, multicast, global}, cs, cta_mask_bits, command_bytes, stages, bytes_per_sm, active_sms, rank, smid, cycles, ns, bytes`.
`16__async__fits.csv`: `path, variant, source, dest, cs, command_bytes, stages, active_sms, bpc_per_sm, intercept_cycles, tbps_gpu_wide, rmse_pct`.
`16__async__ncu.csv`: joined raw ncu metrics per configuration.

## Figures
1. Per-SM B/cycle vs command size for P1/P2/P3/P5/P7 (L2 source) — shows the per-command cost and the measured per-SM read and put ceilings.
2. GPU-wide TB/s vs active SMs for G→S (HBM) and S→G, dotted lines = `05` generic DRAM/L2 TB/s.
3. Multicast: delivered B/cycle/SM vs cluster mask size, unicast reference; L2 sectors read per delivered byte.
4. Reduce vs copy bandwidth (DSMEM and L2 destinations) per data type.
5. Tensor box-shape and swizzle/promotion effects (bars).
6. Completion intercepts (cycles) per mechanism and stage count.

## Sanity checks (agent-only)
Expected ranges for judging plausibility during result analysis. They are never presented as pass/fail and never quoted in the notebook. If a value falls outside, investigate (placement, clocks, SASS, cache state, ncu replay) and re-run. A value that survives is a finding about this device; describe it on its own terms.
P1 from HBM reaches ≥ 85 % of DRAM peak GPU-wide; per-SM read ceiling 28–40 B/cycle; multicast ×4 delivers ≥ 1.6× unicast per SM with ≈ 1/4 of the L2 sector reads; S→R put ≈ 20 ± 3 B/cycle/SM; tensor 2-D 16 KB within 10 % of non-tensor 16 KB.
- Per-SM ceilings vs prior work: read 32 B/cycle (P5), put ≈ 21 B/cycle (P3/P5).

## NVIDIA tools
`ncu --metrics sm__sass_inst_executed_op_tma_ld.sum,sm__sass_inst_executed_op_tma_st.sum,sm__sass_inst_executed_op_tma_red.sum,sm__sass_inst_executed_op_dshared_tma_st.sum,sm__sass_inst_executed_op_dshared_tma_red.sum,l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum,l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld_dest_self.sum,l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld_dest_multicast.sum,l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_st.sum,l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_red.sum,l1tex__m_xbar2l1tex_read_bytes_mem_dshared_op_tma_st.sum,l1tex__m_xbar2l1tex_read_bytes_mem_dshared_op_tma_red.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_op_red.sum,lts__t_sectors_op_write.sum,dram__bytes_read.sum,dram__bytes_write.sum,sm__pipe_tma_cycles_active.avg,sm__cycles_elapsed.avg.per_second --print-metric-instances values --clock-control none --cache-control none`; SASS listing with `UBLKCP.S.G/G.S/S.S`, `UBLKRED`, `UTMALDG.*`, `UTMASTG.*`, `UTMAREDG.*`, `UBLKPF`, `UTMAPF`, `UTMACMDFLUSH`, `DEPBAR.LE`.
