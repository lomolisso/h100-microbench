# TMA and bulk asynchronous copies (the async proxy)

Sources: PTX ISA §9.7.9.25 (`cp.async.bulk*`), §5.5 (tensor maps), CUDA Driver API `cuTensorMapEncodeTiled` (verbatim constraints), CCCL `cuda::ptx` gates, CUTLASS `copy_sm90_tma.hpp` (canonical asm), ptxas 12.9 validation. Measured numbers tagged [LIT] come from `08__prior_work_reference_numbers.md`.

## 1. What the TMA unit is and does
- One TMA unit per SM, shared by the four sub-partitions. A **single thread** issues a bulk instruction; the unit then generates the memory requests (sector-aligned, 32 B), tracks outstanding requests per transaction, applies swizzle / OOB fill / reduction, writes shared memory (or L2 for stores), and finally signals the completion mechanism. ≈ 44-cycle same-warp issue gap, ≤ 16 concurrently admitted transactions per SM, ≈ 32 B/cycle per-SM read path, ≥ 800 outstanding 32 B requests to cover DRAM latency [LIT, RTX 5090 — to verify on H100 in `06`].
- Executes in the **async proxy**; data traffic bypasses L1 (served from L2) — the TMA latency curve has no L1 plateau and sits ≈ 170 cycles above the generic-load curve [LIT, H800].
- Tensor mode needs a 128-byte opaque **tensor map** (`CUtensorMap`, 64-B aligned) created on the host by `cuTensorMapEncodeTiled` / `cuTensorMapEncodeIm2col` (obtain the driver entry point with `cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", ...)`), passed as `const __grid_constant__ CUtensorMap` kernel parameter (→ `.param` space; also `.const` or `.global` allowed), accessed through the *tensormap proxy*; `prefetch.tensormap [p]` warms its cache line. Non-tensor (1-D "bulk") mode needs no descriptor.

## 2. Instruction forms available on sm_90 (exact syntax; see `02__memory_access_taxonomy.md` rows 10–21)

```
// non-tensor bulk
cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes{.L2::cache_hint}              [dst], [src], size, [mbar]{, policy}
cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster{.L2::cache_hint} [dst], [src], size, [mbar], ctaMask{, policy}   // sm_90a
cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes{.L2::cache_hint}                  [dst], [src], size, [mbar]{, policy}   // PTX 8.6
cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes                          [dst], [src], size, [mbar]             // dst must be a different CTA
cp.async.bulk.global.shared::cta.bulk_group{.L2::cache_hint}                                    [dst], [src], size{, policy}
cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.op.T              [dst], [src], size, [mbar]   // add u32/s32/u64, min/max u32/s32, inc/dec u32, and/or/xor b32
cp.reduce.async.bulk.global.shared::cta.bulk_group{.L2::cache_hint}.op.T                        [dst], [src], size{, policy} // + f32/f64/.add.noftz.f16/bf16, s64 min/max, b64 bitwise
cp.async.bulk.prefetch.L2.global{.L2::cache_hint}                                               [src], size{, policy}
// tensor (dim = 1d..5d; load_mode = .tile | .im2col (loads, ≥3d) | .im2col_no_offs (stores))
cp.async.bulk.tensor.Nd.shared::cluster.global{.tile|.im2col}.mbarrier::complete_tx::bytes{.L2::cache_hint}        [dst], [tmap, {c0..}], [mbar]{, {offs}}{, policy}
cp.async.bulk.tensor.Nd.shared::cluster.global{.tile|.im2col}.mbarrier::complete_tx::bytes.multicast::cluster{...} [dst], [tmap, {c0..}], [mbar]{, {offs}}, ctaMask{, policy}  // sm_90a
cp.async.bulk.tensor.Nd.global.shared::cta{.tile|.im2col_no_offs}.bulk_group{.L2::cache_hint}                     [tmap, {c0..}], [src]{, policy}
cp.reduce.async.bulk.tensor.Nd.global.shared::cta.op{.tile|.im2col_no_offs}.bulk_group{.L2::cache_hint}           [tmap, {c0..}], [src]{, policy}
cp.async.bulk.prefetch.tensor.Nd.L2.global{.tile|.im2col}{.L2::cache_hint}                                         [tmap, {c0..}]{, {offs}}{, policy}
cp.async.bulk.commit_group;      cp.async.bulk.wait_group{.read} N;
prefetch.tensormap [p];          tensormap.replace.* (sm_90a only);  tensormap.cp_fenceproxy...;  fence.proxy.tensormap::generic.{release,acquire}.scope [addr], 128;
```
Operand conventions (CUTLASS): smem/mbarrier operands are 32-bit `.shared` addresses (`"r"`, from `__cvta_generic_to_shared`, remote ones from `mapa`); global pointers and the tensor-map address are 64-bit (`"l"`); `size` is a 32-bit register; coordinates are `.s32` registers; `ctaMask` is a 16-bit register (`"h"`); `policy` is a 64-bit `createpolicy` result.

Constraints (verbatim or derived from the driver API): `size` multiple of 16 B, 16-B aligned src/dst; tensor rank 1–5; `boxDim[i] ≤ 256`, `boxDim[0]·elemsize` multiple of 16 B; `globalStrides` multiples of 16 B (< 2^40); `elementStrides ≤ 8`; swizzle 32B/64B/128B requires inner box bytes ≤ 32/64/128 and the smem destination aligned to 256/512/1024 B (128-B alignment minimum for any TMA destination); `l2Promotion` ∈ {NONE, 64B, 128B, 256B} sets the L2 fill granularity; OOB elements are zero- or NaN-filled and still count toward `complete_tx` bytes (the full box size is signalled). Max box bytes = 256 elements × inner dims … but any single box must fit in smem; 1-D tensor boxes are limited to 256 elements (2 KB for 8-byte elements) — use non-tensor bulk for large 1-D copies [LIT].

## 3. Completion mechanisms and the three canonical patterns

### 3.1 Load into smem, mbarrier completion (rows 10–14)
```cpp
__shared__ alignas(1024) uint8_t tile[BYTES];
__shared__ alignas(8) uint64_t bar;
if (threadIdx.x == 0) { mbar_init(smem_u32(&bar), 1); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__syncthreads();                                   // cluster.sync() if peers will arrive on / be signalled through this barrier
if (threadIdx.x == 0) {
  mbar_arrive_expect_tx(smem_u32(&bar), BYTES);     // arrive(1) + expect_tx(BYTES); release.cta by default
  bulk_g2s(smem_u32(tile), gsrc, BYTES, smem_u32(&bar));   // or tma_load_2d(...)
}
mbar_wait_parity(smem_u32(&bar), phase);           // all consumers; phase toggles 0/1 per reuse
// data is now visible to generic loads: NO fence.proxy.async needed (implicit fence at completion)
```
Order of `expect_tx` vs the copy does not matter for correctness (tx-count may go negative within a phase) but the expect must be in the same phase. Each destination CTA of a multicast must `expect_tx(BYTES)` on its own barrier (the signal is replicated to the same CTA-relative offset). A load issued by CTA A into CTA B's smem with B's barrier is legal (`.shared::cluster` destination) — B waits on its local barrier.

### 3.2 Store / reduce from smem, bulk-group completion (rows 17–20)
```cpp
// producers wrote `tile` with generic stores (st.shared)
asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // make generic writes visible to the async proxy
__syncthreads();                                              // all writers done before the elected issuer
if (threadIdx.x == 0) {
  bulk_s2g(gdst, smem_u32(tile), BYTES);                       // or tma_store_2d / tma_reduce_add_2d / bulk_red_s2g_*
  bulk_commit();
  bulk_wait_read<0>();      // smem may be overwritten now (source reads done)
  bulk_wait<0>();           // global writes visible to this thread; add a release (fence/st.release.gpu) before signalling other CTAs/host
}
```
### 3.3 Put into a peer's smem, remote mbarrier completion (rows 15–16)
```cpp
// destination CTA (rank d) before cluster.sync(): mbar_init(bar_d, 1); fence.mbarrier_init.release.cluster;
//                                                 mbar_arrive_expect_tx(bar_d, TOTAL_BYTES_FROM_ALL_SOURCES)
cluster.sync();
// source CTA: generic writes to src tile → fence.proxy.async.shared::cta → __syncthreads → elected thread:
uint32_t dst = mapa_smem(smem_u32(tile_dst), d), rbar = mapa_smem(smem_u32(&bar), d);
bulk_s2s_dsmem(dst, smem_u32(tile_src), BYTES, rbar);          // or bulk_red_s2s_add_u32(...)
// destination: mbar_try_wait_parity_acq_cluster(bar_d, phase, hint) loop; then cluster.sync() before anyone exits
```
Every source's `complete_tx(BYTES)` lands on the destination's barrier; one `expect_tx` with the sum suffices. Use `.acquire.cluster` on the wait when data were produced by other CTAs (the ISA only guarantees release at cluster scope on the complete-tx).

## 4. Facts that shape the TMA experiments (`06`, `16`)
- Issue cost per command ≈ 44–55 cycles [LIT]; a 128 B command therefore reaches only ≈ 2.4 B/cycle; ≥ 4 KB commands are needed for bandwidth, 16 B commands for latency chases.
- Concurrency cap of 16 transactions per SM (issue stalls at the 17th) [LIT, 5090 — verify].
- Per-SM read path ≈ 32 B/cycle; DRAM-path single SM sustained ≈ 31 B/cycle [LIT, 5090]; H100 forum data ≈ 38 B/cycle/SM from L2 and ≈ 70 B/cycle/SM with 4× multicast, no gain at 8× [LIT, unverified].
- Completion observation is quantised by the consumer's polling loop (≈ 42-cycle `try_wait` latency, ≈ 147-cycle steps with a sleep hint / yield loop) [LIT] — measure in `07` and subtract.
- Box shape matters: large innermost dimension best; 3-D boxes with large y/z lose bandwidth [LIT].
- `cp.async.bulk` G→S is served by L2 only; TMA OOB reads are filtered before DRAM (H100 filters part on-chip) and OOB writes generate no traffic [LIT].
- TMA stores/reduces to global are performed at L2 (reduction in the L2 atomic units) [INF from ISA ".relaxed.gpu per element"]; measure with `lts__t_sectors_op_red.sum`.
- `.L2::cache_hint` policies from `createpolicy` apply to the global side of any bulk op (per-sector/per-line semantics unspecified).

## 5. Minimal host-side tensor map for a 2-D float tile
```cpp
CUtensorMap tmap; alignas(64)
cuuint64_t gdim[2] = {cols, rows};                 // innermost first
cuuint64_t gstride[1] = {cols * sizeof(float)};    // bytes, multiple of 16
cuuint32_t box[2] = {64, 64};                      // 64*4 B = 256 B inner span → SWIZZLE_NONE or ≤128 B inner for SWIZZLE_128B
cuuint32_t estride[2] = {1, 1};
encode(&tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, gptr, gdim, gstride, box, estride,
       CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
// kernel: __global__ void k(const __grid_constant__ CUtensorMap tmap, ...)  → tma_load_2d(smem, &tmap, x0, y0, bar)
```
