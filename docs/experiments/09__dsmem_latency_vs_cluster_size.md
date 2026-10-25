# 09 — DSMEM remote-access latency vs cluster size (generic proxy)

> **Device configuration:** runs under [`docs/context/10__device_config.md`](../context/10__device_config.md): H100 NVL (GPU 2), 132 SMs, 60 MiB L2, HBM3 3.94 TB/s, SM clock locked at 1635 MHz, power limit 400 W, ECC on, CUDA 12.9.1 `sm_90a`. Reference numbers quoted in this spec come from **other chips and configurations** (H800 PCIe, H100 PCIe, H100 SXM, H200, RTX 5090; see `10` §4). They are kept unchanged for the agent's sanity checks only, never cited in the notebook and not expectations for this device.

## Goal
Average dependent-load latency, in SM cycles, of a remote shared-memory access as the cluster grows from 2 to 16 CTAs, for every (source rank, destination rank) pair, under the three scheduling policies; plus the local-via-DSMEM-interface baseline and the remote **store→visibility** latency. Figure: histogram of SM-to-SM latencies per cluster size with the GMEM (DRAM) latency from `04` as a dotted line (user's requested figure).

## Background
`docs/context/01__clusters_and_dsmem.md` §4; references: 181 cycles at CS 2, 184–213 at CS 2–16 (H800, 1 thread, 4 KB stride-1 chase) [P1]; 187–218 (H100 SXM, CS 8 heatmap) [P3]; 191 [P4]; 193 mean / 220 dependent round trip / 625 store-visibility (H200) [P5]; local SMEM 29 cycles, local-through-DSMEM-interface 33 [P1].

## Kernel K1 — all-to-all dependent chase (one thread per CTA)
- Block = 32 threads (one warp; lane 0 does the chase, the other lanes stay resident and participate in every `cluster.sync()` so that no thread exits early), dynamic smem = 200 KB (`cudaFuncAttributeMaxDynamicSharedMemorySize`) to force exactly 1 CTA per SM; host asserts that all `%smid` of a cluster are distinct.
- Each CTA fills its own smem with a chain: `s[i] = (i + 1) mod N` as 32-bit indices over N = 1024 words (stride-1, bank-conflict-free, as [P1]) and additionally a **random cyclic permutation over 16 KB** (defeats any hypothetical remote prefetch/combining).
- `cluster.sync()`.
- For `src = 0 … CS-1` (serialised by `cluster.sync()` between sources so only one SM is active at a time): if `rank == src`, for `dst = 0 … CS-1` (including self): `base = mapa(smem_u32(s), dst)`; warm with 64 hops; time 4096 dependent hops with the `01` loop (`ld.shared::cluster.u32 idx, [base + idx*4]` — compute the address with `add`; the chain value is the next index); record `(src, dst, smid_src, smid_dst, cycles_per_hop)`; also record the raw per-hop series for 256 hops (fine-grained variant) to see the distribution.
- Variants: (a) `ld.shared::cluster.u32` with explicit 32-bit address; (b) generic `ld.u32` on the `map_shared_rank` pointer (what CUDA C++ emits); (c) `ld.volatile.shared::cluster`; (d) `ld.relaxed.cluster.shared::cluster` and `ld.acquire.cluster` (memory-model qualifiers — cost of ordering); (e) 64-bit and 128-bit loads (`.v4.u32`) — latency should not change; (f) **loaded per-request latency** (the GPUNetBench `SM2SM/ONE2ONE` latency mode, `docs/research_notes/gpunetbench.md` §2): 1024 threads of the source CTA stream the whole 200 KB remote buffer with 8 independent `ld.shared::cluster.u32` per step (ILP 8, stride 1), `%clock64` around each group of 8 (`read_clock64_after` on the last value), mean per thread and per CTA — a secondary column next to the zero-load chase, comparable with P6's 196–213 cycles.
- Sweep: CS ∈ {1, 2, 4, 8, 16} (CS 1 = local via DSMEM interface); policy ∈ {Default, Spread, LoadBalancing}; 20 clusters per launch (different SM sets); 5 launches.

## Kernel K2 — store visibility (2-point, one-way)
Ranks 0 and 1 only (CS 2…16 but only two ranks active). Rank 1 spins on its own smem flag (`ld.volatile.shared.u32`) and records `%globaltimer` and `%clock64` when it flips; rank 0 records `%globaltimer` immediately before `st.shared::cluster.u32 [flag@1], 1` (or `st.release.cluster`). One-way visibility latency = Δglobaltimer (ns → cycles with the measured clock); repeat 1000 times alternating direction (ping-pong) so that round-trip `%clock64` on one SM gives 2× one-way without relying on `%globaltimer` resolution: rank 0 stores flag to rank 1, rank 1 sees it and stores back, rank 0 measures round trip with `%clock64`. Variants: plain `st.shared::cluster`; `st.release.cluster`; `st.async ... mbarrier::complete_tx::bytes` with the consumer waiting on `mbarrier.try_wait.parity.acquire.cluster` (this is the hardware-signalled path); `red.async` increment; and `atom.shared::cluster.exch` round trip.

## Kernel K3 — local baselines in the same binary
`ld.shared.u32` chase (29–30 cycles) and L2/DRAM chase from `04` are rerun once so the notebook has same-binary, same-clock references.

## CSV columns
`09__dsmem_latency__chase.csv`: `cs, policy, variant, cluster_id, src_rank, dst_rank, smid_src, smid_dst, gpc_src, gpc_dst, same_sm, cycles_per_hop, launch_idx`.
`09__dsmem_latency__fine.csv`: `cs, src_rank, dst_rank, smid_src, smid_dst, hop_idx, cycles`.
`09__dsmem_latency__visibility.csv`: `cs, variant, smid_src, smid_dst, trial, one_way_ns, round_trip_cycles`.
`09__dsmem_latency__loaded.csv`: `cs, src_rank, dst_rank, smid_src, smid_dst, threads, ilp, cta_mean_cycles_per_group, cta_p5, cta_p95, launch_idx`.

## Figures
1. **Histogram** of remote `cycles_per_hop` for each CS (2, 4, 8, 16) overlaid, with dotted vertical lines for local SMEM (K3), L2 near-hit, and DRAM latency (from `04`).
2. Median remote latency vs CS with p5–p95 band; one line per policy.
3. Box plot per variant (a)–(e) at CS 8.
4. Store-visibility one-way latency per variant (plain / release / st.async+mbarrier / red.async) vs load round-trip/2.
5. Fine-grained per-hop series for one pair (to show whether latency is constant or has modes).

## Sanity checks (agent-only)
Expected ranges for judging plausibility during result analysis. They are never presented as pass/fail and never quoted in the notebook. If a value falls outside, investigate (placement, clocks, SASS, cache state, ncu replay) and re-run. A value that survives is a finding about this device; describe it on its own terms.
Remote median ∈ [175, 230] cycles at CS 2 and ≤ 10 % growth to CS 16 (P1/P3); local-via-DSMEM ≈ 33; plain store visibility one-way ≈ 300–650 cycles (P5 saw 625 round-trip-ish) — report whatever is measured; `st.async` + mbarrier path should be faster than polling a plain store (hypothesis to test).

## NVIDIA tools
`ncu --metrics sm__sass_inst_executed_op_dshared_ld.sum,l1tex__t_requests_pipe_lsu_mem_dshared_op_ld.sum,l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_ld.sum,l1tex__m_xbar2l1tex_read_bytes_mem_dshared_op_ld.sum,lts__t_sectors_srcunit_tex.sum --clock-control none` on K1 (prove the loads are DSMEM: dshared counters = 4096 × pairs and `lts__t_sectors_srcunit_tex.sum ≈ 0`); SASS dump showing `LD.E` with the cluster-window address and `CS2R SR_CLOCKLO`.
