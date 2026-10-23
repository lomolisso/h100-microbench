# Open-source repo mining notes — H100 cluster / DSMEM / TMA / mbarrier / memory-hierarchy microbenchmarks

All repos cloned read-only under `/tmp/claude-0/-home-user/05205069-64f2-5289-862b-0ae81f107a84/scratchpad/repos/` (`R` below):

| short | path | what |
|---|---|---|
| ACCEL | `R/accel-sim-framework` (dev, 2026) | trace-driven front end, configs, `ISA_Def/hopper_opcode.h`, tuner scripts |
| GPGPU | `R/gpgpu-sim_distribution` (accel-sim fork, branch `dev`) | the performance model accel-sim actually uses; H100/H200 `gpgpusim.config` live here, NOT in accel-sim-framework |
| UBENCH | `R/gpu-app-collection/src/cuda/GPU_Microbenchmark` (branch `dev`, sparse) | the tuner's microbenchmark suite (`get_ubench.sh` fetches this) |
| CSIM | `R/ClusterSim` | Luhnen et al. IISWC'25 ClusterSim (GPGPU-Sim 4 + clusters + SM-to-SM net) |
| HPMLL | `R/NVIDIA-Hopper-Benchmark` | Luo et al. IPDPS'24 / arXiv 2501.12084 |
| BLOG | `R/blog/site` (clone of `seanzw/seanzw.github.io`) + `R/seanzw-random/dsm_bw` (the benchmark code the post links) | "How does Hopper DSM actually move data?" |
| CUTLASS | `R/cutlass_files/{copy_sm90_tma.hpp,barrier.h,cluster_sm90.hpp,copy_sm90.hpp}` (raw from `main`) | canonical PTX wrappers |
| SAMPLES | `R/cuda-samples` (sparse: `cpp/0_Introduction`, `cpp/3_CUDA_Features`, histogram) | see §5: nothing cluster/TMA in C++ on master |

(seanzw.github.io itself is blocked by the egress proxy; the GitHub Pages source repo was cloned instead and the article HTML was stripped to text.)

---

## 1. accel-sim-framework (+ gpgpu-sim_distribution dev, + GPU_Microbenchmark)

### 1.1 Where things are
* `ACCEL/gpu-simulator/configs/tested-cfgs/{SM90_H100,SM90_H200}/trace.config` — only opcode latency/initiation + specialized units.
* `ACCEL/gpu-simulator/setup_environment.sh:68` → `GPGPUSIM_REPO=https://github.com/accel-sim/gpgpu-sim_distribution.git`, `GPGPUSIM_BRANCH=dev`.
* `GPGPU/configs/tested-cfgs/SM90_H100/{gpgpusim.config,config_hopper_islip.icnt}` and `SM90_H200/` (identical except `-gpgpu_n_mem 48`).
* `ACCEL/util/tuner/{README.md,get_ubench.sh,run_all.sh,tuner.py,config_template/}`; `get_ubench.sh` sparse-clones `accel-sim/gpu-app-collection` (`src/cuda/GPU_Microbenchmark`, `src/cuda/cuda-samples`).
* `ACCEL/gpu-simulator/ISA_Def/hopper_opcode.h` — SASS opcode → unit map (+ GMMA N→latency table).
* `ACCEL/gpu-simulator/trace-driven/trace_driven.cc` — how Hopper SASS (SYNCS/UTMA*/UBLK*/FENCE/ARRIVES) are decoded from NVBit traces.

### 1.2 H100 `gpgpusim.config` (GPGPU/configs/tested-cfgs/SM90_H100/gpgpusim.config) — key values

| knob | value | note |
|---|---|---|
| `-gpgpu_n_clusters` / `-gpgpu_n_cores_per_cluster` | 132 / 1 | 132 SMs, 1 SM per "cluster" (gpgpu-sim cluster = TPC-ish group, NOT a CUDA cluster). **No GPC/CPC grouping at all.** |
| `-gpgpu_n_mem` / `-gpgpu_n_sub_partition_per_mchannel` | 40 / 2 | 40 HBM3 channels (5 stacks × 8), 80 L2 sub-partitions. H200: 48 / 2 (6 stacks). |
| `-gpgpu_n_chiplet_partition 2`, `-inter_chiplet_queue_size 256`, `-gpgpu_chiplet_interleave 0` | | models the 2 L2 halves ("chiplets"); `gpu-sim.h:333`: if interleaved, chiplet = `(tpc/8) % n_chiplet`, else contiguous halves. `l2cache.cc:91` divides `m_n_mem` by chiplet count. |
| `-gpgpu_clock_domains` | `1980:3960:3960:3106` | core : icnt : L2 : DRAM (MHz). icnt & L2 run at 2× core. |
| `-gpgpu_cache:dl2` | `S:256:128:20,L:B:m:L:X,A:384:4,32:0,32` | per sub-partition: 256 sets × 128 B lines × 20-way = 640 KiB; ×80 = 50 MiB. Sectored (`S`), 32 B sectors, 384 MSHR entries, set index fn `X` (xor hash). |
| `-gpgpu_memory_partition_indexing 6` | | comment: "Use IPOLY_Modulo for non-2^k partition count" (`addrdec.cc:135` switch). |
| `-gpgpu_mem_addr_mapping` | `dramid@8;00000000.00000000.00000000.00000000.0000RRRR.RRRRRRRR.RBBBCCCC.BCCSSSSS` | `-gpgpu_mem_address_mask 1` |
| `-gpgpu_lrc_enabled 1`, `-gpgpu_lrc_max_entries 1024`, `-gpgpu_lrc_max_merged 4` | | "L2 Request Coalescer"; comment admits L2 address mapping unknown. |
| `-gpgpu_l2_rop_latency 237`, `-dram_latency 194` | | partition-side latencies (core cycles) |
| `-gpgpu_cache:dl1` | `S:4:128:64,L:T:m:L:L,A:512:64,16:0,32` | + `-gpgpu_adaptive_cache_config 1`, `-gpgpu_unified_l1d_size 256`, `-gpgpu_shmem_option 0,8,16,32,64,228`, `-gpgpu_l1_banks 4`, write-through/alloc `T:m:L`. |
| `-gpgpu_l1_latency 41`, `-gpgpu_smem_latency 29` | | (ClusterSim uses 41 / 20) |
| `-gpgpu_shmem_size 233472` (=228 KiB), `-gpgpu_shmem_per_block 49152`, `-gpgpu_shmem_num_banks 32` | | |
| `-gpgpu_kernel_launch_latency 5000`, `-gpgpu_max_concurrent_kernel 128`, `-gpgpu_TB_launch_latency 0` | | |
| `-gpgpu_shader_registers 65536`, `-gpgpu_shader_core_pipeline 2048:32`, `-gpgpu_shader_cta 32`, 4 schedulers `lrr`, sub-core model | | |
| `-gpgpu_pipeline_widths 4,4,4,4,4,4,4,4,4,4,8,4,4`; 4 SP/SFU/DP/INT/tensor units | | |
| `-network_mode 1`, `-inter_config_file config_hopper_islip.icnt` | | intersim2 (booksim): `topology=fly; k=212; n=1` (132 SM + 80 L2 nodes), `flit_size=40`, `subnets=2`, `internal_speedup=4.0`, islip VC/SW alloc, 1 VC, 256-flit buffers. |
| DRAM | `-gpgpu_dram_buswidth 8 -gpgpu_dram_burst_length 4 -dram_data_command_freq_ratio 2`, FR-FCFS (`-gpgpu_dram_scheduler 1`), queue 128, return queue 384, `-gpgpu_simple_dram_model 1` (!), HBM3 timing `nbk=16:CCD=1:RRD=6:RCD=19:RAS=45:RP=19:RC=50:CL=19:WL=6:CDLR=8:WR=21:nbkgrp=4:CCDL=6:RTPL=8` (old H100 timing kept as comment). |
| `trace.config` | `int 4,2 / sp 4,2 / dp 8,2 / sfu 23,8 / tensor 12,4`; spec units: BRA 4,4; TEX 200,4; TENSOR 12,4; UDP 4,1; GMMA 128,32 | per-opcode latency,initiation |

`hopper_opcode.h` GMMA table (from `lat_gmma` / `MaxFlops_gmma` ubench): N → (latency, initiation) = 256→(128,128), 224→112, 200→100, 192→96, 160→80, 144→72, 128→64, 120→60, 112→56, 104→52, 96→48, 88→44, 80→40, 64→32, 56→56(!), 40→20, 32→24, 24→12, 16→20, 8→18. (Roughly latency ≈ N/2 cycles for N≥64.)

### 1.3 Hopper SASS mnemonics known to accel-sim (`hopper_opcode.h`)
Hopper-specific / relevant entries (unit in parentheses):
* TMA: `UBLKCP`, `UBLKPF`, `UBLKRED`, `UTMALDG`, `UTMAPF`, `UTMAREDG`, `UTMASTG`, `UTMACCTL`, `UTMACMDFLUSH` (all `TMA_OP`), `STAS` (`STAS_OP`, = `st.async` remote store w/ mbarrier).
* Sync: `FENCE` (`FENCE_OP`), `SYNCS` (`SYNCS_OP`, = all mbarrier ops), `ARRIVES` (`ARRIVES_OP`; only `ARRIVES.LDGSTSBAR.64.{TRANSCNT,ARVCNT}` handled), `UCGABAR_ARV` / `UCGABAR_WAIT` → **`OP_NOP, ALU_OP`** (cluster barrier is NOT modeled), `CGAERRBAR`→ERRBAR, `ACQBULK`→NOP, `STSM`→NOP, `USETMAXREG`/`PREEXIT`/`USETSHMSZ`→NOP, `ELECT` (branch unit), `WARPGROUP` (DEPBAR-like), `QGMMA/HGMMA/IGMMA` (spec unit 5).
* Other new: `VIADD`, `VIADDMNMX`, `VIMNMX`, `VIMNMX3`, `LDGSTS`, `LDGDEPBAR`, `LDSM`, `MOVM`, `REPLAY_START/END` (pseudo, spinlocks).
* **Not in the map**: `MAPA`, `LD/ST.*.CLUSTER`-style DSMEM variants (DSMEM accesses in traces are presumably plain `LD`/`ST` with a different address space, not distinguished), `MEMBAR.ALL.CTA` variants etc.

`trace_driven.cc` decode rules worth knowing (they are effectively NVIDIA SASS↔PTX ground truth gathered by the accel-sim authors):
* `SYNCS.EXCH.64` = `mbarrier.init` (count recovered from the register: `((lower>>1) - 0x100000)` negated; bit 31 of upper = "init as one").
* `SYNCS.ARRIVE.TRANS64[.RED]` = `mbarrier.arrive.expect_tx`; `SYNCS.ARRIVE.*.ART0` = arrive count from reg; `.A1T0` = arrive(1); `.A0TR` = expect_tx(reg); `.A0TX` = `complete_tx`; `.A0T1` = tx 1.
* `SYNCS.PHASECHK.TRANS64` = `mbarrier.test_wait`; `SYNCS.PHASECHK.TRANS64.TRYWAIT` = `mbarrier.try_wait` (phase in src reg).
* `FENCE.VIEW.ASYNC.S` = `fence.proxy.async.shared::cta` ("TODO microbenchmark for cluster fence").
* `UBLKCP.S.G` = bulk load (global→shared), `UBLKCP.G.S` = bulk store, `UBLKCP` S→S = "no_memory_op" (DSMEM bulk copy not modeled as traffic!). `UTMALDG/UTMASTG/UTMAREDG/UBLKRED` → `cache_op = CACHE_GLOBAL` ("TMA access L2 only"); `UTMAPF/UBLKPF` → NOP with warning. TMA traces carry `tma_mbar_addr`, `tma_byte_count`, `tma_oob_byte_count`, `tma_is_multicast`, `tma_multicast_cta_mask`, `cluster_id/cluster_rank/cluster_cta_id` (`trace_driven.cc:260,346-367`).
* `DEPBAR` immediate semantics (LDGSTS groups), `UTMACMDFLUSH` = TMA store commit_group.
* gpgpu-sim `shader.cc` has per-cluster mbarrier tables (`ldst_unit::mbarrier_init/expect_tx/complete_tx/arrive/arrive_drop/invalidate`, keyed by `ClusterCTAIdentifier{cluster_id, cluster_rank}`) and multicast complete_tx fan-out via CTA mask (`shader.cc:2961-2970, 3107-3113`). **No SM-to-SM network, no DSMEM latency/bandwidth, no cluster barrier cost** anywhere in gpgpu-sim_distribution dev (grep `dsmem|cluster_shmem|sm_2_sm` → 0 hits).

### 1.4 Tuner microbenchmarks (UBENCH/ubench/*) — what they are and how they're written
Lineage: Khairy et al. "Accel-Sim" (ISCA'20) + Jia et al. Volta dissection. `run_all.sh` order: system_config, core_config, config_{dpu,fpu,int,sfu,tensor,udp} (--blocks 1), regfile_bw, l1_config, l1_lat, l2_config, l2_copy_engine, l2_lat, mem_config, mem_lat, shd_config, shared_lat, kernel_lat. `tuner.py` parses lines starting with `-` printed by the `*_config` binaries into `gpgpusim.config`.

| ubench | shape (file) |
|---|---|
| `l1_cache/l1_lat` | 1 thread; `posArray[i]=&posArray[i+1]` ring over `array_size` u64; warm-up load `ld.global.ca.u64`; `bar.sync 0`; `%clock` start; `repeat_times` dependent `ld.global.ca.u64 %0,[%1]` (reg swap); `%clock` stop; lat = Δ/iters. (`l1_lat.h`) |
| `l2_cache/l2_lat` | same with `ld.global.cg.u64`, ARRAY_SIZE 4096 ×8 B (32 KB), ITERS 32768, 1 block × 1 thread, asserts array < L2. (`l2_lat.h`) |
| `l2_cache/l2_copy_engine` | `l2_lat_no_warmup`: 32768-entry chase, no warm-up, to detect that the cudaMemcpy copy engine leaves data L2-resident. |
| `mem/mem_lat` | THREADS_NUM 4 (4 × 8 B = 32 B = one DRAM atom), one warp, init with `ld.global.cv`, chase with `ld.global.cv.u64`, ITERS 32768, interleaved so each thread touches a different 8 B of each 32 B atom; array = several × L2 size; lat = Δ/(ITERS/4). |
| `shd/shared_lat` | 32 KB static smem of u64, `s[i]=(i+stride)%N`, 1 thread, 2048 iters, `%clock`. |
| `l1_cache/l1_bw_32f` | ARRAY 16384 floats (<L1), REPEAT 4096, per iter 4× `ld.global.ca.f32 [%4+0/+128/+256/+384]` (bank-spread), `%clock64`, accumulates to sink. |
| `l2_cache/l2_bw_32f` | warm-up loop then `ld.global.cg.f32` streaming, `%clock64`, blocks = all SMs. |
| `l2_cache/l2_bw_128`, `l1_bw_128`, `*_64f` | v4 / 64-bit variants. |
| `mem/mem_bw` | 6 arrays (A..F) streaming read/write, blocks = many SMs, `%clock`. |
| `shd/shared_bw`, `shared_bw_64`, `shared_bank_conflicts`, `shared_cp_async` | 32 KB smem, 4096 iters, `%clock64`. |
| `core/MaxFlops_*` | 1024 reps × 8 FMA chain per thread, `%clock`, flops = reps×threads×8/cycles. `MaxFlops_gmma`, `lat_gmma` (wgmma sweep used for the GMMA table). |
| `atomics/Atomic_add_lat` | 1 thread, 4096 dependent atomicAdd, `%clock`. `Atomic_add_bw{,_conflict,_diverge,_profile}`, `Spinlock_simple`. |
| `l2_cache/l2_config` | derives: `L2_CACHE_LINE_SIZE 128`, `L2_SECTOR_SIZE 32`, `L2_CACHE_ASSOC 16`, L2 banks = `L2_BANKS_PER_MEM_CHANNEL(2) × channels`, size per bank = L2/banks → sets; chooses linear vs IPOLY (`ACCELSIM_IPOLY_HASH_SUPPORT 64`), icnt flit = `L2_BANK_WIDTH_in_BYTE(32)+ACCELSIM_ICNT_CONTROL(8)=40`. |
| `l1_cache/{l1_config,l1_associativity,l1_banks,l1_mshr,l1_sector,l1_access_grain,l1_adaptive,l1_write_policy}` | cache-structure probes; `l1_config` hardcodes 128 B line, 4 sets, 32 B sector, MSHR/warp = 8. |
| `l2_cache/l2_access_grain`, `l2_write_policy`, `mem/mem_atom_size`, `mem/mem_config` | stride sweeps (1..32 words) for access granularity. |
| `system/kernel_lat` | empty-ish kernels with 1..2048 TBs of 1024 threads → `gpgpu_kernel_launch_latency`, TB launch cost. |
| `tma/{mbarrier,tma_bulk,tma_tensor}` | **not timing benchmarks**; they exercise every `mbarrier.*` PTX variant and `cp.async.bulk(.tensor)` LD/ST/RED/PF so the authors can map PTX→SASS (`SYNCS.*`, `UTMA*`) in NVBit traces. `tma_tensor`: 1024×1024 int, 32×32 box, 128 iters, `cuda::ptx::cp_async_bulk_tensor(space_cluster, space_global,...)` + `barrier_arrive_tx`. |

`hw_def/hopper_H100_hw_def.h`: `L1_SIZE 256 KB`, `CLK_FREQUENCY 1455`, `DRAM_MODEL HBM`, `WARP_SCHEDS_PER_SM 4`, `L2_BANKS_PER_MEM_CHANNEL 2`, `L2_BANK_WIDTH_in_BYTE 32`, `SASS_hmma_per_PTX_wmma 2`. (common.h: HBM bus width 128 bit, burst 2, freq ratio 2 → 32 B atom.)

---

## 2. ClusterSim (Tim453/ClusterSim)

### 2.1 H100 config (`CSIM/configs/tested-cfgs/SM90_H100/gpgpusim.config`) — only the cluster-relevant deltas vs §1.2
```
-gpgpu_n_clusters 132 / -gpgpu_n_cores_per_cluster 1
-gpgpu_n_mem 32 / -gpgpu_n_sub_partition_per_mchannel 1          # NOT tuned: 32×(64 sets×128B×16way=128KiB) = 4 MiB "L2"
-gpgpu_n_cores_per_gpc 18,18,16,16,16,16,16,8,4,2,2               # 11 GPC-ish groups summing to 132 (derived from scheduler.cu)
-sm_2_sm_network_type crossbar        # {crossbar, ringbus, none, (else) ideal}
-sm_2_sm_network_latency 171          # cycles, fixed, after serialization
-sm_2_sm_network_bw 88                # bits/cycle per input port (= 11 B/cycle)
-gpgpu_cluster_arrive_latency 610     # barrier.cluster.arrive
-gpgpu_cluster_wait_latency 60        # barrier.cluster.wait
-gpgpu_clock_domains 1620:1620:1620:1593:1620   # 5th field = SM-to-SM network clock
-gpgpu_l1_latency 41 / -gpgpu_smem_latency 20 / -gpgpu_l2_rop_latency 242 / -dram_latency 224
-gpgpu_cache:dl2 S:64:128:16,L:B:m:L:P  ; -gpgpu_memory_partition_indexing 2 ; -network_mode 2 (local xbar) ; -gpgpu_kernel_launch_latency 4477
```
The 610/60/171/88 numbers live in the config and in `src/gpgpu-sim/sm_2_sm_network.cc:7-18` as defaults. **The numbers you remembered (198 cycles, ~21 B/cycle) are not in this repo**; they may be in the paper. The paper's GPC map is `18,18,16,16,16,16,16,8,4,2,2` (not `16,14,2,2`).

### 2.2 Network model (`src/gpgpu-sim/sm_2_sm_network.{h,cc}`, `shader.cc`)
* One `SM_2_SM_network` per `gpu_processing_cluster` (= GPC, `shader.cc:4985-5003`), `n_shader = SMs in that GPC`; `sid_to_gid(sid) = sid % n_shader`.
* **Crossbar**: per-source input FIFO; each cycle, per *output* port round-robin pick one requesting input, transfer `min(bw, remaining bits)`; when a message is fully serialized it enters `in_flight` and is delivered after `m_latency` cycles → output queue. No egress contention beyond the one-input-per-output-per-cycle rule; `HasBuffer` always true (infinite buffers). Advanced once per sm2sm clock (`gpu-sim.cc:1952`).
* Message size: `Push(...)` ignores the passed 256 and uses `data->size*8` bits (`*4` for atomics), where `size = (#distinct words accessed on target SM) × WORD_SIZE` computed in `abstract_hardware_model.cc:360-420` (per sub-warp, per target SM, bank-conflict count `max_bank_accesses` stored as request `latency`).
* Receive side (`ldst_unit::cycle`, `shader.cc:2736-2743`): pops at most one response every 6 cycles (`m_dsmem_latency = 6`), during which `shared_cycle` stalls (BK_CONF). Requests are pushed from the dispatch register after the normal shared-memory bank-conflict stage (`shader.cc:2828-2838`). So a remote access costs ≈ serialization(bytes×8/88) + 171 + 6/response + local smem pipeline.
* Ringbus: per-hop latency `m_latency`, clockwise, unbounded. IdealNetwork: fixed latency only.
* Cluster barrier (`barrier_set_t`, `shader.cc:3850-3880`): `ARRIVE` sets warp `m_sync_latency = cluster_arrive_latency`, `WAIT` sets `cluster_wait_latency`; barrier bookkeeping keyed by `cid = sid % shader_per_gpc`.
* Cluster CTA scheduling (`gpu_processing_cluster::issue_parallel_cta_cluster`, `shader.cc:4666`): default & `Spread` policies → place the N CTAs round-robin over TPC-clusters/cores of one GPC starting after the last-issued core, only if the GPC has `ctas_per_cluster` free slots; `LoadBalancing` → `issue_loadBalanced_cta_cluster`. GPCs are polled round-robin (`gpu-sim.cc:1962`).
* PTX: `mapa` (`instructions.cc:4656`), `barrier.cluster`, `%cluster_ctaid/_ctarank/_nctaid/_nctarank`, `.explicitcluster/.maxclusterrank/.reqnctapercluster`, `cudaLaunchKernelExC`. DSMEM generic address encoding: `(target_shader_id+1)*SHARED_MEM_SIZE_MAX + offset` (`instructions.cc:1968`).

### 2.3 Microbenchmarks (`CSIM/benchmarks/network.cu`, `scheduler.cu`)
Launch: `cudaFuncSetAttribute(..., cudaFuncAttributeNonPortableClusterSizeAllowed, 1)`; `cudaLaunchKernelEx` with `cudaLaunchAttributeClusterDimension = {c,1,1}`, grid = `c × tbcs`. CLI: `-c` cluster dim (8), `--config` {0 LATENCY, 1 BW_SEQUANTIAL, 2 BW_BROADCAST, 3 BW_PAIR, 4 BW_RING, 5 BW_ALL, 6 LT_ALL}, `-t` threads/block (default **1**), `-g` #clusters (1), `-i` iterations (10000), `-s` stride (pair). No 1-block/SM forcing beyond the static 40 KB smem (`array<long,5000>`) — with 1024-thread blocks that is enough; with `-t 1` it is not (only the cluster co-scheduling guarantee separates ranks). Results are written as `c×c` CSV with the `%smid` of each rank on row/column headers.

```cpp
// network.cu — shared data + remote pointer
__shared__ array<long, 5000> s_data;                 // 40 KB static, long (8 B) elements
init_smem(s_data.data(), s_data.size());            // smem[i] = (i+stride) % size, stride 0 => i -> i  (!)
auto dsmem = cluster.map_shared_rank(s_data.data(), receiver);
// latency: 1 dependent chain per thread, clock64, divided by iterations
template <typename T> __device__ double measure_latency(T *mem, int size, int iterations) {
  int my_val = 0; __syncthreads(); auto start = clock64();
  for (int i = 0; i < iterations; i++) my_val = mem[my_val];
  const auto endClock = clock64(); __syncthreads();
  auto result = (double)(endClock - start) / iterations;
  atomicAdd(&g_value[blockIdx.x], my_val); return result; }
// bandwidth: same chain per thread, bytes = iterations*sizeof(T)*nthreads / cycles (clock64 after __syncthreads)
```
Patterns: LATENCY = for each loader rank (serialized with `cluster.sync()`), chase into each receiver rank incl. self; BW_SEQUANTIAL = one (broadcaster, reader) pair active at a time; BW_BROADCAST = all ranks read `broadcaster` simultaneously; BW_PAIR = `target = blk%(2*stride)<stride ? blk+stride : blk-stride`; BW_RING = `target = (blk+1)%n`; BW_ALL/LT_ALL = each block's threads split into `n` groups each targeting a different rank. Note `init_smem` with stride 0 makes `smem[i]=i`, so every chase hits the same word (`mem[0]` forever for latency, `mem[tid]` per thread for BW) — effectively a single-address / same-bank-free pattern, not a cache-defeating chase (fine for smem, but means the "pointer chase" is degenerate).

`scheduler.cu`: `busy_kernel` spins ~100 000 `clock64` cycles with 1024 threads/block, cluster 16 × 30 clusters; records `%smid` per (cluster, rank); host unions overlapping SMID sets across clusters (DSU) and prints the merged SM groups → this is how `gpgpu_n_cores_per_gpc 18,18,16,16,16,16,16,8,4,2,2` was derived (groups of SMs that ever co-host a cluster = GPC/co-scheduling domains).

---

## 3. HPMLL/NVIDIA-Hopper-Benchmark (Luo et al.)

### 3.1 DSM (`NewFeatures/DSM/`)
**Latency** (`Latency/OneToAll/dsm_latency.cu` + `main.cpp`): cluster sizes 2..16, **1 thread per block**, dynamic smem = 1024 × uint (4 KB), `smem[i]=i+1` (stride-1 chase, 1024 hops), `clock()`-timed, `(end-start)/array_size` per (src rank, dst rank) incl. self; `cudaFuncAttributeClusterSchedulingPolicyPreference = 0` (default); `%smid` recorded per rank into `d_Data[rank]`; serialized with `cluster.sync()` per source rank.
```cpp
for (uint b = 0; b < cluster_size; ++b) { if (b == blockIdx.x) for (i...) {
  dst_smem = cluster.map_shared_rank(smem, i); start = clock();
  for (int j = 0; j < array_size; j++) index = dst_smem[index];
  end = clock(); d_Data[cluster_size + cluster_size*rank + i] = (end-start)/array_size; } cluster.sync(); }
```
**Throughput** (`Throughput/{Ring,Pair,Broadcast,Local}/dsm_throughput.cu`, `main.cpp`): `arraySize = 16 KiB` dynamic smem, **1024 threads/block**, `block_num = 114*220*2 = 50160` blocks (so many clusters per SM over time), `repeat_times = 65536` dependent loads per thread, cluster sizes {2,4,8,16}; init `smem[i] = (i+1024) % 4096` (each thread's chain cycles through 4 words 1024 apart → conflict-free, same bank per thread). Timed on host with sdkTimer over the whole launch: TB/s = `repeat × 4 B × 1024 × block_num / t`. Target: Ring `(rank+1)%n`; Pair `(rank/2)*2 + (rank+1)%2`; Broadcast: all ranks ≠0 read rank 0 (rank 0 idle) with scheduling policy **2 = LoadBalancing**; Local: own smem (policy Default). Kernel core: `for (i<repeat) temp = dst_smem[temp];`.

### 3.2 TMA (`NewFeatures/TMA/`)
Common issue idiom (note the order: copy first, then `expect_tx`; works because `complete_tx` can precede `expect_tx` within a phase):
```cpp
asm volatile("{\t\n"
  "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes[%0], [%1], %2, [%3];\n\t"
  "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%3], %2;\n\t" "}"
  :: "r"(smem_u32), "l"(gptr), "n"(LOAD_SIZE), "r"(mbar_u32) : "memory");
token = bar.arrive(); bar.wait(std::move(token));   // cuda::barrier<thread_scope_block>, init(&bar, 1)
```
* `Latency/tma_lat_uniform/tma_lat.cu`: 1 block × 1024 threads build a chase (`posArray[i] = &posArray[i+STRIDE]`, STRIDE 2 × 8 B = 16 B) over `ARRAY_SIZE 62914560` u64 (= 480 MiB, > L2) then uid 0 issues ITERS 65536 dependent 16-B bulk copies (`LOAD_SIZE 16`), next address = `ptr[0]` read from smem after wait; `%clock` around the loop; prints Δ/ITERS. Barrier is `init(&bar, 1)` with the single thread arriving. (`-ptx` only `cuda/barrier`.)
* `Latency/tma_lat_random/main.cu`: RRZE-style random chase: `skip_factor 16` (16 × 8 B = 128 B granularity), sizes `LEN = 16 .. 2^24` growing ×1.04+32, random cyclic permutation, iters = max(LEN, 1e6), 7 reps, min time, cycles = `dt/iters × clock` with clock from NVML `nvmlDeviceGetClockInfo(NVML_CLOCK_SM)` while a `powerKernel` warms the GPU; each hop is a 16-B TMA bulk copy with 32× unroll. Plot via `plot.py`.
* `Throughput/tma_bw/tma_bw.cu` (1-D bulk, non-tensor): 4 GiB float array, grid ∈ {114,228,342,456} blocks × **1 thread**, `LOAD_SIZE ∈ {1,2,4,8,12,16} KiB` (default 4 KiB), each block serially copies its chunks (`i += grid*LOAD`), cudaEvent timing → GB/s.
* `Throughput/tma_bw_1d`: tensor 1-D, dtype int64, `cuTensorMapEncodeTiled(rank 1, size={ARRAY_SIZE}, stride={ARRAY_SIZE*8}, box={LOAD/8}, elemstride 1, INTERLEAVE_NONE, SWIZZLE_NONE, L2_PROMOTION_NONE, OOB_FILL_NONE)`, `LOAD_SIZE ∈ {512,768,1024,1280,1536,2048}` B (box ≤ 256 elements), asm `cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2}], [%3];` with `"l"(&tma_desc)` (the `__grid_constant__ CUtensorMap` param address).
* `tma_bw_2d`: float 32768×32768, boxes (W×H) ∈ {16×16, 32×16, 32×32, 64×32, 96×32, 64×64} (default idx 5 = 16 KiB), `...tensor.2d... [%1, {%2, %3}]`.
* `tma_bw_3d`: float 1024³, boxes {16,16,16},{64,64,1},{256,16,1},{16,256,1},{4,4,256} (default idx 4), `...tensor.3d... [%1, {%2,%3,%4}]`.
* `util.h`: `cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", ..., cudaEnableDefault, &driver_status)` → `PFN_cuTensorMapEncodeTiled`.

### 3.3 Memory hierarchy (`RegularUnits/`, derived from shen203/GPU_Microbenchmark = accel-sim's)
* `l1_lat.cu` / `l2_lat.cu`: 1 thread, 4096 × u64 ring, ITERS 32768, `ld.global.ca.u64` vs `ld.global.cg.u64`, `%clock`.
* `mem_lat.cu`: 1 thread, `ARRAY_SIZE 8*1024*128` u64 (8 MiB), `STRIDE 32` B, ITERS 8192, init by 114×1024 threads, warm load `ld.global.cv`, chase `ld.global.cg`, `%clock`. **`#define FINE_GRAINED`** (commented out) wraps each load in `%clock` reads, stores to smem and `printf`s per-access cycles → this is the per-access "fine-grained p-chase" used in the paper to separate near/far L2 partition hits and misses (the discriminating logic is offline on the printed series; nothing in-repo labels partitions).
* `random_latency/main.cu`: RRZE p-chase, `<<<1,4>>>`, 128-B granularity, sizes to 2^24×128 B (2 GiB), 32× unrolled `idx = (int64_t*)*idx`, 7 reps min, cycles via NVML clock — gives the latency-vs-footprint curve (L1 / near-L2 / far-L2 / HBM plateaus).
* `l2_bw_128.cu`: 228 blocks × 1024 thr, `ld.global.cg.v4.f32` of `posArray[((i*32*4)+uid*4) % ARRAY_SIZE]`, REPEAT 2048, array = `TOTAL_THREADS*4 + 2048*32*4` floats (~3.8 MB, must be < L2), warm-up kernel, 10 runs, `%globaltimer` start/stop per thread.
* `mem_bw.cu`: 528 blocks × 1024, 512 Mi floats (2 GiB) streaming, `%globaltimer`, hard-coded `clock_freq_MHZ 2619` (!! 4090-era constant) → B/clk numbers unreliable; GB/s fine.
* `shared_lat.cu`: 48 KB u64 smem, stride 1, 2048 iters, `%clock`. `shared_bw*.cu`: 48 KB, 4096 iters, 1024 threads, `%clock`.

---

## 4. Blog: seanzw.github.io/posts/gpu_dsm_bw/ — "How does Hopper distributed shared memory actually move data?" (+ code `seanzw/random/dsm_bw`)
Other posts on the site index: `gpu_load_bandwidth` (RTX 5090 L2 bandwidth saturation with normal ld / cp.async / TMA; not Hopper), plus gem5/RISC-V/LLM posts — only `gpu_dsm_bw` is Hopper-relevant.

**Setup (ANALYSIS.md, kernels.cuh):** single **H200**, exactly **one cluster** per launch (sizes 2/4/8/16), **1024 threads/block**, **200 KiB dynamic smem** (`kDynamicSharedBytes`) to force one CTA per SM; host rejects any sample where two ranks report the same `%smid`. Source tile 96 KiB + destination tile 96 KiB inside the 200 KiB. Payload sweep 16,24,…,96 KiB per active SM, each address touched once; 30 samples per size, median, then affine fit `T(B)=α+βB` independently on `%clock64` and `%globaltimer`; BW = 1/β (cycle-fit RMSE 0.4–1.4 % of the 16→96 KiB increase). Timers read as a pair; the end read carries a fake data dependency on the load result (`read_timers_after`) so the last remote load is retired before the timestamp. Checksums after the end timestamp. SM clock ≈1.785 GHz (cycle/ns slope ratio 1.77–1.80).
```cpp
// remote load: 16 B per thread, 4-deep ILP, active_warps (16 default) × 32 threads
asm volatile("ld.volatile.shared::cluster.v4.u32 {%0,%1,%2,%3}, [%4];" ...);   // address from map_shared_rank + __cvta_generic_to_shared
// remote store
asm volatile("st.volatile.shared::cluster.v4.u32 [%0], {%1,%2,%3,%4};" ...);
// TMA "put" smem->remote smem, chunks of ≤16 KiB, one remote mbarrier for the whole tile (expect_tx(epoch_bytes) once)
asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: dst_u32, src_u32, bytes, remote_mbar_u32);
// wait: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 loop; init: mbarrier.init.shared::cta.b64 [bar], 1; expect: mbarrier.arrive.expect_tx.shared::cta.b64 _, [bar], bytes
// fence.proxy.async.shared::cta after init/expect, then cluster.sync() before issuing.
```
Patterns: PairUnidirectional (rank0→rank1), PairBidirectional (rank^1, ranks 0,1), Ring (`(rank+param)%n`), PackedRing (groups of `param>>16` with stride `param&0xffff`); Mixed: rank 0 always reads rank 1, a store/TMA put carries a 2nd payload rank1→rank0 (same direction as read responses) or rank0→rank1 (opposite); "pressure" mode adds 1–16 warps of local smem loads on the source SM.

**Numbers (B per SM-cycle unless noted):**
| experiment | result |
|---|---|
| remote `ld.shared::cluster` one-way | **19.87** (35.83 GB/s); fit α≈493 cycles, β=0.0503 cyc/B |
| ordinary remote store one-way | **18.87** (33.68 GB/s) |
| TMA remote put one-way | **21.25** (37.90 GB/s) |
| symmetric 2-way, per direction | load **15.37** (−22.6 %), store **18.12** (−4.0 %), TMA **20.29** (−4.5 %) |
| mixed read + store, same direction / opposite | 21.38 / 29.70 aggregate (+39 %) |
| mixed read + TMA put, same / opposite | 21.45 / 29.26 (+36 %) |
| TMA put ring scaling 2/4/8/16 SMs (aggregate) | 40.59 / 82.02 / 164.30 / 338.89 (= 20.29 / 20.50 / 20.54 / 21.18 per SM); 16-SM within 0.72 % of 16×21.33 |
| fixed-64 KiB concurrent load rings (raw GB/s) | 2-SM 51.2, 4-SM 102.4, 8-SM 204.8, 16-SM 344.9–372.4 (stride 2 & 14 best: 372.4 / 368.2), eight packed adjacent pairs 390.1 (+11 % vs median 16-ring) |
| isolated directed pairs (all 240 of 16 ranks) | 31.51–32.00 raw GB/s — no visible topology; 32-B address-offset scan over 4 KiB changes cycles by 0.39 % |
| source-SM local smem pressure (1–16 warps, conflict-free or 4-bank-camping) | remote-load completion stays 3,628–3,633 cycles / 32.0–32.5 GB/s → destination SRAM/LSU is not the limiter |
| TMA command granularity | 128-B commands only 2.35 B/cycle → ≈54.5 cycles per TMA command front-end cost; 4/16 KiB trains 21.20; cold single command 21.14 |
| completion-poll quantization | `TRYWAIT`+`YIELD` retry loop quantizes in ≈147-cycle steps |
| fitted intercept vs cluster size (TMA put) | ≈1,101 cycles @2 SMs → ≈5,256 @16 SMs |
| per-128 B cost (load) | 6.441 cycles one-way → 8.327 two-way (+1.886 ≈ one 32-B flit at 1.5 cycles) |

**Model / interpretation (patent US12248788B2 "Distributed shared memory", Figs 21A–D):** per CPC six SM paths (TPCARB → GPCMMU/uTLB address hash) converge at GPCARB ingress onto **four 32-B lanes** (`SM2SM4`; 3 CPCs → `SM2SM12`) into two parallel 6×6 switch planes GX0/GX1; egress per CPC = two GPCARB-EG blocks each fanning out to 3 SMs. 4×32 B / 6 SM = **21.33 B/SM-cycle**, implemented as a hard, non-work-conserving per-SM shaper (idle neighbours don't raise one SM's rate). Reads: 128-B response quantum = 4 flits × 1.5 cycles = 6 cycles → 21.33; two-way adds one read-command flit → 7.5 cycles → 17.07, ×(19.87/21.33) = 15.90 vs measured 15.37. Writes carry payload forward and the reverse ACKs are coalesced → only 4–6 % duplex loss. Read responses and write payload share one directional physical resource (VCs are logical). Rejected hypotheses: separate ~1 GHz DSM clock, universal half-duplex port, 64-B flit, TMA-only bypass, destination-SRAM stall, large token bucket, TPC-local bypass. Unresolved: literal TDM vs issue cap, address hash, header layout, per-hop credits, hidden clock crossing.

---

## 5. cuda-samples + CUTLASS ground-truth PTX

**cuda-samples (master, now `cpp/` + `python/` layout):** no C++ sample uses `this_cluster`, `map_shared_rank`, `cp.async.bulk`, `cudaLaunchAttributeClusterDimension` or `CUtensorMap` (grep over `cpp/0_Introduction`, `cpp/3_CUDA_Features`, `cpp/2_Concepts_and_Techniques/histogram` → 0 hits; `3_CUDA_Features` has `globalToShmemAsyncCopy` = cp.async only, tensor-core GEMMs, graphs, CDP). Only `python/2_CoreConcepts/tmaTensorMap/` (cuda.core Python) touches TMA. The old `clusterSync`/cluster-histogram samples are not on this master. Don't bother; use CUTLASS.

**CUTLASS (`include/cutlass/arch/barrier.h`) — exact asm:**
```
ClusterBarrier::init          : mbarrier.init.shared::cta.b64 [%1], %0;                         ("r"(arrive_count), "r"(smem_addr))
ClusterBarrier::wait          : { .reg .pred P1; LAB_WAIT: mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; @P1 bra DONE; bra LAB_WAIT; DONE: }   (addr, phase, ticks)
ClusterBarrier::test_wait     : @P2 mbarrier.test_wait.parity.shared::cta.b64 P1, [%1], %2; selp.b32 %0,1,0,P1;
ClusterBarrier::try_wait      : mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2; selp.b32 %0,1,0,P1;
ClusterBarrier::arrive(remote): { .reg .b32 remAddr32; mapa.shared::cluster.u32 remAddr32, %0, %1; mbarrier.arrive.shared::cluster.b64 _, [remAddr32]; }   (smem_addr, cta_id)
  ..._relaxed_cluster          : mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [remAddr32];
ClusterBarrier::arrive(local) : mbarrier.arrive.shared::cta.b64 _, [%0];
ClusterBarrier::invalidate    : mbarrier.inval.shared::cta.b64 [%0];
ClusterTransactionBarrier::arrive_and_expect_tx        : mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0;    ("r"(bytes), "r"(addr))
  ... (remote, predicated)    : setp.eq.u32 p, %2, 1; @p mapa.shared::cluster.u32 remAddr32, %0, %1; @p mbarrier.arrive.expect_tx.shared::cluster.b64 _, [remAddr32], %3;
  ..._relaxed_cluster          : @p mbarrier.arrive.expect_tx.relaxed.cluster.shared::cluster.b64 _, [remAddr32], %3;
ClusterTransactionBarrier::expect_transaction          : mbarrier.expect_tx.shared::cta.b64 [%1], %0;
ClusterTransactionBarrier::complete_transaction        : setp.eq.u32 p, %2, 1; @p mbarrier.complete_tx.shared::cluster.relaxed.cluster.b64 [%1], %0;   (addr already mapa'd by caller)
fence_barrier_init()          : fence.mbarrier_init.release.cluster;
fence_view_async_shared()     : fence.proxy.async.shared::cta;
(sm100 helper)                : fence.release.sync_restrict::shared::cta.cluster;
cpasync_barrier_arrive        : cp.async.mbarrier.arrive.shared::cta.b64 [%0];   / .noinc variant
```
`include/cute/arch/cluster_sm90.hpp`:
```
cluster_arrive_relaxed(): barrier.cluster.arrive.relaxed.aligned;   cluster_arrive(): barrier.cluster.arrive.aligned;   cluster_wait(): barrier.cluster.wait.aligned;
cluster_grid_dims(): mov.u32 %0, %%nclusterid.{x,y,z};   cluster_id_in_grid(): %%clusterid.{x,y,z};   block_id_in_cluster(): %%cluster_ctaid.{x,y,z};
cluster_shape(): %%cluster_nctaid.{x,y,z};   block_rank_in_cluster(): mov.u32 %0, %%cluster_ctarank;
set_block_rank(smemAddr, rank): mapa.shared::cluster.u32 %0, %1, %2;
elect_one_sync(): { .reg .b32 %rx; .reg .pred %px; elect.sync %rx|%px, %2; @%px mov.s32 %1, 1; mov.s32 %0, %rx; }  ("r"(0xFFFFFFFF))
store_shared_remote(): st.async.shared::cluster.mbarrier::complete_tx::bytes.u32 [%0], %1, [%2];   (dsmem_addr, value, remote_mbar via set_block_rank)
```
`include/cute/arch/copy_sm90_tma.hpp`:
```
SM90_TMA_LOAD_2D         : cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5;
                            : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "r"(crd0), "r"(crd1), "l"(cache_hint)      (1d..5d same pattern; sm120 uses shared::cta)
SM90_TMA_LOAD_MULTICAST_1D: cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%0], [%1, {%4}], [%2], %3, %5;
                            : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "h"(multicast_mask), "r"(crd0), "l"(cache_hint)   (2d: {%4, %5} ... %6)
SM90_TMA_LOAD prefetch   : cp.async.bulk.prefetch.tensor.2d.L2.global [%0, {%1, %2}];
SM90_TMA_STORE_2D        : cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];      ("l"(desc), "r"(smem), crds)
SM90_TMA_REDUCE_ADD_2D   : cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group [%0, {%2, %3}], [%1];
SM90_BULK_COPY_G2S       : cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];   ("r"(smem), "l"(gmem), "r"(bytes), "r"(mbar))
SM90_BULK_COPY_G2S::PREFETCH: cp.async.bulk.prefetch.L2.global [%0], %1;
SM90_BULK_COPY_S2G       : cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;
tma_store_fence()        : fence.proxy.async.shared::cta;   tma_store_arrive(): cp.async.bulk.commit_group;   tma_store_wait<N>(): cp.async.bulk.wait_group.read %0;  ("n"(N))
```
Desc pointer is passed as a 64-bit generic address (`"l"`) of the `__grid_constant__ CUtensorMap` kernel parameter (or a global/const copy); smem/mbarrier addresses as 32-bit shared addresses (`__cvta_generic_to_shared`). The multicast mask is a 16-bit CTA-rank bitmask (`"h"`), and the mbarrier operand is the *issuer's* local smem offset — the same offset is signalled in every destination CTA.

---

## Techniques worth reusing in our microbenchmarks
1. **One CTA per SM via smem footprint** (BLOG: 200 KiB dynamic smem) + **host-side `%smid` uniqueness check**; also record `%smid` per rank in every result row (all four code bases do this). Use `cudaOccupancyMaxActiveClusters` only as a sanity check.
2. **GPC discovery by SMID-set union** (CSIM `scheduler.cu`): launch many 16-CTA clusters of a long-spinning 1024-thread kernel, record SMIDs per cluster, union-find overlapping sets → co-scheduling domains (ClusterSim got 18,18,16,16,16,16,16,8,4,2,2 on their H100).
3. **Size-slope (affine-fit) bandwidth** instead of single-point BW (BLOG): sweep unique-address payload 16–96 KiB, 30 samples, median, fit `T=α+βB` on both `%clock64` and `%globaltimer`, report 1/β and residual RMSE; the α absorbs cluster-sync/mbarrier/TMA-setup cost (which grows ~1.1k→5.3k cycles from 2→16 SMs).
4. **Dual timers** (`%clock64` + `%globaltimer` read back-to-back in one asm block) to detect clock-domain effects; read the end timer through a fake `add.u64` dependency on the last loaded value so the final load is retired (BLOG `read_timers_after`). Cross-check with NVML SM clock (HPMLL/RRZE `getGPUClock`).
5. **Mixed-direction traffic tests** (rank0 reads rank1 while a store/TMA put moves a 2nd payload in the same vs opposite direction) to find shared directional resources; **source-side local-smem pressure** warps to rule out the SRAM port as the bottleneck (BLOG).
6. **TMA smem→remote-smem puts with one remote mbarrier per whole tile, ≤16 KiB per command, `expect_tx(total)` once before `cluster.sync()`**, `fence.proxy.async.shared::cta` after mbarrier init; completion via `mbarrier.try_wait.parity.acquire.cluster` loop. Note per-command front-end cost ≈54.5 cycles (128-B commands reach only 2.35 B/cycle) — pick ≥4 KiB commands for bandwidth, 16-B commands for latency chases.
7. **TMA pointer-chase latency** (HPMLL): 16-B bulk copy whose payload is the next address, `expect_tx` issued *after* the copy (legal), `cuda::barrier` arrive/wait, 480 MiB chase for DRAM, RRZE random permutation at 128-B granularity with min-of-7 timing for the size curve.
8. **DSMEM latency**: 1 thread/block, stride-1 chase in 4 KB (HPMLL) or `mem[mem[...]]` on a 40 KB `long` array (CSIM); all-to-all src×dst matrix incl. self, serialized by `cluster.sync()`; cluster sizes 2..16; also run under `cudaFuncAttributeClusterSchedulingPolicyPreference` {Default, Spread, LoadBalancing} (HPMLL Broadcast uses 2).
9. **DSMEM throughput**: 1024 threads/block, each thread its own dependent chain `temp = dst_smem[temp]` with `smem[i]=(i+1024)%4096` (conflict-free, no address reuse across threads) — HPMLL; or ILP-4 `ld.volatile.shared::cluster.v4.u32` over unique 16-B slots — BLOG. Patterns: ring (+1, +stride), pair (rank^1), broadcast (all→0), packed groups, all-to-all split by thread groups (CSIM BW_ALL).
10. **L2 near/far partition p-chase**: per-access `%clock` deltas around a single `ld.global.cg.u64` (HPMLL `FINE_GRAINED`), 128-B stride random chase (`skip_factor 16`), and `ld.global.cv` for DRAM (bypass L2) — plus accel-sim's `l2_copy_engine` trick (chase without warm-up to see copy-engine L2 residency) and `mem_lat` with 4 threads covering one 32-B atom.
11. **Classic cache probes** (UBENCH): `l1_lat`/`l2_lat` 4096×8 B ring, 32768 iters, `ld.global.ca/.cg`; `l1_bw_32f` with 4 loads at +0/+128/+256/+384 B per iteration; `kernel_lat` with 1..2048 TBs; `atomic_add_lat` dependent chain.
12. **SASS verification idiom** (UBENCH `tma/*`): compile tiny kernels containing every PTX variant and check `cuobjdump -sass` — the `SYNCS.*` / `UTMA*` / `UBLKCP` / `UCGABAR_*` / `STAS` / `MAPA` mnemonics listed in §1.3 are what to expect for cluster/TMA code.
13. **Canonical asm strings** for everything cluster/TMA/mbarrier: copy from §5 (CUTLASS) rather than hand-writing; especially the multicast operand order and `mapa` + `mbarrier.arrive.expect_tx.shared::cluster` for remote barriers.

## Claims to take with a grain of salt
* **ClusterSim's network parameters** (171-cycle latency, 88 bit/cycle = 11 B/cycle per port, 610/60 barrier arrive/wait, 6-cycle response pop) are tuned to match their own PTX-level simulator against their microbenchmarks; the crossbar has infinite buffers, no egress/lane contention, no read/write asymmetry, and the message size is "distinct words × 4 B", so bandwidth per SM in the model (11 B/cycle) is half the BLOG's measured ~20–21 B/cycle. ClusterSim's `BW_*` numbers are B/cycle in a 1.62 GHz "core clock" PTX model, not SASS.
* ClusterSim's H100 `gpgpusim.config` L2/DRAM side is clearly untuned (32 channels × 128 KiB = 4 MiB L2, `memory_partition_indexing 2`, local xbar); only the cluster/network/GPC fields are meaningful.
* `-gpgpu_n_cores_per_gpc 18,18,16,16,16,16,16,8,4,2,2` is a co-scheduling-domain inference from SMID unions on one card with one floorsweeping; the BLOG and patent describe 3 CPC × 3 TPC × 2 SM = 18 SM slots per GPC and H100 SXM exposes 132 SMs — the small groups (8,4,2,2) are likely floorswept-GPC remnants or scheduling artefacts, not physical GPCs.
* Accel-sim's H100 config: `-gpgpu_simple_dram_model 1`, "L2 Request Coalescer" sized by guess ("we don't know L2 address mapping"), `-gpgpu_memory_partition_indexing 6` (IPOLY-modulo) and the DRAM address mapping are heuristics; `lrr` scheduler vs ClusterSim `gto`; icnt `internal_speedup 4` is a knob. `UCGABAR_*` (cluster barrier) is a NOP, DSMEM S→S bulk copy is `no_memory_op`, TMA prefetch is a NOP, TMA is "L2 only" — accel-sim v2 does **not** model the SM-to-SM network at all. `smem_latency 29` (accel-sim) vs 20 (ClusterSim) vs measured ~23–30 depending on method.
* The GMMA latency table has an outlier (`56 → (56,56)`) that looks like a typo relative to the N/2 trend.
* HPMLL throughput numbers are **whole-launch wall-clock TB/s over 50,160 blocks** (host timer, including launch/scheduling, cluster formation and tail effects), so per-SM B/cycle cannot be recovered precisely; the latency test uses `clock()` (32-bit) and divides by only 1024 hops with 1 thread. The Broadcast test silently uses a different scheduling policy (2) than the others (0), confounding comparisons. `mem_bw.cu` hard-codes `clock_freq_MHZ 2619` (an RTX 4090 value) for its B/clk output. TMA BW kernels use 1 thread per block (no overlap inside an SM) and serialize commands with a full arrive/wait per command, so they measure latency-bound TMA throughput per SM, not peak.
* HPMLL TMA latency idiom issues `mbarrier.expect_tx` *after* the copy; correct but unusual — if copied, keep `init(&bar,1)` + single arriving thread, otherwise the phase never completes.
* BLOG is **H200**, not H100 (same GH100 die/SM count; HBM differs, SM clock 1.785 GHz) and a single GPU; the 32-B-lane / 2/3-share / 128-B-quantum model is explicitly an inference (author's own confidence table: "strong inference"/"plausible inference"/"unresolved"). Topology conclusions (adjacent-pair advantage, stride 2/14 favoured) come from raw 64 KiB single-shot rates with ~3 % run-to-run noise. The 147-cycle `TRYWAIT` quantization limits what the burst tests can see.
* Both HPMLL and BLOG rely on `cooperative_groups::map_shared_rank` generic pointers converted with `__cvta_generic_to_shared`; the volatile modifier (`ld.volatile.shared::cluster`) in BLOG prevents the compiler from merging/hoisting but may also disable some hardware coalescing paths — worth testing with/without `.volatile`.
* cuda-samples master has no cluster/DSMEM sample to validate against; the older `clusterSync`-style samples are gone, so don't cite "the NVIDIA sample" for DSMEM idioms — cite CUTLASS.
