# Progress slides — characterising the H100 NVL for DSMEM

Figures: `figs/<name>.png` (matplotlib, 300 dpi). All latencies in cycles at the locked 1635 MHz SM clock
(1 cycle = 0.61 ns); bandwidth in GB/s per SM or TB/s aggregate (21.3 B/cycle = 34.9 GB/s).

---

## 1. Memory hierarchy: latency and bandwidth
Figures: `mem_latency.png` (left), `mem_bandwidth.png` (right)

- DSMEM load: **202 cycles** — 0.75× an L2 hit in the near partition (270), 0.47× a far-partition hit (427)
- Far L2 partition: **+157 cycles**; HBM 506 (near) / 668 (far)
- TMA bulk load: **340 + bytes/107 cycles**; +52 cycles over `ld.global` at both L2 and HBM
- Aggregate: L2 read 7.7 TB/s, HBM read 3.70 TB/s (94 % of peak; 90 % of it with 68 of 132 SMs); DSMEM 2.3–4.0 TB/s

---

## 2. DSMEM access latency
Figures: `dsmem_latency_cs.png` (left), `dsmem_heatmap.png` (right; one 18-SM GPC)

- Remote load latency is independent of cluster size: medians **200–205 cycles** for CS 2–16
- Pairwise latency 186–222 cycles, exactly additive: **L(i, j) = p(i) + p(j)**, p = 93–111.5 cycles per SM
- p depends only on the TPC's position in the GPC; no TPC-local or neighbour shortcut → a central switch per GPC

---

## 3. SM-to-SM fabric bandwidth
Figure: `dsmem_bandwidth.png`

- Per-SM injection bandwidth: **34.9 GB/s (21.3 B/cycle)** for ld, st, st.async and TMA put, to 1 or 15 peers
- Per-SM ejection bandwidth: **53–56 GB/s** with ≥ 3 senders
- Aggregate DSMEM bandwidth: **2.3–4.0 TB/s** (≈ HBM read); decreases with cluster size (idle SMs: 0 / 12 / 12 / 20 for CS 2 / 4 / 8 / 16)
- 16 active SMs per GPC: −10 to −15 % per-SM bandwidth (contention in the GPC switch)

---

## 4. Synchronization overheads
Figures: `sync_costs.png` (left; 256 threads per CTA), `handoff.png` (right; CS 2)

- Every release operation compiles to a GPU-scope `MEMBAR`: `fence.sc.gpu` **281 cycles**, **441** after a prior store
- `barrier.cluster`: **410 cycles** (no prior store), **580** (after a store); relaxed arrive + wait: 46
- mbarrier operations: **29–39 cycles**
- Lowest-cost data hand-off: `st.async` + mbarrier, **565-cycle round trip**; release/acquire flag 1325; global-memory flag 1962–2126

---

## 5. DSMEM collectives: algorithms
Table below (also `collective_algorithms.csv`). One cluster per launch, 256 threads per CTA, N = 2–16.

Synchronization taxonomy:
- **mbarrier-only** — data movement completes on the receiver's mbarrier; no cluster barrier inside the collective
- **single-exchange** — one data exchange (remote loads, stores or atomics) between two cluster barriers
- **round-synchronous** — a cluster barrier after every round

| Collective | Algorithm | Origin | Synchronization | Rounds | Transport |
|---|---|---|---|---|---|
| AllReduce | 2-D mesh | generic | mbarrier-only | 3 | bulk copy |
| AllReduce | Two-shot push | TensorRT-LLM | mbarrier-only | 2 | bulk copy |
| AllReduce | Atomic all-push | generic | single-exchange | 1 | remote atomic |
| AllReduce | One-shot pull | TensorRT-LLM | single-exchange | 1 | remote load |
| AllReduce | Pull RS + push AG | hpc-ops | single-exchange | 1 | remote load/store |
| AllReduce | Replicated direct push | CREDIT | single-exchange | 1 | remote store |
| AllReduce | Binomial reduce + broadcast | MPICH | round-synchronous | 2·log₂N | bulk copy |
| AllReduce | Rabenseifner | MPICH | round-synchronous | 2·log₂N | bulk copy |
| AllReduce | Recursive doubling | ClusterFusion | round-synchronous | log₂N | bulk copy |
| AllReduce | Ring (RS + AG) | NCCL | round-synchronous | 2(N−1) | bulk copy |
| AllReduce | Shift-accumulate (atomic) | ClusterFusion | round-synchronous | N−2 | remote atomic |
| AllReduce | Shift-accumulate (bulk) | ClusterFusion | round-synchronous | N−2 | bulk copy |
| AllGather | Direct pull | generic | single-exchange | 1 | remote load |
| AllGather | Direct push | CREDIT | single-exchange | 1 | remote store |
| AllGather | Binomial gather + broadcast | MPICH | round-synchronous | 2·log₂N | bulk copy |
| AllGather | Bruck | MPICH | round-synchronous | ⌈log₂N⌉ | bulk copy |
| AllGather | Direct shift | ClusterFusion | round-synchronous | N−1 | bulk copy |
| AllGather | Recursive doubling | ClusterFusion | round-synchronous | log₂N | bulk copy |
| AllGather | Ring | NCCL | round-synchronous | N−1 | bulk copy |
| ReduceScatter | Direct push | generic | mbarrier-only | 1 | bulk copy |
| ReduceScatter | Direct pull | generic | single-exchange | 1 | remote load |
| ReduceScatter | Recursive halving | MPICH | round-synchronous | log₂N | bulk copy |
| ReduceScatter | Ring | NCCL | round-synchronous | N−1 | bulk copy |
| AlltoAll | Direct push | generic | mbarrier-only | 1 | bulk copy |
| AlltoAll | Pairwise exchange | FlashAttention-4 | mbarrier-only | 1 (N = 2 only) | bulk copy |
| AlltoAll | Direct pull | generic | single-exchange | 1 | remote load |
| AlltoAll | Bruck | MPICH | round-synchronous | log₂N | bulk copy |
| AlltoAll | Pairwise XOR | MPICH | round-synchronous | N−1 | bulk copy |
| AlltoAll | Shift | generic | round-synchronous | N−1 | bulk copy |


---

## 6. Collective latency (N = 8)
Figure: `coll_latency.png` — one line per synchronization class: the class's algorithm with the lowest latency across message sizes (geometric mean); name at the end of each line

- **mbarrier-only** algorithms have the lowest latency: **1.2–4.4×** below any algorithm with cluster barriers
- Latency is nearly constant up to 4 KiB: **680–1490 cycles** (mbarrier-only)
- Round-synchronous algorithms: **2.6k–5.1k cycles** at 256 B (one cluster barrier per round)

---

## 7. Per-SM bandwidth utilization (N = 8)
Figure: `coll_utilization.png` — same algorithms; measured DSMEM bytes per SM ÷ latency, as % of the 34.9 GB/s injection bandwidth

- ≤ 4 KiB: **≤ 25 %** of the per-SM injection bandwidth for every algorithm
- 64 KiB: **63–86 %** (mbarrier-only), **27–53 %** (algorithms with cluster barriers)
- Single cluster per launch (≤ 16 of 132 SMs): aggregate fabric utilization is not exercised

---

## 8. Per-SM cycle breakdown of AllReduce (rank 0)
Figure: `coll_breakdown.png` — measured time per rank, decomposed into isolated primitive costs; hatched = unattributed

- Ring: cluster barriers are **56–64 %** of the time at 4 KiB (2(N−1) barriers of 580 cycles)
- Two-shot push: bulk-copy issue from one thread is **79 %** at N = 16 (65 cycles per command)
- Ring and two-shot: data transfer **≤ 3 %** of the time up to 4 KiB; dominant only at 64 KiB
- Isolated primitive costs explain a median **85–94 %** of the measured time

---

### Definitions (footnotes)
- *Message size*: payload per CTA (nccl-tests convention: AllReduce M·B; AllGather, ReduceScatter, AlltoAll N·M·B).
- *Injection / ejection bandwidth*: bytes per second one SM sends into / receives from the SM-to-SM fabric.
- *Aggregate DSMEM bandwidth*: sum over all SMs of all co-resident clusters.
- *Utilization*: `l1tex__m_l1tex2xbar_write_bytes_mem_dshared.sum` ÷ N ÷ mean collective latency.
