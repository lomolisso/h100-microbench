"""Markdown text of the experiment-11 notebook (numbers quoted here are read from the executed outputs of the runs listed
in section 3)."""

TITLE = r"""
# 11 — Bandwidth of the SM-to-SM network (distributed shared memory) on the H100

**Goal.** Measure, on one H100 NVL at a locked 1635 MHz SM clock, how fast the CTAs of a thread-block cluster can move data through each other's shared memory:
- **per SM**, in bytes per cycle, for remote loads, remote stores, `st.async` and TMA shared→shared copies (plain and reduce);
- **for the whole GPU**, in TB/s, when every SM that can join a cluster takes part;
- as a function of cluster size (2–16), traffic pattern, threads and loads in flight, TMA command size and scheduling policy.

Two side questions:
- Does serving remote requests slow down the serving SM's own shared-memory traffic?
- Does SM-to-SM traffic compete with SM→L2 traffic inside a GPC?

**Why it matters.** Cluster algorithms (split-K reductions, halo exchange, distributed tiles, collectives) replace L2 round trips by SM-to-SM transfers. Their speed is set by three things: how much one SM can push into the network, how much it can take in, and how the network is shared when many SMs talk at once. These are also the parameters a model of a 3D-stacked GPC needs for its SM-to-SM links.
"""

BACKGROUND = r"""
## 2. Background

**Clusters and DSMEM.**
- A kernel launched with a *cluster dimension* runs groups of up to 16 CTAs at the same time on SMs of one GPC.
- Each CTA can read and write the shared memory of the others through the `.shared::cluster` window. `mapa` turns a local shared-memory address into the address of the same offset in CTA `rank`.
- A remote access leaves the SM, crosses the GPC's SM-to-SM network and is served by the owner SM's shared memory; a load's reply returns the same way.

On this GPU the 132 SMs form 8 GPCs of 12, 16, 16, 16, 18, 18, 18 and 18 SMs; two SMs form a TPC. With one CTA per SM, at most 66 / 30 / 15 / 7 clusters of 2 / 4 / 8 / 16 CTAs are resident at once, covering 132 / 120 / 120 / 112 SMs.

**Five ways to move data to or from a peer.**

```
ld.shared::cluster.v4.u32   {a,b,c,d}, [peer + off];           // remote load: request out, 16 B per thread back
st.shared::cluster.v4.u32   [peer + off], {a,b,c,d};           // remote store: posted; completion only via a fence
st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32  [peer + off], {a,b,c,d}, [peer_bar];
cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes        [peer + off], [own + off], size, [peer_bar];
cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.add.u32  [...same operands...];
```

- The first three are executed by ordinary threads (the load/store unit). The last two are executed by the SM's TMA unit: one thread hands over a command of up to 96 KiB here (`size`), and the unit moves the data.
- `st.async` and the TMA forms signal completion on an **mbarrier in the receiving CTA**. Before the transfer, the receiver arms the barrier with the number of bytes it expects (`mbarrier.arrive.expect_tx`). Every arriving byte is counted down, and the receiver waits until the count reaches zero. That makes the arrival of the data observable at the receiver without polling the data.
- A plain remote store has no such signal. The sender learns that its stores are done with a memory fence (`fence.acq_rel.cluster`, which waits for every outstanding store of the thread).

**Send and receive.** Each SM has an outgoing and an incoming connection to the network. The profiler states the rated throughput of each as **32 B/cycle per SM** (section 4.11). Which side limits a transfer depends on the operation:
- a load's payload is *sent by the owner* of the data and *received by the reader*;
- for a store or a TMA put the payload is sent by the writer and received by the owner.

**Patterns** (ranks inside one cluster; *k* is a parameter):

| pattern | who moves data to whom |
|---|---|
| pair_uni | rank 0 → rank 1 only |
| pair_bi | 0 → 1 and 1 → 0 |
| ring +k | r → (r + k) mod CS, every rank |
| packed pairs (k) | ranks r with ⌊r / k⌋ even → r + k (half of the ranks issue; k = 1: TPC mates) |
| broadcast_in (k) | ranks 1 … k → rank 0 (loads: *k* readers of rank 0; stores/TMA: *k* writers into rank 0) |
| broadcast_out (k) | rank 0 → each of ranks 1 … k, the full payload to each (stores, TMA) |
| all_to_all | every rank sends 1/(CS − 1) of its payload to each other rank (warp groups) |
| all_to_all_full | as all_to_all, but every group reads a peer's whole tile (loads) |
| randperm | r → perm(r), a random derangement drawn per launch |
| uniform_random | a new random partner (≠ r) for every 16 KiB of the payload |
| mixed same / opposite | rank 0 reads rank 1 while rank 1 writes into rank 0 (same physical direction) / while rank 0 writes into rank 1 (opposite) |
"""

SETUP = r"""
**What is measured and how.**
- **Placement.** 1 CTA per SM (200 KB of dynamic shared memory: a 96 KiB source tile, a 96 KiB destination tile, mbarriers). Every launch uses explicit cluster and scheduling-policy attributes. Every CTA records `%smid`, `%clusterid` and `%cluster_ctarank`.
  - *single*: one cluster per launch, on an otherwise idle GPU; the hardware chooses its SMs (it always took the same ones, section 4.7).
  - *full*: as many clusters as can be resident at once (one wave).
  - A launch whose ranks did not sit on distinct SMs, or whose NVML sample after the launch showed a throttle reason or a clock below the lock, would have been discarded and repeated. None was (table above).
- **One sample** = `cluster.sync()`, then the timed transfer:
  - Every 16-byte element of the payload (16–96 KiB per issuing CTA) is moved exactly once.
  - Each warp access covers 512 contiguous bytes.
  - Each thread keeps ILP accesses in flight per step.
  - Timers: `%clock64` and `%globaltimer`.
- **Where each time is taken:**
  - *loads*: from the barrier to the last thread's end timestamp, which depends on the sum of everything it loaded;
  - *plain stores*: until the last thread has passed its fence;
  - *`st.async` and TMA*: at the receiver, from the barrier to the completion of its mbarrier(s). The issuing thread's own time is also recorded (section 4.6).
  - 6 samples per launch (2 on the full GPU) × 5 recorded launches, after 2 warm-up launches.
- **Per-SM bandwidth** = 1/slope of the affine fit *cycles = α + bytes / rate*:
  - The fit uses the medians over all CTAs, samples and launches at the six payload sizes.
  - α holds the fixed costs (barrier exit, fences, the first TMA command).
  - The residual RMSE of the fit is reported. Where it exceeds 2 %, the rate at 96 KiB (bytes / cycles, α included) is shown instead and marked.
- **GPU-wide bandwidth.** All resident clusters run a ring that streams the 96 KiB tile 4, 8, 16 and 32 times per SM, with one sample per launch. The aggregate is the slope of the cudaEvent time over the total bytes, so launch and setup costs cancel. As a cross-check, per-SM rate × active SMs × clock is also given.
- **Reference lines.** L2 and HBM read bandwidth of this card, measured with the same lock by a separate streaming benchmark (16 MiB L2-resident buffer, or a 2 GiB buffer, read with `ld.global.cg` by every SM).
- **Local-pressure control (Fig. 6).**
  - Rank 0 of a 2-CTA cluster streams 32 × 96 KiB out of rank 1 (or into it).
  - Meanwhile 4–32 warps of rank 1 stream their own shared memory (`ld.shared.v4`, 2 loads in flight, flag polled after every pass).
  - The victim alone serves as the baseline.
- **Interference control (Fig. 7).** One cluster of 16 CTAs covers a 16-SM GPC. Roles are assigned by the SM's TPC index in the GPC, using the groups A = {0, 3, 6}, B = {1, 4, 7}, C = {2, 5}:
  - *L2 streamers*: `ld.global.cg.v4` over their own slice of a 16 MiB buffer, after a warm-up pass that leaves every line in the near L2 partition.
  - *DSMEM sources*: remote loads from a *server* SM.
  - Both groups run ≈ 20 ms after one `cluster.sync()`, alone and together; the iteration counts come from calibration launches of each group alone.
  - Role maps:
    - `l2A`: L2 streamers = group A (6 SMs); DSMEM sources = the even SM of each B TPC (3 SMs), reading a server in the same group (`same_cpc`) or in group C (`other_cpc`).
    - `l2all`: the same, but every remaining SM streams L2 (10 SMs).
    - `dsm_ring`: L2 = group A; the 10 SMs of B and C read each other in a ring.

**Sweeps** (1024 threads, ILP 4, Default policy and 16 KiB TMA commands unless swept):

| step | configurations |
|---|---|
| A | ring +1 × {ld, st, st.async, TMA put, TMA reduce} × CS {2, 4, 8, 16} × payload, single and full |
| S | full GPU, ring +1 × the 5 operations × CS, 96 KiB × {4, 8, 16, 32} passes |
| B | threads {128, 256, 512, 1024} × ILP {1, 2, 4, 8} × CS × {pair_uni, ring} for ld and st; st.async × threads |
| C | every pattern × {ld, st, TMA put} and the mixed patterns, CS 2–16 single, CS 8 full |
| D | CS 16: 1–15 readers of rank 0 (ld), rank 0 writing to 1–15 targets (st, TMA) |
| E | TMA command size 128 B – 16 KiB × {put, reduce} × {pair_uni, ring} at CS 8 |
| F | Spread and LoadBalancing policies × ring {ld, st, TMA} × CS, single and full |
| many-block launch | 50 160 blocks of 1024 threads, 16 KiB shared memory, 2 blocks per SM, ring +1, CS 2–16, 1/4/16 passes over the peer's 16 KiB, cudaEvent time only |
| K4 / K5 | local-pressure and interference controls above |

**Raw data.** Every per-CTA, per-sample record is kept (1.37 M rows for the runs analysed here); nothing is reduced.
"""

R41 = r"""
**Reading Fig. 1.**
- **Aggregate.** With every co-resident cluster running a ring, the GPU moves **2.0–4.0 TB/s** through the SM-to-SM network:
  - CS 2: 3.2–4.0 TB/s (132 SMs);
  - CS 4 and 8: 2.4–3.3 TB/s (120 SMs);
  - CS 16: 2.1–3.1 TB/s (112 SMs).
- **Against memory.** That is about the HBM read bandwidth of this card (3.71 TB/s) and one third to one half of its L2 read bandwidth (7.75 TB/s).
- **Ranking.** Plain stores are the fastest generic operation (3.1–3.9 TB/s). The TMA put is the fastest at CS 2 (3.98 TB/s) but falls behind stores at CS 16. The TMA reduce-put is the slowest everywhere.
- **Why the aggregate falls from CS 2 to CS 4 and beyond.** Two effects, each about half of the drop:
  - Fewer SMs take part: CS ≥ 4 cannot use SMs 124–131, CS 16 also not the 12-SM GPC.
  - Each SM gets less once more SMs of its GPC exchange data across TPCs (section 4.2). The CS-2 clusters stay inside a TPC.
- **Event slope vs summed per-SM rates.** The event-time slope is the throughput of the whole launch: it ends when the slowest SM has finished. The cross-check (per-SM rate × SMs) is 1–29 % higher, most for the TMA put at CS 16 (2.97 vs 2.31 TB/s). Under GPC-wide load the SMs do not get equal shares: in one launch some TMA receivers finish their 3 MiB in 102 µs, others in 142 µs (section 4.2, last table).
"""

R42 = r"""
**Reading the per-SM numbers.**
- **One direction: 21.2–21.5 B/cycle per SM for every operation.** One SM sending to one peer reaches 21.3–21.5 (ld), 21.2–21.3 (st) and 21.0–21.3 B/cycle (TMA put), with α of 800–3000 cycles and residuals ≤ 1.5 %. The value is the same at every cluster size and with 15 clusters running at once (pair_uni, packed pairs).
- **Both directions.** When the SMs of a pair (or a ring) send and receive at the same time:
  - loads lose 20 %: 17.1 B/cycle per direction;
  - stores and TMA puts lose only 4–5 %: 20.3–20.6.

  A load's request also travels out of the reader, so in pair_bi and ring each SM's outgoing connection carries its own requests next to the replies it serves.
- **Reduce-put and st.async.** The TMA reduce-put delivers 16.2–16.5 B/cycle in a ring, 20 % below the plain put. `st.async` delivers 20.9, like plain stores.
- **GPC contention.** In a ring the per-SM rate depends on how many SMs of the same GPC take part, not on how many clusters there are:
  - With 8 active SMs (one CS-8 cluster, or the 8 cluster-capable SMs of the 12-SM GPC on the full GPU), ld gives 17.1 B/cycle.
  - With 16 active SMs, ld gives 14.4–14.9: one CS-16 cluster, two CS-8 clusters or four CS-4 clusters behave alike, in every 16-SM GPC.
  - Stores (20.4 → 17.8–18.6) and TMA puts (20.6 → 18.3–19.7) drop the same way.
  - CS-2 clusters, whose traffic stays between the two SMs of a TPC, lose less (ld 16.6, st 18.9) even in 18-SM GPCs.
  - Under that load the share is also uneven across the SMs of a cluster: at CS 16, ld ranks get 12.8–14.8 B/cycle and TMA receivers 12.0–15.0 B/cycle at 96 KiB, with a pattern that repeats from cluster to cluster.
- **Fixed cost α.** 800–930 cycles for loads and stores. For `st.async` and TMA it is 1250–2930 cycles and grows with cluster size, because it includes the receiver's mbarrier handling and the first TMA command (≈ 1150 cycles, section 4.6).
"""

R43 = r"""
**Reading Fig. 2.**
- **Loads are latency-bound until about 8 KiB are in flight per SM.**
  - 128 threads × 1 load × 16 B (2 KiB) give 7.2–7.8 B/cycle, i.e. one round trip of ≈ 280 cycles per 2 KiB.
  - 4 KiB (256 × 1 or 128 × 2) give 12.2–15.5.
  - 8 KiB or more (128 threads × ILP 4, 256 × 2, 512 × 1) reach the 21.3 B/cycle limit.
  - In a ring the same amounts reach 17.1 (both directions).
- **Stores reach the limit with any number of threads.** 21.1–21.5 B/cycle (pair) and 20.1–20.5 (ring) already at 128 threads and ILP 1, because they are posted; `st.async` behaves the same.
- **Cluster size does not matter** except through the GPC effect of section 4.2 (CS 16 ring: ld 14.8–15.4, st 18.5–19.0).
- **Limitation.** At 1024 threads and ILP 8 one step would cover 128 KiB, more than the 96 KiB payload, so the loads in flight are capped by the payload. The fitted 22.6 B/cycle there comes from the payload-dependent number of steps, not from a faster network.
"""

R44 = r"""
**Reading Fig. 3.**
- **One sender, one receiver: always ≈ 21.3 B/cycle.** pair_uni, packed pairs with TPC mates and packed pairs 4 ranks apart give the same 21.0–21.5 for ld, st and TMA, single or full. Distance inside the GPC and the TPC relation do not matter while each SM only sends or only receives.
- **Ring, ring +4, randperm, pair_bi.** Every SM sends and receives, and they behave alike: ld 17.1, st 20.4, TMA 20.5 (one cluster); on the full GPU ld 14.5, st 17.8, TMA 18.5–18.7. The partner's position does not matter, only that both directions are used and how busy the GPC is.
- **One SM shared by 7 others (the shared SM's total):**
  - It serves at most **21.3 B/cycle** to 7 readers, and sends at most 21.3 B/cycle to 7 targets, with stores or with TMA.
  - It *receives* up to **32.1–32.3 B/cycle** from 7 writers.

  Sending is limited at about 2/3 of the rated 32 B/cycle, receiving at the rated value.
- **all_to_all / all_to_all_full / uniform_random.**
  - Loads: all_to_all 16.1 (14.7 full), all_to_all_full 17.1 (15.7 full).
  - Stores: all_to_all 19.4 (15.9 full).
  - Random partners lose another 15–25 % (ld 13.9, st 15.8, TMA 15.2): with a new random target for every 16 KiB, some SMs receive from several senders at once while others idle. The profiler shows the busiest receiving SM getting 1.3–1.8× its share with stores and TMA (section 4.11).
  - The TMA all_to_all is limited by command issue at these payloads (7 commands to 7 different targets take ≈ 2950 cycles to issue, section 4.6), so its fit is not affine; its rate at 96 KiB is 12.7 B/cycle.
- **Two streams at once (right panel).**
  - In the same physical direction (rank 1 serves rank 0's loads and writes into rank 0), the two streams share **21.4 B/cycle** (10.7 each). The limit belongs to the sending SM, whichever operation fills it.
  - In opposite directions they reach **31.4–31.9 B/cycle** together (15.7–16.0 each): each SM sends one stream and receives the other, minus the load requests that rank 0 sends.
"""

R45 = r"""
**Reading Fig. 4.**
- **The shared SM's total stays at 21.3 B/cycle for every N from 1 to 15** (left), whether N readers load from it or it writes the payload to N targets, with stores or with TMA. The average per reader or per target is therefore 21.3 / N; the fit of c / N gives c = 21.33.
- **The shares are not equal.**
  - Readers of the same SM finish at different times: for N = 3 rank 1 (rank 0's TPC mate) reads at 10.6 B/cycle and ranks 2 and 3 at 7.1. For N = 4 ranks 1 and 4 get 7.1 and ranks 2 and 3 get 5.3.
  - The per-reader rates take values 21.3 / m for small integers m, consistent with a server that alternates between groups of requesters and gives the leftover bandwidth to the remaining readers once one group has finished.
  - With TMA the targets are served one after another in issue order: the first target receives at 21.5, the last at 21.3 / N.
- **Receiving.** One SM accepts 32.1–34.4 B/cycle from 3–15 writers (table), i.e. the rated 32 B/cycle of its incoming connection.
"""

R46 = r"""
**Reading Fig. 5.**
- **Below 2 KiB a TMA command is limited by its issue cost, not by the network.** Up to 1 KiB every command costs ≈ 65 cycles at the issuing thread. So 128 / 256 / 512 / 1024-byte commands deliver 1.95 / 3.9 / 7.7 / 15.3–15.7 B/cycle.
- **From 2 KiB on the network limit is reached.** Put: 20.1–22.4 B/cycle (21.2 / 20.6 at 16 KiB, pair / ring). Reduce-put: 16.4–17.7 B/cycle; the element-wise add at the destination costs about 20 % of the put's rate.
- **Issue side:**
  - The issuing thread spends ≈ 1150 cycles until the first command of a sample is accepted, the same for any size up to 16 KiB.
  - Further commands to the same target follow every 65 cycles (small commands) to 180 cycles (16 KiB, back-pressured by the network).
  - Each command to a *different* target adds ≈ 270–430 cycles, independent of its size: 7 commands of 2.3 KiB to 7 targets take as long to issue (2950 cycles) as 7 commands of 16 KiB.
  - Spreading a TMA transfer over many destinations is therefore expensive at the issuer. A single thread can still keep the network saturated when the commands to one target are ≥ 2 KiB.
"""

R47 = r"""
**Scheduling policy.**
- Default, Spread and LoadBalancing give the same bandwidth within ±0.2 B/cycle in every configuration, single or full.
- With one CTA per SM (200 KB of shared memory) LoadBalancing cannot place two CTAs of a cluster on one SM.
- A lone cluster lands on the same SMs under every policy and in every launch:
  - CS 2 on SMs 124/125;
  - CS 4 and 8 in the 12-SM GPC (from SM 0);
  - CS 16 in a 16-SM GPC (from SM 2).
"""

R48 = r"""
**Many-block ring launch.**
- **The launch.** 50 160 blocks (2 per SM, 16 KiB of shared memory each); every block reads (or writes) its ring partner's 16 KiB 1, 4 or 16 times and exits. Measured only by the cudaEvent pair around the launch.
- **Throughput:**
  - With 16 passes: 3.41 (ld), 3.78 (st) and 4.20 TB/s (TMA put) at CS 2; 2.3–3.3 TB/s at CS 4–16.
  - The slope over the passes (block start and exit removed): 3.66 / 4.07 / 4.72 TB/s at CS 2.
  - With a single pass per block, block scheduling dominates: 1.1–1.75 TB/s.
- **Placement.** At CS ≥ 4 the SMs 124–131 run no block, and at CS 16 neither does the 12-SM GPC: those SMs cannot host such clusters. The other SMs run 230–588 blocks each.
"""

R49 = r"""
**Reading Fig. 6.**
- **Being read costs the serving SM local bandwidth; being written does not.**
  - Alone, the victim's warps stream their own shared memory at 54 / 96 / 104 / 113 B/cycle with 4 / 8 / 16 / 32 warps (the loop itself, with its flag check, limits it below the 128 B/cycle of shared memory).
  - While its partner reads 21.2 B/cycle out of it, the victim gets 50 / 72 / 73 / 79 B/cycle: up to 34 B/cycle less, about 1.6 bytes of local bandwidth per remotely read byte.
  - While its partner writes 21.2 B/cycle into it, the victim's rate is unchanged.
- **The remote reader is never slowed down.** It keeps 21.2 B/cycle with 256 or 1024 threads, alone or against 32 local warps.
- **The full GPU (66 pairs) shows the same numbers.**
"""

R410 = r"""
**Reading Fig. 7.**
- **No interference between SM-to-SM and SM→L2 traffic in the same GPC, in any role map.**
  - With both groups running, every group's rate stays within 0.5 % of its rate alone (table; most changes are below 0.1 %), and its loaded latency is unchanged.
  - L2 streamers get 74.4 B/cycle each (6 SMs) or 66.1–66.6 (10 SMs; 1.08 TB/s for the GPC).
  - DSMEM sources keep 21.33 (3 sources) or 14.42 B/cycle (10 SMs in a ring).
- **No CPC-group effect.** Whether the DSMEM server is in the source's TPC-index group (`same_cpc`) or in another group (`other_cpc`) changes nothing.
- **Limitation of the DSMEM ILP axis.** With 1024 threads × 16 B, even one load in flight per thread (16 KiB per SM) saturates a DSMEM source. The swept ILP therefore changes only its loaded latency (713 → 3586 cycles per group from ILP 1 to 8), not its bandwidth. The L2 side spans 35 → 74 B/cycle.
"""

R411 = r"""
**Reading the counters.**
- **Rated limits.** The profiler rates both SM-to-SM connections of every SM at **32 B/cycle** (`l1tex__m_l1tex2xbar_write_bytes_mem_dshared` and `…xbar2l1tex_read_bytes_mem_dshared`, `.peak_sustained`). Measured, an SM receives at that rate (32.1–34.4 B/cycle from several writers) but sends at most 21.3 B/cycle, i.e. 2/3 of it, for loads, stores, `st.async` and TMA alike.
- **Byte counters.**
  - Both counters count exactly the payload: 1.00 × the bytes the CTAs moved, for every operation. For `all_to_all` it is 1.00–1.06 ×, because of partial 32-byte sectors at the split points.
  - They are charged to the SM that *receives* the data (the reader for loads, the owner for stores and TMA): `.max` equals one SM's payload in pair_uni, and equals rank 0's 7 × payload when 7 writers store into it.
  - Request packets of loads are not visible in them.
- **Hot spots.** With uniform_random partners the busiest receiving SM got 1.33 × (st), 1.50 × (TMA) and 1.83 × (`st.async`) its share (`.max/.avg` = 22–30 vs 16.5 for an even spread). Loads show no hot spot in these counters, because they are charged to the reader, which always reads its own payload.
- **Instruction counts.** One `dshared_ld`/`st`/`stas` per warp-level 16-byte access and one `tma_st`/`tma_red` per bulk command (6 per 96 KiB at 16 KiB per command).
"""

INSIGHTS = r"""
## 5. Insights

1. **An SM sends at most 21.3 B/cycle into the SM-to-SM network and receives up to 32 B/cycle.**
   - The send limit is the same for loads (where the owner sends the reply), stores, `st.async` and TMA puts, at every cluster size and distance, and for one or many destinations: an SM serving 15 readers or writing to 15 targets still moves 21.3 B/cycle in total.
   - It is 2/3 of the 32 B/cycle the hardware's counters rate for the connection. The receive side reaches the rated 32 B/cycle when several SMs write into one.
   - For a stacked-GPC model, this is a per-SM output port of 21.3 B/cycle (35 GB/s at 1635 MHz) and an input port of 32 B/cycle.
2. **Traffic in both directions costs loads 20 %, stores and TMA 4–5 %.** A reader's requests leave through the same outgoing connection that carries the replies it serves to others. Exchanges built on writes (stores, `st.async`, TMA puts) therefore use the network better than exchanges built on remote reads: 20.3–20.6 vs 17.1 B/cycle per SM in a ring.
3. **The GPC shares capacity when many of its SMs exchange data across TPCs.**
   - Per-SM rates fall 10–15 % from 8 to 16 active SMs (ld 17.1 → 14.4–14.9, st 20.4 → 17.8–18.6), and the shares become uneven, so the slowest SM sets the completion time.
   - Pairs inside a TPC lose less. Pure one-way traffic (half of the SMs sending) is not affected at all.
   - The number of clusters and the scheduling policy do not matter.
4. **GPU-wide, SM-to-SM bandwidth equals HBM bandwidth, not L2 bandwidth.**
   - 2.0–4.0 TB/s for a ring over all co-resident clusters: highest at CS 2 (TMA put 3.98, st 3.86 TB/s), 2.1–3.1 TB/s at CS 16, where only 112 SMs can take part.
   - That is about the 3.71 TB/s of HBM read and 1/3–1/2 of the 7.75 TB/s of L2 read.
   - DSMEM's advantage is latency and the absence of L2 traffic (Fig. 7 shows the two do not interfere), not raw aggregate bandwidth.
5. **TMA shared→shared copies are cheap to issue only for large commands to one target.**
   - Commands ≤ 1 KiB are issue-bound at ≈ 65 cycles each; 2 KiB or more reach the network limit.
   - The first command after a barrier takes ≈ 1150 cycles, and every change of destination adds ≈ 270–430 cycles.
   - The reduce-put is limited to ≈ 16.5–17.7 B/cycle.
6. **Remote reads take shared-memory bandwidth from the owner; remote writes do not.** Serving 21 B/cycle of remote reads cost the owner up to 34 B/cycle of its local streaming bandwidth. Producer/consumer exchanges that push data (stores, TMA puts) leave the receiver's own shared-memory bandwidth intact.

**Limitations and anomalies.**
- The payload is at most 96 KiB per pass. At 1024 threads with ILP ≥ 4 the loads in flight are capped by the payload, which also makes the 1024-thread ILP-8 load fits read 22.6 B/cycle.
- 33 of the 642 bandwidth fits have a residual RMSE above 2 % (largest 4.6 %). They are the TMA all_to_all and uniform_random points, the opposite-direction mixed streams, the 128-thread ILP-8 loads and a few full-GPU points. For TMA all_to_all the fitted slope is not a rate, because command issue dominates the small payloads; its 96-KiB rate is shown instead.
- The CPC test uses the TPC-index-mod-3 grouping as a hypothesis; no grouping effect was found.
- The DSMEM ILP axis of the interference test is flat because a single load in flight per thread already saturates a source.
- The GMEM reference lines come from a separate benchmark run on this card with the same clock lock, not from these launches.
"""
