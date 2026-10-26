"""Markdown text of the experiment-12 notebook. Numbers quoted in INSIGHTS are read from the executed outputs of the runs
listed in section 3."""

TITLE = r"""
# 12 — Atomics on every path of the H100: shared memory, distributed shared memory, L2 near and far, HBM, host memory

**Goal.** Measure, on one H100 NVL at a locked 1635 MHz SM clock, what an atomic memory operation costs on each path it can take:
- the CTA's own shared memory (`atom/red.shared::cta`);
- the own shared memory addressed through the cluster window (`.shared::cluster` with the CTA's own rank);
- a peer CTA's shared memory in the same cluster (distributed shared memory, DSMEM);
- global memory with the line in L2, homed in the issuer's L2 partition ("near") or in the other one ("far");
- global memory with the line in HBM only (an L2 miss);
- pinned host memory, mapped into the GPU's address space.

For every path: the dependent latency of each operation and memory-ordering semantics, the throughput of one SM and of the whole GPU, with and without contention on one word, one bank or a few hot lines. Plus the DSMEM signalling primitives built on remote writes: `red.async` and `st.async` with hardware completion, how long a hand-off between two SMs takes with each primitive, and whether the width of a remote store changes its cost.

**Why it matters.** Atomics are how GPU programs combine partial results (histograms, reductions, counters, locks, work queues) and how CTAs signal each other. On this chip the same instruction can execute in four different places (the issuing SM, another SM of the GPC, one of two L2 partitions, or the host), each with its own latency, throughput and contention behaviour. A model of a GPU with a different memory hierarchy needs each of these separately, not one "atomic latency".
"""

BACKGROUND = r"""
## 2. Background

**`atom` and `red`.** PTX has two forms of every atomic read-modify-write:

```
atom{.sem}{.scope}.space.op.type  d, [a], b;      // returns the old value d (cas: atom.cas d, [a], compare, new)
red{.sem}{.scope}.space.op.type   [a], b;         // no return value: the issuing thread does not wait for it
```

- Operations measured here: integer `add.u32`, `add.u64`, `inc.u32`, `min.s32`, `max.u32`, `and.b32`, `exch.b32`, `cas.b32`, `cas.b64`; floating-point `add.f32`, `add.f64`, `add.noftz.f16x2`, `add.noftz.bf16x2` (two 16-bit values per 32-bit word); for global memory also the vector forms `add.v2/v4.f32` and `add.noftz.v2/v4.f16x2`.
- **Semantics and scope.** The default is `.relaxed` at `.gpu` scope: the operation is atomic, but it orders nothing else. `.acq_rel` (`atom`) or `.release` (`red`) also orders the thread's earlier and later memory accesses, at the named scope: `.cta`, `.cluster` (the CTAs of a cluster), `.gpu` or `.sys` (the GPU and the host).
- **Where the operation executes.**
  - `.shared::cta`: in the shared-memory unit of the issuing SM (SASS `ATOMS`).
  - `.shared::cluster`: a generic atomic (`ATOM.E`) on an address of the cluster's shared-memory window. If the address belongs to the own CTA, it is served by the own SM; if it belongs to a peer CTA, the request crosses the GPC's SM-to-SM network and executes in the peer SM's shared memory, and the old value travels back.
  - `.global`: in the L2 (`ATOMG` with return, `REDG` without). The L2 of this GPU is split into two partitions; every 128-B line has an address-determined **home** partition. A *load* from the other partition leaves a copy of the line in the reader's partition, so repeated loads are served near. Section 4.2 measures whether atomics behave like that.
  - Host memory (`cudaHostAlloc`, mapped): over PCIe.
- **Compare-and-swap loops.** Not every operation exists in hardware for every space. The compiler then emits a loop that reads the word, computes the new value and retries `cas` until no other thread changed the word in between. Table 3 lists the SASS each operation becomes here.

**`red.async` and `st.async`.** Two remote writes into a peer CTA's shared memory that signal their own completion:

```
red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u32  [peer + off], v, [peer_mbar];
st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32                    [peer + off], {a,b,c,d}, [peer_mbar];
```

- The receiving CTA arms an **mbarrier** (a 64-bit synchronisation object in its own shared memory) with the number of bytes it expects (`mbarrier.arrive.expect_tx`).
- Every arriving write counts its bytes down, and the receiver waits until the count reaches zero (`mbarrier.try_wait`). The receiver therefore observes the arrival of the data without polling the data itself, and the sender needs no memory fence.

**Hand-off (store visibility).** For two CTAs that take turns, the time from one CTA's signal to the other's observation sets the speed of every producer–consumer protocol. A signal is a plain remote store, a release store, a remote `red` or `atom` on a flag, or an `st.async` / `red.async` with the mbarrier completion above. The receiver polls its own shared memory or waits on its mbarrier.
"""

SETUP = r"""
## 3. Experimental setup

**Device and configuration.** Device and clock data are in the table below, from the CSV columns of every run: H100 NVL, 132 SMs in 8 GPCs (12, 16, 16, 16, 18, 18, 18, 18 SMs), SM clock locked at 1635 MHz, HBM3 at 2619 MHz, power limit 400 W, ECC on, CUDA 12.9 (`sm_90a`, `-O3`). Every launch is followed by a clock sample (NVML). A launch with a throttle reason, an SM clock below the lock, or CTAs on other SMs than intended is discarded and repeated; the counts are in the table below.

**L2 partition of every SM (K0a).** The home partition of a line decides whether a global atomic is near or far, so it is measured on this card before anything else:
- Lines of a buffer are labelled by timing: written by SM 0, read once by SM 2, and the reverse. A line read near by SM 2 after SM 0 wrote it is homed in SM 2's partition.
- Then, for every SM, 64 lines of each home are written by an SM of their home partition and read once by that SM. Its partition is the home it reads fast.
- **Partition 0** (by definition the partition of SM 0) = GPCs {0, 5, 6, 7}; **partition 1** = GPCs {1, 2, 3, 4}; 66 SMs each, the same for every SM of a GPC.
- Each binary run labels its own 512 MiB buffer per 4 KiB page, because the home depends on the physical page. The first 16 MiB are also labelled per line, to check that every line of a 4 KiB page has the page's home.

**Dependent latency (K1).** One thread (lane 0 of a one-warp CTA) walks a chain in which each element holds the position of the next one. The atomic's returned value is the next address, so every operation waits for the previous one.
- **Identity operands.** Each operation uses an operand that leaves the element unchanged: `add` +0 in every format, `min.s32` with INT_MAX, `max.u32` with 0, `and.b32` with ~0. The same chain can therefore be walked again. Elements are stored as normal floating-point values (an f32 or f64 in [2, 4), two f16/bf16 halves in their normal range), so floating-point adds return them exactly. `inc.u32` and the `add +1` control add 1, which moves the next access one word further inside the same element. `exch.b32` writes the previous element's word, which reverses the chain. `cas` compares with 0, which never matches, so memory is unchanged (`cas_fail`). CAS that succeeds and `inc` are also timed on one fixed word ("same word", address dependent on the previous result).
- **Address cost.** Turning the returned value into the next address costs one integer multiply-add per hop (or one `LEA`); the f16x2 / bf16x2 encoding needs a few more instructions. The same-binary plain-load chain of the same format carries the same cost. Its value is shown next to every atomic.
- **`red` has no return value.** It is timed as `red` followed by a plain load of the same word, which returns the next position ("red+ld": the time until the issuing thread can read its own reduction).
- **Timing.** Each value is the slope between a 4096-hop and a 16 384-hop walk: (cycles(16 384) − cycles(4096)) / 12 288 SM cycles (`%clock64`), so the fixed cost of the timer and the loop cancels. Exceptions: host memory uses 256 / 1024 hops; HBM and the per-access histograms use one timed operation per sample (`%clock64` before, a timer read that waits for the returned value after; the timer cost is included).

**Placement and targets of K1.**
- **Shared memory** (T1, own smem through the window T2): one CTA per SM (200 KB of dynamic shared memory) on all 132 SMs at once. Each SM chases a random cycle over 128 KiB of its own shared memory; T2 runs as clusters of 2.
- **Peer CTA (DSMEM, T3).**
  - A single cluster of 2, 4 or 16 CTAs; the hardware places them on SMs 124/125, in GPC 0 from SM 0, and on GPC 1. Rank 0 chases every peer's chain in turn.
  - For the per-SM-pair view, full waves of clusters of 16, 8, 4 and 2; every rank chases every other rank in turn.
- **L2 near / far** (T4/T5): one warp pinned to the first SM of every GPC, on a chain over 8 MiB of lines homed in its own partition (near) or in the other one (far). Protocol per launch:
  - Clean the L2.
  - A warp on an SM of the lines' home partition writes the whole chain, so every line lives only at its home.
  - **Pass 1**: the chain is walked once.
  - **Pass 2**: the same lines again, without cleaning.
  - **Pass 2ld**: a plain `ld.global.cg` walk over the same lines, which shows whether the atomics left a near copy.
- **HBM** (T6): one walk of 131 072 operations per launch over a random cycle of all lines of 256 MiB, after cleaning the L2, so every operation misses. Near / far is the home of each line's page.
- **Host memory** (T7): 4096 lines of `cudaHostAlloc`'d memory, mapped; host-native atomics are not supported on this system (`cudaDevAttrHostNativeAtomicSupported` = 0), so the GPU performs them on the memory over PCIe.

**Throughput.**
- **Per SM (K2):** one CTA of 128–1024 threads (or one on every SM at once), each thread issuing 1, 2, 4 or 8 independent operations per step for 4096 steps (results of `atom` consumed after the step). Address patterns:
  - `coal`: 32 consecutive words per warp instruction; conflict-free in shared memory;
  - `sector` / `line`: one 32-B sector / one 128-B line per lane;
  - `bank`: shared memory only; the 32 lanes on different words of one bank;
  - `warp1`: the 32 lanes on one word;
  - `block1`: every thread of the CTA on one word.

  Targets: the 128 KiB of local, own-window and peer smem; 8 MiB of near / far lines (L2-resident after warm-up launches); a single pass over 256 MiB (HBM, cleaned L2 before every launch). Per-CTA cycles are taken between a CTA-wide barrier at the start and the end of the last operation; every thread fences at the end, so posted `red` / `st` are complete.
- **GPU-wide (K3):** 2 × 132 CTAs × 1024 threads, 4 operations per thread and step; cudaEvent time per launch, net of the 3.9 µs cudaEvent floor.
  - Footprint 1–256 MiB; `mixed` (all lines), `local` / `remote` (each partition's SMs on lines homed in their own / the other partition); plain `ld` and `st` of the same addresses in the same kernel as references.
  - The same remote traffic from 1–4 GPCs of one partition, and each GPC alone.
  - Hot spots: every thread on one word, or on N hot lines (each warp instruction one line), N = 1 … 1024, and versus the number of active SMs.
- **Checks.** For every integer-add configuration, the sum of the target memory after the last recorded launch minus the sum before equals the number of issued operations: no atomic was lost or merged away.
- **DSMEM fire-and-forget (K4, K5).**
  - K4: 1024 threads of rank 0 issue 4096 `red`, `atom`, `red.async`, `st.async` per thread into rank 1. `.async` completion = the receiver's last mbarrier phase after the common start (one barrier per 512 KiB of payload).
  - Ring over full waves of clusters of 2–16.
  - K5: 1–15 CTAs of a 16-CTA cluster into one CTA, with the receiver idle or running its own shared-memory atomics on the same words.
- **Hand-off (K6).** All 120 pairs of the 16 SMs of GPC 1 (one cluster of 16), in turn; 200 round trips per pair and launch (`%clock64` only) and 100 with `%globaltimer` stamps on both sides (one-way time, 32 ns resolution). Global flags: two pinned warps on SMs in the same or different partitions, flag lines homed near the consumer, near the producer, both near or both far; 1000 round trips per launch.
- **Store width (K7).** 32 threads of rank 0 write 256 KB into rank 1 with each width of `st.shared::cluster` and `st.async`.
- **Repetitions:** 2 warm-up and 10 recorded launches per configuration (3 for the HBM per-access walks, 5 for the per-pair sweep).
- **Counters:** Nsight Compute (`--clock-control none`); application replay for L2-resident runs, so the profiler does not reset the L2 between passes. A baseline launch of the same grid in which every CTA exits at once is profiled alongside, to separate the profiler's own traffic.
"""

INSIGHTS = r"""
## 5. Insights

**1. An integer atomic costs a fixed, small amount more than a load on every path.** All values are SM cycles at 1635 MHz, from dependent chains (Table 1).

| path | plain load | add.u32 atom | atomic − load |
|---|---|---|---|
| local shared memory | 28.9 | 32.3 | +3.4 |
| own smem through the cluster window | 37.2 | 41.5 | +4.3 |
| peer CTA (DSMEM) | 186–222 by SM pair | load + 8.00 | +8.00 on every one of ≈ 1800 SM pairs (spread 0.003) |
| L2, line homed near | 272.8 | 284 | +11 |
| L2, line homed far | 434 (first read) | 461 | +27 |
| HBM miss, near / far home | 522 / 684 | 538 / 695 | +16 / +11 |
| host memory | 1585 | ≈ 1670 | +85 |

- **Same cost for every single-instruction integer op.** `inc`, `min`, `max` and `and` cost the same as `add`; `exch` costs the same or slightly less (at an HBM miss it costs the same as a load). A `cas` that fails costs +2…+4 more than `add`; a succeeding `cas` on one word costs +9…+16 more.
- **64-bit and vector forms.** Atomics on 64-bit words add 14–25 cycles in L2. The vector global forms (`.v2/.v4.f32`, `.v2/.v4.f16x2`) add 20–30 cycles in L2 and 50–60 at an HBM miss.
- **`red` + load.** Reading back one's own `red` takes 1–6 cycles longer than `atom` in shared memory, 7 longer in near L2 and 20 longer far. Behind a far `red` the load also goes far: the `red` removes the reader's copy of the line.
- **DSMEM model.** The remote atomic is the remote load plus a constant 8 cycles; the cost is not set by the distance. A peer atomic is therefore the same per-SM-pair network round trip as a load plus 8 cycles in the owner SM's atomic unit.

**2. Global atomics always execute at the line's home L2 partition.**
- A line homed in the other partition is far on the first atomic (461 cycles) and stays far on every later one (pass 2: 461). A load becomes near after its first access (434 → 279).
- The atomics leave no copy in the issuer's partition: a plain load after the atomic passes is still far (434).
- The counters agree (Table 8). A far atomic is looked up in both partitions (2 L2 atomic sectors per operation instead of 1) and moves ≈ 1 sector over the partition link; a near atomic is one lookup and no link traffic.
- **GPU-wide consequence.** At 16 MiB, remote-homed lines (each partition's SMs on the other partition's lines) run at 1.8 TB/s for `red` and 1.0 TB/s for `atom`, against 3.2 TB/s for both on local lines. With only one partition's GPCs issuing, remote atomics saturate at ≈ 0.88 TB/s of payload, against ≈ 1.1 TB/s for remote stores (Figure 6). On remote lines `atom` reaches only half of `red`'s rate; `red` behaves like a store.
- **On a random buffer**, half the lines are remote, which explains why `atom` "mixed" (1.9 TB/s) is far below `atom` local (3.2 TB/s) while `red` mixed stays at 3.2.

**3. Where floating-point and 64-bit atomics are slow is decided by the compiler, per state space** (Table 3b):
- **Local shared memory.** Only 32-bit integer ops, `exch` and `cas` are native `ATOMS`. `add.f32`/`.f64`/`.u64`/`.f16x2`/`.bf16x2` become compare-and-swap loops: 100–122 cycles instead of 32, and 5–7 instead of 13.5 operations per cycle per SM.
- **Peer CTA (DSMEM).** `add.u64` and the 16-bit-pair adds are native `ATOM.E` (+23 and +101 cycles over the load). `add.f32`/`.f64` run a CAS loop over the network: 562 / 590 cycles, about 2.8 round trips, and 1.4 operations per cycle per SM.
- **Global memory.** Every op is a native `ATOMG`/`REDG`, including the vector f32/f16x2 forms.

  So the fast place for a floating-point atomic reduction on this GPU is the L2, not shared memory. In shared memory, the only fast reduction is a 32-bit integer one.

**4. Memory-ordering semantics cost a fence, not a different atomic.**
- `.relaxed` at any scope costs the same as the default.
- `.acq_rel.cta` adds 5–9 cycles.
- `.acq_rel.cluster` and `.acq_rel.gpu` are identical on every path: they add a GPU-scope memory barrier and an L1 invalidate. That costs +261 cycles on shared memory, +392 near and +531 far in L2, and +460 on host memory.
- `.sys` costs ≈ 1280 cycles in shared memory and ≈ 1370 / 1550 in near / far L2.
- A `.cluster` scope therefore gives no saving over `.gpu` on this GPU.

**5. One SM's atomic throughput, and where contention serialises** (Table 4, Figure 4):
- **Local shared memory.** A conflict-free `atom.add.u32` reaches 13.5 operations per cycle (54 B/cycle), against 14.9 for a plain 4-B load. Any two lanes on the same bank cost one cycle each: same-bank, same-word-per-warp and same-word-per-CTA patterns all run at exactly 1.0 operation per cycle.
- **Compiler exception.** `red.shared.add.u32` with the constant operand 1 (`atomicAdd(&x, 1)` without using the result) is compiled to `ATOMS.POPC.INC`. It adds the number of lanes that hit the same word in one operation, so a CTA counting into one word reaches 15.9 operations per cycle. With a non-constant operand the same reduction runs at 1.0.
- **Through the window, own CTA.** 10.6 operations per cycle; contention patterns again 1.0.
- **Into a peer CTA.** `red` and `atom` are capped at 3.05 operations per cycle per sender (12 B/cycle), below the 4.27 of 4-B remote loads or stores of the same addresses. The cap is the operation rate, not bytes.
  - A peer word hit by every lane of a warp runs at 0.33 operations per cycle; one word for the whole CTA at 0.17 (≈ 6 cycles per operation at the owner).
  - With one CTA per SM on every SM, the per-SM rates fall to 1.7 (coalesced) and 0.13 (one word).
- **L2 and HBM, one SM.** `red` ≈ 5.9 operations per cycle (≈ 24 B/cycle), the same as a plain store, for near and far lines alike. `atom` reaches 5.5 at 4–8 operations in flight per thread. 64-bit operations run at 3.2 per cycle (25.6 B/cycle).
- **L2 and HBM, every SM at once.** The per-SM share drops to 4.2 (near), 2.2 (far: the partition link) and 1.6 (HBM).
- **Contended global words.** One word per warp instruction, or per CTA, runs at 0.5 (`atom`) / 0.8–0.95 (`red`) operations per cycle from one SM, i.e. one L2 request per lane.

**6. Hot spots in the L2 are served by one slice, one request per lane, without merging** (Table 7, Figure 7).
- Every thread of the GPU on one word reaches 1.2 Gops/s for `add.u32`. That is ≈ 0.8 operations per L2 cycle; four SMs already reach 80 % of it. `add.f32` reaches only 0.52 Gops/s.
- The counters show one L2 request per thread, not per warp instruction. The busiest slice equals the total (`.max == .sum`), and the memory sums match the issued operations: same-address operations are neither merged in the SM nor in the L2; they are serialised in one slice.
- One hot line (each warp instruction one full line of 32 words) runs at 11 Gops/s, 9× one word. Throughput then grows linearly with the number of hot lines up to ≈ 256–512 lines, where it saturates at ≈ 330 Gops/s (`atom`) and ≈ 525 Gops/s (`red`), the throughput of the whole L2 for these patterns.
- Beyond the L2 capacity (≥ 96 MiB), atomics drop to ≈ 1.2–1.4 TB/s of payload, against 2.0–2.9 TB/s for plain loads and stores of the same lines.

**7. DSMEM fire-and-forget forms.**
- **Sending.** `red.async` sends at the same 3.0 operations per cycle as `red` and `atom`. `st.async` sends at 4.2 (`.b32`) and 1.25 (`.v4`: 20 B/cycle) operations per cycle, the highest rate any DSMEM store form reached here.
- **Completion.** For every `.async` form, the receiver's mbarrier completes at the same time as the sender finishes issuing: the mbarrier accounting costs no throughput.
- **Same-word reductions.** `red.async` runs 4× faster than `red`/`atom` on one word (0.66 vs 0.17 operations per cycle). Up to 15 senders into one CTA saturate the receiver at ≈ 4.3–4.5 operations per cycle for spread words, 0.67 (`red`/`atom`) or 1.0 (`red.async`) on one bank or one word.
- **A busy receiver.** A receiver running its own atomics on the same words keeps 15 local operations per cycle on spread words, but drops to 0.34 on one bank or one word, while the remote rate stays at its 0.67 limit.
- **GPU-wide ring.** Every cluster, every rank into the next: 3.7 / 3.1 / 2.2 / 1.45 TB/s for `st.async.v4` / `st.async.b32` / `red.async` / `red` at clusters of 2, and 2.9 / 2.4 / 1.8 / 1.0 at clusters of 16.

**8. Hand-off between two SMs** (Figure 8c, Tables 10–11).
- Within a GPC every SM pair gives the same round trip.
- **Primitive ranking**, with the compiler's unrolled poll loop:
  - plain remote store or remote `red` on a flag: 406–410 cycles (one way ≈ 128 ns);
  - remote `atom` whose return is used: 492;
  - `st.async` / `red.async` + mbarrier: 619 (160 ns);
  - `st.release.cluster`: 923 (288 ns), because of the fence.
- **The poll loop's code moves the flag hand-offs.** With one poll and one scheduler yield per loop iteration, the same flag hand-offs take 721–796 cycles. The mbarrier hand-offs stay at 619, because the wait instruction itself sets their granularity. Absolute flag hand-off times thus depend on how the waiting loop is compiled, by up to ≈ 300 cycles; the mbarrier forms do not. All round trips are multimodal, with steps of 40–190 cycles.
- **Through global flags,** a hand-off takes 1714–1736 cycles when both flag lines are homed in the pair's own partition. It takes 2184–2484 cycles when a flag line, or one of the two SMs, is in the other partition. `red.release` and `atom.acq_rel` cost no less than `st.release`.

**9. Store width.** A remote plain store's issue cost per instruction is constant from 1 to 4 B (time halves per width doubling: 1.9 → 3.8 → 6.7 B/cycle for 32 threads). Above that, wider stores give diminishing returns: 11.3 B/cycle at 8 B, 17.8 at 16 B. `st.async` issues faster than plain stores at every width (8.4 / 15 / 16.5 / 22 B/cycle for 4 / 8 / 8 (v2) / 16 B), and its completion at the receiver follows the issue within 1–5 %.

**Anomalies and limitations.**
- **Clock during heavy L2 kernels.** In-kernel cycles per ns show SM clocks down to ≈ 1510–1570 MHz during the heaviest GPU-wide L2 kernels, although NVML sampled 1635 MHz after them. Per-SM numbers are in SM cycles and unaffected; GPU-wide rates are wall-clock.
- **Discarded launches.** 18 launches of the 132-SM single-word-per-warp far-line `red` configuration were discarded and repeated after a software power-cap event (clock 1380–1620 MHz).
- **HBM tail.** The HBM per-access samples have k-means classes at ≈ 523 / 550 / 701 / 852 (near) and 675 / 711 / 863 / 1014 (far); a slow tail ≈ 150 cycles above each main class.
- **Encoding cost in absolute latencies.** Every chain hop includes the address arithmetic of one multiply-add (the f16x2/bf16x2 encoding ≈ 15 cycles more). It is the same in the plain-load reference, so differences to the load are exact, but absolute values include it.
- **Partition labels** were measured on this card (section 3), not assumed; every near / far result rests on them.
"""
