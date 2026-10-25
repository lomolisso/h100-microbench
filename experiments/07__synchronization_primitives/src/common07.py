"""Shared helpers of experiment 07's scripts and notebook (paths relative to the experiment directory).

  runs()                         the run registry: one row per run_id with started/cmdline/complete
  latest_run(tool, mode, quick)  run_id of the latest *complete* invocation `bin/<tool> <mode>` (quick or not)
  read_run(file, run_id)         rows of results/<file> belonging to one run_id (pandas, chunked)
  bad_launches(run_id)           launch_idx values whose clock sample after the launch was throttled or below the lock
  append_rows(path, header, rows)  append-only CSV writer (flock, header on an empty file, checked otherwise, fsync)
  load_prim(k, quick)            primitive rows of kernel group k (k1 … k5) from the latest complete run of its binary:
                                 launches flagged in the clock log dropped (counted), per_op / per_op_max = (cycles −
                                 timer floor) / n_ops, the floor being the median timer-pair cost of the same run (k0)
  load_prop(k, quick)            propagation rows (one_way_ns, round_trip_cycles) of the same run selection (k = k2, k3,
                                 k4, or k4cs16 for the CS-16 pairs of `sync_flags k4cs16`)
  net_lds(k1)                    K1 CTA-barrier rows net of the dependent shared load that closes their timed region
                                 (the one netting rule of the notebook and of summarize.py)
  stats(df, by, col)             median, p5, p95, mean, n per group
  sass_cfg / timed_windows / window_pcs   the timed region (t0 → t1 %clock64 pair) of every profiled kernel, found in
                                 the SASS of the ncu source export (stall attribution restricted to the timed window)
"""
import csv
import fcntl
import io
import os
import re

import pandas as pd
from pandas.api.types import union_categoricals

EXP = "07__synchronization_primitives"
RES = "results"
MANDATORY = ["gpu_name", "sm_count", "sm_clock_mhz", "mem_clock_mhz", "cuda_version", "driver_version", "run_id", "launch_idx"]


def runs(res=RES):
    r = pd.read_csv(f"{res}/{EXP}__runs.csv", dtype=str, keep_default_na=False)
    started = r[r.status == "started"].set_index("run_id")
    done = set(r.loc[r.status == "complete", "run_id"])
    out = started[["started_iso", "cmdline"]].copy()
    out["complete"] = [i in done for i in out.index]
    return out.reset_index()


def latest_run(tool, mode, quick=False, res=RES):
    r = runs(res)
    toks = r.cmdline.str.split()
    sel = r[toks.map(lambda t: len(t) > 1 and t[0].endswith("/" + tool) and t[1] == mode and (("--quick" in t) == quick))]
    sel = sel[sel.complete]
    if sel.empty:
        raise SystemExit(f"no complete run of {tool} {mode} (quick={quick})")
    return sel.sort_values("started_iso").run_id.iloc[-1]


def read_run(file, run_id, res=RES, cats=(), **kw):
    """cats: string columns parsed as categoricals (one union of categories over all chunks; saves memory)."""
    dt = dict(kw.pop("dtype", {}), **{c: "category" for c in cats})
    parts = [c[c.run_id == run_id] for c in pd.read_csv(f"{res}/{file}", chunksize=2_000_000, dtype=dt or None, **kw)]
    for c in cats:
        u = union_categoricals([p[c] for p in parts]).categories
        for p in parts:
            p[c] = p[c].cat.set_categories(u)
    return pd.concat(parts, ignore_index=True)


def bad_launches(run_id, res=RES):
    c = read_run(f"{EXP}__clock_log.csv", run_id, res)
    c = c[(c.phase == "launch") & ((c.throttle != 0) | (c.below_lock != 0))]
    return set(c.launch_idx.astype(int))


def append_rows(path, header, rows):
    with open(path, "a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0, os.SEEK_END)
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        if f.tell() == 0:
            w.writerow(header)
        else:
            f.seek(0)
            if next(csv.reader(f)) != header:
                raise SystemExit(f"header mismatch in {path}")
            f.seek(0, os.SEEK_END)
        w.writerows(rows)
        f.write(buf.getvalue())
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


TOOL = {"k1": "sync_block", "k2": "sync_cluster", "k3": "sync_mbarrier", "k4": "sync_flags", "k4cs16": "sync_flags",
        "k5": "sync_grid"}
PRIM_FILES = {"k2": ["k2", "k2d"]}
PRIM_COLS = ["run_id", "launch_idx", "sm_clock_mhz", "kernel", "primitive", "cs", "block_threads", "variant", "rep",
             "cycles", "rank", "smid", "n_ops", "cycles_warp_max"]


def load_prim(k, quick=False, res=RES):
    run = latest_run(TOOL[k], k, quick, res)
    cats = ("run_id", "kernel", "primitive", "variant")
    parts = [read_run(f"07__sync__primitives__{f}.csv", run, res, cats=cats, usecols=PRIM_COLS) for f in PRIM_FILES.get(k, [k])]
    for c in cats:
        u = union_categoricals([p[c] for p in parts]).categories
        for p in parts:
            p[c] = p[c].cat.set_categories(u)
    df = pd.concat(parts, ignore_index=True)
    fl = df[df.kernel == "k0"]
    floor = float((fl.cycles / fl.n_ops).median())
    df = df[df.kernel != "k0"].copy()
    bad = bad_launches(run, res)
    n_bad = int(df.launch_idx.isin(bad).groupby(df.launch_idx).first().sum()) if bad else 0
    df = df[~df.launch_idx.isin(bad)]
    df["per_op"] = (df.cycles - floor) / df.n_ops
    df["per_op_max"] = (df.cycles_warp_max - floor) / df.n_ops
    return df, {"run_id": run, "timer_floor": floor, "bad_launches": n_bad,
                "launches": int(df.launch_idx.nunique()), "rows": len(df)}


def load_prop(k, quick=False, res=RES):
    run = latest_run(TOOL[k], k, quick, res)
    df = read_run("07__sync__propagation.csv", run, res, cats=("run_id", "primitive", "variant"))
    bad = bad_launches(run, res)
    return df[~df.launch_idx.isin(bad)].copy(), run


# CTA barriers (K1) are timed up to a dependent ld.volatile.shared after them (bar.sync is BAR.SYNC.DEFER_BLOCKING: a
# clock read right after it can precede its completion). Their net cost subtracts the median of that load alone
# (primitive lds_volatile) at the same block_threads; bar_sync_after_arrive (64 threads, one consumer warp) uses the
# 32-thread value. Divided by n_ops (the load follows the last of 64 back-to-back barriers).
NET_LDS_PRIMS = ["bar_sync0", "bar_sync_named", "bar_sync_after_arrive"]


def net_lds(k1):
    """K1 rows with per_op / per_op_max of the CTA barriers (NET_LDS_PRIMS) net of the dependent shared load.
    Returns (copy of k1 with the netted columns, lds median per block_threads)."""
    k1 = k1.copy()
    lds = k1[k1.primitive == "lds_volatile"].groupby("block_threads").per_op.median()
    isbar = k1.primitive.isin(NET_LDS_PRIMS)
    sub = k1.loc[isbar, "block_threads"].map(lambda b: lds.get(b, lds.get(32))) / k1.loc[isbar, "n_ops"]
    k1.loc[isbar, "per_op"] -= sub
    k1.loc[isbar, "per_op_max"] -= sub
    return k1, lds


def stats(df, by, col="per_op"):
    g = df.groupby(by, observed=True)[col]
    return pd.DataFrame({"median": g.median(), "p5": g.quantile(0.05), "p95": g.quantile(0.95), "mean": g.mean(),
                         "n": g.size()}).reset_index()


# ---------------------------------------------------------------- timed windows in the profiled SASS (stall attribution)
_PRED = re.compile(r"^@!?U?P[T0-9]+\s+")
_BRA = re.compile(r"\bBRA(?:\.\S+)?\s+(0x[0-9a-f]+)")
_CLK64 = re.compile(r"^CS2R\s+R\d+,\s*SR_CLOCKLO\b")          # %clock64 (t0 / t1 of every timed region)
_CLKANY = re.compile(r"SR_(?:CLOCKLO|GLOBALTIMERLO)\b")


def _sass_op(s):
    return _PRED.sub("", s).split()[0].rstrip(";") if s.strip() else ""


def sass_cfg(offs, sass):
    """Control-flow graph of one kernel function from its SASS text (ncu source page; branch targets are absolute).

    offs: instruction offsets (int, from the function start), sass: the instruction texts, same order.
    Returns (succ, pred) as dicts offset -> list of offsets. Fall-through to the next instruction unless the instruction
    is an unpredicated BRA / EXIT / BPT.TRAP / RET; a BRA adds an edge to its target. The function's absolute start
    address comes from the function's final `BRA <itself>` (checked: every branch target must be an instruction).
    """
    order = sorted(range(len(offs)), key=lambda i: offs[i])
    offs = [offs[i] for i in order]
    sass = [sass[i] for i in order]
    have = set(offs)
    tg = [int(m.group(1), 16) if (m := _BRA.search(s)) else None for s in sass]
    abs_t = [t for t in tg if t is not None]
    # Every function ends with an unpredicated `BRA <itself>` (after its last EXIT / BRA, before the NOP padding): that
    # branch gives the absolute start address of the function.
    last = max((k for k in range(len(offs)) if tg[k] is not None), default=None)
    base = tg[last] - offs[last] if last is not None else 0
    if any((t - base) not in have for t in abs_t):
        raise ValueError("branch targets do not map onto the function's instructions")
    succ = {o: [] for o in offs}
    for k, (o, s) in enumerate(zip(offs, sass)):
        op, pred = _sass_op(s), bool(_PRED.match(s))
        if tg[k] is not None:
            succ[o].append(tg[k] - base)
        terminal = op.startswith(("EXIT", "RET", "BPT.TRAP")) or (op == "BRA" or op.startswith("BRA."))
        if (pred or not terminal) and k + 1 < len(offs):
            succ[o].append(offs[k + 1])
    pred_ = {o: [] for o in offs}
    for o, ss in succ.items():
        for t in ss:
            pred_[t].append(o)
    return succ, pred_


def _dominators(succ, pred, entry):
    reach, st = {entry}, [entry]
    while st:
        for t in succ[st.pop()]:
            if t not in reach:
                reach.add(t)
                st.append(t)
    nodes = sorted(reach)
    dom = {n: set(reach) for n in nodes}
    dom[entry] = {entry}
    changed = True
    while changed:
        changed = False
        for n in nodes:
            if n == entry:
                continue
            ps = [dom[p] for p in pred[n] if p in reach]
            new = (set.intersection(*ps) if ps else set()) | {n}
            if new != dom[n]:
                dom[n], changed = new, True
    return dom


def _short_cycle(succ, n, limit=8):
    """Instructions of the shortest CFG cycle through n if it has at most `limit` instructions (a spin loop), else None."""
    frontier, parent = [n], {n: None}
    for _ in range(limit):
        nxt = []
        for x in frontier:
            for t in succ[x]:
                if t == n:
                    cyc = [x]
                    while parent[cyc[-1]] is not None:
                        cyc.append(parent[cyc[-1]])
                    return set(cyc)
                if t not in parent:
                    parent[t] = x
                    nxt.append(t)
        frontier = nxt
    return None


def timed_windows(offs, sass):
    """Timed regions (t0 → t1) of one kernel function, found in its SASS.

    t0 and t1 of every timed region in these kernels are 64-bit %clock64 reads (`CS2R Rd, SR_CLOCKLO`). The other
    clock reads are excluded first:
      * 32-bit reads (`CS2R.32`, `S2UR … SR_CLOCKLO`): register pins of an address (pin_u32), not timers;
      * spin_cycles(): the read inside a CFG cycle of ≤ 8 instructions (the spin loop) and the read that starts it (the
        last %clock64 read at most 4 instructions before the loop's first instruction);
      * %globaltimer reads (`SR_GLOBALTIMERLO`): cross-SM stamps; they never delimit a cycle count here and are reported
        as being inside or outside the windows.
    The remaining reads are paired from the end of the function backwards: two reads adjacent in address order form a
    pair if one dominates the other in the CFG (every path from the kernel entry to the later read passes through the
    earlier one); the dominating read is t0. An unpaired read (e.g. a time stamp of a second warp) is skipped.
    Window = every instruction on a CFG path from t0 to t1 that does not pass through t0 or t1 again, plus t1 itself
    (strictly after t0, up to t1; out-of-line retry loops are included, and t1 may precede t0 in address order).
    Returns a list of dicts: t0, t1 (offsets), pcs (set of offsets), contiguous (window == the address range (t0, t1]),
    other_clock (clock / globaltimer reads inside the window), unpaired (excluded 64-bit reads).
    """
    succ, pred = sass_cfg(offs, sass)
    txt = dict(zip(offs, sass))
    so = sorted(offs)
    idx = {o: i for i, o in enumerate(so)}
    reads = [o for o in so if _CLK64.match(_PRED.sub("", txt[o]))]
    spin = set()
    for o in reads:
        cyc = _short_cycle(succ, o)
        if cyc is None:
            continue
        spin.add(o)
        start = [r for r in reads if r < min(cyc) and idx[min(cyc)] - idx[r] <= 4 and r not in spin]
        if start:
            spin.add(start[-1])
    cand = [o for o in reads if o not in spin]
    dom = _dominators(succ, pred, so[0])
    pairs, j = [], len(cand) - 1
    while j >= 1:
        a, b = cand[j - 1], cand[j]
        if b in dom and a in dom[b]:
            pairs.append((a, b)); j -= 2
        elif a in dom and b in dom[a]:
            pairs.append((b, a)); j -= 2
        else:
            j -= 1
    paired = {x for p in pairs for x in p}
    out = []
    for t0, t1 in sorted(pairs):
        fwd, st = set(), [t for t in succ[t0] if t not in (t0, t1)]
        while st:
            x = st.pop()
            if x in fwd:
                continue
            fwd.add(x)
            st += [t for t in succ[x] if t not in (t0, t1)]
        bwd, st = set(), [p for p in pred[t1] if p not in (t0, t1)]
        while st:
            x = st.pop()
            if x in bwd:
                continue
            bwd.add(x)
            st += [p for p in pred[x] if p not in (t0, t1)]
        pcs = (fwd & bwd) | {t1}
        rng = {o for o in so if t0 < o <= t1}
        out.append({"t0": t0, "t1": t1, "pcs": pcs, "contiguous": pcs == rng,
                    "other_clock": sorted(o for o in pcs if o != t1 and _CLKANY.search(txt[o])),
                    "unpaired": [o for o in cand if o not in paired]})
    return out


def window_pcs(pcs):
    """Timed-window membership of every profiled PC.

    pcs: rows of results/07__sync__ncu_pcs.csv (any number of runs / launches). Each (run_id, launch_idx) is one profiled
    kernel; its windows come from timed_windows() on that launch's SASS. Returns (members, check):
      members  run_id, launch_idx, pc_offset (hex, as in the CSV), window (0, 1, … in address order of t0)
      check    one row per profiled launch: kernel_name, n_windows, t0, t1 (hex; '|' between windows), window PCs,
               contiguous, opcodes at the boundaries (t0, first window instruction, t1), the synchronisation opcodes
               inside the window(s), clock / globaltimer reads inside, unpaired %clock64 reads.
    """
    mem, chk = [], []
    sync_ops = ("BAR.", "UCGABAR", "MEMBAR", "ERRBAR", "CGAERRBAR", "CCTL", "SYNCS", "FENCE", "NANOSLEEP", "STAS", "REDAS",
                "ATOM", "RED", "WARPSYNC", "YIELD", "LD.E.STRONG", "LDG.E.STRONG", "ST.E.STRONG", "STG.E.STRONG")
    for (r, l), g in pcs.groupby(["run_id", "launch_idx"], sort=False):
        offs = [int(h, 16) for h in g.pc_offset]
        sass = [str(s) for s in g.sass]
        txt = dict(zip(offs, sass))
        ws = timed_windows(offs, sass)
        for k, w in enumerate(ws):
            mem += [(r, l, hex(o), k) for o in w["pcs"]]
        allpcs = sorted(set().union(*[w["pcs"] for w in ws])) if ws else []
        so = sorted(offs)
        chk.append({"run_id": r, "launch_idx": l, "kernel_name": g.kernel_name.iloc[0], "n_windows": len(ws),
                    "t0": "|".join(hex(w["t0"]) for w in ws), "t1": "|".join(hex(w["t1"]) for w in ws),
                    "window_pcs": len(allpcs), "contiguous": all(w["contiguous"] for w in ws),
                    "boundaries": " | ".join(f"{_sass_op(txt[w['t0']])} → {_sass_op(txt[so[so.index(w['t0']) + 1]])} … "
                                             f"{_sass_op(txt[w['t1']])}" for w in ws),
                    "sync_ops": " ".join(dict.fromkeys(_sass_op(txt[o]) for o in allpcs
                                                       if _sass_op(txt[o]).startswith(sync_ops))),
                    "clock_reads_inside": " ".join(f"{hex(o)}:{_CLKANY.search(txt[o]).group(0)}" for w in ws
                                                   for o in w["other_clock"]),
                    "unpaired": " ".join(hex(o) for o in (ws[0]["unpaired"] if ws else []))})
    return (pd.DataFrame(mem, columns=["run_id", "launch_idx", "pc_offset", "window"]), pd.DataFrame(chk))
