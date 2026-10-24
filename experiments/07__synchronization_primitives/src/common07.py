"""Shared helpers of experiment 07's scripts and notebook (paths relative to the experiment directory).

  runs()                         the run registry: one row per run_id with started/cmdline/complete
  latest_run(tool, mode, quick)  run_id of the latest *complete* invocation `bin/<tool> <mode>` (quick or not)
  read_run(file, run_id)         rows of results/<file> belonging to one run_id (pandas, chunked)
  bad_launches(run_id)           launch_idx values whose clock sample after the launch was throttled or below the lock
  append_rows(path, header, rows)  append-only CSV writer (flock, header on an empty file, checked otherwise, fsync)
  load_prim(k, quick)            primitive rows of kernel group k (k1 … k5) from the latest complete run of its binary:
                                 launches flagged in the clock log dropped (counted), per_op / per_op_max = (cycles −
                                 timer floor) / n_ops, the floor being the median timer-pair cost of the same run (k0)
  load_prop(k, quick)            propagation rows (one_way_ns, round_trip_cycles) of the same run selection
  stats(df, by, col)             median, p5, p95, mean, n per group
"""
import csv
import fcntl
import io
import os

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


TOOL = {"k1": "sync_block", "k2": "sync_cluster", "k3": "sync_mbarrier", "k4": "sync_flags", "k5": "sync_grid"}
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


def stats(df, by, col="per_op"):
    g = df.groupby(by, observed=True)[col]
    return pd.DataFrame({"median": g.median(), "p5": g.quantile(0.05), "p95": g.quantile(0.95), "mean": g.mean(),
                         "n": g.size()}).reset_index()
