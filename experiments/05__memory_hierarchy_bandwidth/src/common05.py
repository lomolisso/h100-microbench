"""Shared helpers of experiment 05's scripts and notebook (paths relative to the experiment directory).

  runs()                        the run registry: one row per run_id with started/finished/status/cmdline
  latest_run(tool, mode, quick) run_id of the latest *complete* invocation `bin/<tool> <mode>` (quick or not)
  read_run(file, run_id)        rows of results/<file> belonging to one run_id (pandas)
  append_rows(path, header, rows)  append-only CSV writer (flock, header on an empty file, checked otherwise, fsync)
  event_floor_ms()              cudaEvent floor: median elapsed time of an empty event pair (experiment 01, latest run)
  net_tbps(d, floor_ms)         GPU-wide TB/s with the event floor removed from ms_event
  k12_slope(d)                  K1/K2 B/cycle per SM from the slope between 4096 and 8192 steps (fixed cost removed)
  first_pass_slope(d)           first-pass rate from the slope of span over payload (+ bootstrap p5/p95)
  resident_footprints(res)      footprints at which the first-pass residency sample found every line in L2
  hbm_write_factor(ndr, op, payload, cache)  fraction of a K4 launch's payload that reached HBM during the launch (ncu)
"""
import csv
import fcntl
import io
import os

import numpy as np
import pandas as pd

EXP = "05__memory_hierarchy_bandwidth"
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


def read_run(file, run_id, res=RES, **kw):
    it = pd.read_csv(f"{res}/{file}", chunksize=2_000_000, **kw)
    return pd.concat([c[c.run_id == run_id] for c in it], ignore_index=True)


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


EVENT_FLOOR = "../01__timers_and_harness/results/01__timers__event_floor.csv"


def event_floor_ms(path=EVENT_FLOOR):
    """Median cudaEventElapsedTime of an event pair with nothing between the events (01, mode empty_pair, latest run)."""
    d = pd.read_csv(path)
    d = d[d["mode"] == "empty_pair"]
    return d[d.run_id == d.run_id.max()].us_per_launch.median() / 1e3


def net_tbps(d, floor_ms):
    """TB/s of each launch row with the event floor subtracted from its cudaEvent time."""
    return d.tbps * d.ms_event / (d.ms_event - floor_ms)


def k12_slope(d, keys=("level", "op", "width_bytes", "threads", "ilp", "pattern", "placement")):
    """B/cycle per SM from the difference between the 8192- and 4096-step launches: the fixed cost of the timed region
    (barriers, timer reads) cancels. Blocks are paired by SM and by launch order within each configuration (the two
    step counts are separate launches). Returns one row per configuration, launch pair and SM with bpc_slope."""
    d = d.copy()
    d["k"] = d.groupby(list(keys) + ["iters", "smid"]).launch_idx.rank(method="first")
    a = d[d.iters == 4096].set_index(list(keys) + ["smid", "k"])[["bytes", "cycles"]]
    b = d[d.iters == 8192].set_index(list(keys) + ["smid", "k"])[["bytes", "cycles"]]
    j = b.join(a, lsuffix="_8192", rsuffix="_4096", how="inner")
    j["bpc_slope"] = (j.bytes_8192 - j.bytes_4096) / (j.cycles_8192 - j.cycles_4096)
    j["bpc_8192"] = j.bytes_8192 / j.cycles_8192
    return j.reset_index()


def first_pass_slope(d, n_boot=1000, seed=0):
    """Rate of one pass from launch rows (bytes, span_ns) of several payloads: span = start-up + bytes / rate, so the
    rate is 1 / slope of a least-squares line. Bootstrap (launches resampled within each payload) gives p5/p95.
    Returns (rate_bytes_per_ns, intercept_ns, p5, p95)."""
    x, y = d.bytes.to_numpy(float) / 2**20, d.span_ns.to_numpy(float)  # MiB, ns

    def fit(x, y):  # least-squares line y = a·x + b
        a = np.cov(x, y, bias=True)[0, 1] / np.var(x)
        return a, y.mean() - a * x.mean()

    a, b = fit(x, y)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(d.bytes.to_numpy() == v) for v in np.unique(d.bytes)]
    boot = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        boot.append(2**20 / fit(x[idx], y[idx])[0])
    return 2**20 / a, b, np.percentile(boot, 5), np.percentile(boot, 95)


HBM_CYCLES = 480  # first-read latency above every L2 hit (near ≈ 275, far ≈ 435 cycles) and below HBM


def resident_footprints(res, max_missing=0.001):
    """Footprints whose residency sample (firstpass mode) found at most `max_missing` of the lines outside L2."""
    m = (res.cycles > HBM_CYCLES).groupby(res.footprint_bytes).mean()
    return m[m <= max_missing].index


WRITE_SHARE = {"read": 0.0, "write": 1.0, "mix": 1 / 6, "mix11": 0.5}  # written bytes / payload bytes (bw_dram.cu)


def hbm_write_factor(ndr, op, payload, cache="cg"):
    """Share of a K4 launch's payload (bytes read + written) that reached HBM by the end of the launch: written lines
    still dirty in the L2 at kernel end (payload writes − dram__bytes_write of the ncu launch of the same configuration,
    kernel replay with the L2 flushed before) are not HBM traffic of the timed launch."""
    r = ndr[(ndr.op == op) & (ndr.cache_op == cache)].iloc[0]
    dirty = max(0.0, payload * WRITE_SHARE[op] - float(r["dram__bytes_write.sum"]))
    return (payload - dirty) / payload
