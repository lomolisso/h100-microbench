"""Shared helpers of experiment 04's scripts and notebook (paths relative to the experiment directory).

  runs()                       the run registry: one row per run_id with started/finished/status/cmdline
  latest_run(tool, mode, quick) run_id of the latest *complete* invocation `bin/<tool> <mode>` (quick or not)
  read_run(file, run_id)       rows of results/<file> belonging to one run_id (pandas)
  append_rows(path, header, rows)  append-only CSV writer (flock, header on an empty file, checked otherwise, fsync)
  kmeans1d(values, counts, k)  weighted 1-D K-means (sklearn, n_init = 20, seed 0): centres ascending, labels, SD
  sm_clocks()                  real SM clock of every SM under the 1635 MHz lock, per GPC (03's clock-domain run)
  at_nominal(cycles, mhz)      SM cycles measured at `mhz` → cycles at the nominal 1635 MHz (= ns × 1.635)
"""
import csv
import fcntl
import io
import os

import numpy as np
import pandas as pd

EXP = "04__memory_hierarchy_latency"
RES = "results"
NOMINAL_MHZ = 1635.0
RES03 = "../03__sm_gpc_topology/results"
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


def kmeans1d(values, counts=None, k=2):
    """Weighted K-means on 1-D data. Returns (centres ascending, label per value, SD per cluster, weight per cluster)."""
    from sklearn.cluster import KMeans

    x = np.asarray(values, dtype=float).reshape(-1, 1)
    w = np.ones(len(x)) if counts is None else np.asarray(counts, dtype=float)
    km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(x, sample_weight=w)
    order = np.argsort(km.cluster_centers_.ravel())
    remap = np.empty(k, int)
    remap[order] = np.arange(k)
    lab = remap[km.labels_]
    cen = km.cluster_centers_.ravel()[order]
    sd = np.array([np.sqrt(np.average((x[lab == j, 0] - cen[j]) ** 2, weights=w[lab == j])) for j in range(k)])
    wt = np.array([w[lab == j].sum() for j in range(k)])
    return cen, lab, sd, wt


def sm_clocks(res03=RES03):
    """Real SM clock under the lock, per SM and per GPC, from experiment 03.

    Clock: latest complete `gpc_probe clockdom` run of 03 (results/03__sm_gpc_topology__clock_domains.csv: one row
    per SM and launch, sm_mhz = span_cycles / span_ns over a 33.5 ms %globaltimer window); per-SM value = median over
    the launches. GPC: latest run_id of 03's map (03__sm_gpc_topology__map.csv). Per-GPC clock = median of the per-SM
    medians of the GPC's SMs (they agree to < 0.002 MHz within a GPC).
    Returns (DataFrame indexed by smid with columns gpc, sm_mhz, gpc_mhz; clockdom run_id; map run_id).
    """
    r = pd.read_csv(f"{res03}/03__sm_gpc_topology__runs.csv", dtype=str, keep_default_na=False)
    done = set(r.loc[r.status == "complete", "run_id"])
    st = r[(r.status == "started") & r.run_id.isin(done)]
    toks = st.cmdline.str.split()
    st = st[toks.map(lambda t: len(t) > 1 and t[0].endswith("/gpc_probe") and t[1] == "clockdom" and "--quick" not in t)]
    if st.empty:
        raise SystemExit("no complete clockdom run of experiment 03")
    rid = st.sort_values("started_iso").run_id.iloc[-1]
    c = pd.read_csv(f"{res03}/03__sm_gpc_topology__clock_domains.csv")
    sm = c[c.run_id == rid].groupby("smid").sm_mhz.median()
    m = pd.read_csv(f"{res03}/03__sm_gpc_topology__map.csv")
    map_rid = m.run_id.max()
    gpc = m[m.run_id == map_rid].set_index("smid").gpc.astype(int)
    out = pd.DataFrame({"gpc": gpc, "sm_mhz": sm}).dropna()
    out["gpc"] = out.gpc.astype(int)
    out["gpc_mhz"] = out.gpc.map(out.groupby("gpc").sm_mhz.median())
    return out, rid, map_rid


def at_nominal(cycles, mhz):
    """SM cycles counted at the real clock `mhz` → cycles at the nominal 1635 MHz: ns = cycles / mhz × 1e3, × 1.635."""
    return np.asarray(cycles, float) * NOMINAL_MHZ / np.asarray(mhz, float)
