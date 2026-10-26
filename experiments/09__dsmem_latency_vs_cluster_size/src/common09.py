"""Shared helpers of experiment 09's scripts and notebook (paths relative to the experiment directory).

  runs()                         the run registry: one row per run_id with started/cmdline/complete
  latest_run(tool, mode, quick)  run_id of the latest *complete* invocation `bin/<tool> <mode>` (quick or not)
  read_run(file, run_id)         rows of results/<file> belonging to one run_id (pandas, chunked)
  append_rows(path, header, rows)  append-only CSV writer (flock, header on an empty file, checked otherwise, fsync)
  sm_map()                       latest 03 map of this device: smid → gpc (+ tpc = smid // 2)
  stats(df, by, col)             median, p5, p95, mean, std, n per group
  load(kind, quick)              rows of one results file from the latest complete run of the binary/mode that writes it
                                 (kind: chase, fine, loaded, loaded_hist, baselines, visibility, ncu_k1) and that run_id;
                                 adds `pair` = self / same_tpc / other_tpc where smid_src/smid_dst exist (TPC = smid // 2)
"""
import csv
import fcntl
import io
import os

import numpy as np
import pandas as pd

EXP = "09__dsmem_latency_vs_cluster_size"
RES = "results"
MAP03 = "../03__sm_gpc_topology/results/03__sm_gpc_topology__map.csv"
MANDATORY = ["gpu_name", "sm_count", "sm_clock_mhz", "mem_clock_mhz", "cuda_version", "driver_version", "run_id", "launch_idx"]


def runs(res=RES):
    r = pd.read_csv(f"{res}/{EXP}__runs.csv", dtype=str, keep_default_na=False)
    started = r[r.status == "started"].set_index("run_id")
    done = set(r.loc[r.status == "complete", "run_id"])
    out = started[["started_iso", "cmdline", "gpu_uuid"]].copy()
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
    parts = [c[c.run_id == run_id] for c in pd.read_csv(f"{res}/{file}", chunksize=2_000_000, **kw)]
    return pd.concat(parts, ignore_index=True)


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


def sm_map(path=MAP03):
    m = pd.read_csv(path)
    m = m[m.run_id == m.run_id.max()][["smid", "gpc", "gpc_size", "notes"]].copy()
    m["tpc"] = m.smid // 2
    return m.set_index("smid")


def stats(df, by, col):
    g = df.groupby(by, observed=True)[col]
    return pd.DataFrame({"median": g.median(), "p5": g.quantile(0.05), "p95": g.quantile(0.95), "mean": g.mean(),
                         "std": g.std(), "n": g.size()})


SOURCE = {"chase": ("dsmem_lat", "k1"), "fine": ("dsmem_lat", "k1"), "loaded": ("dsmem_lat", "loaded"),
          "loaded_hist": ("dsmem_lat", "loaded"), "baselines": ("dsmem_lat", "k3"), "visibility": ("dsmem_vis", "k2"),
          "ncu_k1": ("dsmem_lat", "ncu")}


def pair_kind(src, dst):
    return np.where(src == dst, "self", np.where(src // 2 == dst // 2, "same_tpc", "other_tpc"))


def load(kind, quick=False, res=RES, **kw):
    tool, mode = SOURCE[kind]
    run = latest_run(tool, mode, quick, res)
    df = read_run(f"09__dsmem_latency__{kind}.csv", run, res, **kw)
    if "smid_src" in df and "smid_dst" in df:
        df["pair"] = pair_kind(df.smid_src.to_numpy(), df.smid_dst.to_numpy())
    return df, run
