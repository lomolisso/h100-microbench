"""Shared helpers of experiment 05's scripts and notebook (paths relative to the experiment directory).

  runs()                        the run registry: one row per run_id with started/finished/status/cmdline
  latest_run(tool, mode, quick) run_id of the latest *complete* invocation `bin/<tool> <mode>` (quick or not)
  read_run(file, run_id)        rows of results/<file> belonging to one run_id (pandas)
  append_rows(path, header, rows)  append-only CSV writer (flock, header on an empty file, checked otherwise, fsync)
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
