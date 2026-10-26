"""Shared helpers of experiment 12 (analysis scripts and the notebook).

runs()            the run registry with a `complete` flag per run_id.
latest_runs(...)  complete run_ids of one binary/mode (quick runs excluded unless asked), oldest first.
append_rows(...)  crash-safe append under an exclusive lock; header written only into an empty file, checked otherwise.
sm_map()          latest 03 map (smid, gpc, gpc_size, max_cs_observed) with the TPC index inside the GPC.
"""
import csv
import fcntl
import io
import os

import pandas as pd

EXP = "12__dsmem_atomics_and_async_stores"
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


def latest_runs(binary, mode, quick=False, res=RES):
    r = runs(res)
    toks = r.cmdline.str.split()

    def ok(t):
        return len(t) >= 2 and t[0].endswith("/" + binary) and t[1] == mode and (("--quick" in t) == quick)

    sel = r[toks.map(ok) & r.complete]
    return list(sel.sort_values("started_iso").run_id)


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
    m = m[m.run_id == m.run_id.max()][["smid", "gpc", "gpc_size", "max_cs_observed"]].copy()
    m["tpc"] = m.smid // 2
    idx = {}
    for g, d in m.groupby("gpc"):
        for i, t in enumerate(sorted(d.tpc.unique())):
            idx[t] = i
    m["tpc_index"] = m.tpc.map(idx)
    return m.sort_values("smid").reset_index(drop=True)


def sm_partition(res=RES):
    """Per-SM L2 partition on this card (K0a, latest complete k0a run)."""
    p = pd.read_csv(f"{res}/12__atomics__sm_partition.csv")
    full = latest_runs("atom_lat", "k0a", res=res)
    rid = full[-1] if full else p.run_id.max()
    return p[p.run_id == rid].sort_values("smid").reset_index(drop=True)
