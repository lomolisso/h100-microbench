"""Data loading for the experiment-12 notebook: the latest complete (non-quick) run of every step, the latest analysis run,
and small helpers. Run from the experiment directory."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common12 import EXP, RES, latest_runs, runs, sm_partition  # noqa: E402

STEPS = {"atom_lat": ["k0a", "k0b", "calib", "smem", "dsmem", "pairs", "l2", "l2fine", "hbm", "sys", "flags", "ncu_l2"],
         "atom_tput": ["k2", "k2all", "k2x", "k3a", "k3b", "k3c", "k3d"],
         "dsmem_async": ["k4", "k4ring", "k5", "k6", "k6poll1", "k7"]}
EVENT_FLOOR_MS = 3.9e-3  # cudaEvent pair floor (experiment 01)
TARGET_LABEL = {"smem": "local smem", "smem_dsmem_if": "own smem via window", "dsmem": "peer smem (DSMEM)", "l2_near": "L2 near home",
                "l2_far": "L2 far home", "hbm_near": "HBM, near home", "hbm_far": "HBM, far home", "sysmem": "host memory"}


def sources():
    out = {}
    for b, modes in STEPS.items():
        for m in modes:
            ids = latest_runs(b, m)
            if ids:
                out[m] = ids[-1]
    return out


def analysis_run():
    r = runs()
    a = r[(r.cmdline == "src/analyze.py") & r.complete].sort_values("started_iso")
    return a.run_id.iloc[-1]


def read_ids(file, ids, **kw):
    it = pd.read_csv(f"{RES}/{file}", chunksize=1_000_000, low_memory=False, **kw)
    return pd.concat([c[c.run_id.isin(ids)] for c in it], ignore_index=True)


def wquantile(values, counts, q):
    o = np.argsort(np.asarray(values))
    v, c = np.asarray(values)[o], np.asarray(counts)[o]
    cum = np.cumsum(c) / c.sum()
    return float(v[min(np.searchsorted(cum, q), len(v) - 1)])


def kmeans1d(x, k, w=None, iters=100):
    """Lloyd's 1-D k-means on values x (weights w); returns sorted centres, boundaries and the weighted SSE."""
    x = np.asarray(x, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    c = np.quantile(np.repeat(x, 1) if len(x) < 2 else x, np.linspace(0.1, 0.9, k))
    for _ in range(iters):
        lab = np.argmin(np.abs(x[:, None] - c[None, :]), axis=1)
        nc = np.array([np.average(x[lab == j], weights=w[lab == j]) if (lab == j).any() else c[j] for j in range(k)])
        if np.allclose(nc, c):
            break
        c = nc
    c = np.sort(c)
    lab = np.argmin(np.abs(x[:, None] - c[None, :]), axis=1)
    sse = float(np.sum(w * (x - c[lab]) ** 2))
    return c, (c[1:] + c[:-1]) / 2, sse


def kmeans_elbow(x, w=None, kmax=4, gain=0.1):
    """smallest k whose next k improves the SSE by less than `gain` (relative)"""
    res = [kmeans1d(x, k, w) for k in range(1, kmax + 1)]
    for k in range(1, kmax):
        if res[k - 1][2] == 0 or (res[k - 1][2] - res[k][2]) / res[k - 1][2] < gain:
            return k, res[k - 1]
    return kmax, res[-1]


def fix_2ld(df):
    """2ld rows (plain loads after a protocol) → variant after_<kind of the protocol's pass 1> (same run_id, launch_idx)."""
    if "pass" not in df or not (df["pass"] == "2ld").any():
        return df
    p1 = df[df["pass"] == "1"].drop_duplicates(["run_id", "launch_idx"]).set_index(["run_id", "launch_idx"]).kind
    m = df["pass"] == "2ld"
    df = df.copy()
    df["after"] = ""
    df.loc[m, "after"] = [p1.get(i, "?") for i in zip(df.loc[m, "run_id"], df.loc[m, "launch_idx"])]
    return df
