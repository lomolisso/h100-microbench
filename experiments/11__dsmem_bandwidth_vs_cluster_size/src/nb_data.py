"""Data loading for the experiment-11 notebook: the latest complete run of every step, the latest analysis, the GMEM
reference lines from experiment 05's L2 and HBM runs on the same card, and small helpers. Run from the experiment directory.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common11 import EXP, RES, latest_runs, runs, sm_map  # noqa: E402

R05 = "../05__memory_hierarchy_bandwidth/results"
STEPS = ["A", "S", "B", "C", "D", "E", "F", "P1", "K4", "ncu"]
ASYNC = ("st_async", "tma_put", "tma_red")
OPS = ["ld", "st", "st_async", "tma_put", "tma_red"]
OPLAB = {"ld": "ld.shared::cluster.v4", "st": "st.shared::cluster.v4", "st_async": "st.async.v4 + mbarrier",
         "tma_put": "TMA put (cp.async.bulk s→s)", "tma_red": "TMA reduce-put (add.u32)"}


def sources():
    src = {s: latest_runs("dsmem_bw", s)[-1] for s in STEPS}
    src["K5"] = latest_runs("dsmem_l2mix", "k5")[-1]
    return src


def analysis_run():
    r = runs()
    a = r[(r.cmdline == "src/analyze.py") & r.complete].sort_values("started_iso")
    return a.run_id.iloc[-1]


def read_ids(file, ids, usecols=None):
    it = pd.read_csv(f"{RES}/{file}", chunksize=1_000_000, usecols=usecols)
    return pd.concat([c[c.run_id.isin(ids)] for c in it], ignore_index=True)


def bw_side(stream):
    return "target" if stream in ASYNC else "issuer"


def gmem_reference(uuid):
    """L2 read and HBM read TB/s on this card from 05's latest complete K3/K5 and K4 runs (event floor removed),
    with the same selection as 05's summary: L2 = 16 MiB, ld.global.cg, 2 × 132 × 1024 threads, ILP 4, mixed homes;
    HBM = read, 4 × 132 × 1024 threads, ILP 4, .cg."""
    sys.path.insert(0, "../05__memory_hierarchy_bandwidth/src")
    import common05 as c5
    reg = pd.read_csv(f"{R05}/05__memory_hierarchy_bandwidth__runs.csv", dtype=str)
    r3, r4 = c5.latest_run("bw_l2", "k3", res=R05), c5.latest_run("bw_dram", "k4", res=R05)
    card = {r: reg[reg.run_id == r].gpu_uuid.iloc[0] for r in (r3, r4)}
    floor = c5.event_floor_ms()
    l2 = c5.read_run("05__bandwidth__l2.csv", r3, res=R05)
    l2 = l2[(l2.discard.isna()) | (l2.discard == "")]
    l2 = l2.assign(tbps=c5.net_tbps(l2, floor))
    l2 = l2[(l2["pass"] == 0) & (l2.ilp == 4) & (l2.blocks == 264) & (l2.cache_op == "cg") & (l2.op == "ld") &
            (l2.variant == "mixed") & (l2.footprint_bytes == 16 << 20)]
    dr = c5.read_run("05__bandwidth__dram.csv", r4, res=R05)
    dr = dr[(dr.discard.isna()) | (dr.discard == "")]
    dr = dr.assign(tbps=c5.net_tbps(dr, floor))
    dr = dr[(dr.blocks == 4 * 132) & (dr.ilp == 4) & (dr.cache_op == "cg") & (dr.op == "read")]
    return {"l2_read_tbps": float(l2.tbps.median()), "hbm_read_tbps": float(dr.tbps.median()), "n_l2": len(l2), "n_hbm": len(dr),
            "l2_run": r3, "hbm_run": r4, "same_card": all(v == uuid for v in card.values())}


def affine(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    b, a = np.polyfit(x, y, 1)
    return a, b
