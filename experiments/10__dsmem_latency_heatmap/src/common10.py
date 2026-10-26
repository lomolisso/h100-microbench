"""Shared helpers of experiment 10's scripts and notebook (paths relative to the experiment directory).

  runs()                          the run registry: one row per run_id with started/cmdline/complete
  latest_runs(mode, schedule)     run_ids of complete `bin/heatmap <mode> [--schedule S]` invocations (not --quick), oldest first
  append_rows(path, header, rows) append-only CSV writer (flock, header on an empty file, checked otherwise, fsync)
  sm_map()                        latest 03 map: smid → gpc, max_cs_observed, tpc (= smid // 2), tpc_index (rank of the TPC
                                  among its GPC's TPCs in ascending smid order), capable (TPC joins clusters ≥ 4)
"""
import csv
import fcntl
import io
import os

import pandas as pd

EXP = "10__dsmem_latency_heatmap"
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


def latest_runs(mode, schedule=None, quick=False, res=RES):
    r = runs(res)
    toks = r.cmdline.str.split()

    def ok(t):
        if len(t) < 2 or not t[0].endswith("/heatmap") or t[1] != mode or (("--quick" in t) != quick):
            return False
        if schedule is None:
            return True
        return "--schedule" in t and t[t.index("--schedule") + 1] == schedule

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
    m["tpc_index"] = m.groupby("gpc").tpc.rank(method="dense").astype(int) - 1
    cap = m.groupby("tpc").max_cs_observed.min() >= 4
    m["capable"] = m.tpc.map(cap)
    return m.set_index("smid")


# ---------------------------------------------------------------- analysis helpers (analyze.py and the notebook)
import itertools
import math

import numpy as np

SCHEDULES = ["E", "A", "C", "B", "D", "D8"]


def full_runs(res=RES, quick=False):
    """{schedule: run_id} of the latest complete full (non --quick) invocation per schedule, plus 'steer' and 'ncu'."""
    out = {}
    for s in SCHEDULES:
        r = latest_runs("sched", s, quick=quick, res=res)
        if r:
            out[s] = r[-1]
    for mode in ("steer", "ncu"):
        r = latest_runs(mode, quick=quick, res=res) or latest_runs(mode, res=res)
        if r:
            out[mode] = r[-1]
    return out


def read_runs(file, run_ids, res=RES, usecols=None):
    parts = [ch[ch.run_id.isin(run_ids)] for ch in pd.read_csv(f"{res}/{file}", chunksize=2_000_000, usecols=usecols)]
    return pd.concat(parts, ignore_index=True)


def mad(x):
    x = np.asarray(x, float)
    return float(np.median(np.abs(x - np.median(x)))) if len(x) else float("nan")


def pair_table(df, col, by=("smid_src", "smid_dst")):
    """Per pair: median, MAD over launches, n launches, p5, p95, min, max of `col`."""
    g = df.groupby(list(by))[col]
    t = pd.DataFrame({"median": g.median(), "mad": g.agg(mad), "n": g.size(), "p5": g.quantile(0.05), "p95": g.quantile(0.95),
                      "min": g.min(), "max": g.max()})
    return t.reset_index()


def coschedulable_pairs(m):
    """Directed SM pairs that can share a cluster: both cluster-capable SMs of one GPC, or TPC mates."""
    out = []
    for g, sub in m.groupby("gpc"):
        cap = sorted(sub.index[sub.capable])
        out += [(a, b) for a in cap for b in cap if a != b]
        for s in sorted(sub.index[~sub.capable]):
            out.append((s, s ^ 1))
    return out


def fit_additive(pairs, val, symmetric=False):
    """Least-squares fit val(i,j) = mu + a(i) + b(j) (symmetric: b = a). pairs = (N, 2) SM ids. Returns dict with mu, a, b
    (dicts smid → offset, centred per connected GPC so that mean a = mean b = 0 within it), fitted, resid, r2."""
    pairs = np.asarray(pairs)
    y = np.asarray(val, float)
    sms = np.unique(pairs)
    idx = {s: k for k, s in enumerate(sms)}
    n = len(sms)
    ncol = 1 + (n if symmetric else 2 * n)
    X = np.zeros((len(y), ncol))
    X[:, 0] = 1
    for r, (i, j) in enumerate(pairs):
        X[r, 1 + idx[i]] += 1
        X[r, 1 + (idx[j] if symmetric else n + idx[j])] += 1
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coef
    resid = y - fitted
    a = coef[1:1 + n]
    b = a if symmetric else coef[1 + n:]
    ss = ((y - y.mean()) ** 2).sum()
    return {"mu": float(coef[0]), "a": dict(zip(sms, a)), "b": dict(zip(sms, b)), "fitted": fitted, "resid": resid,
            "r2": float(1 - (resid ** 2).sum() / ss) if ss > 0 else float("nan")}


def centred_offsets(fit, m):
    """Offsets with the per-GPC gauge removed: e_src(i) = a(i) − mean_GPC(a), e_dst(j) = b(j) − mean_GPC(b), and
    mu_gpc = mu + mean_GPC(a) + mean_GPC(b) (each GPC's matrix is a separate connected component)."""
    rows = []
    for s in fit["a"]:
        rows.append({"smid": s, "gpc": int(m.gpc[s]), "a": fit["a"][s], "b": fit["b"][s]})
    d = pd.DataFrame(rows)
    ga, gb = d.groupby("gpc").a.transform("mean"), d.groupby("gpc").b.transform("mean")
    d["mu_gpc"] = fit["mu"] + ga + gb
    d["e_src"], d["e_dst"] = d.a - ga, d.b - gb
    return d.set_index("smid")


def f_two(x, y):
    """One-way ANOVA F for two groups and the difference of means (y − x)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or len(y) < 2:
        return float("nan"), float("nan")
    allv = np.concatenate([x, y])
    gm = allv.mean()
    ssb = len(x) * (x.mean() - gm) ** 2 + len(y) * (y.mean() - gm) ** 2
    ssw = ((x - x.mean()) ** 2).sum() + ((y - y.mean()) ** 2).sum()
    dfw = len(allv) - 2
    return (float(ssb / (ssw / dfw)) if ssw > 0 else float("inf")), float(y.mean() - x.mean())


def partitions_le3(items):
    """Every partition of `items` into ceil(n/3) unlabeled groups of at most 3 (280 for n = 8 or 9, 7 for n = 4)."""
    items = list(items)
    k = math.ceil(len(items) / 3)
    out = []

    def rec(rest, groups):
        if not rest:
            if len(groups) == k:
                out.append([tuple(g) for g in groups])
            return
        if len(groups) + math.ceil(len(rest) / 3) > k:
            return
        first, others = rest[0], rest[1:]
        for size in (2, 1, 0):  # partners of `first`: 2, 1 or 0 more items
            for comb in itertools.combinations(others, size):
                rec([o for o in others if o not in comb], groups + [[first, *comb]])

    rec(items, [])
    return out


def cpc_label(part):
    return "|".join("-".join(str(i) for i in g) for g in sorted(part))


def cpc_tests(d, col):
    """d: one GPC's remote pairs with tpc_src, tpc_dst (TPC index in the GPC) and `col`. Same-TPC pairs are excluded from
    the CPC tests. Returns TPC F, mod-3 F, every partition's F (same-CPC vs other-CPC) and the best one.
    step = mean(other-CPC) − mean(same-CPC) (positive: pairs inside a group are faster)."""
    same_tpc = d.tpc_src == d.tpc_dst
    f_tpc, d_tpc = f_two(d.loc[same_tpc, col], d.loc[~same_tpc, col])
    o = d[~same_tpc]
    tp = sorted(set(o.tpc_src) | set(o.tpc_dst))
    res = []
    for part in partitions_le3(tp):
        gid = {t: k for k, g in enumerate(part) for t in g}
        same = o.tpc_src.map(gid).to_numpy() == o.tpc_dst.map(gid).to_numpy()
        f, step = f_two(o.loc[same, col], o.loc[~same, col])
        res.append((cpc_label(part), f, step))
    r = pd.DataFrame(res, columns=["partition", "f", "step"])
    mod3 = cpc_label([tuple(t for t in tp if t % 3 == k) for k in range(3) if any(t % 3 == k for t in tp)])
    rm = r[r.partition == mod3]
    best = r.loc[r.f.idxmax()]
    f_mod3 = float(rm.f.iloc[0]) if len(rm) else float("nan")
    return {"f_tpc": f_tpc, "tpc_step": d_tpc, "f_mod3": f_mod3, "mod3_step": float(rm.step.iloc[0]) if len(rm) else float("nan"),
            "mod3_rank_p": float((r.f >= f_mod3).mean()) if len(rm) else float("nan"), "mod3_partition": mod3,
            "f_best": float(best.f), "best_partition": best.partition, "best_step": float(best.step), "n_partitions": len(r),
            "all": r}


def relation3(d):
    """same_tpc / same_cpc_mod3 / other_cpc_mod3 from tpc_src, tpc_dst (TPC index within the GPC)."""
    return np.where(d.tpc_src == d.tpc_dst, "same TPC",
                    np.where(d.tpc_src % 3 == d.tpc_dst % 3, "same CPC (mod 3)", "other CPC (mod 3)"))
