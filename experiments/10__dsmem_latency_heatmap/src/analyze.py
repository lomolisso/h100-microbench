"""Experiment 10 derived tables from the latest complete full run of every schedule (no GPU). Run from the repository root:
    tools/run --no-gpu -- bash -c "cd experiments/10__dsmem_latency_heatmap && python3 src/analyze.py"

Matrices (per directed SM pair, remote pairs only, idle placeholders, every recorded launch of schedules A, B, C and the
idle sessions of D and D8, all cluster sizes pooled):
  chase        median over launches of cycles_per_hop (4096-hop mean of the dependent ld.shared::cluster chain)
  loaded       median over launches of the per-launch p50 of the 6144 loaded samples (8 loads × 1024 threads)
  vis_st_async one-way cycles = median over launches of the per-launch median round trip / 2 (st_async_mbarrier_nogt)
  vis_st_plain the same for st_plain_nogt
Outputs (append, own run_id = time of this analysis, registered in the runs registry as `analyze`; source runs in
source_run_ids):
  10__heatmap__pair_stats.csv  matrix, cs ("all" / 16 / 8 / 2), smid_src, smid_dst, gpc, tpc_idx_src, tpc_idx_dst, relation,
                                 n_launches, median, mad, p5, p95, min, max (over launches)
  10__heatmap__visibility.csv    variant, smid_src, smid_dst, gpc, one_way_cycles_median (= round-trip median / 2), mad, n
                                 (round trips of all trials pooled; n = trials), one_way_ns_mean (q→p, %globaltimer variant)
  10__heatmap__model.csv         gpc ("all" or 0–7), matrix, n_pairs, mu, r2_additive, r2_symmetric, resid_std, resid_max,
                                 f_stat_tpc, tpc_step, f_stat_cpc_mod3, cpc_mod3_step, cpc_mod3_rank_p, f_stat_cpc_best,
                                 best_cpc_partition, best_cpc_step, n_partitions, the same five CPC/TPC columns on the
                                 additive residuals (prefix resid_), median_pair_mad
  10__heatmap__offsets.csv       matrix, smid, gpc, tpc_index, mu_gpc, e_src, e_dst (additive fit, per-GPC gauge removed)
  10__heatmap__cpc_map.csv       smid, gpc, tpc_index_in_gpc, cpc (group of the CPC grouping if found, else "none")
  10__heatmap__summary.csv       level, config, median, p5, p95, mean, unit, n, source, source_run_ids
"""
import datetime
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common10 import (MANDATORY, append_rows, centred_offsets, coschedulable_pairs, cpc_tests, fit_additive, full_runs, pair_table,
                      read_runs, sm_map, mad)

QUICK = "--quick" in sys.argv  # test mode: the latest --quick runs, outputs to --out DIR (never into results/)
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "results"
if QUICK and OUT == "results":
    raise SystemExit("--quick needs --out DIR")
runs = full_runs(quick=QUICK)
need = ["A", "B", "C", "D", "D8", "E"]
missing = [s for s in need if s not in runs]
if missing:
    raise SystemExit(f"no complete full run of schedule(s) {missing}")
ids = [runs[s] for s in need]
m = sm_map()
tpc_idx = m.tpc_index.to_dict()
IDLE = ["A", "B", "C", "D", "D8"]

pairs = read_runs("10__heatmap__pairs.csv", ids)
loaded = read_runs("10__heatmap__loaded.csv", ids)
vis = read_runs("10__heatmap__visibility_trials.csv", ids)
clk = pairs.sm_clock_mhz.median()


def idle_remote(df):
    return df[df.schedule.isin(IDLE) & (df.load_condition == "idle") & (df.smid_src != df.smid_dst)]


mats = {
    "chase": (idle_remote(pairs), "cycles_per_hop"),
    "loaded": (idle_remote(loaded), "p50"),
}
vr = idle_remote(vis).copy()
vr["one_way_cycles"] = vr.rt_median_cycles / 2
mats["vis_st_async"] = (vr[vr.method == "st_async_mbarrier_nogt"], "one_way_cycles")
mats["vis_st_plain"] = (vr[vr.method == "st_plain_nogt"], "one_way_cycles")

run_id = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")
first = pairs.iloc[0]
pre = [first.gpu_name, first.sm_count, int(round(clk)), first.mem_clock_mhz, first.cuda_version, first.driver_version, run_id, -1]
src = ";".join(f"{s}={runs[s]}" for s in need)


def rel(a, b):
    return "self" if a == b else ("same_tpc" if a // 2 == b // 2 else "other_tpc")


# ---- pair summary
rows, tables = [], {}
for name, (df, col) in mats.items():
    for cs in ["all", 16, 8, 2]:
        sub = df if cs == "all" else df[df.cs == cs]
        if sub.empty:
            continue
        t = pair_table(sub, col)
        if cs == "all":
            tables[name] = t
        for r in t.itertuples():
            rows.append(pre + [name, cs, r.smid_src, r.smid_dst, int(m.gpc[r.smid_src]), tpc_idx[r.smid_src], tpc_idx[r.smid_dst],
                               rel(r.smid_src, r.smid_dst), r.n, r.median, r.mad, r.p5, r.p95, r.min, r.max, src])
append_rows(f"{OUT}/10__heatmap__pair_stats.csv",
            MANDATORY + ["matrix", "cs", "smid_src", "smid_dst", "gpc", "tpc_idx_src", "tpc_idx_dst", "relation", "n_launches", "median",
                         "mad", "p5", "p95", "min", "max", "source_run_ids"], rows)

# ---- visibility (spec table): pooled trial histograms of the idle sessions
vh = read_runs("10__heatmap__visibility_hist.csv", ids)
vh = vh[vh.schedule.isin(IDLE) & (vh.load_condition == "idle")]
vrows = []
for (meth, q), g in vh.groupby(["method", "quantity"]):
    if q != "rt_cycles":
        continue
    ow = vh[(vh.method == meth) & (vh.quantity == "ow_ns")]
    owm = (ow.assign(w=ow.value * ow["count"]).groupby(["smid_src", "smid_dst"]).w.sum() /
           ow.groupby(["smid_src", "smid_dst"])["count"].sum()) if len(ow) else pd.Series(dtype=float)
    for (a, b), h in g.groupby(["smid_src", "smid_dst"]):
        h = h.groupby("value")["count"].sum().sort_index()
        v, c = h.index.to_numpy(float) / 2, h.to_numpy()
        cum = np.cumsum(c)
        med = v[np.searchsorted(cum, cum[-1] / 2)]
        dev = np.abs(v - med)
        o = np.argsort(dev)
        madv = dev[o][np.searchsorted(np.cumsum(c[o]), cum[-1] / 2)]
        vrows.append(pre + [meth, a, b, int(m.gpc[a]), med, madv, int(cum[-1]), owm.get((a, b), np.nan), src])
append_rows(f"{OUT}/10__heatmap__visibility.csv",
            MANDATORY + ["variant", "smid_src", "smid_dst", "gpc", "one_way_cycles_median", "mad", "n", "one_way_ns_mean", "source_run_ids"], vrows)

# ---- model + CPC tests
cosched = coschedulable_pairs(m)
mrows, orows, cpc_found = [], [], {}
for name, t in tables.items():
    t = t.copy()
    t["gpc"] = t.smid_src.map(m.gpc)
    t["tpc_src"], t["tpc_dst"] = t.smid_src.map(tpc_idx), t.smid_dst.map(tpc_idx)
    cap = t.smid_src.map(m.capable) & t.smid_dst.map(m.capable)  # cluster ≥ 4 pairs (mate-only TPCs excluded from fits)
    fa = fit_additive(t.loc[cap, ["smid_src", "smid_dst"]].to_numpy(), t.loc[cap, "median"].to_numpy())
    fs = fit_additive(t.loc[cap, ["smid_src", "smid_dst"]].to_numpy(), t.loc[cap, "median"].to_numpy(), symmetric=True)
    t.loc[cap, "resid"] = fa["resid"]
    off = centred_offsets(fa, m)
    for s, r in off.iterrows():
        orows.append(pre + [name, s, int(r.gpc), tpc_idx[s], r.mu_gpc, r.e_src, r.e_dst, src])
    for g in ["all"] + sorted(t.gpc.unique()):
        d = t[cap] if g == "all" else t[cap & (t.gpc == g)]
        if g == "all":
            res = d.resid
            mrows.append(pre + ["all", name, len(d), fa["mu"], fa["r2"], fs["r2"], res.std(), res.abs().max()] + [np.nan] * 17 +
                         [d["mad"].median(), src])
            continue
        y = d["median"]
        ss = ((y - y.mean()) ** 2).sum()
        r2 = 1 - (d.resid ** 2).sum() / ss if ss > 0 else np.nan
        sres = fs["resid"][(t[cap].gpc == g).to_numpy()]
        r2s = 1 - (sres ** 2).sum() / ss if ss > 0 else np.nan
        c = cpc_tests(d, "median")
        cr = cpc_tests(d, "resid")
        if name == "chase":
            cpc_found[g] = (c, cr, d["mad"].median())
        mrows.append(pre + [g, name, len(d), float(off[off.gpc == g].mu_gpc.iloc[0]), r2, r2s, d.resid.std(), d.resid.abs().max(),
                            c["f_tpc"], c["tpc_step"], c["f_mod3"], c["mod3_step"], c["mod3_rank_p"], c["f_best"], c["best_partition"],
                            c["best_step"], c["n_partitions"], cr["f_tpc"], cr["tpc_step"], cr["f_mod3"], cr["mod3_step"], cr["mod3_rank_p"],
                            cr["f_best"], cr["best_partition"], cr["best_step"], d["mad"].median(), src])
append_rows(f"{OUT}/10__heatmap__model.csv",
            MANDATORY + ["gpc", "matrix", "n_pairs", "mu", "r2_additive", "r2_symmetric", "resid_std", "resid_max", "f_stat_tpc", "tpc_step",
                         "f_stat_cpc_mod3", "cpc_mod3_step", "cpc_mod3_rank_p", "f_stat_cpc_best", "best_cpc_partition", "best_cpc_step",
                         "n_partitions", "resid_f_stat_tpc", "resid_tpc_step", "resid_f_stat_cpc_mod3", "resid_cpc_mod3_step",
                         "resid_cpc_mod3_rank_p", "resid_f_stat_cpc_best", "resid_best_cpc_partition", "resid_best_cpc_step",
                         "median_pair_mad", "source_run_ids"], mrows)
append_rows(f"{OUT}/10__heatmap__offsets.csv",
            MANDATORY + ["matrix", "smid", "gpc", "tpc_index", "mu_gpc", "e_src", "e_dst", "source_run_ids"], orows)

# ---- CPC map. Rule fixed before the run: a grouping is "found" if, on the chase residuals, the best partition's step
# (other − same) exceeds 3 × the median per-pair MAD in every GPC and the best partition is the same in all GPCs with the
# same number of TPCs. Otherwise every SM gets cpc = none.
found = True
by_n = {}
for g, (c, cr, md) in cpc_found.items():
    n_tpc = len(set(m[(m.gpc == g) & m.capable].tpc_index))
    by_n.setdefault(n_tpc, set()).add(cr["best_partition"])
    if not (cr["best_step"] > 3 * md):
        found = False
if any(len(v) > 1 for v in by_n.values()):
    found = False
crow = []
for s, r in m.iterrows():
    label = "none"
    if found and int(r.gpc) in cpc_found:
        part = cpc_found[int(r.gpc)][1]["best_partition"]
        for k, grp in enumerate(part.split("|")):
            if str(int(r.tpc_index)) in grp.split("-"):
                label = str(k)
    crow.append(pre + [s, int(r.gpc), int(r.tpc_index), label, src])
append_rows(f"{OUT}/10__heatmap__cpc_map.csv", MANDATORY + ["smid", "gpc", "tpc_index_in_gpc", "cpc", "source_run_ids"], crow)

# ---- summary for later experiments
srows = []


def add(level, cfg, x, unit, source):
    x = pd.Series(x).dropna()
    srows.append(pre + [level, cfg, x.median(), x.quantile(0.05), x.quantile(0.95), x.mean(), unit, len(x), source, src])


ch = tables["chase"]
cov = set(zip(ch.smid_src, ch.smid_dst))
add("chase/remote", "idle, all CS, pair medians", ch["median"], "cycles", "K1")
add("chase/pair_mad", "idle, all CS", ch["mad"], "cycles", "K1")
add("chase/n_launches_per_pair", "idle, all CS", ch["n"], "launches", "K1")
add("coverage/pairs_measured", f"of {len(cosched)} co-schedulable", [len(cov & set(cosched))], "pairs", "K1")
selfs = pairs[(pairs.smid_src == pairs.smid_dst) & (pairs.load_condition == "idle")]
add("chase/self", "own CTA via DSMEM path", selfs.cycles_per_hop, "cycles", "K1")
for g in sorted(ch.smid_src.map(m.gpc).unique()):
    add(f"chase/remote/G{g}", "idle", ch.loc[ch.smid_src.map(m.gpc) == g, "median"], "cycles", "K1")
add("loaded/remote", "pair medians of launch p50 (8 loads)", tables["loaded"]["median"], "cycles", "K1f")
for v in ("vis_st_async", "vis_st_plain"):
    add(f"{v}/one_way_cycles", "round trip / 2, pair medians", tables[v]["median"], "cycles", "K2")
for cond in ("light", "heavy"):
    for sch in ("D", "D8"):
        dd = pairs[(pairs.schedule == sch) & (pairs.smid_src != pairs.smid_dst)]
        pm = dd.groupby(["load_condition", "smid_src", "smid_dst"]).cycles_per_hop.median().unstack(0)
        if cond in pm and "idle" in pm:
            add(f"chase/{cond}_minus_idle/{sch}", "per pair", pm[cond] - pm["idle"], "cycles", "K1")
append_rows(f"{OUT}/10__heatmap__summary.csv",
            MANDATORY + ["level", "config", "median", "p5", "p95", "mean", "unit", "n", "source", "source_run_ids"], srows)

# register the analysis as a run of its own
reg = "results/10__dsmem_latency_heatmap__runs.csv"
now = datetime.datetime.now().astimezone().isoformat(timespec="milliseconds")
uuid = pd.read_csv(reg, dtype=str).gpu_uuid.dropna().iloc[-1]
if not QUICK:
  append_rows(reg, ["run_id", "started_iso", "finished_iso", "status", "gpu_uuid", "cmdline"],
            [[run_id, now, "", "started", uuid, "src/analyze.py"], [run_id, "", now, "complete", uuid, "src/analyze.py"]])
print(f"analysis {run_id}: {len(rows)} pair rows, {len(vrows)} visibility rows, {len(mrows)} model rows; CPC grouping found: {found}")
print(pd.DataFrame([r[8:] for r in mrows if r[9] == "chase"]).iloc[:, :20].to_string())
