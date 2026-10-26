"""Derived tables of experiment 12 from the latest complete (non-quick) run of every step. Run from the experiment directory:
    python3 src/analyze.py
Registers itself in the run registry (cmdline src/analyze.py) and writes, with its own run_id and the mandatory columns
taken from the source run of each row:
  results/12__atomics__lat_summary.csv   latency per (target, cs, op, kind, variant, sem, scope, pass): median over SMs
      (T1/T2: 132 SMs; T3 lone: per destination; T4/T5: 8 issuing SMs; T7: 2 SMs) of the per-SM slope
      (median cycles of the 16384-hop runs − median of the 4096-hop runs) / 12288 (T7: 1024/256 hops); T6 (hbm): median of
      the per-access samples by near/far home (raw rows for add.u32 atom, exact histograms for the rest); p5/p95 over SMs
      (T6: over samples); ld_ref_cycles = the same statistic of the plain-load chain of the same target and word format.
  results/12__atomics__pairs.csv          T3 per directed SM pair (pairs step: 4096 hops, cycles_per_op median over launches)
      for add.u32 atom, cas.b32 (cas_fail), add.u32 red+ld and ld, with experiment 10's dependent-load median of the pair.
  results/12__atomics__summary.csv        one row per (target, op, kind) at the default semantics: latency (median),
      per-SM throughput (K2, 1024 threads, coalesced, best ILP: ops/cycle and payload B/cycle), GPU-wide throughput (K3 or
      K4 ring: Gops/s and payload TB/s); consumed by experiment 18.
"""
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common12 import EXP, MANDATORY, RES, append_rows, latest_runs, sm_partition  # noqa: E402

EVENT_FLOOR_MS = 3.9e-3  # cudaEvent pair floor (experiment 01)
P10 = "../10__dsmem_latency_heatmap/results/10__heatmap__pair_stats.csv"


def read(file, ids, **kw):
    it = pd.read_csv(f"{RES}/{file}", chunksize=1_000_000, low_memory=False, **kw)
    return pd.concat([c[c.run_id.isin(ids)] for c in it], ignore_index=True)


def src(binary, mode):
    ids = latest_runs(binary, mode)
    if not ids:
        raise SystemExit(f"no complete run of {binary} {mode}")
    return ids[-1]


def wquantile(values, counts, q):
    o = np.argsort(values)
    v, c = np.asarray(values)[o], np.asarray(counts)[o]
    cum = np.cumsum(c) / c.sum()
    return float(v[np.searchsorted(cum, q)])


KEY = ["target", "cs", "op", "kind", "variant", "sem", "scope", "pass"]


def fix_2ld(df):
    """The 2ld rows (plain-load pass after a protocol) carry the op of the protocol but kind ld: give them the variant
    after_<kind of pass 1> and the semantics of pass 1 (same run_id and launch_idx)."""
    if "pass" not in df or not (df["pass"] == "2ld").any():
        return df
    p1 = df[df["pass"] == "1"].drop_duplicates(["run_id", "launch_idx"]).set_index(["run_id", "launch_idx"])
    m = df["pass"] == "2ld"
    k = list(zip(df.loc[m, "run_id"], df.loc[m, "launch_idx"]))
    df = df.copy()
    df.loc[m, "variant"] = ["after_" + p1.kind.get(i, "?") for i in k]
    for c in ["sem", "scope"]:
        if c in df:
            df.loc[m, c] = [p1[c].get(i, "?") for i in k]
    return df


def slopes(lat, h1=4096, h2=16384, by=("smid_src", "dst_rank")):
    """per-SM (and destination) slope latency of every configuration"""
    g = lat.groupby(KEY + list(by) + ["hops"]).total_cycles.median().unstack("hops")
    g["lat"] = (g[h2] - g[h1]) / (h2 - h1)
    return g.reset_index()


def summarize(per, ref=None):
    s = per.groupby(KEY).lat.agg(lat_cycles="median", p5=lambda x: x.quantile(.05), p95=lambda x: x.quantile(.95), n="count").reset_index()
    return s


def main():
    started = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    run_id = started.replace("+00:00", "+00:00")
    S = {m: src("atom_lat", m) for m in ["smem", "dsmem", "pairs", "l2", "l2fine", "hbm", "sys", "calib", "flags"]}
    S.update({m: src("atom_tput", m) for m in ["k2", "k2all", "k3a", "k3b", "k3c", "k3d"]})
    S.update({m: src("dsmem_async", m) for m in ["k4", "k4ring", "k5", "k6", "k7"]})
    reg = os.path.join(RES, f"{EXP}__runs.csv")
    append_rows(reg, ["run_id", "started_iso", "finished_iso", "status", "gpu_uuid", "cmdline"],
                [[run_id, started, "", "started", "", "src/analyze.py"]])

    lat = read("12__atomics__latency.csv", [S[m] for m in ["smem", "dsmem", "l2", "sys", "hbm"]], dtype={"pass": str})
    mand = lat.iloc[0][MANDATORY].tolist()
    parts = []
    # T1/T2 and T3 lone: per SM (and destination) slopes
    for m in ["smem", "dsmem"]:
        d = lat[lat.run_id == S[m]]
        parts.append(summarize(slopes(d)))
    # T4/T5: slope per issuing SM and pass
    d = fix_2ld(lat[lat.run_id == S["l2"]])
    parts.append(summarize(slopes(d, by=("smid_src",))))
    # T7: 256 / 1024 hops
    d = lat[lat.run_id == S["sys"]]
    parts.append(summarize(slopes(d, 256, 1024, by=("smid_src",))))
    # T6: per access
    fine = read("12__atomics__fine.csv", [S["hbm"]], dtype={"pass": str})
    hist = read("12__atomics__fine_hist.csv", [S["hbm"]], dtype={"pass": str})
    part = sm_partition().set_index("smid").partition
    rows = []
    for (op, kind), f in fine.groupby(["op", "kind"]):
        f = f.assign(rel=np.where(f.home_partition == f.smid.map(part), "near", "far"))
        for rel, g in f.groupby("rel"):
            rows.append(dict(target="hbm", cs=1, op=op, kind=kind, variant="identity", sem="default", scope="default", **{"pass": f"1_{rel}"},
                             lat_cycles=g.cycles.median(), p5=g.cycles.quantile(.05), p95=g.cycles.quantile(.95), n=len(g)))
    hist = hist.assign(rel=np.where(hist.home_partition == hist.smid.map(part), "near", "far"))
    for (op, kind, rel), g in hist.groupby(["op", "kind", "rel"]):
        rows.append(dict(target="hbm", cs=1, op=op, kind=kind, variant="identity", sem="default", scope="default", **{"pass": f"1_{rel}"},
                         lat_cycles=wquantile(g.cycles, g["count"], .5), p5=wquantile(g.cycles, g["count"], .05),
                         p95=wquantile(g.cycles, g["count"], .95), n=int(g["count"].sum())))
    parts.append(pd.DataFrame(rows))
    L = pd.concat(parts, ignore_index=True)
    L["pass"] = L["pass"].astype(str)
    # same-target plain-load reference with the same word format (u32, u64, f16x2)
    fmt = {"add.u64": "add.u64", "cas.b64": "add.u64", "add.f64": "add.u64", "add.f16x2": "add.f16x2", "add.bf16x2": "add.f16x2",
           "add.v2.f16x2": "add.f16x2", "add.v4.f16x2": "add.f16x2"}
    ldr = L[(L.kind == "ld") & (L.variant.isin(["identity", "after_atomic"]))]
    def ref(r):
        p = r["pass"] if r.target != "hbm" else r["pass"]
        cand = ldr[(ldr.target == r.target) & (ldr.cs == r.cs) & (ldr.op == fmt.get(r.op, "add.u32")) & (ldr["pass"] == p) &
                   (ldr.variant == "identity")]
        if cand.empty and r.target.startswith("l2"):
            cand = ldr[(ldr.target == r.target) & (ldr.op == "add.u32") & (ldr["pass"] == p) & (ldr.variant == "identity")]
        return float(cand.lat_cycles.iloc[0]) if len(cand) else np.nan
    L["ld_ref_cycles"] = L.apply(ref, axis=1)
    out = [mand[:6] + [run_id, -1] + r for r in L[KEY + ["lat_cycles", "p5", "p95", "n", "ld_ref_cycles"]].values.tolist()]
    append_rows(f"{RES}/12__atomics__lat_summary.csv", MANDATORY + KEY + ["lat_cycles", "p5", "p95", "n", "ld_ref_cycles"], out)

    # T3 pairs vs experiment 10
    pr = read("12__atomics__latency.csv", [S["pairs"]], dtype={"pass": str})
    pp = pr.groupby(["op", "kind", "variant", "smid_src", "smid_dst"]).cycles_per_op.median().reset_index()
    if os.path.exists(P10):
        p10 = pd.read_csv(P10)
        p10 = p10[(p10.run_id == p10.run_id.max()) & (p10.matrix == "chase") & (p10.cs == "all")]
        p10 = p10.rename(columns={"median": "ld10_cycles"})[["smid_src", "smid_dst", "ld10_cycles"]]
        pp = pp.merge(p10, on=["smid_src", "smid_dst"], how="left")
    else:
        pp["ld10_cycles"] = np.nan
    out = [mand[:6] + [run_id, -1] + r for r in pp[["op", "kind", "variant", "smid_src", "smid_dst", "cycles_per_op", "ld10_cycles"]].values.tolist()]
    append_rows(f"{RES}/12__atomics__pairs.csv", MANDATORY + ["op", "kind", "variant", "smid_src", "smid_dst", "cycles_per_op", "ld10_cycles"], out)

    # summary for 18: latency + per-SM + GPU-wide throughput at the default semantics
    k2 = read("12__atomics__throughput_sm.csv", [S["k2"]])
    k2 = k2[(k2.discard.isna()) & (k2.threads == 1024) & (k2.pattern == "coal")]
    k2b = k2.groupby(["target", "op", "kind", "ilp"]).ops_per_cycle.median().reset_index()
    k2b = k2b.loc[k2b.groupby(["target", "op", "kind"]).ops_per_cycle.idxmax()]
    k3 = read("12__atomics__throughput_gpu.csv", [S["k3a"]])
    k3 = k3[k3.discard.isna()].copy()
    k3["gops"] = k3.ops / ((k3.ms_event - EVENT_FLOOR_MS) * 1e-3) / 1e9
    k3s = k3[(k3.pattern == "coal")].groupby(["op", "kind", "variant", "footprint_bytes"]).gops.median().reset_index()
    rows = []
    default = L[(L["sem"] == "default") & (L.variant.isin(["identity", "cas_fail", "reverse_chain", "host_native_atomic_0"])) & (L["pass"].isin(["0", "1", "1_near", "1_far"]))]
    for _, r in default.iterrows():
        tgt = r.target
        tk = {"red+ld": "red"}.get(r.kind, r.kind)
        if tgt == "hbm":
            tgt = f"hbm_{r['pass'].split('_')[1]}"
        q = k2b[(k2b.target == r.target.replace("hbm", "hbm")) & (k2b.op == r.op) & (k2b.kind == tk)]
        if r.target.startswith("l2") or r.target == "hbm":
            var, fp = ("local", 16 << 20) if r.target == "l2_near" else ("remote", 16 << 20) if r.target == "l2_far" else ("mixed", 256 << 20)
            g = k3s[(k3s.op == r.op) & (k3s.kind == tk) & (k3s.variant == var) & (k3s.footprint_bytes == fp)]
            gops = float(g.gops.iloc[0]) if len(g) else np.nan
        else:
            gops = np.nan
        opb = 8 if r.op in ("add.u64", "cas.b64", "add.f64") else 4
        rows.append([tgt, r.cs, r.op, r.kind, r.variant, r.lat_cycles, r.ld_ref_cycles,
                     float(q.ops_per_cycle.iloc[0]) if len(q) else np.nan, float(q.ops_per_cycle.iloc[0]) * opb if len(q) else np.nan,
                     int(q.ilp.iloc[0]) if len(q) else -1, gops, gops * opb / 1e3 if gops == gops else np.nan])
    cols = ["target", "cs", "op", "kind", "variant", "latency_cycles", "ld_ref_cycles", "sm_ops_per_cycle", "sm_bpc", "sm_best_ilp", "gpu_gops",
            "gpu_tbps_payload"]
    append_rows(f"{RES}/12__atomics__summary.csv", MANDATORY + cols, [mand[:6] + [run_id, -1] + r for r in rows])
    append_rows(reg, ["run_id", "started_iso", "finished_iso", "status", "gpu_uuid", "cmdline"],
                [[run_id, "", datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "complete", "", "src/analyze.py"]])
    print(f"analysis {run_id}: {len(L)} latency rows, {len(pp)} pair rows, {len(rows)} summary rows")


if __name__ == "__main__":
    main()
