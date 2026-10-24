"""Analysis of experiment 08 shared by src/summarize.py and the notebook (paths relative to the experiment directory).

  gpc_map()                    smid → GPC from the latest run of 03's map
  load_k5(rid)                 K5 rows with first-read and repeat (samples 1-63) medians; per-sample arrays on request
  sm_signature(k5, tp)         per-SM partition, sub-node, near/far/copy latencies (timer pair tp subtracted):
     partition = the home label (0: SM 0's partition, 1: SM 2's) whose lines the SM reads fastest on first read;
     sub-node: for each partition, the matrix X[SM, line] of first-read medians of the lines homed there, read by the
     SMs of that partition, is double-centred; the signs of its first singular vectors split SMs and lines into two
     groups each. An SM's "same sub-node" lines are the line group its SM group reads faster.
     Sub-node ids: 2·partition + (0 for the SM group that contains the partition's smallest smid, else 1).
  line_groups(...)             the same split for any (SM set, line set, value) matrix (used for far and copy reads)
  line_subnode(k5, sig, groups) home sub-node of every K5 line (the sub-node of its home partition that reads it fastest)
  quad_table(k5, sig, groups, tp) first reads labelled same/other partition × same/other sub-node position (sub-nodes
                               paired across partitions by the fastest far reads)
  k2_bits(hb)                  per bit: share of (base, base ^ 2^b) pairs whose home differs; linear coefficients
  k2_linearity(hb, coef)       agreement of pair / random-mask home changes with the XOR of the single-bit coefficients
"""
import numpy as np
import pandas as pd

from common08 import read_run

MAP03 = "../03__sm_gpc_topology/results/03__sm_gpc_topology__map.csv"


def gpc_map():
    g = pd.read_csv(MAP03)
    g = g[g.run_id == g.run_id.max()]
    return g.set_index("smid").gpc.astype(int)


def load_k5(rid, samples=False):
    d = read_run("08__l2__sm_signature_samples.csv", rid)
    seq = np.array([np.array(s.split(";"), dtype=np.int32) for s in d.cycles_seq])
    d["repeat_median"] = np.median(seq[:, 1:], axis=1)
    d = d.drop(columns="cycles_seq")
    return (d, seq) if samples else d


def line_groups(d, sms, lines, value):
    """Rank-1 split of the matrix value[SM, line] (medians over reps): returns (sm_group Series, line_group Series, share
    of the centred variance explained by the first component)."""
    x = d[d.probe_smid.isin(sms) & d.line_offset.isin(lines)].groupby(["probe_smid", "line_offset"])[value].median().unstack()
    x = x.dropna(axis=1)
    c = x.values - x.values.mean(axis=1, keepdims=True) - x.values.mean(axis=0, keepdims=True) + x.values.mean()
    u, s, vt = np.linalg.svd(c, full_matrices=False)
    sg = pd.Series((u[:, 0] > 0).astype(int), index=x.index)
    lg = pd.Series((vt[0] > 0).astype(int), index=x.columns)
    if sg.loc[min(sg.index)] == 1:  # group 0 = the group of the smallest smid
        sg, lg = 1 - sg, 1 - lg
    return sg, lg, float(s[0] ** 2 / (s ** 2).sum())


def sm_signature(k5, tp):
    d = k5.copy()
    d["lat_first"] = d.first_cycles - tp
    d["lat_repeat"] = d.repeat_median - tp
    gpc = gpc_map()
    med = d.groupby(["probe_smid", "line_home"]).first_cycles.median().unstack()
    part = (med[1] < med[0]).astype(int)  # faster on home-1 lines → partition 1
    rows, groups = [], {}
    for p in (0, 1):
        sms = part.index[part == p]
        near_lines = d.loc[d.line_home == p, "line_offset"].unique()
        far_lines = d.loc[d.line_home != p, "line_offset"].unique()
        sg, lg, ev = line_groups(d, sms, near_lines, "lat_first")
        fsg, flg, fev = line_groups(d, sms, far_lines, "lat_first")
        csg, clg, cev = line_groups(d, sms, far_lines, "lat_repeat")
        groups[p] = dict(sm=sg, line=lg, ev=ev, far_sm=fsg, far_line=flg, far_ev=fev, copy_sm=csg, copy_line=clg, copy_ev=cev)
        for s in sms:
            x = d[d.probe_smid == s]
            near, far = x[x.line_home == p], x[x.line_home != p]
            nf = near.assign(g=near.line_offset.map(lg))
            # which line group this SM's group reads faster
            gm = nf.groupby("g").lat_first.median()
            same = int(gm.idxmin())
            ff = far.assign(g=far.line_offset.map(flg)).groupby("g").lat_first.median()
            rows.append(dict(
                smid=s, gpc=int(gpc[s]), partition=p, subnode=2 * p + int(sg[s]),
                near_median=near.lat_first.median(), far_median=far.lat_first.median(),
                near_same_subnode_median=gm[same], near_other_subnode_median=gm[1 - same],
                subnode_offset=gm[1 - same] - gm[same],
                far_lo_median=ff.min(), far_hi_median=ff.max(),
                copy_median=far.lat_repeat.median(), home_repeat_median=near.lat_repeat.median(),
                n_first_near=len(near), n_first_far=len(far)))
    return pd.DataFrame(rows).sort_values("smid").reset_index(drop=True), groups


def k2_bits(hb):
    s = hb[hb.set == "single"]
    frac = s.groupby("bit").flipped_class_changed.mean()
    coef = {int(b): int(f > 0.5) for b, f in frac.items() if f >= 0.9 or f <= 0.1}
    return frac, coef


def k2_linearity(hb, coef):
    """Pairs and random masks restricted to bits 7..20: observed home change vs parity of the participating bits."""
    out = {}
    for set_ in ("pair", "mask"):
        x = hb[hb.set == set_].copy()
        m = x.mask_hex.map(lambda h: int(h, 16))
        M = sum(1 << b for b, c in coef.items() if c and b <= 20)
        x["pred"] = (m & M).map(lambda v: bin(v).count("1") & 1)
        out[set_] = (float((x.pred == x.flipped_class_changed).mean()), len(x))
    return out


def line_subnode(k5, sig, groups):
    """Home sub-node of every K5 line: the sub-node whose SMs read it fastest among the SMs of its home partition."""
    s = sig.set_index("smid")
    out = {}
    for p in (0, 1):
        sg, lg = groups[p]["sm"], groups[p]["line"]
        x = k5[(k5.line_home == p) & k5.probe_smid.isin(sg.index)]
        t = x.assign(g=x.probe_smid.map(sg), l=x.line_offset.map(lg)).groupby(["g", "l"]).first_cycles.median().unstack()
        sub_of_g = {g: int(s.subnode[sg.index[sg == g][0]]) for g in (0, 1)}
        for g in (0, 1):
            fast = int(t.loc[g].idxmin())
            for o in lg.index[lg == fast]:
                out[o] = sub_of_g[g]
    return pd.Series(out, name="line_subnode")


def quad_table(k5, sig, groups, tp):
    """First-read latency by (same / other partition) × (same / other sub-node position), where sub-nodes are paired
    across the partitions by the far-read pattern: the far sub-node a sub-node reads fastest is its partner."""
    s = sig.set_index("smid")
    ls = line_subnode(k5, sig, groups)
    d = k5.assign(c=k5.first_cycles - tp, sp=k5.probe_smid.map(s.partition), sn=k5.probe_smid.map(s.subnode),
                  ln=k5.line_offset.map(ls)).dropna(subset=["ln"])
    far = d[d.line_home != d.sp]
    partner = far.groupby(["sn", "ln"]).c.median().unstack().idxmin(axis=1).astype(int).to_dict()
    d["partition"] = np.where(d.line_home == d.sp, "same partition", "other partition")
    d["position"] = [("same" if (ln == sn or partner[sn] == ln) else "other") + " sub-node position" for sn, ln in zip(d.sn, d.ln)]
    return d, partner
