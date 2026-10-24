"""Choose the variant-B toucher of every K3 probe SM from the latest complete toucher x probe matrix.

Input: results/04__latency__fine_hist.csv (latest complete `lat_fine matrix` run), 8 MiB, pass 0.
Pooled histogram → 2-means threshold between the low and the high latency class. For every probe P, frac_low(T, P) =
fraction of P's accesses in the low class when the lines were first touched by T. B toucher of P = the T ≠ P with
the lowest frac_low (the toucher that makes P's accesses "farthest"). If frac_low varies by less than 0.05 over all
T for a probe (no first-touch effect), the fallback is the first SM of GPC (g + 4) mod 8 (quick mode: the other probe).
Prints `P:T,P:T,...` on stdout (consumed by src/run.sh) and the frac_low table on stderr.
--k2-sms: instead print the 4 K2 probe SMs: partition groups = sets of probes whose accesses are ≥ 90 % in the low class
for each other's touchers; the 2 lowest-GPC probes of each of the two largest groups (fallback: GPCs 1, 3, 5, 7).
"""
import argparse
import sys

import numpy as np
import pandas as pd

from common04 import kmeans1d, latest_run, read_run

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--k2-sms", action="store_true")
a = ap.parse_args()

rid = latest_run("lat_fine", "matrix", a.quick)
h = read_run("04__latency__fine_hist.csv", rid)
h = h[(h.footprint_bytes == 8 << 20) & (h["pass"] == 0)]
pooled = h.groupby("cycles")["count"].sum()
cen, lab, _, _ = kmeans1d(pooled.index.values, pooled.values, 2)
thr = pooled.index.values[lab == 0].max()
h = h.assign(low=h.cycles <= thr)
frac = (h.assign(lc=h["count"] * h.low).groupby(["probe_smid", "first_touch_smid"]).apply(lambda g: g.lc.sum() / g["count"].sum())
        .unstack())
print(f"matrix run {rid}: 2-means centres {cen.round(1)}, threshold {thr} cycles; frac_low (rows probe, cols toucher):",
      file=sys.stderr)
print(frac.round(3).to_string(), file=sys.stderr)

gmap = pd.read_csv("../03__sm_gpc_topology/results/03__sm_gpc_topology__map.csv")
gmap = gmap[gmap.run_id == gmap.run_id.max()].set_index("smid").gpc
first = {g: int(s.index.min()) for g, s in gmap.groupby(gmap)}
probes = list(frac.index)
if a.k2_sms:
    groups = sorted({tuple(sorted(int(t) for t in frac.columns if frac.loc[p, t] >= 0.9)) for p in probes}, key=lambda g: -len(g))
    groups = [sorted(g, key=lambda s: int(gmap[s])) for g in groups if len(g) >= 2]
    if len(groups) >= 2 and not set(groups[0]) & set(groups[1]):
        sms = groups[0][:2] + groups[1][:2]
        print(f"partition groups {groups}: K2 SMs {sms}", file=sys.stderr)
    else:
        sms = [first[g] for g in (1, 3, 5, 7)]
        print(f"no two disjoint partition groups ({groups}): K2 SMs {sms} (fallback)", file=sys.stderr)
    print(",".join(str(s) for s in sorted(sms, key=lambda s: int(gmap[s]))))
    sys.exit(0)
pairs = []
for p in probes:
    row = frac.loc[p]
    others = row.drop(p)
    if row.max() - row.min() >= 0.05:
        t = int(others.idxmin())
        why = f"min frac_low {others.min():.3f}"
    else:
        t = [q for q in probes if q != p][0] if a.quick else first[(int(gmap[p]) + 4) % 8]
        why = f"no first-touch effect (spread {row.max() - row.min():.3f}): fallback"
    print(f"probe {p}: toucher {t} ({why})", file=sys.stderr)
    pairs.append(f"{p}:{t}")
print(",".join(pairs))
