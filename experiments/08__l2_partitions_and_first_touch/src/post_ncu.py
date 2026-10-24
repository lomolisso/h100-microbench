"""Turn one ncu export of experiment 08 into result rows with the mandatory columns.

The profiled binary appends one manifest row per profiled k_ncu* launch, in launch order, to
results/08__l2__ncu_manifest.csv (mode, ncu_order, kernel, probe/toucher SM, lines, line offset, what, first/repeat
median cycles). Under application replay the binary runs once per ncu pass, each run with its own run_id; the launch
sequence is identical in every pass, so the manifest of the first run started after --since is joined by order.
  --tag lines|fabric|capacity   output results/08__l2__ncu_<tag>.csv
  --mode ncu_lines|ncu_fabric|ncu   binary mode (manifest mode: ncu_lines, ncu_fabric, ncu_capacity)
  --since ISO-UTC               start time of the ncu invocation
  --ncu-csv bin/ncu_<tag>.csv   ncu -i ... --csv --page raw --print-units base --print-metric-instances values
  --metrics a.sum,b.pct,...     metrics passed to ncu; only these (all rollups) plus ID and Kernel Name are kept.
Per-instance values (lts__ metrics: one value per L2 slice) stay in their cells as exported (';'-separated).
"""
import argparse
import csv

import pandas as pd

from common08 import MANDATORY, append_rows

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--mode", required=True)
ap.add_argument("--since", required=True)
ap.add_argument("--ncu-csv", required=True)
ap.add_argument("--metrics", required=True)
a = ap.parse_args()
mmode = "ncu_capacity" if a.mode == "ncu" else a.mode

man = pd.read_csv("results/08__l2__ncu_manifest.csv", dtype=str, keep_default_na=False)
man = man[(man["mode"] == mmode) & (man.run_id >= a.since)]
rids = sorted(man.run_id.unique())
if not rids:
    raise SystemExit(f"no manifest rows of {mmode} since {a.since}")
first = man[man.run_id == rids[0]].sort_values("ncu_order", key=lambda s: s.astype(int))
print(f"ncu {a.tag}: {len(rids)} binary runs (passes) since {a.since}; manifest of {rids[0]}: {len(first)} launches")

with open(a.ncu_csv, newline="") as f:
    rows = list(csv.reader([l for l in f if l.startswith('"')]))  # drop ==PROF== messages
header, data = rows[0], [r for r in rows[1:] if r and r[0].strip().isdigit()]  # numeric ID: drops the units row
bases = {m.split(".")[0] for m in a.metrics.split(",")}
keep = [i for i, c in enumerate(header) if c in ("ID", "Kernel Name") or c.split(".")[0] in bases]
keep.sort(key=lambda i: (header[i] not in ("ID", "Kernel Name"), header[i]))
header, data = [header[i] for i in keep], [[r[i] for i in keep] for r in data]
if len(data) != len(first):
    print(f"WARNING: {len(data)} profiled launches, manifest has {len(first)}; joining the first {min(len(data), len(first))}")
CONFIG = ["mode", "ncu_order", "kernel", "probe_smid", "first_touch_smid", "lines", "line_offset", "what", "first_median", "repeat_median"]
out = [[m[k] for k in MANDATORY + CONFIG] + [str(len(rids))] + r for (_, m), r in zip(first.iterrows(), data)]
append_rows(f"results/08__l2__ncu_{a.tag}.csv", MANDATORY + CONFIG + ["binary_runs"] + header, out)
show = [c for c in header if c.endswith(".sum") or c.endswith(".pct") or c.startswith("profiler__")]
for (_, m), r in list(zip(first.iterrows(), data))[:6]:
    print(f"  {m['kernel']} P {m['probe_smid']} {m['what']} off {m['line_offset']}: "
          + ", ".join(f"{c}={r[header.index(c)][:60]}" for c in show))
print(f"ncu {a.tag}: {len(out)} rows → results/08__l2__ncu_{a.tag}.csv")
