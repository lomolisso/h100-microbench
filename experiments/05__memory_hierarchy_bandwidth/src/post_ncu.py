"""Turn an ncu export of a `bin/bw_* ncu` invocation into result rows with the mandatory columns.

The binary writes a manifest (bin/ncu_manifest_<variant>.csv): one row per profiled launch, in launch order, with the
mandatory columns (run_id, NVML clocks sampled after the launch) and the configuration. Profiled rows are joined by
launch order (application replay: ncu reports each profiled kernel once, in launch order).
  --ncu-csv bin/ncu_<variant>.csv   (ncu -i ... --csv --page raw --print-units base)
  --manifest bin/ncu_manifest_<variant>.csv
  --metrics a.sum,b.pct,...         the metrics passed to ncu; only these (all their rollups) plus ID and Kernel Name are
                                    kept, so the header does not depend on what else the raw page exports
  --out results/05__bandwidth__ncu_<variant>.csv   (append only)
"""
import argparse
import csv

from common05 import append_rows

ap = argparse.ArgumentParser()
ap.add_argument("--ncu-csv", required=True)
ap.add_argument("--manifest", required=True)
ap.add_argument("--metrics", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
with open(a.manifest, newline="") as f:
    rd = csv.reader(f)
    man_header = next(rd)
    man = list(rd)
with open(a.ncu_csv, newline="") as f:
    rows = list(csv.reader([l for l in f if l.startswith('"')]))  # drop ==PROF== messages
header, data = rows[0], [r for r in rows[1:] if r and r[0].strip().isdigit()]  # numeric ID: drops the units row
bases = {m.split(".")[0] for m in a.metrics.split(",")}
keep = [i for i, c in enumerate(header) if c in ("ID", "Kernel Name") or c.split(".")[0] in bases]
keep.sort(key=lambda i: (header[i] not in ("ID", "Kernel Name"), header[i]))
header, data = [header[i] for i in keep], [[r[i] for i in keep] for r in data]
if len(data) != len(man):
    print(f"WARNING: {len(data)} profiled launches, manifest has {len(man)}; joining the first {min(len(data), len(man))}")
out = [m + r for m, r in zip(man, data)]
append_rows(a.out, man_header + header, out)
show = [c for c in header if c.endswith(("per_second", "pct_of_peak_sustained_elapsed", "ltcfabric.sum", "hit_rate.pct", "_st.sum", "_ld.sum"))]
for m, r in zip(man, data):
    print(" ".join(m[8:14]) + ": " + ", ".join(f"{c}={r[header.index(c)]}" for c in show))
print(f"ncu: {len(out)} rows → {a.out}")
