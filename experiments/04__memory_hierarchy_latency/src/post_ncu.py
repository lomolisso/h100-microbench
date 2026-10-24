"""Turn the ncu export of `bin/lat_footprint ncu` into result rows with the mandatory columns.

The binary writes bin/ncu_manifest.csv: one row per profiled k2_ncu launch, in launch order, with the mandatory
columns (run_id, NVML clocks sampled after the launch) and the configuration. Profiled rows are joined by launch order.
  --ncu-csv bin/ncu_k2.csv  (ncu -i ... --csv --page raw --print-units base)
  --metrics a.sum,b.pct,...  the metrics passed to ncu; only these (all their rollups .sum/.avg/.min/.max) plus ID and
                             Kernel Name are kept, so the header does not depend on what else the raw page exports
  → results/04__latency__ncu_k2.csv: mandatory + configuration + the kept ncu columns (units row dropped). Append only.
"""
import argparse
import csv

from common04 import MANDATORY, append_rows

CONFIG = ["load_variant", "footprint_bytes", "hops", "cycles_per_hop_under_ncu", "smid", "gpc", "retries"]

ap = argparse.ArgumentParser()
ap.add_argument("--ncu-csv", required=True)
ap.add_argument("--metrics", required=True)
ap.add_argument("--quick", action="store_true")
a = ap.parse_args()
with open("bin/ncu_manifest.csv", newline="") as f:
    man = list(csv.DictReader(f))
with open(a.ncu_csv, newline="") as f:
    rows = list(csv.reader([l for l in f if l.startswith('"')]))  # drop ==PROF== messages
header, data = rows[0], [r for r in rows[1:] if r and r[0].strip().isdigit()]  # numeric ID: drops the units row
bases = {m.split(".")[0] for m in a.metrics.split(",")}
keep = [i for i, c in enumerate(header) if c in ("ID", "Kernel Name") or c.split(".")[0] in bases]
keep.sort(key=lambda i: (header[i] not in ("ID", "Kernel Name"), header[i]))
header, data = [header[i] for i in keep], [[r[i] for i in keep] for r in data]
if len(data) != len(man):
    print(f"WARNING: {len(data)} profiled launches, manifest has {len(man)}; joining the first {min(len(data), len(man))}")
out = [[m[k] for k in MANDATORY + CONFIG] + r for m, r in zip(man, data)]
append_rows("results/04__latency__ncu_k2.csv", MANDATORY + CONFIG + header, out)
show = [c for c in header if "hit_rate" in c or c.startswith("dram__bytes") or "ltcfabric" in c or c == "launch__shared_mem_config_size"]
for m, r in zip(man, data):
    print(f"{m['load_variant']} F {int(m['footprint_bytes']):>10}: " + ", ".join(f"{c}={r[header.index(c)]}" for c in show))
print(f"ncu: {len(out)} rows")
