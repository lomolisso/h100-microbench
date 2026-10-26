"""Turn the ncu export of `bin/dsmem_lat ncu` into result rows with the mandatory columns.

The binary writes bin/ncu_manifest_k1.csv: one row per k1_chase launch, in launch order, with the mandatory columns
(run_id, NVML clocks sampled after the launch), CS, number of clusters, chase (1) or all-exit baseline (0) and the expected
number of DSMEM load instructions. Profiled rows (bin/ncu_k1_raw.csv, ncu -i … --csv --page raw --print-units base) are
joined by launch order.
  → results/09__dsmem_latency__ncu_k1.csv: mandatory + configuration + ID, Kernel Name and every metric column whose base
    name starts with sm__, l1tex__, lts__, launch__, gpu__ or profiler__ (units row dropped). Append only.
"""
import csv

from common09 import MANDATORY, append_rows

CONFIG = ["kernel", "cs", "nclusters", "chase", "expected_dshared_ld"]
PREFIX = ("sm__", "l1tex__", "lts__", "launch__", "gpu__", "profiler__")

with open("bin/ncu_manifest_k1.csv", newline="") as f:
    man = list(csv.DictReader(f))
with open("bin/ncu_k1_raw.csv", newline="") as f:
    rows = list(csv.reader([l for l in f if l.startswith('"')]))  # drop ==PROF== messages
header, data = rows[0], [r for r in rows[1:] if r and r[0].strip().isdigit()]  # numeric ID: drops the units row
keep = [i for i, c in enumerate(header) if c in ("ID", "Kernel Name") or c.startswith(PREFIX)]
keep.sort(key=lambda i: (header[i] not in ("ID", "Kernel Name"), header[i]))
header, data = [header[i] for i in keep], [[r[i] for i in keep] for r in data]
if len(data) != len(man):
    raise SystemExit(f"{len(data)} profiled launches, manifest has {len(man)}")
out = [[m[k] for k in MANDATORY + CONFIG] + r for m, r in zip(man, data)]
append_rows("results/09__dsmem_latency__ncu_k1.csv", MANDATORY + CONFIG + header, out)
show = [c for c in header if "dshared" in c or c.startswith("lts__")]
for m, r in zip(man, data):
    print(f"cs {m['cs']} chase {m['chase']} expected dshared ld {m['expected_dshared_ld']}: " +
          ", ".join(f"{c}={r[header.index(c)]}" for c in show))
print(f"ncu: {len(out)} rows")
