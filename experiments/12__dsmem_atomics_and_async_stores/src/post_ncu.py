"""Join one ncu export (bin/ncu_raw_<tag>.csv) with the binary's manifest (bin/ncu_manifest.csv) by launch order.

Usage: python3 src/post_ncu.py <tag> <raw csv> <manifest csv>
The manifest has one row per launch that matters, in launch order, with the mandatory columns and the configuration.
Under application replay the binary runs once per pass and appends its manifest every time: only the rows of the first
run_id are used. Profiled launches and manifest rows are matched as follows:
  tag l2     every k1_glob launch is profiled: one manifest row per launch.
  tag tputN  4 profiled k_tput launches (baseline recorded, profiled warm-up × 2, profiled recorded): rows 0 and 3 ↔ the
             manifest's baseline and profiled rows.
  tag dsN    2 profiled launches ↔ baseline and profiled rows.
  → results/12__atomics__ncu.csv: tag, the manifest's configuration columns (as one `config` string), ID, Kernel Name and
    every metric column (base names starting with sm__, smsp__, l1tex__, lts__, dram__, launch__, gpu__, profiler__).
    Long format: one row per (launch, metric) so that the different binaries share one file. Append only.
"""
import csv
import sys

from common12 import MANDATORY, append_rows

tag, raw, manifest = sys.argv[1:4]
PREFIX = ("sm__", "smsp__", "l1tex__", "lts__", "dram__", "launch__", "gpu__", "profiler__")
with open(manifest, newline="") as f:
    man = list(csv.DictReader(f))
first = man[0]["run_id"]
man = [m for m in man if m["run_id"] == first]
with open(raw, newline="") as f:
    rows = list(csv.reader([l for l in f if l.startswith('"')]))
header, data = rows[0], [r for r in rows[1:] if r and r[0].strip().isdigit()]
if tag.startswith("tput"):
    data = [data[0], data[3]] if len(data) == 4 else data
if len(data) != len(man):
    raise SystemExit(f"{tag}: {len(data)} profiled launches, manifest has {len(man)}")
cfg_cols = [k for k in man[0] if k not in MANDATORY]
keep = [i for i, c in enumerate(header) if c.startswith(PREFIX)]
out = []
for m, r in zip(man, data):
    config = ";".join(f"{k}={m[k]}" for k in cfg_cols)
    for i in keep:
        out.append([m[k] for k in MANDATORY] + [tag, config, r[header.index("ID")], r[header.index("Kernel Name")], header[i], r[i]])
append_rows("results/12__atomics__ncu.csv", MANDATORY + ["tag", "config", "id", "kernel_name", "metric", "value"], out)
print(f"ncu {tag}: {len(data)} launches, {len(out)} metric rows")
