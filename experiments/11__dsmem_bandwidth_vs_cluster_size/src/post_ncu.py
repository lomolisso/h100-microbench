"""Turn the ncu export of `bin/dsmem_bw ncu` into result rows with the mandatory columns.

The binary writes bin/ncu_manifest.csv: one row per k_stream launch, in launch order, with the mandatory columns (run_id,
NVML clocks sampled after the launch), the configuration and the bytes its issuers moved. Profiled rows (bin/ncu_raw.csv,
ncu -i … --csv --page raw --print-units base) are joined by launch order.
  → results/11__dsmem_bw__ncu_v2.csv: mandatory + configuration + ID, Kernel Name and every metric column whose base name
    starts with sm__, l1tex__, lts__, launch__, gpu__ or profiler__ (units row dropped). Append only.
"""
import csv

from common11 import MANDATORY, append_rows

CONFIG = ["kernel", "pattern", "pattern_k", "cs", "launch_mode", "nclusters", "threads", "ilp", "chunk_bytes", "payload_bytes", "issuer_bytes"]
PREFIX = ("sm__", "l1tex__", "lts__", "launch__", "gpu__", "profiler__")

with open("bin/ncu_manifest.csv", newline="") as f:
    man = list(csv.DictReader(f))
with open("bin/ncu_raw.csv", newline="") as f:
    rows = list(csv.reader([l for l in f if l.startswith('"')]))  # drop ==PROF== messages
header, data = rows[0], [r for r in rows[1:] if r and r[0].strip().isdigit()]  # numeric ID: drops the units row
keep = [i for i, c in enumerate(header) if c in ("ID", "Kernel Name") or c.startswith(PREFIX)]
keep.sort(key=lambda i: (header[i] not in ("ID", "Kernel Name"), header[i]))
header, data = [header[i] for i in keep], [[r[i] for i in keep] for r in data]
if len(data) != len(man):
    raise SystemExit(f"{len(data)} profiled launches, manifest has {len(man)}")
out = [[m[k] for k in MANDATORY + CONFIG] + r for m, r in zip(man, data)]
append_rows("results/11__dsmem_bw__ncu_v2.csv", MANDATORY + CONFIG + header, out)
show = [c for c in header if c.startswith(("l1tex__m_l1tex2xbar_write_bytes_mem_dshared.sum", "l1tex__m_xbar2l1tex_read_bytes_mem_dshared.sum"))]
for m, r in zip(man, data):
    print(f"{m['kernel']:13s} {m['pattern']:16s} k{m['pattern_k']} {m['launch_mode']:6s} issuer {m['issuer_bytes']:>9s} B: " +
          ", ".join(f"{c.split('_mem_')[0][-12:]}={r[header.index(c)]}" for c in show))
print(f"ncu: {len(out)} rows")
