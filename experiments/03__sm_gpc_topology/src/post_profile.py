"""Turn the ncu export of `bin/gpc_probe ncu` into result rows with the mandatory columns.

The binary writes bin/ncu_manifest.csv: one row per sleep_kernel launch in launch order, with the mandatory columns
(run_id, NVML clocks sampled after the launch) and the configuration (cs, policy, smem_bytes, block_threads,
grid_blocks, waves, api_max_active_clusters, sleep_cycles). Profiled rows are joined with it by launch order.

  --ncu-csv bin/ncu_gpc.csv  (ncu -i ... --csv --page raw --print-units base --print-metric-instances values)
  → results/03__sm_gpc_topology__ncu.csv: mandatory + configuration + every raw ncu column (units row dropped).
Appends only (flock, header written to an empty file, checked otherwise), fsync at the end.
"""
import argparse
import csv
import fcntl
import io
import os

MANDATORY = ["gpu_name", "sm_count", "sm_clock_mhz", "mem_clock_mhz", "cuda_version", "driver_version", "run_id", "launch_idx"]
CONFIG = ["cs", "policy", "smem_bytes", "block_threads", "grid_blocks", "waves", "api_max_active_clusters", "sleep_cycles"]
OUT = "results/03__sm_gpc_topology__ncu.csv"


def append_rows(path, header, rows):
    """Append whole rows under an exclusive lock; header only if the file is empty; fsync at the end."""
    with open(path, "a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0, os.SEEK_END)
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        if f.tell() == 0:
            w.writerow(header)
        else:
            f.seek(0)
            if next(csv.reader(f)) != header:
                raise SystemExit(f"header mismatch in {path}")
            f.seek(0, os.SEEK_END)
        w.writerows(rows)
        f.write(buf.getvalue())
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncu-csv", required=True)
    a = ap.parse_args()
    with open("bin/ncu_manifest.csv", newline="") as f:
        man = list(csv.DictReader(f))
    with open(a.ncu_csv, newline="") as f:
        rows = list(csv.reader([l for l in f if l.startswith('"')]))  # drop ==PROF== messages
    header, data = rows[0], [r for r in rows[1:] if r and r[0].strip().isdigit()]  # numeric ID: drops the units row
    if len(data) != len(man):
        print(f"WARNING: {len(data)} profiled launches, manifest has {len(man)}; joining the first {min(len(data), len(man))}")
    out = [[m[k] for k in MANDATORY + CONFIG] + r for m, r in zip(man, data)]
    append_rows(OUT, MANDATORY + CONFIG + header, out)
    cols = [c for c in header if c.startswith("gpc__cgas_launched")]
    for m, r in zip(man, data):
        print(f"cs {m['cs']:>2} smem {m['smem_bytes']:>6}: " + ", ".join(f"{c.split('__')[1]}={r[header.index(c)]}" for c in cols))
    print(f"ncu: {len(out)} rows")


if __name__ == "__main__":
    main()
