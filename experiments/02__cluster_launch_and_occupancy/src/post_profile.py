"""Turn ncu / nsys exports of `bin/cluster_launch {ncu|nsys}` into result rows with the mandatory columns.

The binary writes a manifest (bin/<tool>_manifest.csv): one row per sleep_kernel launch in launch order, with the
mandatory columns (run_id, NVML clocks sampled after the launch) and the configuration (probe, cs, policy,
grid_blocks, smem_bytes, sleep_cycles). Profiled rows are joined with it by launch order.

  ncu : --ncu-csv bin/ncu_launch.csv  (ncu -i ... --csv --page raw --print-units base --print-metric-instances values)
        → results/02__cluster_launch__ncu_launch.csv: mandatory + configuration + every raw ncu column (units row dropped)
  nsys: --nsys-prefix bin/nsys        (nsys stats --format csv --output bin/nsys, one file per report)
        → results/02__cluster_launch__nsys_<report>.csv. Per-launch reports (cuda_gpu_trace: sleep_kernel rows;
          cuda_api_trace: cudaLaunchKernelEx rows) get the configuration of their launch; summary reports
          (cuda_gpu_kern_sum, cuda_api_sum) get the run's mandatory columns with launch_idx = -1.
Appends only (flock, header written to an empty file, checked otherwise), fsync at the end.
"""
import argparse
import csv
import fcntl
import io
import os

MANDATORY = ["gpu_name", "sm_count", "sm_clock_mhz", "mem_clock_mhz", "cuda_version", "driver_version", "run_id", "launch_idx"]
CONFIG = ["probe", "cs", "policy", "grid_blocks", "smem_bytes", "sleep_cycles"]
OUT = "results/02__cluster_launch__"


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
            existing = next(csv.reader(f))
            if existing != header:
                raise SystemExit(f"header mismatch in {path}")
            f.seek(0, os.SEEK_END)
        w.writerows(rows)
        f.write(buf.getvalue())
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def read_manifest(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_csv_lines(path, keep):
    with open(path, newline="") as f:
        lines = [l for l in f if keep(l)]
    return list(csv.reader(lines))


def ncu(a):
    man = read_manifest("bin/ncu_manifest.csv")
    rows = read_csv_lines(a.ncu_csv, lambda l: l.startswith('"'))  # drop ==PROF== messages
    header, data = rows[0], [r for r in rows[1:] if r and r[0].strip().isdigit()]  # numeric ID: drops units row
    if len(data) != len(man):
        print(f"WARNING: {len(data)} profiled launches, manifest has {len(man)}; joining the first {min(len(data), len(man))}")
    out = [[m[k] for k in MANDATORY + CONFIG] + r for m, r in zip(man, data)]
    append_rows(OUT + "ncu_launch.csv", MANDATORY + CONFIG + header, out)
    print(f"ncu: {len(out)} rows")


def nsys(a):
    man = read_manifest("bin/nsys_manifest.csv")
    for report, match in [("cuda_gpu_trace", "sleep_kernel"), ("cuda_api_trace", "cudaLaunchKernelEx"),
                          ("cuda_gpu_kern_sum", None), ("cuda_api_sum", None)]:
        path = f"{a.nsys_prefix}_{report}.csv"
        if not os.path.exists(path):
            print(f"WARNING: {path} missing")
            continue
        rows = read_csv_lines(path, lambda l: l.strip() != "")
        header, data = rows[0], rows[1:]
        if match is None:
            m0 = man[0]
            out = [[m0[k] for k in MANDATORY[:-1]] + ["-1"] + [""] * len(CONFIG) + r for r in data]
        else:
            name_col = header.index("Name")
            data = [r for r in data if match in r[name_col]]
            if len(data) != len(man):
                print(f"WARNING {report}: {len(data)} matching rows, manifest has {len(man)}")
            out = [[m[k] for k in MANDATORY + CONFIG] + r for m, r in zip(man, data)]
        append_rows(OUT + f"nsys_{report}.csv", MANDATORY + CONFIG + header, out)
        print(f"nsys {report}: {len(out)} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncu-csv")
    ap.add_argument("--nsys-prefix")
    a = ap.parse_args()
    if a.ncu_csv:
        ncu(a)
    if a.nsys_prefix:
        nsys(a)


if __name__ == "__main__":
    main()
