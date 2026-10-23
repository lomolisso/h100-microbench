"""K4: turn one ncu raw-page CSV (bin/k4_<mode>.csv) into result rows.

Inputs:  the ncu CSV of `bin/timers k4 --label k4_<mode>` (10 profiled launches: 5 x k3_loop, then 5 x k4_spin),
         and the clock-log rows that this k4 run wrote (label k4_<mode>, phase after). These provide run_id and
         the mandatory columns. sm_clock_mhz/mem_clock_mhz are the NVML values sampled after profiling.
Outputs (append-only, flock, header written only to an empty file, checked otherwise):
  results/01__timers__ncu_clock_domains_<mode>.csv   mandatory columns + every raw ncu column (units row dropped)
  results/01__timers__clock_domains.csv              mandatory columns + clock_control, kernel, ncu_id, sm_hz,
                                                     lts_hz, dram_hz, gpc_hz, lts_over_sm, gpu_time_ns
launch_idx = index of the profiled launch within its kernel (0..4).
"""
import argparse
import csv
import fcntl
import io
import os

MANDATORY = ["gpu_name", "sm_count", "sm_clock_mhz", "mem_clock_mhz", "cuda_version", "driver_version", "run_id", "launch_idx"]
EXP = "01__timers_and_harness"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["none", "base"])
    ap.add_argument("--ncu-csv", required=True)
    a = ap.parse_args()

    with open(f"results/{EXP}__clock_log.csv", newline="") as f:
        log = [r for r in csv.DictReader(f) if r["label"] == f"k4_{a.mode}" and r["phase"] == "after"]
    if not log:
        raise SystemExit(f"no clock-log row for k4_{a.mode}")
    meta = max(log, key=lambda r: r["run_id"])  # this run (ISO timestamps sort chronologically)

    with open(a.ncu_csv, newline="") as f:
        lines = [l for l in f if l.startswith('"')]  # drop ncu's own ==PROF== messages, if any
    rows = list(csv.reader(lines))
    header, data = rows[0], [r for r in rows[1:] if r and r[0].strip().isdigit()]  # numeric ID: drops units row
    col = {name: i for i, name in enumerate(header)}
    if len(data) != 10:
        print(f"WARNING: expected 10 profiled launches, found {len(data)}")

    raw_rows, dom_rows, seen = [], [], {}
    for r in data:
        kernel = "k4_spin" if "k4_spin" in r[col["Kernel Name"]] else "k3_loop"
        idx = seen.get(kernel, 0)
        seen[kernel] = idx + 1
        mand = [meta[k] for k in MANDATORY[:-1]] + [str(idx)]
        raw_rows.append(mand + r)
        g = lambda m: float(r[col[m]].replace(",", ""))
        sm, lts = g("sm__cycles_elapsed.avg.per_second"), g("lts__cycles_elapsed.avg.per_second")
        dom_rows.append(mand + [a.mode, kernel, r[col["ID"]], f"{sm:.0f}", f"{lts:.0f}",
                                f"{g('dram__cycles_elapsed.avg.per_second'):.0f}",
                                f"{g('gpc__cycles_elapsed.avg.per_second'):.0f}", f"{lts / sm:.5f}",
                                f"{g('gpu__time_duration.sum'):.0f}"])
        print(f"k4 {a.mode:4s} {kernel:8s} #{idx}: sm {sm / 1e6:8.2f} MHz  lts {lts / 1e6:8.2f} MHz  "
              f"lts/sm {lts / sm:.4f}  dram {g('dram__cycles_elapsed.avg.per_second') / 1e6:8.2f} MHz")

    append_rows(f"results/01__timers__ncu_clock_domains_{a.mode}.csv", MANDATORY + header, raw_rows)
    append_rows("results/01__timers__clock_domains.csv",
                MANDATORY + ["clock_control", "kernel", "ncu_id", "sm_hz", "lts_hz", "dram_hz", "gpc_hz",
                             "lts_over_sm", "gpu_time_ns"], dom_rows)


if __name__ == "__main__":
    main()
