"""Experiment 11 derived tables from the latest complete run of every step (no GPU). Run from the repository root:
    tools/run --no-gpu -- bash -c "cd experiments/11__dsmem_bandwidth_vs_cluster_size && python3 src/analyze.py"
(--quick: the latest --quick runs, outputs to --out DIR, never into results/; --out DIR alone: full runs, written to DIR
and not registered).

Fits (steps A, B, C, D, E, F; passes 1): per configuration (step, kernel, pattern, pattern_k, cs, policy, threads, ilp,
chunk_bytes, launch_mode, side, stream) and payload level, the median of cycles and of bytes_moved over every record
(CTAs × samples × launches); affine least-squares fit cycles = alpha + beta · bytes over the payload levels:
bpc_per_sm = 1 / beta, alpha_cycles = alpha, rmse_pct = 100 · rms((median − fit) / fit). bpc_at_max_payload = median of
bytes / cycles of the records at the largest payload (includes alpha). Full-chip configurations also get
tbps_gpu_wide = 1 / slope of the cudaEvent time over the launch's total bytes (launches.csv, same payload levels) and
tbps_from_bpc = bpc_per_sm × active SMs × measured SM clock.
Long stream (step S, full, 96 KiB × passes): per (kernel, cs) event-time slope over total bytes → TB/s, and the per-CTA
bytes / cycles median at the largest pass count.
P1 style: TB/s = total bytes / event time per launch (median) and the event-time slope over the passes.
Outputs (append, own run_id = time of this analysis, registered in the runs registry as `analyze`; source runs in
source_run_ids):
  11__dsmem_bw__fits.csv     MANDATORY + configuration + n_payloads, n_records, bpc_per_sm, alpha_cycles, rmse_pct,
                             bpc_at_max_payload, tbps_gpu_wide, tbps_from_bpc, active_sms, source_run_ids
  11__dsmem_bw__gpuwide.csv  MANDATORY + source (S / P1), kernel, cs, n_points, tbps_slope, alpha_ns, tbps_max_point,
                             bpc_cta_median, active_sms, source_run_ids
  11__dsmem_bw__summary.csv  MANDATORY + level, config, value, unit, n, source, source_run_ids (the numbers later
                             experiments consume)
"""
import datetime
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common11 import EXP, MANDATORY, append_rows, latest_runs

QUICK = "--quick" in sys.argv
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "results"
if QUICK and OUT == "results":
    raise SystemExit("--quick needs --out DIR")

STEPS = ["A", "S", "B", "C", "D", "E", "F", "P1", "K4"]
src = {}
for s in STEPS:
    r = latest_runs("dsmem_bw", s, quick=QUICK)
    if r:
        src[s] = r[-1]
k5 = latest_runs("dsmem_l2mix", "k5", quick=QUICK)
if k5:
    src["K5"] = k5[-1]
missing = [s for s in STEPS + ["K5"] if s not in src]
if missing:
    print(f"warning: no complete run of step(s) {missing}")
ids = sorted(set(src.values()))
ids_str = " ".join(ids)

samples = pd.read_csv("results/11__dsmem_bw__samples.csv")
samples = samples[samples.run_id.isin(ids)]
launches = pd.read_csv("results/11__dsmem_bw__launches.csv")
launches = launches[launches.run_id.isin(ids)]
clk = samples.sm_clock_mhz.median()
first = samples.iloc[0]
run_id = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")
pre = [first.gpu_name, first.sm_count, int(round(clk)), first.mem_clock_mhz, first.cuda_version, first.driver_version, run_id, -1]

KEY = ["step", "kernel", "pattern", "pattern_k", "cs", "policy", "threads", "ilp", "chunk_bytes", "launch_mode", "side", "stream"]
LKEY = ["step", "kernel", "pattern", "pattern_k", "cs", "policy", "threads", "ilp", "chunk_bytes", "launch_mode"]


def affine(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.ptp(x) == 0:
        return np.nan, np.nan, np.nan
    b, a = np.polyfit(x, y, 1)
    fit = a + b * x
    return a, b, 100 * np.sqrt(np.mean(((y - fit) / fit) ** 2))


fits = []
sp = samples[samples.passes == 1].copy()
sp["bpc"] = sp.bytes_moved / sp.cycles
lev = sp.groupby(KEY + ["payload_bytes"]).agg(cyc=("cycles", "median"), byt=("bytes_moved", "median"), n=("cycles", "size")).reset_index()
lmed = sp.groupby(KEY + ["payload_bytes"]).bpc.median()
lsel = launches[launches.passes == 1]
for key, d in lev.groupby(KEY):
    d = d.sort_values("payload_bytes")
    a, b, rm = affine(d.byt, d.cyc)
    k = dict(zip(KEY, key))
    bmax = lmed.loc[key + (d.payload_bytes.max(),)]
    tb, act, tbb = np.nan, np.nan, np.nan
    if k["launch_mode"] == "full":
        lk = lsel
        for c in LKEY:
            lk = lk[lk[c] == k[c]]
        if len(lk):
            la, lb, _ = affine(lk.total_bytes, lk.event_ns)
            tb = 1e-3 / lb if lb > 0 else np.nan
            act = lk.active_ctas.median()
            tbb = (1 / b) * act * clk * 1e6 / 1e12 if b > 0 else np.nan
    fits.append(pre + list(key) + [len(d), int(d.n.sum()), 1 / b if b > 0 else np.nan, a, rm, bmax, tb, tbb, act, ids_str])
append_rows(f"{OUT}/11__dsmem_bw__fits.csv",
            MANDATORY + KEY + ["n_payloads", "n_records", "bpc_per_sm", "alpha_cycles", "rmse_pct", "bpc_at_max_payload", "tbps_gpu_wide",
                               "tbps_from_bpc", "active_sms", "source_run_ids"], fits)
F = pd.DataFrame(fits, columns=MANDATORY + KEY + ["n_payloads", "n_records", "bpc_per_sm", "alpha_cycles", "rmse_pct", "bpc_at_max_payload",
                                                  "tbps_gpu_wide", "tbps_from_bpc", "active_sms", "source_run_ids"])

# GPU-wide: long stream (S) and P1 style
gw = []
S = launches[launches.step == "S"]
SS = samples[samples.step == "S"].copy()
SS["bpc"] = SS.bytes_moved / SS.cycles
for (kern, cs), d in S.groupby(["kernel", "cs"]):
    a, b, _ = affine(d.total_bytes, d.event_ns)
    mx = d[d.passes == d.passes.max()]
    side = "target" if kern in ("st_async", "tma_put", "tma_red") else "issuer"
    s = SS[(SS.kernel == kern) & (SS.cs == cs) & (SS.passes == d.passes.max()) & (SS.side == side)]
    gw.append(pre + ["S", kern, cs, len(d), 1e-3 / b if b > 0 else np.nan, a, (mx.total_bytes / mx.event_ns).median() * 1e-3,
                     s.bpc.median(), d.active_ctas.median(), ids_str])
P = launches[launches.step == "P1"]
for (kern, cs), d in P.groupby(["kernel", "cs"]):
    a, b, _ = affine(d.total_bytes, d.event_ns)
    mx = d[d.passes == d.passes.max()]
    gw.append(pre + ["P1", kern, cs, len(d), 1e-3 / b if b > 0 else np.nan, a, (mx.total_bytes / mx.event_ns).median() * 1e-3, np.nan,
                     d.active_ctas.median(), ids_str])
GWC = ["source", "kernel", "cs", "n_points", "tbps_slope", "alpha_ns", "tbps_max_point", "bpc_cta_median", "active_sms", "source_run_ids"]
append_rows(f"{OUT}/11__dsmem_bw__gpuwide.csv", MANDATORY + GWC, gw)
G = pd.DataFrame(gw, columns=MANDATORY + GWC)

# summary
srows = []


def add(level, config, value, unit, n, source):
    srows.append(pre + [level, config, value, unit, n, source, ids_str])


def bw_side(kern):
    return "target" if kern in ("st_async", "tma_put", "tma_red") else "issuer"


for _, r in F[(F.step == "A")].iterrows():
    if r.side == bw_side(r.kernel):
        add(f"ring_{r.kernel}_cs{r.cs}_{r.launch_mode}", "ring +1, 1024 thr, ILP 4, 16 KiB chunks, payload slope 16-96 KiB", r.bpc_per_sm,
            "B/cycle/SM", r.n_records, "A fit")
for _, r in F[(F.step == "C") & (F.cs == 8) & (F.launch_mode == "single")].iterrows():
    if r.side == bw_side(r.stream) or r.kernel.startswith("mixed"):
        add(f"pattern_{r.pattern}{r.pattern_k}_{r.kernel}_{r.stream}_{r.side}_cs8", "CS 8 single cluster", r.bpc_per_sm, "B/cycle/SM",
            r.n_records, "C fit")
for _, r in G.iterrows():
    add(f"gpu_wide_{r.source}_{r.kernel}_cs{r.cs}", "event-time slope" if r.source == "S" else "P1 style, event-time slope", r.tbps_slope,
        "TB/s", r.n_points, r.source)
if "K4" in src:
    V = pd.read_csv("results/11__dsmem_bw__victim.csv")
    V = V[V.run_id == src["K4"]]
    for (kern, rt, w, mode, side), d in V.groupby(["kernel", "reader_threads", "victim_warps", "launch_mode", "side"]):
        add(f"victim_{side}_{kern}_r{rt}_w{w}_{mode}", "K4 local-pressure control", d.bpc.median(), "B/cycle/SM", len(d), "K4")
if "K5" in src:
    I = pd.read_csv("results/11__dsmem_bw__interference.csv")
    I = I[I.run_id == src["K5"]]
    for key, d in I.groupby(["role_map", "dsmem_dest", "ilp_l2", "ilp_dsmem", "group", "alone_or_both"]):
        add("interf_" + "_".join(map(str, key)), "K5 one CS-16 cluster", d.bpc.median(), "B/cycle/SM", len(d), "K5")
        add("interf_lat_" + "_".join(map(str, key)), "K5 mean loaded latency per ILP group", d.loaded_latency_cycles.median(), "cycles",
            len(d), "K5")
append_rows(f"{OUT}/11__dsmem_bw__summary.csv", MANDATORY + ["level", "config", "value", "unit", "n", "source", "source_run_ids"], srows)

if OUT == "results":
    reg = f"results/{EXP}__runs.csv"
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")
    uuid = pd.read_csv(reg, dtype=str).query("run_id in @ids").gpu_uuid.iloc[0]
    append_rows(reg, ["run_id", "started_iso", "finished_iso", "status", "gpu_uuid", "cmdline"],
                [[run_id, now, "", "started", uuid, "src/analyze.py"], [run_id, "", now, "complete", uuid, "src/analyze.py"]])
print(f"analysis {run_id}: {len(fits)} fits, {len(gw)} GPU-wide rows, {len(srows)} summary rows; sources {src}")
