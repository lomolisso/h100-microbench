"""Experiment 07 summary: the synchronisation costs later experiments (09, 12, 16, 17, 18) consume, from the latest
complete full run of each binary (no GPU).

Primitive costs (K1–K3, K5; cycles per operation = (t1 − t0 − timer floor) / n_ops, median/p5/p95 over every rep of
every recorded launch, launches flagged in the clock log dropped): one row per (kernel, primitive, cs, block_threads,
variant), level = "<kernel>/<primitive>/<variant>", config = "cs <cs>, threads <block_threads>".
Signalling (K2 delay2000, K3 remote arrive, K4 ping-pong; propagation file): per (primitive, cs, variant,
block_threads) the median one-way %globaltimer delay (ns) and, where measured, the round trip (cycles on rank 0's SM).
→ results/07__sync__summary.csv (append; own run_id = time of this summary; source runs in source_run_ids).
"""
import datetime

import pandas as pd

from common07 import MANDATORY, append_rows, load_prim, load_prop, stats

rows, srcs, clk = [], [], []
for k in ["k1", "k2", "k3", "k5"]:
    df, meta = load_prim(k)
    srcs.append(meta["run_id"])
    clk.append(df.sm_clock_mhz.median())
    for _, r in stats(df, ["kernel", "primitive", "cs", "block_threads", "variant"]).iterrows():
        rows.append([f"{r.kernel}/{r.primitive}/{r.variant}", r["median"], r.p5, r.p95, "cycles", int(r.n),
                     f"cs {r.cs}, threads {r.block_threads}", k.upper(), meta["run_id"]])
for k in ["k2", "k3", "k4"]:
    p, run = load_prop(k)
    srcs.append(run)
    by = ["primitive", "cs", "variant", "block_threads"]
    for col, unit in [("one_way_ns", "ns"), ("round_trip_cycles", "cycles")]:
        q = p.dropna(subset=[col])
        if q.empty:
            continue
        for _, r in stats(q, by, col).iterrows():
            rows.append([f"{k}/{r.primitive}/{r.variant}/{col}", r["median"], r.p5, r.p95, unit, int(r.n),
                         f"cs {r.cs}, threads {r.block_threads}", k.upper(), run])
first = pd.read_csv("results/07__synchronization_primitives__clock_log.csv", nrows=1)
run_id = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")
pre = [first.gpu_name[0], first.sm_count[0], round(sum(clk) / len(clk)), first.mem_clock_mhz[0], first.cuda_version[0],
       first.driver_version[0], run_id, 0]
append_rows("results/07__sync__summary.csv",
            MANDATORY + ["level", "median", "p5", "p95", "unit", "n", "config", "source", "source_run_ids"],
            [pre + r for r in rows])
print(f"summary: {len(rows)} rows from runs {sorted(set(srcs))}")
