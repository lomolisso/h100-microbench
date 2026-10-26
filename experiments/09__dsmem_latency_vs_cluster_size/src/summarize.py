"""Experiment 09 summary: the DSMEM latencies later experiments (10, 11, 12, 16, 17, 18) consume, from the latest complete
full run of each binary/mode (no GPU). Run from the repository root:
    tools/run --no-gpu -- bash -c "cd experiments/09__dsmem_latency_vs_cluster_size && python3 src/summarize.py"

Every recorded launch is used (launches with a bad placement or clock sample were already discarded and repeated by the
binaries). Levels (statistics over every row of the selection; cycles of the measuring SM):
  k1/<variant>/<chain>/<policy>/<pair>   cycles_per_hop of the 4096-hop chase, pair = remote (all src ≠ dst), self
                                         (CS 1 = own CTA through the DSMEM path; for CS > 1 the self rows of the cluster),
                                         same_tpc / other_tpc (remote pairs split by TPC = smid // 2); config "cs N"
  k1_fine/ld_cluster/stride1/Default/<pair>   per-hop cycles of the fine pass (includes the timer pair)
  loaded/<pair>                           cta_mean_cycles_per_group (8 loads, 1024 threads) per pair and launch
  k2/<variant>/r0<p>/round_trip_cycles    ping-pong round trip; k2/<variant>/r0<p>/one_way_ns (mean column = mean of
                                         the 32-ns-tick samples) for the variants with %globaltimer
  k3/<level>/<chain>                      reference chases (all SMs of K3 pooled; per SM in the notebook)
→ results/09__dsmem_latency__summary.csv (append; own run_id = time of this summary; source run in source_run_id).
"""
import datetime

import pandas as pd

from common09 import MANDATORY, append_rows, load, stats

rows = []


def add(df, by, col, prefix, unit, source, run, cfg=lambda r: f"cs {r['cs']}", suffix=""):
    for _, r in stats(df, by, col).reset_index().iterrows():
        parts = [str(r[b]) for b in by if b != "cs"]
        rows.append(["/".join([prefix] + parts) + suffix, cfg(r), r["median"], r.p5, r.p95, r["mean"], unit, int(r.n), source, run])


ch, run = load("chase", usecols=MANDATORY + ["cs", "policy", "variant", "chain", "smid_src", "smid_dst", "cycles_per_hop"])
clk = [ch.sm_clock_mhz.median()]
ch_r = ch[ch.pair != "self"].assign(pair="remote")
add(pd.concat([ch_r, ch[ch.pair == "self"]]), ["variant", "chain", "policy", "pair", "cs"], "cycles_per_hop", "k1", "cycles", "K1", run)
add(ch[ch.pair.isin(["same_tpc", "other_tpc"])], ["variant", "chain", "policy", "pair", "cs"], "cycles_per_hop", "k1", "cycles", "K1", run)
del ch, ch_r
fi, run = load("fine", usecols=MANDATORY + ["cs", "policy", "variant", "smid_src", "smid_dst", "cycles"])
fi["chain"] = "stride1"
add(fi, ["variant", "chain", "policy", "pair", "cs"], "cycles", "k1_fine", "cycles", "K1-fine", run)
del fi
lo, run = load("loaded")
add(lo, ["pair", "cs"], "cta_mean_cycles_per_group", "loaded", "cycles", "K1f", run)
vi, run = load("visibility")
vi["pairname"] = "r0" + vi.dst_rank.astype(str)
add(vi, ["variant", "pairname", "cs"], "round_trip_cycles", "k2", "cycles", "K2", run, suffix="/round_trip_cycles")
add(vi.dropna(subset=["one_way_ns"]), ["variant", "pairname", "cs"], "one_way_ns", "k2", "ns", "K2", run, suffix="/one_way_ns")
clk.append(vi.sm_clock_mhz.median())
ba, run = load("baselines")
add(ba, ["level", "chain"], "cycles_per_hop", "k3", "cycles", "K3", run, cfg=lambda r: "SMs " + ",".join(map(str, sorted(ba.smid.unique()))))

first = pd.read_csv("results/09__dsmem_latency_vs_cluster_size__clock_log.csv", nrows=1)
run_id = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")
pre = [first.gpu_name[0], first.sm_count[0], round(sum(clk) / len(clk)), first.mem_clock_mhz[0], first.cuda_version[0],
       first.driver_version[0], run_id, 0]
append_rows("results/09__dsmem_latency__summary.csv",
            MANDATORY + ["level", "config", "median", "p5", "p95", "mean", "unit", "n", "source", "source_run_id"],
            [pre + r for r in rows])
print(f"summary: {len(rows)} rows; run_id {run_id}")
