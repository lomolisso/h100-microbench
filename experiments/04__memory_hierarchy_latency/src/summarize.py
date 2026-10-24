"""Experiment 04 summary: one latency per level of the generic-proxy hierarchy, from the latest complete runs.

  smem, smem_dsmem_if  K1 (lds / dsmem_if, all strides and sizes): per launch, (total_cycles(16384 hops) − median
                       total_cycles(4096 hops) of the same configuration) / 12288 — the slope removes the fixed region.
  l1                   K2 ld.global.ca, footprints ≤ 32 KiB (inside L1 for every carveout), every recorded launch.
  l2_near, l2_far      K3 raw, 8 MiB, every sample (variants A and B, both passes): 2-means classes.
  dram_near, dram_far, dram_slow  K3 raw, 256 MiB, every sample: 6-means classes 2, 3 and 4+5 (classes 0, 1 are the
                       near and far L2 hits; with 4 classes the far L2 hits would merge into the near-HBM class).
  K3 samples above 10 000 cycles (isolated interruptions, ~1 per launch at most) are excluded and counted in `source`.
  l2_chase, dram_chase K2 ld.global.cg plateau medians of a plain chase (1–16 MiB; 256–512 MiB), every launch.
K3 values are per-access timings: the timer-pair cost (K3 calib median − K1 lds latency) is subtracted and reported in
timer_correction_cycles. median/p5/p95 over the samples (K3) or launches (K1, K2); n = number of those.
→ results/04__latency__summary.csv (append, own run_id = time of this summary; source runs in source_run_ids).
"""
import datetime

import numpy as np
import pandas as pd

from common04 import MANDATORY, append_rows, kmeans1d, latest_run, read_run

OUT = "results/04__latency__summary.csv"
OUTLIER = 10000
COLS = ["level", "median_cycles", "p5", "p95", "n", "timer_correction_cycles", "source", "source_run_ids"]


def wq(values, counts, q):
    """Weighted quantile of a histogram (values ascending not required)."""
    o = np.argsort(values)
    v, c = np.asarray(values)[o], np.asarray(counts, float)[o]
    cum = np.cumsum(c) / c.sum()
    return float(v[np.searchsorted(cum, q)])


def k1_slopes(k1, variant):
    d = k1[k1.variant == variant]
    t4 = d[d.hops == 4096].groupby(["stride", "n_words"]).total_cycles.median()
    d16 = d[d.hops == 16384]
    return np.array([(r.total_cycles - t4[(r.stride, r.n_words)]) / 12288 for r in d16.itertuples()])


def main():
    rid = {k: latest_run(*k) for k in [("lat_smem", "k1"), ("lat_footprint", "k2"), ("lat_fine", "raw"), ("lat_fine", "calib")]}
    k1 = read_run("04__latency__smem.csv", rid["lat_smem", "k1"], dtype={"stride": str})
    k2 = read_run("04__latency__footprint.csv", rid["lat_footprint", "k2"])
    cal = read_run("04__latency__fine_calib.csv", rid["lat_fine", "calib"])
    fine = read_run("04__latency__fine.csv", rid["lat_fine", "raw"], usecols=["run_id", "footprint_bytes", "cycles"])

    rows = []
    lds = k1_slopes(k1, "lds")
    corr = float(np.median(cal.cycles) - np.median(lds))

    def add(level, x, source, run, c=0.0):
        x = np.asarray(x, float)
        rows.append([level, np.median(x), np.percentile(x, 5), np.percentile(x, 95), len(x), c, source, run])

    add("smem", lds, "K1 lds slope", rid["lat_smem", "k1"])
    add("smem_dsmem_if", k1_slopes(k1, "dsmem_if"), "K1 dsmem_if slope", rid["lat_smem", "k1"])
    add("l1", k2[(k2.load_variant == "ca") & (k2.footprint_bytes <= 32 << 10)].cycles_per_hop, "K2 ca <= 32 KiB", rid["lat_footprint", "k2"])
    for F, k, names in [(8 << 20, 2, {(0,): "l2_near", (1,): "l2_far"}),
                        (256 << 20, 6, {(2,): "dram_near", (3,): "dram_far", (4, 5): "dram_slow"})]:
        h = fine[fine.footprint_bytes == F].cycles.value_counts().sort_index()
        n_out = int(h[h.index > OUTLIER].sum())
        h = h[h.index <= OUTLIER]
        cen, lab, _, _ = kmeans1d(h.index.values, h.values, k)
        for js, name in names.items():
            j = js[0]
            sel = np.isin(lab, js)
            v, c = h.index.values[sel] - corr, h.values[sel]
            rows.append([name, wq(v, c, 0.5), wq(v, c, 0.05), wq(v, c, 0.95), int(c.sum()), corr,
                         f"K3 {F >> 20} MiB {k}-means class {'+'.join(map(str, js))} (centre {', '.join(f'{cen[i] - corr:.1f}' for i in js)}; "
                         f"{n_out} samples > {OUTLIER} excluded)",
                         rid["lat_fine", "raw"]])
    cg = k2[k2.load_variant == "cg"]
    add("l2_chase", cg[(cg.footprint_bytes >= 1 << 20) & (cg.footprint_bytes <= 16 << 20)].cycles_per_hop, "K2 cg 1-16 MiB",
        rid["lat_footprint", "k2"])
    add("dram_chase", cg[(cg.footprint_bytes >= 256 << 20) & (cg.footprint_bytes <= 512 << 20)].cycles_per_hop, "K2 cg 256-512 MiB",
        rid["lat_footprint", "k2"])

    src = k2.iloc[0]
    me = datetime.datetime.now().astimezone().isoformat(timespec="milliseconds")
    pre = [src.gpu_name, src.sm_count, int(k2.sm_clock_mhz.median()), int(k2.mem_clock_mhz.median()), src.cuda_version,
           src.driver_version, me, -1]
    out = [pre + [r[0]] + [f"{x:.2f}" for x in r[1:4]] + r[4:] for r in rows]
    append_rows(OUT, MANDATORY + COLS, out)
    print(pd.DataFrame([r[:6] for r in rows], columns=COLS[:6]).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
