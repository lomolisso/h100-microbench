"""Experiment 04 summary: one latency per level of the generic-proxy hierarchy, from the latest complete runs.

  smem, smem_dsmem_if  K1 (lds / dsmem_if, all strides and sizes): per launch, (total_cycles(16384 hops) − median
                       total_cycles(4096 hops) of the same configuration) / 12288 — the slope removes the fixed region.
  l1                   K2 ld.global.ca, footprints ≤ 32 KiB (inside L1 for every carveout), every recorded launch.
  l2_near, l2_far      K3 raw, 8 MiB, every sample (variants A and B, both passes): 2-means classes.
  dram_near, dram_far, dram_slow  K3 raw, 256 MiB, every sample: 6-means classes 2, 3 and 4+5. The 256 MiB samples
                       are not all HBM accesses: ≈ 14 % are L2 hits (class 0 near ≈ 8.3 %, class 1 far ≈ 5.6 % of the
                       samples), so k = 6 keeps them in their own classes; with k = 4 the far L2 hits (≈ 430 cycles)
                       would merge into the near-HBM class (≈ 505) and bias it.
  K3 samples above 10 000 cycles (isolated interruptions, ~1 per launch at most) are excluded and counted in `source`.
  l2_chase, dram_chase K2 ld.global.cg plateau medians of a plain chase (1–16 MiB; 256–512 MiB), every launch.
K3 values are per-access timings: the timer-pair cost (K3 calib median − K1 lds latency) is subtracted and reported in
timer_correction_cycles. median/p5/p95 over the samples (K3) or launches (K1, K2); n = number of those.
Clock normalisation: under the 1635 MHz lock every GPC runs at its own real SM clock (1628.5–1634.9 MHz, medians of
03's clock-domain run, common04.sm_clocks). Each sample/launch is converted with the clock of the GPC of the SM that
measured it: ns = cycles / f_GPC, cycles_at_1635 = ns × 1.635; the quantiles of the converted values are reported next
to the raw cycles (median_cycles/p5/p95 are unchanged, still raw SM cycles). sm_mhz_min/max = range of the clocks
applied to the level's samples; smids = the measuring SMs; clock_run_ids = 03 clockdom run and 03 map run.
→ results/04__latency__summary_v2.csv (append, own run_id = time of this summary; source runs in source_run_ids).
  (results/04__latency__summary.csv holds earlier summaries with the old header, raw cycles only.)
"""
import datetime

import numpy as np
import pandas as pd

from common04 import MANDATORY, NOMINAL_MHZ as NOMINAL, append_rows, at_nominal, kmeans1d, latest_run, read_run, sm_clocks

OUT = "results/04__latency__summary_v2.csv"
OUTLIER = 10000
COLS = ["level", "median_cycles", "p5", "p95", "median_cycles_at_1635", "p5_cycles_at_1635", "p95_cycles_at_1635",
        "median_ns", "n", "timer_correction_cycles", "sm_mhz_min", "sm_mhz_max", "smids", "source", "source_run_ids",
        "clock_run_ids"]


def wq(values, counts, q):
    """Weighted quantile of a histogram (values ascending not required)."""
    o = np.argsort(values)
    v, c = np.asarray(values)[o], np.asarray(counts, float)[o]
    cum = np.cumsum(c) / c.sum()
    return float(v[np.searchsorted(cum, q)])


def k1_slopes(k1, variant):
    """Per 16384-hop launch: slope against the median 4096-hop launch of the same configuration; and its smid."""
    d = k1[k1.variant == variant]
    t4 = d[d.hops == 4096].groupby(["stride", "n_words"]).total_cycles.median()
    d16 = d[d.hops == 16384]
    return np.array([(r.total_cycles - t4[(r.stride, r.n_words)]) / 12288 for r in d16.itertuples()]), d16.smid.values


def main():
    rid = {k: latest_run(*k) for k in [("lat_smem", "k1"), ("lat_footprint", "k2"), ("lat_fine", "raw"), ("lat_fine", "calib")]}
    k1 = read_run("04__latency__smem.csv", rid["lat_smem", "k1"], dtype={"stride": str})
    k2 = read_run("04__latency__footprint.csv", rid["lat_footprint", "k2"])
    cal = read_run("04__latency__fine_calib.csv", rid["lat_fine", "calib"])
    fine = read_run("04__latency__fine.csv", rid["lat_fine", "raw"], usecols=["run_id", "footprint_bytes", "probe_smid", "cycles"])
    clk, clk_rid, map_rid = sm_clocks()
    f_sm = clk.gpc_mhz  # smid → real SM clock of its GPC [MHz]
    clock_runs = f"clockdom {clk_rid}; map {map_rid}"

    rows = []
    lds, lds_sm = k1_slopes(k1, "lds")
    corr = float(np.median(cal.cycles) - np.median(lds))

    def add(level, x, smid, source, run, c=0.0, w=None):
        """x: raw SM cycles per sample (or histogram values with counts w), smid: measuring SM per value."""
        x = np.asarray(x, float)
        mhz = f_sm.loc[np.asarray(smid)].values
        xn = at_nominal(x, mhz)
        if w is None:  # one value per launch: interpolated percentiles (as before)
            w = np.ones(len(x))
            q = lambda v, p: float(np.percentile(v, 100 * p))
        else:          # histogram values with counts (K3)
            w = np.asarray(w, float)
            q = lambda v, p: wq(v, w, p)
        rows.append([level, q(x, 0.5), q(x, 0.05), q(x, 0.95), q(xn, 0.5), q(xn, 0.05), q(xn, 0.95),
                     q(xn, 0.5) / (NOMINAL / 1e3), int(w.sum()), c, mhz.min(), mhz.max(),
                     " ".join(str(int(s)) for s in sorted(set(np.asarray(smid).tolist()))), source, run, clock_runs])

    add("smem", lds, lds_sm, "K1 lds slope", rid["lat_smem", "k1"])
    add("smem_dsmem_if", *k1_slopes(k1, "dsmem_if"), "K1 dsmem_if slope", rid["lat_smem", "k1"])
    l1 = k2[(k2.load_variant == "ca") & (k2.footprint_bytes <= 32 << 10)]
    add("l1", l1.cycles_per_hop, l1.smid, "K2 ca <= 32 KiB", rid["lat_footprint", "k2"])
    for F, k, names in [(8 << 20, 2, {(0,): "l2_near", (1,): "l2_far"}),
                        (256 << 20, 6, {(2,): "dram_near", (3,): "dram_far", (4, 5): "dram_slow"})]:
        hp = fine[fine.footprint_bytes == F].groupby(["probe_smid", "cycles"]).size().rename("count").reset_index()
        n_out = int(hp.loc[hp.cycles > OUTLIER, "count"].sum())
        hp = hp[hp.cycles <= OUTLIER]
        h = hp.groupby("cycles")["count"].sum().sort_index()  # pooled histogram over probes: the K-means input
        cen, lab, _, _ = kmeans1d(h.index.values, h.values, k)
        cls = pd.Series(lab, index=h.index.values)
        hp = hp.assign(cls=cls.loc[hp.cycles.values].values)
        # k = 6 at 256 MiB: classes 0 (near L2 hit) and 1 (far L2 hit) hold ≈ 14 % of the samples (≈ 8.3 % + 5.6 %);
        # with k = 4 the far L2 hits would be absorbed by the near-HBM class.
        for js, name in names.items():
            d = hp[hp.cls.isin(js)]
            add(name, d.cycles.values - corr, d.probe_smid.values,
                f"K3 {F >> 20} MiB {k}-means class {'+'.join(map(str, js))} (centre {', '.join(f'{cen[i] - corr:.1f}' for i in js)}; "
                f"{n_out} samples > {OUTLIER} excluded)",
                rid["lat_fine", "raw"], corr, d["count"].values)
    cg = k2[k2.load_variant == "cg"]
    d = cg[(cg.footprint_bytes >= 1 << 20) & (cg.footprint_bytes <= 16 << 20)]
    add("l2_chase", d.cycles_per_hop, d.smid, "K2 cg 1-16 MiB", rid["lat_footprint", "k2"])
    d = cg[(cg.footprint_bytes >= 256 << 20) & (cg.footprint_bytes <= 512 << 20)]
    add("dram_chase", d.cycles_per_hop, d.smid, "K2 cg 256-512 MiB", rid["lat_footprint", "k2"])

    src = k2.iloc[0]
    me = datetime.datetime.now().astimezone().isoformat(timespec="milliseconds")
    pre = [src.gpu_name, src.sm_count, int(k2.sm_clock_mhz.median()), int(k2.mem_clock_mhz.median()), src.cuda_version,
           src.driver_version, me, -1]
    out = [pre + [r[0]] + [f"{x:.2f}" for x in r[1:8]] + [r[8], f"{r[9]:.2f}", f"{r[10]:.3f}", f"{r[11]:.3f}"] + r[12:]
           for r in rows]
    append_rows(OUT, MANDATORY + COLS, out)
    print(pd.DataFrame([r[:13] for r in rows], columns=COLS[:13]).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
