"""Experiment 05 summary: one bandwidth per level and access type, from the latest complete runs (no GPU needed).

Per-SM levels (bpc_per_sm = B/cycle of one SM; tbps = bpc_per_sm × 132 SMs × the SM clock measured by the blocks):
  smem_ld, smem_st, l1_ld   K1/K2 all_sms placement (132 blocks at once), W = 16 B: the configuration (threads, ILP)
                            with the highest median; per launch pair, the slope (bytes, cycles) between the 4096- and
                            8192-step launches of every SM (the fixed cost of the timed region cancels; common05.k12_slope).
  l1_st_all_sms             K2 st.global.v4 from all 132 SMs at once (the write-through path to L2).
GPU-wide levels (tbps = bytes / (cudaEvent time − event floor), the floor = median elapsed time of an empty event pair
from experiment 01; bpc_chip = tbps / SM clock; bpc_per_sm = bpc_chip / 132):
  l2_ld_{mixed,local,remote}  K3 steady state, 16 MiB, 2 × 132 blocks × 1024 threads, ILP 4, .cg.
  l2_ld_max                 K3 steady state: highest median over all footprints and variants (footprint in `config`).
  l2_st_{mixed,local,remote}, l2_mix_mixed   K5 at L2, 16 MiB.
  l2_link_one_way           link mode: st remote from the 66 SMs of one partition (lines homed in the other one), 16 MiB
                            steady state (each store is written through to its home and crosses the partition link once,
                            although it also leaves a copy in the writer's partition); both partitions' runs pooled.
  l2_link_both_ways         the same from both partitions at once.
  l2_ld_one_gpc             link mode: ld local from the SMs of one GPC (G1, 16 SMs, and G4, 18 SMs, pooled).
  l2_ld_first_pass_{near,far}  firstpass mode, pass 1, the 66 SMs of one partition (1 × 1024 threads per SM, gentle home
                            touch), lines held only at home: rate = 1 / slope of the %globaltimer span over the payload
                            for the footprints whose residency sample found ≤ 0.1 % of lines missing from L2 (first read
                            > 480 cycles); B/cycle per SM at the locked 1635 MHz; p5/p95 by bootstrap; n = launches.
  dram_{read,write,mix,mix11}  K4 full grid, .cg; written lines still dirty in L2 at kernel end (ncu dram__bytes_write)
                            are taken out of the payload (common05.hbm_write_factor).
  dram_read_per_sm          K4 active-SM sweep, read: median GB/s per active SM for n ≤ 16 (the linear region).
median/p5/p95 over the recorded (valid) launches; n = number of launches.
Run_ids read: the latest complete run of each mode (firstpass for the first-pass levels).
→ results/05__bandwidth__summary.csv (append; own run_id = time of this summary; source runs in source_run_ids).
"""
import datetime

import numpy as np
import pandas as pd

from common05 import (MANDATORY, append_rows, event_floor_ms, first_pass_slope, hbm_write_factor, k12_slope, latest_run,
                      net_tbps, read_run, resident_footprints)

OUT = "results/05__bandwidth__summary.csv"
COLS = ["level", "median", "p5", "p95", "unit", "n", "bpc_per_sm", "tbps", "config", "source", "source_run_ids"]
SMS = 132
LOCK_MHZ = 1635  # fixed run configuration (docs/context/10)


def q(v):
    v = np.asarray(v, float)
    return np.median(v), np.percentile(v, 5), np.percentile(v, 95), len(v)


def main():
    rows = []
    floor = event_floor_ms()
    r12, r3, r4 = latest_run("bw_smem_l1", "k12"), latest_run("bw_l2", "k3"), latest_run("bw_dram", "k4")
    k12 = read_run("05__bandwidth__smem_l1.csv", r12)
    k12 = k12[(k12.discard.isna()) | (k12.discard == "")]
    k12["mhz"] = k12.cycles / k12.ns * 1e3
    for level, op, name in [("smem", "ld", "smem_ld"), ("smem", "st", "smem_st"), ("l1", "ld", "l1_ld"), ("l1", "st", "l1_st_all_sms")]:
        d = k12[(k12.level == level) & (k12.op == op) & (k12.width_bytes == 16) & (k12.placement == "all_sms") &
                (k12.pattern == "stride")]
        sl = k12_slope(d)
        med = sl.groupby(["threads", "ilp"]).bpc_slope.median()
        t, i = med.idxmax()
        sel = sl[(sl.threads == t) & (sl.ilp == i)]
        per_launch = sel.groupby("k").bpc_slope.median()
        m, p5, p95, n = q(per_launch)
        mhz = d[(d.threads == t) & (d.ilp == i)].mhz.median()
        rows.append([name, m, p5, p95, "B/cycle/SM", n, m, m * SMS * mhz * 1e6 / 1e12,
                     f"{t} threads, ILP {i}, 132 SMs at once; slope 4096->8192 steps (8192 alone: {sel.bpc_8192.median():.2f}); "
                     f"clock {mhz:.1f} MHz", "K1/K2", r12])

    l2 = read_run("05__bandwidth__l2.csv", r3)
    l2 = l2[(l2.discard.isna()) | (l2.discard == "")]
    l2 = l2.assign(tbps=net_tbps(l2, floor))
    l2 = l2.assign(bpc_chip=l2.tbps * 1e12 / (l2.sm_mhz_blocks * 1e6))
    st = l2[(l2["pass"] == 0) & (l2.ilp == 4) & (l2.blocks == 2 * SMS) & (l2.cache_op == "cg")]
    for op, variants in [("ld", ["mixed", "local", "remote"]), ("st", ["mixed", "local", "remote"]), ("mix", ["mixed"])]:
        for v in variants:
            d = st[(st.op == op) & (st.variant == v) & (st.footprint_bytes == 16 << 20)]
            m, p5, p95, n = q(d.tbps)
            bpc = d.bpc_chip.median()
            rows.append([f"l2_{op}_{v}", m, p5, p95, "TB/s", n, bpc / SMS, m, "16 MiB, 2 x 132 x 1024 threads, ILP 4", "K3/K5 steady", r3])
    ld = st[(st.op == "ld") & st.variant.isin(["mixed", "local", "remote"])]
    best = ld.groupby(["variant", "footprint_bytes"]).tbps.median().idxmax()
    d = ld[(ld.variant == best[0]) & (ld.footprint_bytes == best[1])]
    m, p5, p95, n = q(d.tbps)
    rows.append(["l2_ld_max", m, p5, p95, "TB/s", n, d.bpc_chip.median() / SMS, m, f"{best[0]}, {best[1] >> 20} MiB", "K3 steady", r3])

    rl = latest_run("bw_l2", "link")
    lk = read_run("05__bandwidth__l2_link.csv", rl)
    lk = lk[lk.discard.isna()]
    lk = lk.assign(tbps=net_tbps(lk, floor))
    lk = lk.assign(bpc_chip=lk.tbps * 1e12 / (lk.sm_mhz_blocks * 1e6))
    one_part = lk.reader_gpc_mask.isin([0x8E, 0x71])
    for name, sel, cfg in [
        ("l2_link_one_way", (lk["pass"] == 0) & (lk.op == "st") & (lk.variant == "remote") & one_part, "st remote, 66 SMs of one partition, 16 MiB"),
        ("l2_link_both_ways", (lk["pass"] == 0) & (lk.op == "st") & (lk.variant == "remote") & (lk.reader_gpc_mask == 0xFF), "st remote, all 132 SMs, 16 MiB"),
        ("l2_ld_one_gpc", (lk["pass"] == 0) & (lk.op == "ld") & (lk.variant == "local") & lk.reader_gpc_mask.isin([0x02, 0x10]), "ld local, one GPC (16 or 18 SMs), 16 MiB"),
    ]:
        d = lk[sel]
        m, p5, p95, n = q(d.tbps)
        rows.append([name, m, p5, p95, "TB/s", n, d.bpc_chip.median() / SMS, m, cfg, "link mode", rl])
    rf = latest_run("bw_l2", "firstpass")
    fp = read_run("05__bandwidth__l2_link.csv", rf)
    fp = fp[fp.discard.isna() & (fp["pass"] == 1) & (fp.threads == 1024) & (fp.blocks == SMS)]
    res = read_run("05__bandwidth__firstpass_residency.csv", rf)
    ok = resident_footprints(res)
    for v, name in [("local", "near"), ("remote", "far")]:
        d = fp[(fp.variant == v) & fp.footprint_bytes.isin(ok)]
        rate, icpt, p5, p95 = first_pass_slope(d)
        tb = rate * 1e9 / 1e12
        # B/cycle at the locked clock: the block timers' cycles/ns over µs-long passes read high (%globaltimer quantised)
        rows.append([f"l2_ld_first_pass_{name}", tb, p5 * 1e-3, p95 * 1e-3, "TB/s", len(d), rate * 1e9 / (LOCK_MHZ * 1e6) / 66, tb,
                     f"first read of lines held only at home; 66 SMs x 1 x 1024 threads; slope of span over "
                     f"{d.bytes.min() / 2**20:.0f}-{d.bytes.max() / 2**20:.0f} MiB per direction (start-up {icpt / 1e3:.2f} us)",
                     "firstpass mode, pass 1", rf])

    dr = read_run("05__bandwidth__dram.csv", r4)
    dr = dr[(dr.discard.isna()) | (dr.discard == "")]
    dr = dr.assign(tbps_raw=dr.tbps, tbps=net_tbps(dr, floor))
    ndr = pd.read_csv("results/05__bandwidth__ncu_dram.csv")
    ndr = ndr[ndr.run_id == ndr.run_id.max()]
    full = dr[(dr.blocks == 4 * SMS) & (dr.ilp == 4) & (dr.cache_op == "cg")]
    for op in ["read", "write", "mix", "mix11"]:
        d = full[full.op == op]
        f = hbm_write_factor(ndr, op, d.bytes.median())
        m, p5, p95, n = q(d.tbps * f)
        mhz = d.sm_mhz_blocks.median()
        rows.append([f"dram_{op}", m, p5, p95, "TB/s", n, m * 1e12 / (mhz * 1e6) / SMS, m,
                     f"4 x 132 x 1024 threads, ILP 4, .cg; x {f:.4f} for lines still dirty in L2 at kernel end", "K4", r4])
    sw = dr[(dr.op == "read") & (dr.blocks <= 16) & (dr.ilp == 4) & (dr.threads == 1024) & (dr.blocks < 4 * SMS)]
    m, p5, p95, n = q(sw.gbps_per_sm * sw.ms_event / (sw.ms_event - floor))
    rows.append(["dram_read_per_sm", m, p5, p95, "GB/s per SM", n, m * 1e9 / (sw.sm_mhz_blocks.median() * 1e6), np.nan,
                 "1 block x 1024 threads per SM, ILP 4, 1-16 active SMs", "K4 active SMs", r4])

    run_id = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")
    first = k12.iloc[0]
    pre = [first.gpu_name, first.sm_count, first.sm_clock_mhz, first.mem_clock_mhz, first.cuda_version, first.driver_version, run_id, -1]
    out = []
    for r in rows:
        r = [f"{x:.4f}" if isinstance(x, (float, np.floating)) and np.isfinite(x) else ("" if isinstance(x, float) else x) for x in r]
        out.append(pre + r)
    append_rows(OUT, MANDATORY + COLS, out)
    print(pd.DataFrame([r[:9] for r in rows], columns=COLS[:9]).to_string(index=False))


if __name__ == "__main__":
    main()
