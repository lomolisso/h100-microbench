"""Experiment 05 summary: one bandwidth per level and access type, from the latest complete runs (no GPU needed).

Per-SM levels (bpc_per_sm = B/cycle of one SM; tbps = bpc_per_sm × 132 SMs × the SM clock measured by the blocks):
  smem_ld, smem_st, l1_ld   K1/K2 all_sms placement (132 blocks at once), W = 16 B: the configuration (threads, ILP)
                            with the highest median; per launch, bytes / cycles of every block (iters 8192).
  l1_st_all_sms             K2 st.global.v4 from all 132 SMs at once (the write-through path to L2).
GPU-wide levels (tbps = bytes / cudaEvent time; bpc_chip = tbps / SM clock; bpc_per_sm = bpc_chip / 132):
  l2_ld_{mixed,local,remote}  K3 steady state, 16 MiB, 2 × 132 blocks × 1024 threads, ILP 4, .cg.
  l2_ld_max                 K3 steady state: highest median over all footprints and variants (footprint in `config`).
  l2_st_{mixed,local,remote}, l2_mix_mixed   K5 at L2, 16 MiB.
  l2_link_one_way           link mode: st remote from the 66 SMs of one partition (lines homed in the other one), 16 MiB
                            steady state (stores never leave a copy, so every store crosses the partition link); both
                            partitions' runs pooled.
  l2_link_both_ways         the same from both partitions at once.
  l2_ld_one_gpc             link mode: ld local from the SMs of one GPC (G1, 16 SMs, and G4, 18 SMs, pooled).
  l2_ld_first_pass_{near,far}  link mode, first-pass test, pass 1, the 66 SMs of one partition (512 threads per SM),
                            lines held only at home: GPU-wide rate over the %globaltimer span of the reading blocks.
  dram_{read,write,mix,mix11}  K4 full grid, .cg.
  dram_read_per_sm          K4 active-SM sweep, read: median GB/s per active SM for n ≤ 16 (the linear region).
median/p5/p95 over the recorded (valid) launches; n = number of launches.
→ results/05__bandwidth__summary.csv (append; own run_id = time of this summary; source runs in source_run_ids).
"""
import datetime

import numpy as np
import pandas as pd

from common05 import MANDATORY, append_rows, latest_run, read_run

OUT = "results/05__bandwidth__summary.csv"
COLS = ["level", "median", "p5", "p95", "unit", "n", "bpc_per_sm", "tbps", "config", "source", "source_run_ids"]
SMS = 132


def q(v):
    v = np.asarray(v, float)
    return np.median(v), np.percentile(v, 5), np.percentile(v, 95), len(v)


def main():
    rows = []
    r12, r3, r4 = latest_run("bw_smem_l1", "k12"), latest_run("bw_l2", "k3"), latest_run("bw_dram", "k4")
    k12 = read_run("05__bandwidth__smem_l1.csv", r12)
    k12 = k12[(k12.discard.isna()) | (k12.discard == "")]
    k12["mhz"] = k12.cycles / k12.ns * 1e3
    for level, op, name in [("smem", "ld", "smem_ld"), ("smem", "st", "smem_st"), ("l1", "ld", "l1_ld"), ("l1", "st", "l1_st_all_sms")]:
        d = k12[(k12.level == level) & (k12.op == op) & (k12.width_bytes == 16) & (k12.placement == "all_sms") &
                (k12.iters == 8192) & (k12.pattern == "stride")]
        med = d.groupby(["threads", "ilp"]).bpc_per_sm.median()
        t, i = med.idxmax()
        sel = d[(d.threads == t) & (d.ilp == i)]
        per_launch = sel.groupby("launch_idx").bpc_per_sm.median()
        m, p5, p95, n = q(per_launch)
        mhz = sel.mhz.median()
        rows.append([name, m, p5, p95, "B/cycle/SM", n, m, m * SMS * mhz * 1e6 / 1e12,
                     f"{t} threads, ILP {i}, 132 SMs at once; clock {mhz:.1f} MHz", "K1/K2", r12])

    l2 = read_run("05__bandwidth__l2.csv", r3)
    l2 = l2[(l2.discard.isna()) | (l2.discard == "")]
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
    one_part = lk.reader_gpc_mask.isin([0x8E, 0x71])
    for name, sel, cfg in [
        ("l2_link_one_way", (lk["pass"] == 0) & (lk.op == "st") & (lk.variant == "remote") & one_part, "st remote, 66 SMs of one partition, 16 MiB"),
        ("l2_link_both_ways", (lk["pass"] == 0) & (lk.op == "st") & (lk.variant == "remote") & (lk.reader_gpc_mask == 0xFF), "st remote, all 132 SMs, 16 MiB"),
        ("l2_ld_one_gpc", (lk["pass"] == 0) & (lk.op == "ld") & (lk.variant == "local") & lk.reader_gpc_mask.isin([0x02, 0x10]), "ld local, one GPC (16 or 18 SMs), 16 MiB"),
    ]:
        d = lk[sel]
        m, p5, p95, n = q(d.tbps)
        rows.append([name, m, p5, p95, "TB/s", n, d.bpc_chip.median() / SMS, m, cfg, "link mode", rl])
    for v, name in [("local", "near"), ("remote", "far")]:
        d = lk[(lk["pass"] == 1) & (lk.variant == v) & one_part]
        m, p5, p95, n = q(d.tbps_span)
        rows.append([f"l2_ld_first_pass_{name}", m, p5, p95, "TB/s", n, np.nan, m,
                     "first read of lines held only at home; 66 SMs x 512 threads; %globaltimer span", "link mode, first pass", rl])

    dr = read_run("05__bandwidth__dram.csv", r4)
    dr = dr[(dr.discard.isna()) | (dr.discard == "")]
    full = dr[(dr.blocks == 4 * SMS) & (dr.ilp == 4) & (dr.cache_op == "cg")]
    for op in ["read", "write", "mix", "mix11"]:
        d = full[full.op == op]
        m, p5, p95, n = q(d.tbps)
        mhz = d.sm_mhz_blocks.median()
        rows.append([f"dram_{op}", m, p5, p95, "TB/s", n, m * 1e12 / (mhz * 1e6) / SMS, m, "4 x 132 x 1024 threads, ILP 4, .cg", "K4", r4])
    sw = dr[(dr.op == "read") & (dr.blocks <= 16) & (dr.ilp == 4) & (dr.threads == 1024) & (dr.blocks < 4 * SMS)]
    m, p5, p95, n = q(sw.gbps_per_sm)
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
