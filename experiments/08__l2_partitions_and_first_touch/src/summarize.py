"""Experiment 08 summary from the latest complete runs (no GPU): per-SM partition / sub-node labels and key numbers.

  sm_signature.csv   one row per SM (analysis08.sm_signature on the K5 run): partition, sub-node, near/far first-read
                     medians, same/other sub-node near medians and their difference (subnode_offset), far sub-class
                     medians, repeat (copy) medians. Cycles with the timer pair (calib median − 23) subtracted.
  sm_map.csv         03's map (latest run) with the partition and subnode columns filled; the same rows are appended
                     to ../03__sm_gpc_topology/results/03__sm_gpc_topology__map.csv under a new run_id, which is
                     registered as complete in 03's run registry (skipped if the latest 03 map already came from this
                     K5 run; never with --quick).
  summary.csv        level, value, unit, p5, p95, n: L2 near / same / other sub-node / far / copy latencies (K5, every
                     first-read sample), first reads by same/other partition × same/other sub-node position, K3 phase medians, partition hash bits (K2), K4 plain-chase cycles per hop and
                     the footprint where the steady-state chase leaves the near plateau.
All three append with their own run_id (time of this summary); source runs in source_run_ids.
"""
import argparse
import datetime

import numpy as np
import pandas as pd

from analysis08 import MAP03, gpc_map, k2_bits, k2_linearity, load_k5, quad_table, sm_signature
import common08
from common08 import MANDATORY, latest_run, read_run, timer_pair

append_rows = common08.append_rows

R = "results"
REG03 = "../03__sm_gpc_topology/results/03__sm_gpc_topology__runs.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print only, append nothing")
    a = ap.parse_args()
    global append_rows
    if a.dry_run:
        append_rows = lambda path, header, rows: print(f"(dry run) {len(rows)} rows → {path}")
    q = a.quick
    rid = {m: latest_run(t, m, q) for t, m in [("l2_probe", "calib"), ("l2_probe", "k2"), ("l2_probe", "k3"), ("l2_probe", "k5"),
                                               ("l2_capacity", "k4")]}
    tp = timer_pair(quick=q)
    k5 = load_k5(rid["k5"])
    sig, groups = sm_signature(k5, tp)
    me = datetime.datetime.now().astimezone().isoformat(timespec="milliseconds")
    src = k5.iloc[0]
    pre = [src.gpu_name, src.sm_count, int(k5.sm_clock_mhz.median()), int(k5.mem_clock_mhz.median()), src.cuda_version, src.driver_version, me, -1]

    cols = list(sig.columns) + ["timer_correction_cycles", "source_run_ids"]
    append_rows(f"{R}/08__l2__sm_signature.csv", MANDATORY + cols,
                [pre + [f"{v:.2f}" if isinstance(v, float) else v for v in r] + [f"{tp:.1f}", rid["k5"]] for r in sig.itertuples(index=False)])

    # 03 map with partition / subnode filled
    m03 = pd.read_csv(MAP03, dtype=str, keep_default_na=False)
    hdr = list(m03.columns)
    last = m03[m03.run_id == m03.run_id.max()].copy()
    lab = sig.set_index("smid")
    last["partition"] = last.smid.astype(int).map(lab.partition).astype(str)
    last["subnode"] = last.smid.astype(int).map(lab.subnode).astype(str)
    note = f"partition/subnode from 08 K5 run {rid['k5']}"
    last["notes"] = [(n + "; " if n else "") + note for n in last.notes]
    rows = [pre[:6] + [me, "-1"] + r[8:] for r in last[hdr].values.tolist()]
    append_rows(f"{R}/08__l2__sm_map.csv", hdr, rows)
    reg = pd.read_csv(REG03, dtype=str, keep_default_na=False)
    already = reg[reg.cmdline.str.contains(rid["k5"], regex=False)]
    if q:
        print("--quick: 03's map is not touched")
    elif len(already):
        print(f"03 map already holds the labels of K5 run {rid['k5']} (run {already.run_id.iloc[0]}): not appended again")
    else:
        cmd = f"08__l2_partitions_and_first_touch/src/summarize.py (partition/subnode from K5 run {rid['k5']})"
        append_rows(REG03, list(reg.columns), [[me, me, "", "started", "", cmd]])
        append_rows(MAP03, hdr, rows)
        append_rows(REG03, list(reg.columns), [[me, "", me, "complete", "", cmd]])
        print(f"03 map: {len(rows)} rows appended under run_id {me}")

    # summary
    out = []
    def add(level, v, unit, src_, rids, n=None):
        v = np.asarray(v, float)
        v = v[~np.isnan(v)]
        if not len(v):
            print(f"  {level}: no data, skipped")
            return
        out.append([level, f"{np.median(v):.2f}", unit, f"{np.percentile(v, 5):.2f}", f"{np.percentile(v, 95):.2f}", n if n is not None else len(v),
                    f"{tp:.1f}", src_, rids])
    d = k5.assign(lat_first=k5.first_cycles - tp, lat_repeat=k5.repeat_median - tp)
    d["part"] = d.probe_smid.map(lab.partition)
    near, far = d[d.line_home == d.part], d[d.line_home != d.part]
    same = []
    for p in (0, 1):
        g = groups[p]
        x = near[near.part == p].copy()
        x["smg"], x["lg"] = x.probe_smid.map(g["sm"]), x.line_offset.map(g["line"])
        gm = x.groupby(["smg", "lg"]).lat_first.median().unstack()
        fast = {s: int(gm.loc[s].idxmin()) for s in gm.index}
        same.append(x.assign(same=[fast[s] == l for s, l in zip(x.smg, x.lg)]))
    same = pd.concat(same)
    add("l2_near", near.lat_first, "cycles", "K5 first reads of lines homed in the SM's partition", rid["k5"])
    add("l2_near_same_subnode", same[same.same].lat_first, "cycles", "K5 near first reads, line in the SM's sub-node group", rid["k5"])
    add("l2_near_other_subnode", same[~same.same].lat_first, "cycles", "K5 near first reads, line in the other sub-node group", rid["k5"])
    add("l2_far", far.lat_first, "cycles", "K5 first reads of lines homed in the other partition", rid["k5"])
    quad, partner = quad_table(k5, sig, groups, tp)
    for (pa, po), g in quad.groupby(["partition", "position"]):
        add("l2_" + pa.split()[0] + "_partition_" + po.split()[0] + "_subnode_position", g.c, "cycles",
            f"K5 first reads; sub-node partners across partitions {partner}", rid["k5"])
    add("l2_copy", far.lat_repeat, "cycles", "K5 samples 1-63 of far-homed lines (per-line median), served by the copy", rid["k5"])
    add("l2_home_repeat", near.lat_repeat, "cycles", "K5 samples 1-63 of near-homed lines (per-line median)", rid["k5"])

    rep = read_run("08__l2__replication.csv", rid["k3"])
    cross = rep[~((rep.a_smid == 0) & (rep.b_smid == 8))]
    for ph in ["b_first", "b_repeat", "c_first", "a_after_first", "b_after_a_write_first"]:
        for hm, nm in [("a", "homed_with_A"), ("b", "homed_with_B")]:
            x = cross[cross.phase == ph]
            pa = x.a_smid.map(lab.partition)
            x = x[(x.line_home == pa) if hm == "a" else (x.line_home != pa)]
            add(f"k3_{ph}_{nm}", x.median_cycles - tp, "cycles", "K3 cross-partition triples, per-line values", rid["k3"])

    hb = read_run("08__l2__hashbits.csv", rid["k2"])
    frac, coef = k2_bits(hb)
    bits = [b for b, c in sorted(coef.items()) if c]
    lin = k2_linearity(hb, coef)
    out.append(["partition_hash_bits_7_20", str(len([b for b in bits if b <= 20])), "bits", "", "", int((hb.set == "single").sum()), "",
                "home flips for bits " + " ".join(map(str, bits)) + "; bits 21-30 flip in "
                + ", ".join(f"{b}:{100 * frac[b]:.0f}%" for b in range(21, 31) if b in frac.index)
                + f"; XOR model agreement pair {100 * lin['pair'][0]:.2f}% (n={lin['pair'][1]}), mask {100 * lin['mask'][0]:.2f}% (n={lin['mask'][1]})",
                rid["k2"]])

    cap = read_run("08__l2__capacity.csv", rid["k4"])
    plain = cap[cap.pass_kind == "plain"]
    for F in (16 << 20, 80 << 20):
        add(f"k4_plain_chase_{F >> 20}MiB", plain[plain.footprint_bytes == F].cycles_per_hop, "cycles", "K4 plain pass, all patterns/SMs/reps", rid["k4"])
    curve = plain.groupby("footprint_bytes").cycles_per_hop.median()
    base = curve.iloc[0]
    onset = curve.index[np.argmax(curve.values > 1.1 * base)] if (curve.values > 1.1 * base).any() else np.nan
    out.append(["k4_steady_state_onset", f"{onset / 2**20:.0f}", "MiB", "", "", len(plain), "",
                "smallest footprint whose median plain-pass cycles/hop exceeds 1.1 x the 16 MiB value", rid["k4"]])

    COLS = ["level", "value", "unit", "p5", "p95", "n", "timer_correction_cycles", "source", "source_run_ids"]
    append_rows(f"{R}/08__l2__summary.csv", MANDATORY + COLS, [pre + r for r in out])
    print(pd.DataFrame([r[:6] for r in out], columns=COLS[:6]).to_string(index=False))
    print(sig.groupby(["gpc", "partition", "subnode"]).size().rename("SMs").to_string())


if __name__ == "__main__":
    main()
