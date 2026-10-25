"""Experiment 08 summary from the latest complete runs (no GPU): per-SM partition / sub-node labels and key numbers.

  sm_signature.csv   one row per SM (analysis08.sm_signature on the K5 run): partition, sub-node, near/far first-read
                     medians, same/other sub-node near medians and their difference (subnode_offset), far sub-class
                     medians, repeat (copy) medians. Cycles with the timer pair (calib median − 23) subtracted.
  sm_map.csv         03's map (latest run) with the partition and subnode columns filled; the same rows are appended
                     to ../03__sm_gpc_topology/results/03__sm_gpc_topology__map.csv under a new run_id, which is
                     registered as complete in 03's run registry (skipped if the latest 03 map already came from this
                     K5 run; never with --quick).
  summary.csv        level, value, unit, p5, p95, n: L2 near / same / other sub-node / far / copy latencies (K5, every
                     first-read sample), first reads by same/other partition × same/other sub-node position, the
                     permutation test of the sub-node split, K3 phase medians (cross-partition triples of the latest K3
                     run, replication_v2.csv), partition hash bits (K2: in-page bits; page term of the 2 GiB buffer), the
                     page term on cuMemCreate allocations (k2vmm), K4 plain-chase cycles per hop, the footprint where the
                     steady-state chase leaves the near plateau, the K4 access classes (Gaussian mixture) and their shares,
                     and the first-read counters per line (ncu_fabric1, all-exit baseline subtracted).
All three append with their own run_id (time of this summary); source runs in source_run_ids.
"""
import argparse
import datetime

import numpy as np
import pandas as pd

from analysis08 import (MAP03, fabric1, gmm1d, k2_bits, k2_first_reads, k2_linearity, k2_page_terms, k2vmm, load_k5, quad_table,
                        share_null, sm_signature)
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
                                               ("l2_probe", "k2vmm"), ("l2_capacity", "k4")]}
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
    last["notes"] = [n if note in n else (n + "; " if n else "") + note for n in last.notes]  # the 03 map may carry it already
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

    rep = read_run("08__l2__replication_v2.csv", rid["k3"])
    cross = rep[rep.triple_kind == "cross"]
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
    inpage = [b for b in bits if b <= 20]
    out.append(["partition_hash_bits_7_20", str(len(inpage)), "bits", "", "", int((hb.set == "single").sum()), "",
                "in-page home flips for bits " + " ".join(map(str, inpage))
                + f"; XOR model agreement pair {100 * lin['pair'][0]:.2f}% (n={lin['pair'][1]}), mask {100 * lin['mask'][0]:.2f}% (n={lin['mask'][1]});"
                " above bit 20 one term per 2 MiB physical page (rows k2_page_term_*, k2vmm_*)", rid["k2"]])
    pt = k2_page_terms(k2_first_reads(rid["k2"]), coef)
    out.append(["k2_page_term_consistent_pages", f"{100 * pt['pages_consistent']:.2f}", "%", "", "", pt["n_pages"], "",
                f"2 GiB cudaMalloc buffer: pages whose {pt['offsets_per_page']} probed offsets give one page term; g = 1 in "
                f"{100 * pt['g1']:.1f}% of pages", rid["k2"]])
    out.append(["k2_page_term_flip_bits_21_30", f"{pt['flips']['g differs [%]'].loc[21]:.1f}", "%", "", "", int(pt["flips"]["page pairs"].iloc[0]), "",
                "2 GiB cudaMalloc buffer, page pairs differing in page-index bit b whose page term differs: "
                + ", ".join(f"{b}:{v:.1f}%" for b, v in pt["flips"]["g differs [%]"].items())
                + " (properties of this allocation's physical page layout, not hash coefficients; value = bit 21)", rid["k2"]])

    kv = k2vmm(read_run("08__l2__k2vmm_pages.csv", rid["k2vmm"]))
    fl = kv["flips"]
    out.append(["k2vmm_page_term_same_per_handle", f"{100 * min(kv['agree'].values()):.1f}", "%", "", "", len(kv["handles"]), "",
                "64 cuMemCreate 2 MiB handles mapped at 3 VAs (A2: VA bits 21,22,24 flipped; A3: permuted): share of handles with the same "
                f"page term in A1 and A2 ({100 * kv['agree']['A2']:.0f}%) / A3 ({100 * kv['agree']['A3']:.0f}%); all 8 lines of every page agree in "
                f"{kv['consistency']['rows with all 8 lines agreeing [%]'].min():.0f}% of rows", rid["k2vmm"]])
    out.append(["k2vmm_page_term_flip_bits_21_22", f"{min(fl[21].min(), fl[22].min()):.1f}", "%", "", "", len(fl), "",
                "min over allocations (4x32 MiB, 2x64 MiB cuMemCreate, 256 MiB cudaMalloc, handles in creation order) of the share of page "
                "pairs differing in bit 21 or 22 whose page term differs; bit 23: " + ", ".join(f"{a.split()[0]} {v:.0f}%" for a, v in fl[23].items())
                + "; bit 24: " + ", ".join(f"{a.split()[0]} {v:.0f}%" for a, v in fl[24].items()), rid["k2vmm"]])

    cap = read_run("08__l2__capacity.csv", rid["k4"])
    plain = cap[cap.pass_kind == "plain"]
    for F in (16 << 20, 80 << 20):
        add(f"k4_plain_chase_{F >> 20}MiB", plain[plain.footprint_bytes == F].cycles_per_hop, "cycles", "K4 plain pass, all patterns/SMs/reps", rid["k4"])
    curve = plain.groupby("footprint_bytes").cycles_per_hop.median()
    base = curve.iloc[0]
    onset = curve.index[np.argmax(curve.values > 1.1 * base)] if (curve.values > 1.1 * base).any() else np.nan
    out.append(["k4_steady_state_onset", f"{onset / 2**20:.0f}", "MiB", "", "", len(plain), "",
                "smallest footprint whose median plain-pass cycles/hop exceeds 1.1 x the 16 MiB value", rid["k4"]])

    # K4 access classes: 4-component Gaussian mixture of every pass-3 access (all footprints, both patterns, SMs 0 and 2)
    hist = read_run("08__l2__capacity_hist.csv", rid["k4"])
    hist["cyc"] = hist.cycles - tp
    h2 = hist[hist["pass"] == 2].groupby("cyc")["count"].sum()
    gm = gmm1d(h2.index.values, h2.values, k=4, n_starts=50, seed=0)
    cls_names = ["near_hit", "far_hit", "hbm_near_class", "hbm_far_class"]
    for k, nm in enumerate(cls_names):
        out.append([f"k4_class_{nm}_mean", f"{gm['mean'][k]:.1f}", "cycles", "", "", int(h2.sum()), f"{tp:.1f}",
                    f"K4 pass 3, 4-component Gaussian mixture (EM, best of 50 starts): SD {gm['sd'][k]:.1f}, weight {100 * gm['weight'][k]:.1f}%"
                    + (f", upper boundary (equal posterior) {gm['boundaries'][k]:.1f}" if k < 3 else ""), rid["k4"]])
    hist["cls"] = np.searchsorted(gm["boundaries"], hist.cyc.values)
    for lo, hi in [(38, 52), (64, 80)]:
        x = hist[(hist["pass"] == 2) & (hist.footprint_bytes >= lo << 20) & (hist.footprint_bytes <= hi << 20)]
        sh = 100 * x.groupby("cls")["count"].sum() / x["count"].sum()
        for k, nm in enumerate(cls_names):
            if k == 0:
                continue
            out.append([f"k4_share_{nm}_{lo}_{hi}MiB", f"{sh.get(k, 0.0):.2f}", "%", "", "", int(x["count"].sum()), f"{tp:.1f}",
                        f"K4 pass 3, footprints {lo}-{hi} MiB, both touch patterns: share of accesses in the class", rid["k4"]])

    # sub-node split vs permutation null (1000 permutations, seed 0, lines permuted within each SM row)
    dd = k5.assign(lat_first=k5.first_cycles - tp, lat_repeat=k5.repeat_median - tp)
    obs_all, null_all = [], []
    for p in (0, 1):
        sms = sig.smid[sig.partition == p]
        nl_ = dd.loc[dd.line_home == p, "line_offset"].unique()
        fl_ = dd.loc[dd.line_home != p, "line_offset"].unique()
        for lines_, val in [(nl_, "lat_first"), (fl_, "lat_first"), (fl_, "lat_repeat")]:
            o, nn = share_null(dd, sms, lines_, val, n_perm=1000, seed=0, axis="row")
            obs_all.append(o)
            null_all.append(nn)
    out.append(["subnode_split_first_component_share_min", f"{100 * min(obs_all):.1f}", "%", "", "", 6, "",
                f"K5 SM x line matrices (near, far, copy; both partitions): observed {100 * min(obs_all):.1f}-{100 * max(obs_all):.1f}%; "
                f"null (lines permuted within each SM, 1000 each): median {100 * min(np.median(n) for n in null_all):.1f}-"
                f"{100 * max(np.median(n) for n in null_all):.1f}%, p99 <= {100 * max(np.percentile(n, 99) for n in null_all):.1f}%, "
                f"max {100 * max(n.max() for n in null_all):.1f}%; p = 1/1001 for every matrix" if all((n < o).all() for o, n in zip(obs_all, null_all))
                else "see notebook", rid["k5"]])

    # first-read counters per line (ncu_fabric1, one profiled kernel per binary run, all-exit baseline subtracted)
    per, raw, frids = fabric1()
    for w in ("far_first", "near_first", "far_second"):
        for c, nm in [("L2 lookup misses", "lookup_miss"), ("L2 lookup hits", "lookup_hit"), ("partition-link sectors", "link")]:
            out.append([f"ncu_{w}_{nm}_sectors_per_line", f"{per.loc[w, c]:.3f}", "sectors/line", "", "", int(per.loc[w, "lines"]), "",
                        f"ncu_fabric1, SM 2 reads lines SM 0 wrote, once; all-exit baseline subtracted; each line also reads its page's "
                        f"TLB-warm line (1 sector) and 0.5 sector of warmed lists; HBM {per.loc[w, 'HBM bytes (min–max of the 3 runs)']} B/line",
                        f"{frids[0]}..{frids[-1]}"])

    COLS = ["level", "value", "unit", "p5", "p95", "n", "timer_correction_cycles", "source", "source_run_ids"]
    append_rows(f"{R}/08__l2__summary.csv", MANDATORY + COLS, [pre + r for r in out])
    print(pd.DataFrame([r[:6] for r in out], columns=COLS[:6]).to_string(index=False))
    print(sig.groupby(["gpc", "partition", "subnode"]).size().rename("SMs").to_string())


if __name__ == "__main__":
    main()
