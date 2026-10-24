"""Experiment 06 summary: the numbers later experiments (07, 16, 18) consume, from the latest complete full runs (no GPU).

Latency of one dependent 16-B load (K1/K5, cycles per hop; median/p5/p95 over the recorded launches of both probe SMs):
  chase_<variant>_l2    footprints 1–16 MiB (L2-resident; single-SM near plateau)
  chase_<variant>_hbm   footprints ≥ 512 MiB (L2 misses)
  for variant ∈ bulk16, tensor1d, tensor2d, bulk16_evict_last, bulk16_prefetch, bulk16_hint0, bulk16_testwait, cp_async16,
  ld_cg, ld_ca; plus tma_minus_generic_{l2,hbm} = median(bulk16) − median(ld_cg).
mbarrier costs (K2 predicate kernels, cycles per call, median over launches):
  mbar_<form>_pred        wait call on a completed phase minus the glue chain (setp + selp + mad), form ∈ try, test, hint0
  mbar_<form>_incomplete  one call on a phase that never completes (time until it returns false)
  mbar_expect_tx_latency / mbar_expect_tx_issue   arrive.expect_tx(16) dependent / back to back
Completion of one bulk copy (K2, issue → observed completion, timer floor subtracted):
  copy16_{l2,hbm}_try     16-B payload, try_wait loop
  copy_fit_T0, copy_fit_bpc   linear fit of the L2 try_wait medians over payloads ≥ 8 KiB: T0 + bytes / bpc
Issue and admission (K3, one SM):
  issue_gap_1warp         slope of the issue span over N = 1..32, 1 warp, 4-KiB copies, L2 source (cycles per copy)
  issue_gap_4warps        slope over N = 1..16, 4 warps, 4-KiB copies, L2 (aggregate cycles per copy)
  issue_gap_4warps_full   slope over N = 17..32, 4 warps, 4-KiB copies, L2 (issue tracks the data rate)
  batch_bpc_4warps_{l2,hbm}   (32 − 16) · 4 KiB / (completion(32) − completion(16)), 4 warps
Tensor boxes (K4, 16-KiB box, warm descriptor; cycles): box_<box>_<swizzle>_<source>; tensormap_cold_extra_l2 =
  median over boxes of (cold − warm), L2 source.
→ results/06__tma__summary.csv (append; own run_id = time of this summary; source runs in source_run_ids).
"""
import datetime

import numpy as np
import pandas as pd

from common06 import MANDATORY, append_rows, latest_run, read_run

OUT = "results/06__tma__summary.csv"
COLS = ["level", "median", "p5", "p95", "unit", "n", "config", "source", "source_run_ids"]


def q(v):
    v = np.asarray(v, float)
    return np.median(v), np.percentile(v, 5), np.percentile(v, 95), len(v)


def bad_launches():
    c = pd.read_csv("results/06__tma_latency_and_issue__clock_log.csv")
    return c[(c.phase == "launch") & ((c.throttle == 1) | (c.below_lock == 1))][["run_id", "launch_idx"]]


def valid(d, bad):
    k = d.merge(bad.assign(_bad=1), on=["run_id", "launch_idx"], how="left")
    return d[k._bad.isna().values]


def slope(m, lo, hi):
    s = m[(m.index >= lo) & (m.index <= hi)]
    return np.polyfit(s.index, s.values, 1)[0]


def main():
    bad = bad_launches()
    rows = []
    r1 = latest_run("tma_chase", "chase")
    ch = valid(read_run("06__tma__chase.csv", r1), bad)
    regions = {"l2": (1 << 20, 16 << 20), "hbm": (512 << 20, 4 << 30)}
    for v in ["bulk16", "tensor1d", "tensor2d", "bulk16_evict_last", "bulk16_prefetch", "bulk16_hint0", "bulk16_testwait",
              "cp_async16", "ld_cg", "ld_ca"]:
        for reg, (lo, hi) in regions.items():
            d = ch[(ch.variant == v) & (ch.footprint_bytes >= lo) & (ch.footprint_bytes <= hi)]
            m, p5, p95, n = q(d.cycles_per_hop)
            rows.append([f"chase_{v}_{reg}", m, p5, p95, "cycles", n, f"footprint {lo >> 20}–{hi >> 20} MiB, SMs {sorted(d.smid.unique())}",
                         "K1/K5", r1])
    for reg, (lo, hi) in regions.items():
        sel = (ch.footprint_bytes >= lo) & (ch.footprint_bytes <= hi)
        dlt = ch[sel & (ch.variant == "bulk16")].cycles_per_hop.median() - ch[sel & (ch.variant == "ld_cg")].cycles_per_hop.median()
        rows.append([f"tma_minus_generic_{reg}", dlt, np.nan, np.nan, "cycles", np.nan, "median(bulk16) − median(ld_cg)", "K1/K5", r1])

    r2 = latest_run("tma_completion", "k2")
    co = valid(read_run("06__tma__completion.csv", r2), bad)
    for form in ["try", "test", "hint0"]:
        p = co[(co.kind == "pred") & (co.wait_variant == form)].cycles_predicate_only
        b = co[(co.kind == "pred_baseline") & (co.wait_variant == form)].cycles_predicate_only
        rows.append([f"mbar_{form}_pred", p.median() - b.median(), np.nan, np.nan, "cycles", len(p),
                     "dependent chain on a completed phase, glue subtracted", "K2", r2])
        i = co[(co.kind == "pred_incomplete") & (co.wait_variant == form)].cycles_predicate_only
        m, p5, p95, n = q(i)
        rows.append([f"mbar_{form}_incomplete", m, p5, p95, "cycles", n, "one call on a phase that never completes", "K2", r2])
    for kind, name in [("expect_tx_lat", "mbar_expect_tx_latency"), ("expect_tx_issue", "mbar_expect_tx_issue")]:
        m, p5, p95, n = q(co[co.kind == kind].cycles_predicate_only)
        rows.append([name, m, p5, p95, "cycles", n, "arrive.expect_tx(16) on a local barrier", "K2", r2])
    tf = co[co.kind == "timer_floor"].cycles_issue_to_complete.median()
    cp = co[co.kind == "copy"]
    for src in ["l2", "hbm"]:
        d = cp[(cp.source == src) & (cp.wait_variant == "try") & (cp.payload_bytes == 16)].cycles_issue_to_complete - tf
        m, p5, p95, n = q(d)
        rows.append([f"copy16_{src}_try", m, p5, p95, "cycles", n, "issue → observed completion, 16 B, try_wait loop", "K2", r2])
    med = cp[(cp.source == "l2") & (cp.wait_variant == "try")].groupby("payload_bytes").cycles_issue_to_complete.median() - tf
    s = med[med.index >= 8192]
    b, a = np.polyfit(s.index, s.values, 1)
    rows.append(["copy_fit_T0", a, np.nan, np.nan, "cycles", len(s), "fit of L2 try_wait medians over payloads 8–64 KiB", "K2", r2])
    rows.append(["copy_fit_bpc", 1 / b, np.nan, np.nan, "B/cycle/SM", len(s), "fit of L2 try_wait medians over payloads 8–64 KiB", "K2", r2])

    r3 = latest_run("tma_issue", "k3")
    iss = valid(read_run("06__tma__issue.csv", r3, dtype={"issue_cycles_list": str, "warp_t0_list": str}), bad)
    one = iss[iss.sms == 1]
    def med_of(w, z, src, col):
        d = one[(one.issuers == w) & (one.copy_bytes == z) & (one.source == src)]
        return d.groupby("n_in_flight")[col].median()
    sp1, sp4 = med_of(1, 4096, "l2", "issue_span_cycles"), med_of(4, 4096, "l2", "issue_span_cycles")
    rows.append(["issue_gap_1warp", slope(sp1, 1, 32), np.nan, np.nan, "cycles/copy", np.nan, "1 warp, 4 KiB, L2, N 1–32", "K3", r3])
    rows.append(["issue_gap_4warps", slope(sp4, 1, 16), np.nan, np.nan, "cycles/copy", np.nan, "4 warps, 4 KiB, L2, N 1–16", "K3", r3])
    rows.append(["issue_gap_4warps_full", slope(sp4, 17, 32), np.nan, np.nan, "cycles/copy", np.nan, "4 warps, 4 KiB, L2, N 17–32", "K3", r3])
    for src in ["l2", "hbm"]:
        c = med_of(4, 4096, src, "completion_cycles")
        rows.append([f"batch_bpc_4warps_{src}", 16 * 4096 / (c[32] - c[16]), np.nan, np.nan, "B/cycle/SM", np.nan,
                     "4 warps, 4-KiB copies, completion(32) − completion(16)", "K3", r3])

    r4 = latest_run("tma_boxes", "k4")
    bx = valid(read_run("06__tma__boxes.csv", r4), bad)
    bx = bx[bx.encode_ok == 1]
    for (box, sw, src), d in bx[bx.descriptor_state == "warm"].groupby(["box", "swizzle", "source"]):
        m, p5, p95, n = q(d.cycles)
        rows.append([f"box_{box}_{sw}_{src}", m, p5, p95, "cycles", n, "16-KiB tensor box, warm descriptor", "K4", r4])
    D = bx[bx.source == "l2"].groupby(["box", "swizzle", "descriptor_state"]).cycles.median().unstack()
    rows.append(["tensormap_cold_extra_l2", (D.cold - D.warm).median(), (D.cold - D.warm).min(), (D.cold - D.warm).max(), "cycles",
                 len(D), "first use of a tensor map in a launch minus warm, L2 source (min/max over boxes in p5/p95)", "K4", r4])

    now = datetime.datetime.now().astimezone().isoformat(timespec="milliseconds")
    ref = ch.iloc[0]
    pre = [ref.gpu_name, ref.sm_count, ref.sm_clock_mhz, ref.mem_clock_mhz, ref.cuda_version, ref.driver_version, now, 0]
    out = [pre + [r[0]] + [("" if (isinstance(x, float) and np.isnan(x)) else (round(x, 3) if isinstance(x, float) else x)) for x in r[1:]]
           for r in rows]
    append_rows(OUT, MANDATORY + COLS, out)
    df = pd.DataFrame([r[:7] for r in rows], columns=COLS[:7])
    with pd.option_context("display.width", 200, "display.max_rows", 200, "display.max_colwidth", 60):
        print(df.round(2).to_string(index=False))
    print(f"→ {OUT}: {len(out)} rows")


if __name__ == "__main__":
    main()
