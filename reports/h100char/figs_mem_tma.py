"""Figures of the memory-hierarchy and TMA sections.

Every function returns (Figure, info): info holds the numbers the text quotes, computed from the local CSVs.
Data provenance (run ids pinned where a later run of the same file belongs to another card or campaign):
- footprint sweep, latency ladder, bandwidth ladder (SMEM/L1/L2/HBM), L2 NUCA, TMA: card A (serial ...9411);
- DSMEM latency and bandwidth: card B (serial ...8254). Both cards gave identical topology and the same
  generic-load latencies to < 1 % where both were measured (L2 chase 269 / HBM chase 618).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from . import data as D
from . import style as S
from .style import PAL, GREY, LIGHTGREY, INK, INK2, MUTED, COL, FULL

# ---- runs -------------------------------------------------------------------------------------------------------
RUN_FOOTPRINT = "2026-10-24T01:12:33.113"   # 04 K2: ca/cg/cv x 72 footprints x SMs 0,2,4,8
RUN_DRAM = "2026-10-24T02:48:27.349"        # 05 K4 HBM sweep (later runs in the file: card B re-runs)
RUN_L2 = "2026-10-24T02:48:03.494"          # 05 K3 L2 sweep
RUN_COMPLETION = "2026-10-24T17:37:55.878"  # 06 K2 (latest complete `tma_completion k2`)
RUN_ISSUE = "2026-10-24T17:38:07.220"       # 06 K3 (latest complete `tma_issue k3`)
RUN_11_C = "2026-10-26T05:30:46.286"        # 11 step C (patterns per CS, single cluster)
RUN_11_E = "2026-10-26T05:31:22.231"        # 11 step E (TMA command size)
SMS = 132
SEND_LIMIT = 64 / 3   # B/cycle: 2/3 of the 32 B/cycle rated DSMEM port; measured one-way rates sit at 21.1-21.5
HBM_PEAK = 3.94       # TB/s: 2 x 2619 MHz x 6016 bit / 8 (device constants, not a measurement)


def _valid(d):
    return d[d.discard.isna() | (d.discard.astype(str) == "")]


def _r(x, n=1):
    return float(round(float(x), n))


def _label(ax, x, y, s, **kw):
    kw.setdefault("fontsize", 7)
    kw.setdefault("color", INK2)
    ax.text(x, y, s, **kw)


# =================================================================================================================
# §1 memory hierarchy
# =================================================================================================================
def _footprint():
    d = D.pinned(D.read(D.res("04", "04__latency__footprint.csv")), RUN_FOOTPRINT)
    return d.groupby(["load_variant", "footprint_bytes"]).cycles_per_hop.median().unstack(0)


def fig_latency_footprint():
    m = _footprint()
    KiB, MiB = 1 << 10, 1 << 20
    w = lambda c, lo, hi: m.loc[(m.index >= lo) & (m.index <= hi), c].median()
    l1 = w("ca", 1 * KiB, 128 * KiB)
    near = w("cg", 1 * MiB, 16 * MiB)
    mixed = w("cg", 35 * MiB, 51 * MiB)
    hbm = w("cg", 256 * MiB, 512 * MiB)
    hbm_2g = m.cg.iloc[-1]
    l1_last = m.index[(m.ca <= l1 + 1) & (m.index <= 1 * MiB)].max()
    l1_first_miss = m.index[m.index > l1_last].min()
    near_last = m.index[(m.cg <= near * 1.02) & (m.index >= 1 * MiB)].max()
    near_first_out = m.index[m.index > near_last].min()

    f, ax = S.fig(width=COL, height=2.35)
    x = m.index.values
    ax.plot(x, m.cg, color=GREY, lw=1.3, ls=(0, (3, 1.5)), zorder=2)
    ax.plot(x, m.ca, color=PAL[0], lw=1.5, zorder=3)
    S.bytes_axis(ax)
    ax.xaxis.set_major_locator(FixedLocator([2 ** k for k in (10, 15, 20, 25, 30)]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(x.min() / 1.3, x.max() * 1.3)
    ax.set_ylim(0, 720)
    ax.set_xlabel("footprint (one SM, random cycle of 128-B lines)")
    ax.set_ylabel("latency per dependent load (cycles)")
    S.ns_axis(ax)
    # capacity markers
    for v, s in [(224 * KiB, "L1 224 KiB"), (30 * MiB, "½ L2")]:
        ax.axvline(v, ls=":", lw=0.9, color=MUTED, zorder=1)
        ax.text(v * 1.12, 712, s, fontsize=6.5, color=MUTED, ha="left", va="top")
    # plateau labels
    _label(ax, 1.4 * KiB, l1 + 22, f"L1 hit {l1:.0f}  (ld.ca)", color=PAL[0])
    _label(ax, 1.4 * KiB, near + 22, f"L2 {near:.0f}  (ld.cg: L1 bypassed)", color=INK2)
    _label(ax, 20 * MiB, mixed + 18, f"near/far\nmix {mixed:.0f}", ha="right", va="bottom", linespacing=0.95)
    _label(ax, 1.5 * (1 << 30), hbm - 22, f"HBM {hbm:.0f}", ha="right", va="top")
    info = dict(l1_plateau=_r(l1), l2_near_plateau=_r(near), l2_mixed_plateau_35_51MiB=_r(mixed),
                hbm_plateau_256_512MiB=_r(hbm), hbm_at_2GiB=_r(hbm_2g),
                l1_last_hit_KiB=_r(l1_last / KiB), l1_first_miss_KiB=_r(l1_first_miss / KiB),
                ca_at_first_miss=_r(m.ca[l1_first_miss]),
                near_last_flat_MiB=_r(near_last / MiB), near_first_above_MiB=_r(near_first_out / MiB),
                cg_minus_ca_above_L1=_r((m.cg - m.ca)[m.index > 512 * KiB].abs().max(), 2),
                ns_l2=_r(near / S.CYC_PER_NS), ns_hbm=_r(hbm / S.CYC_PER_NS))
    return f, info


def _lat_ladder():
    s = D.latest(D.read(D.res("04", "04__latency__summary_v2.csv"))).set_index("level")
    k = D.latest(D.read(D.res("09", "09__dsmem_latency__summary.csv")))
    ds = k[k.level == "k1/ld_cluster/stride1/Default/remote"].set_index("config")["median"]
    return s, ds


def _bw_ladder():
    b = D.pinned(D.read(D.res("05", "05__bandwidth__summary.csv")), D.PIN["05_summary"]).set_index("level")
    g = D.latest(D.read(D.res("11", "11__dsmem_bw__summary.csv")))
    gw = g[g.level.str.match(r"gpu_wide_S_(ld|st|tma_put)_cs\d+$")].copy()
    gw["op"] = gw.level.str.extract(r"gpu_wide_S_(\w+?)_cs")[0]
    gw["cs"] = gw.level.str.extract(r"_cs(\d+)$")[0].astype(int)
    pu = g[g.level.str.match(r"pattern_pair_uni1_(ld_ld_issuer|st_st_issuer|tma_put_tma_put_target)_cs8$")].value
    return b, gw, pu


def fig_ladders():
    s, ds = _lat_ladder()
    b, gw, pu = _bw_ladder()
    f, (a1, a2) = S.fig(ncols=2, width=FULL, height=2.55)

    # (a) latency ladder ------------------------------------------------------------------------------------
    c = lambda lv: s.loc[lv, "median_cycles_at_1635"]
    rows = [("SMEM (ld.shared)", c("smem"), PAL[0], None),
            ("own SMEM via cluster window", c("smem_dsmem_if"), LIGHTGREY, None),
            ("L1 hit", c("l1"), PAL[0], None),
            ("DSMEM (other SM of cluster)", ds.median(), PAL[1], (ds.min(), ds.max())),
            ("L2, near partition", c("l2_near"), PAL[0], None),
            ("L2, far partition", c("l2_far"), PAL[0], None),
            ("HBM, near partition", c("dram_near"), PAL[0], None),
            ("HBM, far partition", c("dram_far"), PAL[0], None)]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v, col, rng) in zip(y, rows):
        a1.barh(yi, v, height=0.62, color=col, zorder=2)
        txt = f"{v:.0f}" if not rng else f"{v:.0f}  (CS 2–16: {rng[0]:.0f}–{rng[1]:.0f})"
        a1.text(v + 10, yi, txt, va="center", fontsize=7, color=INK2)
    a1.set_yticks(y, [r[0] for r in rows])
    a1.tick_params(axis="y", length=0)
    a1.set_xlim(0, 780)
    a1.set_xlabel("latency of a dependent load (SM cycles)")
    a1.grid(axis="y", visible=False)
    S.ns_axis(a1, which="x")
    S.panel(a1, "a", x=-0.5)

    # (b) bandwidth ladder ----------------------------------------------------------------------------------
    t = lambda lv: b.loc[lv, "tbps"]
    p = lambda lv: b.loc[lv, "bpc_per_sm"]
    dmin, dmax, dmed = gw.value.min(), gw.value.max(), gw.value.median()
    brow = [("SMEM load", t("smem_ld"), PAL[0], None, f"{p('smem_ld'):.0f} B/cyc/SM"),
            ("L1 load", t("l1_ld"), PAL[0], None, f"{p('l1_ld'):.0f}"),
            ("DSMEM (ld/st/TMA, CS 2–16)", dmed, PAL[1], (dmin, dmax), f"≤ {pu.median():.1f} per sender"),
            ("L2 read", t("l2_ld_mixed"), PAL[0], None, f"{p('l2_ld_mixed'):.0f}"),
            ("L2 store", t("l2_st_mixed"), PAL[0], None, f"{p('l2_st_mixed'):.0f}"),
            ("HBM read", t("dram_read"), PAL[0], None, f"{p('dram_read'):.0f}"),
            ("HBM write", t("dram_write"), PAL[0], None, f"{p('dram_write'):.0f}")]
    y = np.arange(len(brow))[::-1]
    L0 = 0.5
    for yi, (lab, v, col, rng, ps) in zip(y, brow):
        a2.barh(yi, v - L0, left=L0, height=0.62, color=col, zorder=2)
        right = v
        if rng:
            a2.plot(rng, [yi, yi], color=INK, lw=0.9, zorder=3)
            for e in rng:
                a2.plot([e, e], [yi - .14, yi + .14], color=INK, lw=0.9, zorder=3)
            right = rng[1] * 1.12     # clear the whisker's end cap
            txt = f"{rng[0]:.1f}–{rng[1]:.1f} TB/s · {ps}"
        else:
            txt = f"{v:.2f} TB/s · {ps}" if v < 10 else f"{v:.1f} TB/s · {ps}"
        a2.text(right * 1.15, yi, txt, va="center", fontsize=7, color=INK2)
    a2.set_xscale("log")
    a2.set_xlim(L0, 400)
    a2.xaxis.set_major_locator(FixedLocator([1, 3, 10, 30]))
    a2.xaxis.set_minor_locator(NullLocator())
    a2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a2.set_yticks(y, [r[0] for r in brow])
    a2.tick_params(axis="y", length=0)
    a2.grid(axis="y", visible=False)
    a2.set_xlabel("GPU-wide bandwidth (TB/s, log scale); labels: B/cycle per SM")
    a2.spines["bottom"].set_bounds(L0, 50)
    S.panel(a2, "b", x=-0.42)

    info = dict(
        lat_cycles={k: _r(v) for k, v in [("smem", c("smem")), ("own_smem_via_window", c("smem_dsmem_if")), ("l1", c("l1")),
                                          ("dsmem_median_cs2_16", ds.median()), ("l2_near", c("l2_near")),
                                          ("l2_far", c("l2_far")), ("hbm_near", c("dram_near")), ("hbm_far", c("dram_far"))]},
        dsmem_by_cs={k: _r(v) for k, v in ds.items()},
        dsmem_over_l2_near=_r(ds.median() / c("l2_near"), 2),
        bw_tbps={k: _r(t(k), 2) for k in ["smem_ld", "l1_ld", "l2_ld_mixed", "l2_st_mixed", "dram_read", "dram_write"]},
        bw_bpc_per_sm={k: _r(p(k), 1) for k in ["smem_ld", "l1_ld", "l2_ld_mixed", "l2_st_mixed", "dram_read", "dram_write"]},
        dsmem_gpu_wide_tbps=dict(min=_r(dmin, 2), median=_r(dmed, 2), max=_r(dmax, 2),
                                 min_at=gw.loc[gw.value.idxmin(), "level"], max_at=gw.loc[gw.value.idxmax(), "level"]),
        dsmem_one_way_pair_bpc=dict(zip(["ld", "st", "tma_put"], [_r(v, 2) for v in pu.values])),
        dsmem_send_limit_bpc=_r(SEND_LIMIT, 2))
    return f, info


def _hbm_l2_scaling():
    c05 = D.import_file(D.res("05", "../src/common05.py"), "common05")
    floor = c05.event_floor_ms(path=D.res("01", "01__timers__event_floor.csv"))
    dr = D.pinned(D.read(D.res("05", "05__bandwidth__dram.csv")), RUN_DRAM)
    dr["tbps"] = c05.net_tbps(dr, floor)
    v = _valid(dr)
    sw = v[(v.threads == 1024) & (v.blocks <= SMS) & (v.ilp == 4) & (v.cache_op == "cg")]
    hbm = sw.groupby(["op", "blocks"]).tbps.median().unstack(0)
    full = v[(v.blocks == 4 * SMS) & (v.ilp == 4) & (v.cache_op == "cg")].groupby("op").tbps.median()
    l2 = D.pinned(D.read(D.res("05", "05__bandwidth__l2.csv")), RUN_L2)
    l2["tbps"] = c05.net_tbps(l2, floor)
    v = _valid(l2)
    a = v[(v["pass"] == 0) & (v.variant == "mixed") & (v.op == "ld") & (v.cache_op == "cg") & (v.ilp == 4)
          & (v.threads == 1024) & (v.blocks <= SMS) & (v.footprint_bytes == 16 << 20)]
    l2m = a.groupby("blocks").tbps.median()
    l2bpc = a.groupby("blocks").bpc_chip.median() if "bpc_chip" in a else None
    l2full = v[(v["pass"] == 0) & (v.variant == "mixed") & (v.op == "ld") & (v.cache_op == "cg") & (v.ilp == 4)
               & (v.blocks == 2 * SMS) & (v.footprint_bytes == 16 << 20)].tbps.median()
    return hbm, full, l2m, l2full, floor, len(dr) - len(_valid(dr))


def fig_l2_nuca():
    sig = D.latest(D.read(D.res("08", "08__l2__sm_signature.csv")))
    q = D.latest(D.read(D.res("08", "08__l2__summary.csv"))).set_index("level").value
    grid = {"same": q["l2_same_partition_same_subnode_position"], "other": q["l2_same_partition_other_subnode_position"],
            "far_partner": q["l2_other_partition_same_subnode_position"],
            "far_other": q["l2_other_partition_other_subnode_position"]}
    hbm, full, l2m, l2full, floor, ndisc = _hbm_l2_scaling()

    f, (a1, a2) = S.fig(ncols=2, width=FULL, height=2.6, gridspec_kw={"width_ratios": [1.25, 1]})
    # (a) per-SM first-read latency: four classes per GPC, colour = partition -------------------------------
    order = sig.groupby("gpc").partition.first().sort_values(kind="stable").index.tolist()
    cls = [("near_same_subnode_median", "same partition, same sub-node", grid["same"]),
           ("near_other_subnode_median", "same partition, other sub-node", grid["other"]),
           ("far_lo_median", "other partition, partner sub-node", grid["far_partner"]),
           ("far_hi_median", "other partition, other sub-node", grid["far_other"])]
    offs = np.array([-0.27, -0.09, 0.09, 0.27])
    rng = np.random.default_rng(0)
    for i, g in enumerate(order):
        x = sig[sig.gpc == g]
        col = PAL[0] if x.partition.iloc[0] == 0 else PAL[2]
        for o, (cname, _, _) in zip(offs, cls):
            a1.scatter(i + o + rng.uniform(-0.05, 0.05, len(x)), x[cname], s=5, color=col, lw=0, alpha=0.85, zorder=3)
    for cname, lab, gv in cls:
        a1.axhline(gv, color=LIGHTGREY, lw=0.8, zorder=1)
        a1.text(len(order) - 0.45, gv, f"{gv:.0f}  {lab}", fontsize=6.5, color=INK2, va="center", ha="left")
    a1.set_xticks(range(len(order)), [f"G{g}" for g in order])
    a1.set_xlim(-0.6, len(order) - 0.5)
    a1.set_ylim(225, 475)
    a1.set_ylabel("first read of a line held only at its home (cycles)")
    a1.set_xlabel("GPC (SMs as dots; four line classes per GPC)")
    a1.grid(axis="x", visible=False)
    n0 = sum(sig.groupby("gpc").partition.first() == 0)
    a1.text((n0 - 1) / 2, 350, "SMs of L2 partition 0", color=PAL[0], fontsize=7, ha="center", va="center")
    a1.text(n0 + (len(order) - n0 - 1) / 2, 350, "SMs of L2 partition 1", color=PAL[2], fontsize=7, ha="center", va="center")
    a1.axvline(n0 - 0.5, color=S.GRID, lw=0.8)
    S.panel(a1, "a")
    # make room for the class labels on the right
    a1.spines["right"].set_visible(False)

    # (b) bandwidth vs active SMs ---------------------------------------------------------------------------
    r_full = full["read"]
    knee = int(hbm.read[hbm.read >= 0.9 * r_full].index.min())
    a2.plot(l2m.index, l2m.values, color=PAL[0], lw=1.4, marker="o", ms=2.5)
    a2.plot(hbm.index, hbm.read, color=PAL[1], lw=1.4)
    a2.plot(hbm.index, hbm.write, color=PAL[1], lw=1.1, ls=(0, (3, 1.5)))
    S.ref_line(a2, HBM_PEAK, f"HBM peak {HBM_PEAK:.2f}", pos=0.99, ha="right", va="bottom", fs=6.5)
    a2.plot([knee], [hbm.read[knee]], "o", ms=5, mfc="none", mec=INK, mew=0.8, zorder=4)
    a2.annotate(f"90 % of full-grid read\nat {knee} SMs", (knee, hbm.read[knee]), xytext=(knee + 6, 1.6),
                fontsize=6.5, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    _label(a2, 70, 7.1, "L2 read (16 MiB)", color=PAL[0], ha="left", va="top")
    _label(a2, 55, 3.3, "HBM read", color=PAL[1], ha="right", va="bottom")
    _label(a2, 30, hbm.write[30] + 0.0, "HBM write", color=PAL[1], ha="left", va="top")
    a2.set_xlim(0, 134)
    a2.set_ylim(0, 8.6)
    a2.set_xlabel("active SMs (1 block × 1024 threads each)")
    a2.set_ylabel("GPU-wide bandwidth (TB/s)")
    S.panel(a2, "b")

    per_sm_l2 = (l2m / l2m.index * 1e12 / (S.SM_MHZ * 1e6))
    info = dict(
        grid_cycles={k: _r(v) for k, v in grid.items()},
        subnode_step=_r(grid["other"] - grid["same"]), partition_step=_r(grid["far_partner"] - grid["same"]),
        far_subnode_step=_r(grid["far_other"] - grid["far_partner"]),
        per_sm_near_range=[_r(sig.near_median.min()), _r(sig.near_median.max())],
        per_sm_far_range=[_r(sig.far_median.min()), _r(sig.far_median.max())],
        per_sm_same_subnode_range=[_r(sig.near_same_subnode_median.min()), _r(sig.near_same_subnode_median.max())],
        partitions={int(p): sorted(int(g) for g in sig[sig.partition == p].gpc.unique()) for p in (0, 1)},
        subnodes={int(s): sorted(int(g) for g in sig[sig.subnode == s].gpc.unique()) for s in sorted(sig.subnode.unique())},
        hbm_read_full_grid_tbps=_r(r_full, 2), hbm_write_full_grid_tbps=_r(full["write"], 2),
        hbm_read_132_blocks_tbps=_r(hbm.read.iloc[-1], 2), hbm_write_132_blocks_tbps=_r(hbm.write.iloc[-1], 2),
        hbm_read_1sm_gbps=_r(hbm.read[1] * 1e3), hbm_write_1sm_gbps=_r(hbm.write[1] * 1e3),
        hbm_read_1sm_bpc=_r(hbm.read[1] * 1e12 / (S.SM_MHZ * 1e6)),
        hbm_read_90pct_sms=knee, hbm_read_pct_of_peak=_r(100 * r_full / HBM_PEAK),
        hbm_write_dip=dict(at_70=_r(hbm.write[70], 2), at_71=_r(hbm.write[71], 2), at_86=_r(hbm.write[86], 2),
                           at_87=_r(hbm.write[87], 2)),
        l2_read_1sm_tbps=_r(l2m.iloc[0], 3), l2_read_1sm_bpc=_r(per_sm_l2.iloc[0]),
        l2_read_64sms_tbps=_r(l2m[64], 2), l2_read_132_blocks_tbps=_r(l2m.iloc[-1], 2),
        l2_read_full_grid_tbps=_r(l2full, 2), event_floor_us=_r(floor * 1e3, 2), dram_discarded_rows=int(ndisc))
    return f, info



# =================================================================================================================
# §3 TMA
# =================================================================================================================
def _completion():
    co = D.pinned(D.read(D.res("06", "06__tma__completion.csv")), RUN_COMPLETION)
    cl = D.read(D.res("06", "06__tma_latency_and_issue__clock_log.csv"))
    bad = cl[(cl.phase == "launch") & ((cl.throttle == 1) | (cl.below_lock == 1))][["run_id", "launch_idx"]]
    k = co.merge(bad.assign(_bad=1), on=["run_id", "launch_idx"], how="left")
    nbad = int(k._bad.notna().sum())
    co = co[k._bad.isna().values]
    tf = co[co.kind == "timer_floor"].cycles_issue_to_complete.median()
    cp = co[(co.kind == "copy") & (co.wait_variant == "try")]
    g = cp.groupby(["source", "payload_bytes"]).cycles_issue_to_complete
    m = (g.median() - tf).unstack(0)
    lo = (g.quantile(.05) - tf).unstack(0)
    hi = (g.quantile(.95) - tf).unstack(0)
    return m, lo, hi, tf, nbad


def fig_tma_copy_latency():
    m, lo, hi, tf, nbad = _completion()
    l2 = m.l2.dropna()
    fit = l2[l2.index >= 8192]
    A = np.vstack([np.ones(len(fit)), fit.index.values]).T
    (T0, sl), *_ = np.linalg.lstsq(A, fit.values, rcond=None)
    B = 1 / sl
    sm = D.latest(D.read(D.res("06", "06__tma__summary.csv"))).set_index("level")["median"]
    ld_l2, ld_hbm = sm["chase_ld_cg_l2"], sm["chase_ld_cg_hbm"]
    tma_l2, tma_hbm = sm["chase_bulk16_l2"], sm["chase_bulk16_hbm"]
    hb = m.hbm.dropna()

    f, ax = S.fig(width=COL, height=2.45)
    x = l2.index.values
    ax.fill_between(x, lo.l2[x], hi.l2[x], color=PAL[0], alpha=0.15, lw=0)
    ax.plot(x, l2.values, color=PAL[0], lw=1.4)
    ax.fill_between(hb.index, lo.hbm[hb.index], hi.hbm[hb.index], color=PAL[1], alpha=0.13, lw=0)
    ax.plot(hb.index, hb.values, color=PAL[1], lw=1.1, marker="o", ms=2.6, zorder=4)
    xx = np.geomspace(2048, 65536, 40)
    ax.plot(xx, T0 + xx / B, color=INK, lw=0.9, ls=(0, (1, 1.5)), zorder=5)
    S.bytes_axis(ax)
    ax.xaxis.set_major_locator(FixedLocator([16, 128, 1024, 8192, 65536]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(12, 90000)
    ax.set_ylim(0, 1480)
    S.ref_line(ax, ld_l2, f"ld.global L2 {ld_l2:.0f}", pos=0.02, ha="left", va="top", fs=6.5)
    S.ref_line(ax, ld_hbm, f"ld.global HBM {ld_hbm:.0f}", pos=0.02, ha="left", va="bottom", fs=6.5)
    _label(ax, 64, l2[64] + 25, "from L2", color=PAL[0], ha="center", va="bottom")
    _label(ax, 64, hi.hbm[64] + 25, "from HBM", color=PAL[1], ha="center", va="bottom")
    _label(ax, 500, 215, f"dotted: {T0:.0f} + bytes / {B:.0f}\n(fit over ≥ 8 KiB)", color=INK, ha="left", va="top", fontsize=6.5)
    ax.set_xlabel("payload of one cp.async.bulk")
    ax.set_ylabel("issue → mbarrier completion (cycles)")
    S.ns_axis(ax)
    small = l2[l2.index <= 1024]
    info = dict(T0=_r(T0), B_bytes_per_cycle=_r(B), B_GBps=_r(B * S.CYC_PER_NS),
                l2_16B=_r(l2[16]), l2_le_1KiB_range=[_r(small.min()), _r(small.max())],
                l2_64KiB=_r(l2[65536]), hbm_16B=_r(hb[16]), hbm_64KiB=_r(hb[65536]),
                hbm_minus_l2_16B=_r(hb[16] - l2[16]), hbm_minus_l2_64KiB=_r(hb[65536] - l2[65536]),
                timer_floor_subtracted=_r(tf), discarded_launches=nbad,
                chase_ld_cg=dict(l2=_r(ld_l2), hbm=_r(ld_hbm)), chase_tma16=dict(l2=_r(tma_l2), hbm=_r(tma_hbm)),
                tma_minus_generic=dict(l2=_r(tma_l2 - ld_l2), hbm=_r(tma_hbm - ld_hbm)),
                half_bandwidth_payload_B=_r(T0 * B, 0))
    return f, info


def _issue():
    cols = ["launch_idx", "n_in_flight", "issuers", "sms", "source", "copy_bytes", "issue_span_cycles"]
    I = D.read_run(D.res("06", "06__tma__issue.csv"), RUN_ISSUE, usecols=cols)
    d = I[(I.sms == 1) & (I.copy_bytes == 4096)]
    return d.groupby(["source", "issuers", "n_in_flight"]).issue_span_cycles.median()


def _tma_s2s_issue():
    """Issuer records of 11's TMA shared→shared kernels (raw samples are 0.5 GB: reduce once and cache)."""
    def build():
        cols = ["step", "kernel", "pattern", "cs", "side", "launch_mode", "payload_bytes", "chunk_bytes", "cycles"]
        q = ('side == "issuer" & ((step == "C" & kernel == "tma_put" & launch_mode == "single" & '
             'pattern in ["ring", "all_to_all", "broadcast_out"]) | (step == "E" & payload_bytes == 98304))')
        d = D.read_run(D.res("11", "11__dsmem_bw__samples.csv"), None, usecols=cols, query=q)
        d = d[d.run_id.str.startswith(RUN_11_C) | d.run_id.str.startswith(RUN_11_E)]
        return (d.groupby(["step", "kernel", "pattern", "cs", "payload_bytes", "chunk_bytes"]).cycles
                .agg(["median", "count"]).reset_index())
    return D.cached("mem_tma__11_tma_issuer.csv", build)


def fig_tma_issue():
    sp = _issue()
    F = D.latest(D.read(D.res("11", "11__dsmem_bw__fits.csv")))
    E = F[(F.step == "E") & (F.side == "target") & (F.pattern == "pair_uni")]
    put = E[E.kernel == "tma_put"].set_index("chunk_bytes").bpc_per_sm.sort_index()
    red = E[E.kernel == "tma_red"].set_index("chunk_bytes").bpc_per_sm.sort_index()
    ring = F[(F.step == "E") & (F.side == "target") & (F.pattern == "ring")]
    cyc_cmd = (put.index / put)[put.index <= 1024]   # cycles per command where commands are issue-bound
    c_issue = cyc_cmd.median()

    f, (a1, a2) = S.fig(ncols=2, width=FULL, height=2.45)
    # (a) issue span vs batch size -------------------------------------------------------------------------
    l2 = sp.loc["l2"]
    slopes = {}
    for j, w in enumerate([1, 2, 4]):
        s = l2.loc[w]
        a1.plot(s.index, s.values, color=PAL[j], lw=1.4, marker="o", ms=2.2)
        lo_hi = (1, 32) if w < 4 else (1, 16)
        z = s[(s.index >= lo_hi[0]) & (s.index <= lo_hi[1])]
        slopes[w] = np.polyfit(z.index, z.values, 1)[0]
    z4 = l2.loc[4][(l2.loc[4].index >= 17)]
    slope4_hi = np.polyfit(z4.index, z4.values, 1)[0]
    rate_line = 4096 / 128   # cycles per 4-KiB copy at 128 B/cycle (TMA processing rate)
    nn = np.array([16, 36])
    a1.plot(nn, l2.loc[4][16] + (nn - 16) * rate_line, color=INK, lw=0.9, ls=(0, (1, 1.5)))
    _label(a1, 31, l2.loc[1][31] + 30, f"1 warp: {slopes[1]:.0f} cycles/copy", color=PAL[0], ha="right", va="bottom")
    _label(a1, 24, 420, f"2 warps: {slopes[2]:.0f}", color=PAL[1], ha="left", va="top")
    _label(a1, 11, 110, f"4 warps: {slopes[4]:.0f} (N ≤ 16)", color=PAL[2], ha="left", va="center")
    _label(a1, 36.5, l2.loc[4][16] + 20 * rate_line + 30, f"{rate_line:.0f}/copy\n= 128 B/cycle", color=INK,
           ha="right", va="bottom", fontsize=6.5)
    a1.set_xlim(0, 37)
    a1.set_ylim(0, 1700)
    a1.set_xlabel("4-KiB copies issued back to back (one SM)")
    a1.set_ylabel("first issue → last issue returns (cycles)")
    S.panel(a1, "a")

    # (b) TMA shared→shared put rate vs command size ------------------------------------------------------
    a2.plot(put.index, put.values, color=PAL[1], lw=1.4, marker="o", ms=3)
    a2.plot(red.index, red.values, color=PAL[1], lw=1.1, ls=(0, (3, 1.5)), marker="o", ms=2.5, mfc="white")
    xs = np.array([128, 1024])
    a2.plot(xs, xs / c_issue, color=INK, lw=0.9, ls=(0, (1, 1.5)))
    S.ref_line(a2, SEND_LIMIT, f"{SEND_LIMIT:.2f} B/cycle per-SM send limit", pos=0.02, ha="left", va="bottom", fs=6.5)
    _label(a2, 135, 17.5, f"issue-bound:\n{c_issue:.0f} cycles/command", color=INK, ha="left", va="center",
           fontsize=6.5)
    _label(a2, 16384, put[16384] - 0.6, "put", color=PAL[1], ha="right", va="top")
    _label(a2, 16384, red[16384] - 0.6, "reduce-add put", color=PAL[1], ha="right", va="top")
    S.bytes_axis(a2)
    a2.xaxis.set_minor_locator(NullLocator())
    a2.set_xlim(100, 21000)
    a2.set_ylim(0, 26)
    a2.set_xlabel("bytes per TMA command (shared::cta → shared::cluster)")
    a2.set_ylabel("delivered to the target SM (B/cycle)")
    S.panel(a2, "b")

    # issue-side costs from the raw issuer records (11, step C, 16-KiB commands)
    ti = _tma_s2s_issue()
    C = ti[(ti.step == "C")]
    first = C[(C.pattern == "ring") & (C.payload_bytes == 16384)]["median"]
    a2a = C[(C.pattern == "all_to_all") & (C.payload_bytes == 16384)].set_index("cs")["median"]
    t1 = first.median()
    per_dest = {int(cs): _r((v - t1) / (cs - 2), 0) for cs, v in a2a.items() if cs > 2}
    Eiss = ti[(ti.step == "E") & (ti.kernel == "tma_put") & (ti.pattern == "pair_uni")].set_index("chunk_bytes")
    iss_per_cmd = (Eiss["median"] / (98304 // Eiss.index)).sort_index()
    crossover = int(put.index[put >= 0.9 * SEND_LIMIT].min())
    info = dict(issue_cycles_per_copy={f"{w}_warps": _r(v) for w, v in slopes.items()},
                issue_4warps_N17_32=_r(slope4_hi), rate_bound_cycles_per_4KiB=_r(rate_line),
                issue_span_N1=_r(l2.loc[1][1]),
                l2_vs_hbm_issue_span_max_abs_diff=_r((sp.loc["l2"] - sp.loc["hbm"]).abs().max()),
                put_bpc={int(k): _r(v, 2) for k, v in put.items()},
                red_bpc={int(k): _r(v, 2) for k, v in red.items()},
                ring_put_16KiB=_r(ring[(ring.kernel == "tma_put") & (ring.chunk_bytes == 16384)].bpc_per_sm.iloc[0], 2),
                cycles_per_command_issue_bound=_r(c_issue, 1), crossover_bytes=crossover,
                put_plateau_2_16KiB=[_r(put[put.index >= 2048].min(), 2), _r(put[put.index >= 2048].max(), 2)],
                red_plateau_2_16KiB=[_r(red[red.index >= 2048].min(), 2), _r(red[red.index >= 2048].max(), 2)],
                first_command_after_barrier_cycles=_r(t1, 0),
                first_command_by_cs={int(k): _r(v, 0) for k, v in
                                     C[(C.pattern == "ring") & (C.payload_bytes == 16384)].set_index("cs")["median"].items()},
                all_to_all_issue_16KiB_by_cs={int(k): _r(v, 0) for k, v in a2a.items()},
                extra_cycles_per_new_destination=per_dest,
                issuer_cycles_per_command_96KiB={int(k): _r(v, 1) for k, v in iss_per_cmd.items()})
    return f, info


# =================================================================================================================
def facts():
    m = _footprint()
    s, ds = _lat_ladder()
    b, gw, pu = _bw_ladder()
    hbm, full, l2m, l2full, floor, _ = _hbm_l2_scaling()
    q = D.latest(D.read(D.res("08", "08__l2__summary.csv"))).set_index("level").value
    cm, _, _, _, _ = _completion()
    l2c = cm.l2.dropna()
    fit = l2c[l2c.index >= 8192]
    T0, sl = np.polyfit(fit.index.values, fit.values, 1)[::-1]
    sm = D.latest(D.read(D.res("06", "06__tma__summary.csv"))).set_index("level")["median"]
    sp = _issue().loc["l2"]
    F = D.latest(D.read(D.res("11", "11__dsmem_bw__fits.csv")))
    put = F[(F.step == "E") & (F.side == "target") & (F.pattern == "pair_uni") & (F.kernel == "tma_put")] \
        .set_index("chunk_bytes").bpc_per_sm
    v = lambda lv: _r(s.loc[lv, "median_cycles_at_1635"])
    knee = int(hbm.read[hbm.read >= 0.9 * full["read"]].index.min())
    return {
        "lat_smem_cycles": v("smem"), "lat_own_smem_via_cluster_window_cycles": v("smem_dsmem_if"),
        "lat_l1_cycles": v("l1"), "lat_dsmem_remote_cycles": _r(ds.median()),
        "lat_dsmem_remote_range_cs2_16": [_r(ds.min()), _r(ds.max())],
        "lat_l2_near_cycles": v("l2_near"), "lat_l2_far_cycles": v("l2_far"),
        "lat_hbm_near_cycles": v("dram_near"), "lat_hbm_far_cycles": v("dram_far"),
        "lat_l2_chase_cycles": v("l2_chase"), "lat_hbm_chase_cycles": v("dram_chase"),
        "lat_mixed_plateau_35_51MiB_cycles": _r(m.loc[(m.index >= 35 << 20) & (m.index <= 51 << 20), "cg"].median()),
        "l1_capacity_KiB_last_hit": _r(m.index[(m.ca <= 33) & (m.index <= 1 << 20)].max() / 1024),
        "l2_grid_cycles": [_r(q["l2_same_partition_same_subnode_position"]),
                           _r(q["l2_same_partition_other_subnode_position"]),
                           _r(q["l2_other_partition_same_subnode_position"]),
                           _r(q["l2_other_partition_other_subnode_position"])],
        "bw_smem_tbps": _r(b.loc["smem_ld", "tbps"], 2), "bw_smem_bpc_per_sm": _r(b.loc["smem_ld", "bpc_per_sm"], 1),
        "bw_l1_tbps": _r(b.loc["l1_ld", "tbps"], 2),
        "bw_l2_read_tbps": _r(b.loc["l2_ld_mixed", "tbps"], 2), "bw_l2_store_tbps": _r(b.loc["l2_st_mixed", "tbps"], 2),
        "bw_l2_read_bpc_per_sm": _r(b.loc["l2_ld_mixed", "bpc_per_sm"], 1),
        "bw_l2_link_one_way_tbps": _r(b.loc["l2_link_one_way", "tbps"], 2),
        "bw_l2_one_gpc_tbps": _r(b.loc["l2_ld_one_gpc", "tbps"], 2),
        "bw_hbm_read_tbps": _r(b.loc["dram_read", "tbps"], 2), "bw_hbm_write_tbps": _r(b.loc["dram_write", "tbps"], 2),
        "bw_hbm_read_pct_of_peak": _r(100 * b.loc["dram_read", "tbps"] / HBM_PEAK),
        "bw_hbm_read_one_sm_gbps": _r(hbm.read[1] * 1e3), "bw_hbm_read_90pct_at_sms": knee,
        "bw_l2_read_one_sm_bpc": _r(l2m.iloc[0] * 1e12 / (S.SM_MHZ * 1e6)),
        "bw_dsmem_gpu_wide_tbps_range": [_r(gw.value.min(), 2), _r(gw.value.max(), 2)],
        "bw_dsmem_send_limit_bpc": _r(SEND_LIMIT, 2),
        "bw_dsmem_one_way_pair_bpc": [_r(pu.min(), 2), _r(pu.max(), 2)],
        "tma_copy_T0_cycles": _r(T0), "tma_copy_bpc": _r(1 / sl),
        "tma_copy_16B_l2_cycles": _r(l2c[16]),
        "tma_chase_l2_cycles": _r(sm["chase_bulk16_l2"]), "tma_chase_hbm_cycles": _r(sm["chase_bulk16_hbm"]),
        "tma_minus_generic_cycles": [_r(sm["tma_minus_generic_l2"]), _r(sm["tma_minus_generic_hbm"])],
        "tma_issue_1warp_cycles_per_copy": _r(np.polyfit(sp.loc[1].index, sp.loc[1].values, 1)[0]),
        "tma_issue_4warps_cycles_per_copy": _r(np.polyfit(sp.loc[4].index[:16], sp.loc[4].values[:16], 1)[0]),
        "tma_s2s_issue_bound_cycles_per_command": _r((put.index / put)[put.index <= 1024].median(), 1),
        "tma_s2s_put_16KiB_bpc": _r(put[16384], 2),
        "tma_prefetch_l2_hbm_chase_cycles": _r(sm["chase_bulk16_prefetch_hbm"]),
    }


if __name__ == "__main__":
    import os
    import json
    out = os.path.join(D.CACHE, "preview")
    os.makedirs(out, exist_ok=True)
    for name in ["latency_footprint", "ladders", "l2_nuca", "tma_copy_latency", "tma_issue"]:
        fg, inf = globals()[f"fig_{name}"]()
        fg.savefig(os.path.join(out, f"{name}.png"))
        plt.close(fg)
        print(name, json.dumps(inf, default=str))
    print("facts", json.dumps(facts(), default=str))
