"""Figures for the distributed-shared-memory (DSMEM) and synchronisation sections of the report.

Every `fig_<name>()` returns (Figure, info) where `info` holds the numbers the text quotes, read from the local CSVs.
`facts()` collects the cheat-sheet numbers of these sections. `python3 -m reports.h100char.figs_dsmem_sync` renders
every figure to reports/cache/preview/<name>.png.

Data sources (all local): 02 occupancy, 03 SM->GPC map (card A); 07 synchronisation summary (card A);
09 chase / summary, 10 pair statistics, 11 fits / GPU-wide / summary, 12 summary / pairs (card B; same model,
identical topology and placement).
"""
import os
import re

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from . import style as S
from .style import (COL, FULL, PAL, BLUE, ORANGE, AQUA, YELLOW, GREY, LIGHTGREY, INK, INK2, MUTED,
                    panel, ns_axis, ref_line, bytes_axis)
from . import data as D

PIN_MAP = D.PIN["03_map"]
PIN_OCC = "2026-10-23T22:59:12.029"        # card A occupancy run (the card-B repeat is identical)
PIN_CHASE09 = "2026-10-26T03:29:45.158"    # 09 K1 chase run (all variants / policies / chains)


# ----------------------------------------------------------------------------------------------- helpers
def _map():
    m = D.pinned(D.read(D.res("03", "03__sm_gpc_topology__map.csv")), PIN_MAP)
    return m.set_index("smid").sort_index()


def _sync():
    return D.latest(D.read(D.res("07", "07__sync__summary_v2.csv")))


def _sv(d, level, config, col="median"):
    r = d[(d.level == level) & (d.config == config)]
    assert len(r) == 1, (level, config, len(r))
    return float(r[col].iloc[0])


def _clean(ax, grid_axis="x"):
    ax.grid(False)
    ax.grid(True, axis=grid_axis)


# ======================================================================================= §2 DSMEM
def fig_gpc_occupancy():
    """(a) the chip as 8 GPCs of TPC tiles; (b) max simultaneously active clusters and idle SMs per cluster size."""
    m = _map()
    pair_only = sorted(m.index[m.max_cs_observed <= 2])
    occ = D.pinned(D.read(D.res("02", "02__cluster_launch__occupancy.csv")), PIN_OCC)
    occ = occ[(occ.policy == "default") & (occ.smem_bytes == 204800) & (occ.max_ctas_per_sm == 1)]
    css = [2, 4, 8, 16]
    clusters = {c: int(occ[occ.cs == c].measured_first_wave_clusters.median()) for c in css}
    api = {c: int(occ[occ.cs == c].api_max_active_clusters.median()) for c in css}
    idle = {c: 132 - c * clusters[c] for c in css}
    assert all(idle[c] == int(occ[occ.cs == c].sms_idle.median()) for c in css)

    f, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.25), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.75, 1]})
    # (a) one row per GPC, tiles = SMs, pairs of tiles = TPCs (SMs 2t, 2t+1)
    w, gap, tgap = 1.0, 0.1, 0.4
    sizes = {}
    for g in range(8):
        sms = sorted(m.index[m.gpc == g])
        sizes[g] = len(sms)
        y = 7 - g
        for k, s in enumerate(sms):
            x = k * (w + gap) + (k // 2) * tgap
            po = s in pair_only
            a.add_patch(Rectangle((x, y), w, 0.82, facecolor=ORANGE if po else "#c9dcf3",
                                  edgecolor="none"))
            a.text(x + w / 2, y + 0.41, str(s), ha="center", va="center", fontsize=4.1,
                   color="white" if po else INK2)
        a.text(-0.35, y + 0.41, f"G{g}", ha="right", va="center", fontsize=7.5, color=INK)
        xe = len(sms) * (w + gap) + (len(sms) // 2) * tgap
        cap = int(m[(m.gpc == g) & (m.max_cs_observed > 2)].max_cs_observed.max())
        a.text(xe + 0.1, y + 0.41, f"{len(sms)} SMs" + ("" if cap == 16 else f"  (clusters ≤ {cap})"),
               ha="left", va="center", fontsize=6.5, color=INK2)
    a.set_xlim(-1.6, 18 * (w + gap) + 9 * tgap + 3.6)
    a.set_ylim(-0.25, 8.0)
    a.axis("off")
    a.legend(handles=[Rectangle((0, 0), 1, 1, fc="#c9dcf3"), Rectangle((0, 0), 1, 1, fc=ORANGE)],
             labels=["SM (adjacent pair = one TPC)", "SM only ever paired with its TPC mate"], loc="upper center",
             bbox_to_anchor=(0.45, 0.0), ncol=2, fontsize=6.5, handlelength=1.0, columnspacing=1.2)
    panel(a, "a", x=0.0, y=1.0)

    # (b) stacked bar of the 132 SMs: in clusters vs idle
    for i, c in enumerate(css):
        used = c * clusters[c]
        b.barh(i, used, color=BLUE, height=0.62)
        b.barh(i, idle[c], left=used, color=LIGHTGREY, height=0.62)
        b.text(used / 2, i, f"{clusters[c]} × {c}", ha="center", va="center", fontsize=7, color="white")
        b.text(133.5, i, f"{idle[c]} idle" if idle[c] else "0 idle", ha="left", va="center", fontsize=7,
               color=INK2)
    b.set_yticks(range(len(css)), [f"CS {c}" for c in css])
    b.invert_yaxis()
    b.set_xlim(0, 160)
    b.set_xticks([0, 33, 66, 99, 132])
    b.set_xlabel("SMs (1 CTA per SM)")
    b.spines["left"].set_visible(False)
    b.tick_params(axis="y", length=0)
    _clean(b)
    panel(b, "b", x=-0.12)
    info = {"gpc_sizes": sizes, "pair_only_sms": pair_only, "max_active_clusters": clusters,
            "api_max_active_clusters": api, "idle_sms": idle,
            "cluster_capable_sms": int((m.max_cs_observed > 2).sum()),
            "gpcs_hosting_cs16": sorted(int(g) for g in m[m.max_cs_observed == 16].gpc.unique()),
            "occupancy_run": PIN_OCC, "map_run": PIN_MAP}
    return f, info


def _chase09():
    def build():
        d = D.read_run(D.res("09", "09__dsmem_latency__chase.csv"), PIN_CHASE09,
                       usecols=["cs", "policy", "variant", "chain", "same_sm", "smid_src", "smid_dst",
                                "cycles_per_hop"],
                       query="variant == 'ld_cluster' and chain == 'stride1' and policy == 'Default' and same_sm == 0")
        return d[["cs", "smid_src", "smid_dst", "cycles_per_hop"]]
    return D.cached("09_chase_ldcluster_stride1_default_remote.csv", build)


def _refs09():
    s = D.latest(D.read(D.res("09", "09__dsmem_latency__summary.csv")))
    g = lambda lv: float(s[s.level == lv]["median"].iloc[0])
    own = float(s[(s.level == "k1/ld_cluster/stride1/Default/self") & (s.config == "cs 2")]["median"].iloc[0])
    s4 = D.latest(D.read(D.res("04", "04__latency__summary_v2.csv")))
    l1 = float(s4[s4.level == "l1"].median_cycles_at_1635.iloc[0])
    return {"smem": g("k3/lds_idx/stride1"), "l1": l1, "l2": g("k3/l2_chase/random"),
            "hbm": g("k3/dram_chase/random"), "own_via_dsmem": own}


def fig_dsmem_latency_cs():
    """Remote DSMEM dependent-load latency per cluster size: (a) against the memory ladder, (b) zoom (ECDF)."""
    d = _chase09()
    R = _refs09()
    css = [2, 4, 8, 16]
    f, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.2), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.35, 1]})
    info = {"per_cs": {}, "refs": R}
    for i, c in enumerate(css):
        x = d[d.cs == c].cycles_per_hop.to_numpy()
        q = np.percentile(x, [5, 25, 50, 75, 95])
        info["per_cs"][c] = {"median": round(q[2], 1), "p5": round(q[0], 1), "p95": round(q[4], 1),
                             "min": round(x.min(), 1), "max": round(x.max(), 1), "n_pairs_samples": len(x),
                             "ratio_to_l2": round(q[2] / R["l2"], 3), "ratio_to_smem": round(q[2] / R["smem"], 2)}
        a.plot([q[0], q[4]], [i, i], color=BLUE, lw=1.0, solid_capstyle="butt")
        a.plot([q[1], q[3]], [i, i], color=BLUE, lw=5.0, solid_capstyle="butt")
        a.plot(q[2], i, "o", color="white", mec=BLUE, mew=1.0, ms=4.0, zorder=4)
        a.text(q[4] * 1.04, i, f"{q[2]:.0f}", va="center", fontsize=6.5, color=INK2)
        b.ecdf(x, color=PAL[i], lw=1.3, label=f"CS {c}")
    a.set_xscale("log")
    a.set_xlim(15, 900)
    a.set_ylim(-1.5, 3.5)
    a.set_yticks(range(4), [f"CS {c}" for c in css])
    a.invert_yaxis()
    for v, lab, ha in [(R["smem"], "local smem", "right"), (R["l1"], "L1 hit", "left"),
                       (R["l2"], "L2", "left"), (R["hbm"], "HBM", "left")]:
        a.axvline(v, ls=":", lw=1.0, color=INK2, zorder=1)
        a.text(v * (1.04 if ha == "left" else 0.96), -1.0, f"{lab}\n{v:.0f}", fontsize=6.5, color=INK2,
               va="center", ha=ha)
    a.set_xticks([20, 50, 100, 200, 500], ["20", "50", "100", "200", "500"])
    a.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    a.set_xlabel("dependent-load latency (SM cycles)")
    a.tick_params(axis="y", length=0)
    a.spines["left"].set_visible(False)
    _clean(a)
    sec = ns_axis(a, "x")
    sec.set_xticks([10, 20, 50, 100, 200, 500], ["10", "20", "50", "100", "200", "500"])
    sec.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    panel(a, "a", x=-0.08, y=1.12)
    b.set_xlim(180, 225)
    b.set_ylim(0, 1)
    b.set_xlabel("remote load latency (SM cycles)")
    b.set_ylabel("fraction of SM pairs")
    b.legend(loc="lower right", fontsize=7)
    ns_axis(b, "x")
    panel(b, "b", x=-0.12, y=1.12)
    meds = [info["per_cs"][c]["median"] for c in css]
    info["median_range"] = (min(meds), max(meds))
    info["own_cta_via_dsmem_path"] = round(R["own_via_dsmem"], 1)
    info["chase_run"] = PIN_CHASE09
    return f, info


def _pairs10():
    p = D.latest(D.read(D.res("10", "10__heatmap__pair_stats.csv")))
    return p[(p.matrix == "chase") & (p.cs == "all")].copy()


def _fit_additive(p, m):
    """Symmetric additive model L(i,j) = p(i) + p(j), least squares over cluster-capable pairs (GPC blocks)."""
    cap = set(m.index[m.max_cs_observed > 2])
    q = p[p.smid_src.isin(cap) & p.smid_dst.isin(cap)]
    sms = sorted(set(q.smid_src) | set(q.smid_dst))
    ix = {s: k for k, s in enumerate(sms)}
    X = np.zeros((len(q), len(sms)))
    X[np.arange(len(q)), q.smid_src.map(ix).to_numpy()] += 1
    X[np.arange(len(q)), q.smid_dst.map(ix).to_numpy()] += 1
    y = q["median"].to_numpy()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ coef
    return pd.Series(coef, index=sms), res, len(q)


def fig_dsmem_heatmap(gpc=6):
    """(a) SM x SM remote-load latency in one 18-SM GPC; (b) additive per-SM terms by TPC slot for every GPC."""
    p = _pairs10()
    m = _map()
    pv, res, npairs = _fit_additive(p, m)
    # TPC slot within the GPC: rank of the TPC (smid // 2) among the GPC's cluster-capable TPCs
    cap = m[m.max_cs_observed > 2].copy()
    cap["tpc"] = cap.index // 2
    cap["slot"] = cap.groupby("gpc").tpc.rank(method="dense").astype(int) - 1
    cap["p"] = pv.reindex(cap.index)

    f, (a, b) = plt.subplots(1, 2, figsize=(FULL, 3.0), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.12, 1]})
    sms = list(cap[cap.gpc == gpc].sort_values(["slot"]).index)
    sms = sorted(sms, key=lambda s: (cap.slot[s], s))
    M = np.full((len(sms), len(sms)), np.nan)
    ix = {s: k for k, s in enumerate(sms)}
    for r in p[p.smid_src.isin(sms) & p.smid_dst.isin(sms)].itertuples():
        M[ix[r.smid_src], ix[r.smid_dst]] = r.median
    vmin, vmax = np.floor(np.nanmin(M)), np.ceil(np.nanmax(M))
    cmap = mpl.colormaps["Blues"].copy()
    cmap.set_bad("white")
    im = a.imshow(M, cmap=cmap, vmin=vmin - 4, vmax=vmax, origin="upper", interpolation="nearest")
    a.set_xticks(range(len(sms)), [str(s) for s in sms], rotation=90, fontsize=5.5)
    a.set_yticks(range(len(sms)), [str(s) for s in sms], fontsize=5.5)
    for k in range(2, len(sms), 2):
        a.axhline(k - 0.5, color="white", lw=0.8)
        a.axvline(k - 0.5, color="white", lw=0.8)
    a.grid(False)
    a.set_xlabel(f"destination SM (GPC {gpc}, TPC order)")
    a.set_ylabel("source SM")
    for sp in a.spines.values():
        sp.set_visible(False)
    a.tick_params(length=0)
    cb = f.colorbar(im, ax=a, shrink=0.85, pad=0.02, aspect=25)
    cb.set_label("remote load latency (SM cycles)")
    cb.outline.set_visible(False)
    panel(a, "a", x=-0.02, y=1.03)

    # (b) per-SM additive term, mean of the TPC's two SMs, by GPC and TPC slot
    T = cap.groupby(["gpc", "slot"]).p.mean().unstack()
    cm2 = mpl.colormaps["Blues"].copy()
    cm2.set_bad("white")
    b.imshow(T.to_numpy(), cmap=cm2, vmin=T.min().min() - 6, vmax=T.max().max() + 1, aspect="auto",
             interpolation="nearest")
    for (gi, sl), v in np.ndenumerate(T.to_numpy()):
        if np.isfinite(v):
            dark = v > (T.min().min() + T.max().max()) / 2 + 2
            b.text(sl, gi, f"{v:.1f}", ha="center", va="center", fontsize=5.8,
                   color="white" if dark else INK)
    b.set_xticks(range(T.shape[1]), [str(k) for k in T.columns])
    b.set_yticks(range(T.shape[0]), [f"G{g}" for g in T.index])
    for k in range(1, T.shape[0]):
        b.axhline(k - 0.5, color="white", lw=1.0)
    for k in range(1, T.shape[1]):
        b.axvline(k - 0.5, color="white", lw=1.0)
    b.grid(False)
    for sp in b.spines.values():
        sp.set_visible(False)
    b.tick_params(length=0)
    b.set_xlabel("TPC slot within the GPC (ascending SM id)")
    b.xaxis.set_label_position("bottom")
    b.text(0.0, 1.01, "p(i), SM cycles (mean of the TPC's two SMs);  L(i,j) = p(i) + p(j)",
           transform=b.transAxes, fontsize=6.5, color=INK2, ha="left", va="bottom")
    panel(b, "b", x=-0.06, y=1.07)

    allp = p["median"]
    mates = cap.assign(mate=cap.index ^ 1)
    tp = cap.groupby("tpc").p.agg(["min", "max"])
    dmate = (tp["max"] - tp["min"])
    per_gpc = p.groupby("gpc")["median"].median().round(1).to_dict()
    slot_tab = cap.groupby(["gpc", "slot"]).p.mean().round(1).unstack()
    info = {"pair_latency_min": round(allp.min(), 1), "pair_latency_max": round(allp.max(), 1),
            "pair_latency_median": round(allp.median(), 1), "n_pairs": int(len(p)), "n_pairs_fit": npairs,
            "pair_mad_max": float(p["mad"].max()),
            "p_min": round(pv.min(), 2), "p_max": round(pv.max(), 2), "p_distinct_values": int(pv.round(1).nunique()),
            "resid_std": float(res.std()), "resid_max": float(np.abs(res).max()),
            "tpc_mates_equal": f"{int((dmate < 0.05).sum())}/{len(dmate)}", "tpc_mate_diff_max": round(dmate.max(), 2),
            "per_gpc_median_pair": per_gpc, "heatmap_gpc": gpc,
            "p_by_gpc_and_slot": slot_tab.to_dict(orient="index"), "run": str(p.run_id.iloc[0])}
    return f, info


def _gmem_ref_cardB():
    """L2 read and HBM read TB/s on the DSMEM card, same selection as the 11 notebook (05 binaries, event floor removed)."""
    def build():
        c5 = D.import_file(os.path.join(D.EXP, D.EXPDIR["05"], "src", "common05.py"), "_c05")
        R05 = D.res("05")
        uuid11 = D.runs("11").gpu_uuid.dropna().iloc[-1]
        r3, r4 = c5.latest_run("bw_l2", "k3", res=R05), c5.latest_run("bw_dram", "k4", res=R05)
        reg = D.read(D.res("05", "05__memory_hierarchy_bandwidth__runs.csv"), dtype=str)
        same = all(reg[reg.run_id == r].gpu_uuid.iloc[0] == uuid11 for r in (r3, r4))
        floor = c5.event_floor_ms(os.path.join(D.EXP, D.EXPDIR["01"], "results", "01__timers__event_floor.csv"))
        l2 = c5.read_run("05__bandwidth__l2.csv", r3, res=R05)
        l2 = l2[(l2.discard.isna()) | (l2.discard == "")]
        l2 = l2.assign(tbps=c5.net_tbps(l2, floor))
        l2 = l2[(l2["pass"] == 0) & (l2.ilp == 4) & (l2.blocks == 264) & (l2.cache_op == "cg") & (l2.op == "ld") &
                (l2.variant == "mixed") & (l2.footprint_bytes == 16 << 20)]
        dr = c5.read_run("05__bandwidth__dram.csv", r4, res=R05)
        dr = dr[(dr.discard.isna()) | (dr.discard == "")]
        dr = dr.assign(tbps=c5.net_tbps(dr, floor))
        dr = dr[(dr.blocks == 4 * 132) & (dr.ilp == 4) & (dr.cache_op == "cg") & (dr.op == "read")]
        return pd.DataFrame([{"l2_read_tbps": float(l2.tbps.median()), "hbm_read_tbps": float(dr.tbps.median()),
                              "l2_run": r3, "hbm_run": r4, "same_card_as_11": same}])
    return D.cached("11_gmem_reference_cardB.csv", build).iloc[0].to_dict()


def fig_dsmem_bandwidth():
    """(a) per-SM B/cycle vs bytes in flight; (b) one SM sending to / receiving from k peers; (c) GPU-wide vs CS."""
    F = D.latest(D.read(D.res("11", "11__dsmem_bw__fits.csv")))
    G = D.latest(D.read(D.res("11", "11__dsmem_bw__gpuwide.csv")))
    Sm = D.latest(D.read(D.res("11", "11__dsmem_bw__summary.csv")))
    ref = _gmem_ref_cardB()
    LIM = 64 / 3
    f, (a, b, c) = plt.subplots(1, 3, figsize=(FULL, 2.3), constrained_layout=True,
                                gridspec_kw={"width_ratios": [1.15, 1, 1]})
    info = {"send_limit_bpc": round(LIM, 2)}
    # (a) step B, one-way pair: threads x ILP x 16 B (v4) in flight per SM, mean over CS 2-16
    B = F[(F.step == "B") & (F.pattern == "pair_uni") & (F.side == "issuer") & F.kernel.isin(["ld", "st"])].copy()
    B["inflight"] = B.threads * B.ilp * 16
    agg = B.groupby(["kernel", "threads", "ilp", "inflight"]).bpc_per_sm.mean().reset_index()
    for k, col, lab, mk in [("ld", BLUE, "ld.shared::cluster", "o"), ("st", ORANGE, "st.shared::cluster", "s")]:
        t = agg[agg.kernel == k]
        line = t.groupby("inflight").bpc_per_sm.median()
        a.plot(line.index, line.values, "-", color=col, lw=1.3, zorder=3)
        a.plot(t.inflight, t.bpc_per_sm, mk, color=col, ms=3.0, mec="white", mew=0.4, zorder=4, label=lab)
        info[f"{k}_bpc_by_inflight"] = line.round(2).to_dict()
    ld2k = agg[(agg.kernel == "ld") & (agg.inflight == 2048)].bpc_per_sm.mean()
    lat = 2048 / ld2k
    xs = np.array([1024, 16384])
    a.plot(xs, xs / lat, "--", color=GREY, lw=0.9, zorder=2)
    a.text(2900, 23.6, f"in flight ÷ {lat:.0f} cycles", fontsize=6.3, color=INK2, ha="left", va="center")
    ref_line(a, LIM, "send limit 21.33", pos=0.99, va="top", fs=6.5)
    bytes_axis(a, "x")
    a.set_xlim(1536, 1 << 17)
    a.set_xticks([2048, 8192, 32768, 131072])
    a.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    a.set_ylim(0, 25)
    a.set_xlabel("bytes in flight per SM")
    a.set_ylabel("B/cycle per SM")
    a.legend(loc="lower right", bbox_to_anchor=(1.0, 0.12), fontsize=6.5, handletextpad=0.3)
    panel(a, "a", x=-0.1)
    info["ld_effective_latency_cycles_at_2KiB"] = round(lat, 0)
    info["ld_bpc_2KiB"] = round(ld2k, 2)

    # (b) fan-out (step D, CS 16, one SM stores the payload to k targets) and fan-in (step C, k writers into one SM)
    Do = F[(F.step == "D") & (F.kernel == "st") & (F.pattern == "broadcast_out") & (F.side == "issuer")]
    Do = Do.groupby("pattern_k").bpc_per_sm.mean()
    Ci = F[(F.step == "C") & (F.pattern == "broadcast_in")]
    st_in = Ci[(Ci.kernel == "st") & (Ci.side == "issuer")].groupby("pattern_k").bpc_per_sm.mean()
    st_in = st_in * st_in.index          # k writers, per-writer rate -> total into the receiver
    tma_in = Ci[(Ci.kernel == "tma_put") & (Ci.side == "target")].groupby("pattern_k").bpc_per_sm.mean()
    b.plot(Do.index, Do.values, "-o", color=BLUE, ms=2.6, mec="white", mew=0.3, lw=1.3)
    b.plot(st_in.index, st_in.values, "-s", color=ORANGE, ms=3.2, mec="white", mew=0.4, lw=1.3)
    b.plot(tma_in.index, tma_in.values, "D", color=ORANGE, ms=3.4, mfc="white", mew=0.9, ls="none")
    b.text(15, Do.values[-1] - 1.4, "one SM → k targets (st)", ha="right", va="top", fontsize=6.5, color=INK)
    b.text(15, st_in.values[-1] + 1.0, "k writers → one SM", ha="right", va="bottom", fontsize=6.5, color=INK)
    b.legend(handles=[Line2D([], [], marker="s", color=ORANGE, ls="none", ms=3.2, mec="white"),
                      Line2D([], [], marker="D", color=ORANGE, mfc="white", ls="none", ms=3.4)],
             labels=["st", "TMA put"], loc="lower right", fontsize=6.3, handletextpad=0.2,
             title="into one SM", title_fontsize=6.3)
    ref_line(b, 32, "rated port 32", pos=0.30, ha="left", va="top", fs=6.3)
    b.axhline(LIM, ls=":", lw=1.0, color=INK2, zorder=1)
    b.set_xlim(0, 16)
    b.set_xticks([1, 3, 7, 11, 15])
    b.set_ylim(0, 40)
    b.set_xlabel("peers k (CS = k + 1)")
    b.set_ylabel("B/cycle at the shared SM")
    panel(b, "b", x=-0.1)
    info["fan_out_st_total_bpc"] = Do.round(2).to_dict()
    info["fan_in_st_total_bpc"] = st_in.round(2).to_dict()
    info["fan_in_tma_put_total_bpc"] = tma_in.round(2).to_dict()

    # (c) GPU-wide, all resident clusters in a ring, cudaEvent slope
    Gs = G[G.source == "S"]
    for k, col, lab, mk in [("ld", BLUE, "ld", "o"), ("st", ORANGE, "st", "s"), ("tma_put", AQUA, "TMA put", "D")]:
        t = Gs[Gs.kernel == k].sort_values("cs")
        c.plot(t.cs, t.tbps_slope, "-" + mk, color=col, ms=3.2, mec="white", mew=0.4, lw=1.3)
        c.text(17.5, {"ld": 2.68, "st": 3.22, "tma_put": 2.18}[k], lab, fontsize=6.5, va="center", color=INK)
        info[f"gpu_wide_{k}_tbps"] = dict(zip(t.cs.astype(int), t.tbps_slope.round(2)))
    info["gpu_wide_active_sms"] = dict(zip(Gs[Gs.kernel == "ld"].cs.astype(int), Gs[Gs.kernel == "ld"].active_sms.astype(int)))
    ref_line(c, ref["l2_read_tbps"], f"L2 read {ref['l2_read_tbps']:.2f}", pos=0.99, fs=6.5)
    ref_line(c, ref["hbm_read_tbps"], f"HBM read {ref['hbm_read_tbps']:.2f}", pos=0.99, fs=6.5)
    c.set_xscale("log", base=2)
    c.set_xticks([2, 4, 8, 16], ["2", "4", "8", "16"])
    c.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    c.set_xlim(1.7, 32)
    c.set_ylim(0, 8.6)
    c.set_xlabel("cluster size")
    c.set_ylabel("GPU-wide TB/s")
    panel(c, "c", x=-0.1)
    info["l2_read_tbps_ref"] = round(ref["l2_read_tbps"], 2)
    info["hbm_read_tbps_ref"] = round(ref["hbm_read_tbps"], 2)
    info["gmem_ref_same_card"] = bool(ref["same_card_as_11"])
    ring = {}
    for op in ["ld", "st", "st_async", "tma_put", "tma_red"]:
        for cs in [2, 4, 8, 16]:
            for mode in ["single", "full"]:
                r = Sm[Sm.level == f"ring_{op}_cs{cs}_{mode}"]
                if len(r):
                    ring[f"{op}_cs{cs}_{mode}"] = round(float(r.value.iloc[0]), 2)
    info["ring_bpc_per_sm"] = ring
    return f, info


def fig_dsmem_atomics():
    """(a) dependent latency, atomic add.u32 vs plain load per path; (b) per-SM ops/cycle at 1024 threads."""
    s = D.latest(D.read(D.res("12", "12__atomics__summary.csv")))
    pr = D.latest(D.read(D.res("12", "12__atomics__pairs.csv")))

    def row(target, op, kind, cs=1, col="latency_cycles"):
        r = s[(s.target == target) & (s.op == op) & (s.kind == kind) & (s.cs == cs)]
        return float(r[col].iloc[0])

    paths = [("smem", 1, "local smem"), ("smem_dsmem_if", 2, "own smem via DSMEM window"),
             ("dsmem", 2, "remote DSMEM (peer SM)"), ("l2_near", 1, "L2, near partition"),
             ("l2_far", 1, "L2, far partition"), ("hbm_near", 1, "HBM, near partition"),
             ("hbm_far", 1, "HBM, far partition")]
    f, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.45), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.15, 1]})
    lat = {}
    for i, (t, cs, lab) in enumerate(paths):
        ld, at = row(t, "add.u32", "atom", cs, "ld_ref_cycles"), row(t, "add.u32", "atom", cs)
        lat[lab] = {"load": round(ld, 1), "atom_add_u32": round(at, 1), "delta": round(at - ld, 1)}
        a.plot([ld, at], [i, i], color=LIGHTGREY, lw=2.2, zorder=2, solid_capstyle="butt")
        a.plot(ld, i, "o", color=GREY, ms=4.2, zorder=3)
        a.plot(at, i, "o", color=BLUE, ms=4.2, zorder=4)
        a.text(max(ld, at) * 1.07, i, f"+{at - ld:.1f}", va="center", fontsize=6.3, color=INK2)
    f32 = row("dsmem", "add.f32", "atom", 2)
    idsm = [p[0] for p in paths].index("dsmem")
    a.plot(f32, idsm, "D", color=ORANGE, ms=3.8, zorder=4)
    a.text(f32 * 1.1, idsm, f"{f32:.0f}, CAS\nloop", ha="left", va="center", fontsize=6.0, color=INK2)
    a.set_xscale("log")
    a.set_xlim(20, 1100)
    a.set_xticks([20, 50, 100, 200, 500, 1000], ["20", "50", "100", "200", "500", "1000"])
    a.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    a.set_yticks(range(len(paths)), [p[2] for p in paths])
    a.set_ylim(len(paths) - 0.5, -0.9)
    a.set_xlabel("dependent latency (SM cycles)")
    a.tick_params(axis="y", length=0)
    a.spines["left"].set_visible(False)
    _clean(a)
    a.legend(handles=[Line2D([], [], marker="o", color=GREY, ls="none", ms=4.2),
                      Line2D([], [], marker="o", color=BLUE, ls="none", ms=4.2),
                      Line2D([], [], marker="D", color=ORANGE, ls="none", ms=3.8)],
             labels=["load", "atom.add.u32", "atom.add.f32"], loc="upper right", fontsize=6.3,
             handletextpad=0.2, borderaxespad=0.1)
    sec = ns_axis(a, "x")
    sec.set_xticks([20, 50, 100, 200, 500], ["20", "50", "100", "200", "500"])
    sec.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    panel(a, "a", x=-0.02, y=1.12)

    # (b) ops per cycle per SM
    tp = [("smem", 1, "local smem"), ("smem_dsmem_if", 2, "own smem via window"), ("dsmem", 2, "remote DSMEM"),
          ("l2_near", 1, "L2 near"), ("hbm_near", 1, "HBM")]
    kinds = [("ld", GREY, "load"), ("atom", BLUE, "atom"), ("red+ld", ORANGE, "red")]
    h = 0.26
    thr = {}
    for i, (t, cs, lab) in enumerate(tp):
        thr[lab] = {}
        for j, (k, col, kl) in enumerate(kinds):
            v = row(t, "add.u32", k, cs, "sm_ops_per_cycle")
            thr[lab][kl] = round(v, 2)
            y = i + (j - 1) * h
            b.barh(y, v, height=h * 0.9, color=col)
            b.text(v + 0.25, y, f"{v:.1f}", va="center", fontsize=5.8, color=INK2)
    b.text(thr["local smem"]["red"] + 2.8, 0 + h, "constant +1:\nwarp-aggregated", va="center", fontsize=5.6,
           color=INK2)
    b.set_yticks(range(len(tp)), [t[2] for t in tp])
    b.set_ylim(len(tp) - 0.5, -0.55)
    b.set_xlim(0, 26)
    b.set_xlabel("operations per cycle per SM (1024 threads)")
    b.tick_params(axis="y", length=0)
    b.spines["left"].set_visible(False)
    _clean(b)
    b.legend(handles=[Rectangle((0, 0), 1, 1, fc=c) for _, c, _ in kinds], labels=[k[2] for k in kinds],
             loc="lower right", fontsize=6.3, handlelength=0.9, handletextpad=0.3)
    panel(b, "b", x=-0.02, y=1.12)

    q = pr[(pr.op == "add.u32") & (pr.kind == "atom") & (pr.variant == "identity")]
    dd = q.cycles_per_op - q.ld10_cycles
    info = {"latency": lat, "dsmem_pair_used": "CS 2 (one SM pair)",
            "dsmem_f32_add_cas_cycles": round(f32, 1),
            "dsmem_all_pairs": {"n_pairs": int(len(q)), "atom_median": round(q.cycles_per_op.median(), 1),
                                "load_median": round(q.ld10_cycles.median(), 1),
                                "delta_median": round(dd.median(), 2), "delta_min": round(dd.min(), 2),
                                "delta_max": round(dd.max(), 2)},
            "throughput_ops_per_cycle": thr,
            "l2_far_atom_ops_per_cycle": round(row("l2_far", "add.u32", "atom", 1, "sm_ops_per_cycle"), 2),
            "run": str(s.run_id.iloc[0])}
    return f, info


# ======================================================================================= §4 synchronisation
def fig_sync_costs():
    """Cost of each primitive (log cycles), grouped by scope; one warp vs 1024 threads; after a global store."""
    d = _sync()
    g = lambda lv, cfg, col="median": _sv(d, lv, cfg, col)
    w1, wT = "cs 1, threads 32", "cs 1, threads 1024"
    c16, c16T = "cs 16, threads 32", "cs 16, threads 1024"
    wake = d[d.level.str.startswith("k3/mbar_wakeup/")]["median"]
    st_after = [g("k2/cluster_arrive_release/after_st_global", f"cs {c}, threads 32") for c in (2, 4, 8, 16)]
    rows = [  # (group, label, one warp, 1024 threads, after a global store, range)
        ("CTA", "__syncthreads()", g("k1/bar_sync0/single", w1, "net_cycles"), g("k1/bar_sync0/single", wT, "net_cycles"), None, None),
        ("CTA", "bar.arrive (named)", g("k1/bar_arrive/single", "cs 1, threads 64"), None, None, None),
        ("CTA", "fence.acq_rel.cta", g("k1/fence_acq_rel_cta/single", w1), g("k1/fence_acq_rel_cta/single", wT), None, None),
        ("mbarrier", "mbarrier.init", g("k3/mbar_init/single", "cs 0, threads 1"), None, None, None),
        ("mbarrier", "mbarrier.arrive", g("k3/mbar_arrive/single", "cs 0, threads 32"), g("k3/mbar_arrive/single", "cs 0, threads 1024"), None, None),
        ("mbarrier", "arrive.expect_tx", g("k3/mbar_arrive_expect_tx/single", "cs 0, threads 1"), None, None, None),
        ("mbarrier", "try_wait, phase done", g("k3/mbar_try_wait/complete", "cs 0, threads 1"), None, None, None),
        ("mbarrier", "wake-up after completion", None, None, None, (wake.min(), wake.max())),
        ("mbarrier", "fence.proxy.async.shared::cta", g("k1/fence_proxy_async_shared_cta/single", w1), g("k1/fence_proxy_async_shared_cta/single", wT), None, None),
        ("mbarrier", "fence.proxy.async (all spaces)", g("k1/fence_proxy_async/single", w1), g("k1/fence_proxy_async/single", wT), None, None),
        ("cluster", "arrive.relaxed", g("k2/cluster_arrive_relaxed/single", c16), g("k2/cluster_arrive_relaxed/single", c16T), None, None),
        ("cluster", "wait, all arrived", g("k2/cluster_wait_acquire/all_arrived", c16), g("k2/cluster_wait_acquire/all_arrived", c16T), None, None),
        ("cluster", "arrive.release", g("k2/cluster_arrive_release/single", c16), g("k2/cluster_arrive_release/single", c16T), float(np.median(st_after)), None),
        ("cluster", "cluster.sync()", g("k2/cluster_sync/single", c16), g("k2/cluster_sync/single", c16T), None, None),
        ("GPU", "fence.sc.gpu", g("k1/fence_sc_gpu/single", w1), g("k1/fence_sc_gpu/single", wT), g("k1/fence_sc_gpu/after_st_global", w1), None),
        ("GPU", "grid.sync() (132 CTAs)", None, g("k5/grid_sync/b132", "cs 0, threads 1024"), g("k5/grid_sync/b132_after_st_global", "cs 0, threads 1024"), None),
    ]
    f, ax = plt.subplots(figsize=(COL, 3.9), constrained_layout=True)
    y, yt, ylab, gpos = 0, [], [], []
    last = None
    for grp, lab, v1, vT, vs, rg in rows:
        if grp != last:
            if last is not None:
                y += 0.6
            gpos.append((grp, y))
            last = grp
        yt.append(y)
        ylab.append(lab)
        if rg is not None:
            ax.plot(rg, [y, y], color=BLUE, lw=3.0, alpha=0.45, solid_capstyle="round")
        if v1 is not None and vT is not None:
            ax.plot([v1, vT], [y, y], color=LIGHTGREY, lw=1.6, zorder=2)
        if vT is not None:
            ax.plot(vT, y, "o", mfc="white", mec=BLUE, mew=1.1, ms=4.2, zorder=3)
        if v1 is not None:
            ax.plot(v1, y, "o", color=BLUE, ms=4.2, zorder=4)
        if vs is not None:
            ax.plot(vs, y, "x", color=ORANGE, ms=4.5, mew=1.3, zorder=5)
        y += 1
    ax.set_xscale("log")
    ax.set_xlim(4, 3000)
    ax.set_xticks([5, 10, 30, 100, 300, 1000], ["5", "10", "30", "100", "300", "1000"])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_yticks(yt, ylab, fontsize=6.8)
    ax.set_ylim(y - 0.4, -1.3)
    for grp, gy in gpos:
        ax.text(1.0, gy - 0.05, grp, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=6.8, color=INK2, fontstyle="italic")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    _clean(ax)
    ax.set_xlabel("cost (SM cycles)")
    ax.legend(handles=[Line2D([], [], marker="o", color=BLUE, ls="none", ms=4.2),
                       Line2D([], [], marker="o", mfc="white", mec=BLUE, mew=1.1, ls="none", ms=4.2),
                       Line2D([], [], marker="x", color=ORANGE, mew=1.3, ls="none", ms=4.5)],
              labels=["≤ 64 threads", "1024 threads", "after a global store"], loc="upper center",
              bbox_to_anchor=(0.42, 1.0), ncol=3, fontsize=6.3, handletextpad=0.1, columnspacing=0.8)
    info = {"rows": [{"scope": r[0], "primitive": r[1], "le_64_threads": r[2], "threads_1024": r[3],
                      "after_global_store": r[4], "range": None if r[5] is None else (float(r[5][0]), float(r[5][1]))}
                     for r in rows]}
    info["syncthreads_net_by_warps"] = {w: g("k1/bar_sync0/single", f"cs 1, threads {32 * w}", "net_cycles") for w in (1, 4, 16, 32)}
    info["cluster_sync_by_cs_32thr"] = {c: g("k2/cluster_sync/single", f"cs {c}, threads 32") for c in (1, 2, 4, 8, 16)}
    info["cluster_sync_by_cs_512thr"] = {c: g("k2/cluster_sync/single", f"cs {c}, threads 512") for c in (2, 4, 8, 16)}
    info["cluster_arrive_release_after_store_cs2_16"] = st_after
    info["fence_sc_gpu_by_threads"] = {t: g("k1/fence_sc_gpu/single", f"cs 1, threads {t}") for t in (32, 128, 512, 1024)}
    info["fence_sc_gpu_after_store_by_threads"] = {t: g("k1/fence_sc_gpu/after_st_global", f"cs 1, threads {t}") for t in (32, 128, 512, 1024)}
    info["fence_proxy_async_by_threads"] = {t: g("k1/fence_proxy_async/single", f"cs 1, threads {t}") for t in (32, 128, 512, 1024)}
    info["fence_proxy_async_shared_cta_by_threads"] = {t: g("k1/fence_proxy_async_shared_cta/single", f"cs 1, threads {t}") for t in (32, 128, 512, 1024)}
    info["grid_sync"] = {"b1": g("k5/grid_sync/b1", "cs 0, threads 1024"), "b132": g("k5/grid_sync/b132", "cs 0, threads 1024"),
                         "b132_after_store": g("k5/grid_sync/b132_after_st_global", "cs 0, threads 1024")}
    info["mbar_wakeup_range"] = (float(wake.min()), float(wake.max()))
    info["mbar_try_wait_incomplete_timeout"] = g("k3/mbar_try_wait/incomplete_nohint", "cs 0, threads 1")
    info["run"] = str(d.run_id.iloc[0])
    return f, info


def _handoff12():
    """Hand-off round trips of the atomics campaign (card B) from its exported results text (Table 10)."""
    path = os.path.join(D.EXP, D.EXPDIR["12"], f"{D.EXPDIR['12']}__results_summary.txt")
    txt = open(path).read()
    blk = txt.split("**Table 10")[1].split("**Table 11")[0]
    out = {}
    for prim, var, rt in re.findall(r"^(\w+)\s+(nogt(?:_poll1)?)\s+([\d.]+)", blk, flags=re.M):
        out[f"{prim}/{var}"] = float(rt)
    return out


def fig_handoff():
    """Round trip between two SMs of one CS-2 cluster per hand-off method (07, SMs 124/125)."""
    d = _sync()
    cfg = lambda lv: d[(d.level == lv) & d.config.str.startswith("cs 2,")]
    rt = lambda m, v: float(cfg(f"k4/{m}/{v}/round_trip_cycles")["median"].iloc[0])
    ow = lambda m, v: float(cfg(f"k4/{m}/{v}/one_way_ns")["median"].iloc[0])
    M = [  # (label, method, variant, class, second variant for a range)
        ("remote mbarrier arrive (relaxed)", "remote_mbarrier_arrive_relaxed", "push_r01", 0, None),
        ("poll peer's flag (remote ld.volatile)", "flag_pull_ld_volatile", "pull_r01", 0, None),
        ("st.async + mbarrier", "st_async_mbarrier", "push_r01", 1, None),
        ("red.async + mbarrier", "red_async_mbarrier", "push_r01", 1, None),
        ("two cluster.sync()", "cluster_sync", "push_r01", 2, None),
        ("remote st + local ld.volatile poll", "flag_st_ld_volatile", "push_r01", 0, None),
        ("remote mbarrier arrive (release)", "remote_mbarrier_arrive_release", "push_r01", 2, None),
        ("atom.exch flag", "flag_atom_exch", "push_r01", 2, None),
        ("st.release / ld.acquire flag", "flag_release_acquire", "push_r01", 2, None),
        ("relaxed flag + fence each side", "flag_relaxed_fence", "push_r01", 2, None),
        ("global flag (L2, release/acquire)", "gmem_flag_release_acquire", "off4096_r01_home0", 2, "off0_r01_home1"),
    ]
    cls = {0: (GREY, "flag only"), 1: (BLUE, "delivered data visible"), 2: (ORANGE, "all prior writes visible")}
    vals = [(lab, rt(m, v), ow(m, v), c, None if v2 is None else rt(m, v2)) for lab, m, v, c, v2 in M]
    vals.sort(key=lambda t: t[1])
    f, ax = plt.subplots(figsize=(COL, 3.05), layout="constrained")
    for i, (lab, r, o, c, r2) in enumerate(vals):
        ax.barh(i, r, color=cls[c][0], height=0.66)
        if r2 is not None:
            ax.plot([r, r2], [i, i], color=INK2, lw=0.9)
            ax.plot(r2, i, "|", color=INK2, ms=5, mew=0.9)
            ax.text(r2 + 40, i, f"{r:.0f} / {r2:.0f}", va="center", fontsize=6.2, color=INK2)
        else:
            ax.text(r + 40, i, f"{r:.0f}", va="center", fontsize=6.2, color=INK2)
    ax.set_yticks(range(len(vals)), [v[0] for v in vals], fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 2850)
    ax.set_xlabel("round trip (SM cycles)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    _clean(ax)
    ns_axis(ax, "x", label="round trip (ns)")
    f.legend(handles=[Rectangle((0, 0), 1, 1, fc=cls[k][0]) for k in (0, 1, 2)],
             labels=[cls[k][1] for k in (0, 1, 2)], loc="outside lower center", ncol=3, fontsize=6.2,
             handlelength=0.9, handletextpad=0.3, columnspacing=0.9, title="what the receiver may assume",
             title_fontsize=6.2)

    # pair independence (card B heatmap) and code-layout sensitivity of polling hand-offs
    s10 = D.latest(D.read(D.res("10", "10__heatmap__summary.csv")))
    v10 = D.latest(D.read(D.res("10", "10__heatmap__visibility.csv")))
    s09 = D.latest(D.read(D.res("09", "09__dsmem_latency__summary.csv")))
    g09 = lambda lv: float(s09[(s09.level == lv) & (s09.config == "cs 2")]["median"].iloc[0])
    one = lambda lv: s10[s10.level == lv].iloc[0]
    vs = v10.groupby("variant").one_way_cycles_median.agg(["min", "median", "max", "size"])
    h12 = _handoff12()
    info = {"cs2_round_trip_cycles": {v[0]: v[1] for v in vals},
            "cs2_one_way_ns": {v[0]: v[2] for v in vals},
            "global_flag_home_other_partition": [v[4] for v in vals if v[4] is not None][0],
            "pair_independence_10": {
                "st_async_one_way_cycles_median": float(one("vis_st_async/one_way_cycles")["median"]),
                "st_async_p5_p95": (float(one("vis_st_async/one_way_cycles")["p5"]), float(one("vis_st_async/one_way_cycles")["p95"])),
                "plain_one_way_cycles_median": float(one("vis_st_plain/one_way_cycles")["median"]),
                "plain_p5_p95": (float(one("vis_st_plain/one_way_cycles")["p5"]), float(one("vis_st_plain/one_way_cycles")["p95"])),
                "n_pairs": int(one("vis_st_async/one_way_cycles")["n"]),
                "per_pair_one_way_by_variant": vs.round(1).to_dict(orient="index")},
            "code_layout": {
                "09_st_plain_round_trip": g09("k2/st_plain_nogt/r01/round_trip_cycles"),
                "09_st_async_round_trip": g09("k2/st_async_mbarrier_nogt/r01/round_trip_cycles"),
                "10_st_plain_round_trip": 2 * float(one("vis_st_plain/one_way_cycles")["median"]),
                "10_st_async_round_trip": 2 * float(one("vis_st_async/one_way_cycles")["median"]),
                "12_round_trips": h12,
                "07_st_plain_round_trip": rt("flag_st_ld_volatile", "push_r01"),
                "07_st_async_round_trip": rt("st_async_mbarrier", "push_r01")},
            "run": str(d.run_id.iloc[0])}
    return f, info


# ======================================================================================= cheat sheet
def facts():
    _, i4 = fig_gpc_occupancy()
    _, i5 = fig_dsmem_latency_cs()
    _, i6 = fig_dsmem_heatmap()
    _, i7 = fig_dsmem_bandwidth()
    _, i8 = fig_dsmem_atomics()
    _, i11 = fig_sync_costs()
    _, i12 = fig_handoff()
    plt.close("all")
    rt = i12["cs2_round_trip_cycles"]
    return {
        "gpcs": 8, "gpc_sizes": i4["gpc_sizes"], "pair_only_sms": i4["pair_only_sms"],
        "max_active_clusters_1cta": i4["max_active_clusters"], "idle_sms_1cta": i4["idle_sms"],
        "dsmem_remote_load_median_by_cs": {c: v["median"] for c, v in i5["per_cs"].items()},
        "dsmem_pair_latency_range": (i6["pair_latency_min"], i6["pair_latency_max"]),
        "dsmem_pair_latency_median": i6["pair_latency_median"],
        "dsmem_additive_resid_std": i6["resid_std"], "dsmem_p_range": (i6["p_min"], i6["p_max"]),
        "own_cta_via_dsmem_path": i5["own_cta_via_dsmem_path"],
        "local_smem_load": i5["refs"]["smem"], "l2_chase": i5["refs"]["l2"], "hbm_chase": i5["refs"]["hbm"],
        "dsmem_send_limit_bpc": i7["send_limit_bpc"], "dsmem_receive_max_bpc": max(i7["fan_in_st_total_bpc"].values()),
        "dsmem_gpu_wide_tbps": {k: i7[f"gpu_wide_{k}_tbps"] for k in ("ld", "st", "tma_put")},
        "l2_read_tbps": i7["l2_read_tbps_ref"], "hbm_read_tbps": i7["hbm_read_tbps_ref"],
        "ring_ld_8_vs_16_sms": (i7["ring_bpc_per_sm"]["ld_cs8_single"], i7["ring_bpc_per_sm"]["ld_cs16_single"]),
        "dsmem_atom_minus_load": i8["dsmem_all_pairs"]["delta_median"],
        "dsmem_atom_ops_per_cycle": i8["throughput_ops_per_cycle"]["remote DSMEM"]["atom"],
        "dsmem_f32_add_cas": i8["dsmem_f32_add_cas_cycles"],
        "syncthreads_net": i11["syncthreads_net_by_warps"], "cluster_sync_32thr": i11["cluster_sync_by_cs_32thr"],
        "fence_sc_gpu": i11["fence_sc_gpu_by_threads"], "fence_sc_gpu_after_store": i11["fence_sc_gpu_after_store_by_threads"],
        "grid_sync": i11["grid_sync"],
        "handoff_st_async_rt": rt["st.async + mbarrier"], "handoff_relaxed_arrive_rt": rt["remote mbarrier arrive (relaxed)"],
        "handoff_release_acquire_rt": rt["st.release / ld.acquire flag"],
        "handoff_global_flag_rt": (rt["global flag (L2, release/acquire)"], i12["global_flag_home_other_partition"]),
    }


FIGS = ["gpc_occupancy", "dsmem_latency_cs", "dsmem_heatmap", "dsmem_bandwidth", "dsmem_atomics", "sync_costs",
        "handoff"]

if __name__ == "__main__":
    import json
    out = os.path.join(D.CACHE, "preview")
    os.makedirs(out, exist_ok=True)
    for name in FIGS:
        fg, info = globals()[f"fig_{name}"]()
        fg.savefig(os.path.join(out, f"{name}.png"))
        plt.close(fg)
        print(f"== {name}\n{json.dumps(info, default=str, indent=None)[:4000]}\n")
    print("== facts\n", json.dumps(facts(), default=str))
