"""Figures, tables and numbers for the progress slides (presentation/slides.md).

Every number comes from the local measurement results through reports/h100char (the report's data layer).
Run: tools/run --no-gpu -- python3 presentation/make_figures.py
Writes presentation/figs/<name>.png (matplotlib, 300 dpi), presentation/collective_algorithms.csv, presentation/numbers.txt.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "reports"))
sys.path.insert(0, HERE)
import slide_style as S                                                     # noqa: E402
from h100char import data, breakdown as B, figs_coll as C                   # noqa: E402
from h100char import figs_mem_tma as A, figs_dsmem_sync as D                # noqa: E402

S.apply()
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)
NUM = []                       # (slide, quantity, value) dumped to numbers.txt


def note(slide, what, val):
    NUM.append((slide, what, val))


def save(fig, name):
    fig.savefig(os.path.join(FIGS, f"{name}.png"))
    plt.close(fig)


def bar_labels(ax, rects, fmt="{:.0f}", horizontal=False, pad=3, size=10.5):
    for r in rects:
        v = r.get_width() if horizontal else r.get_height()
        if horizontal:
            ax.annotate(fmt.format(v), (r.get_x() + v, r.get_y() + r.get_height() / 2), xytext=(pad, 0),
                        textcoords="offset points", va="center", ha="left", fontsize=size, color=S.INK2)
        else:
            ax.annotate(fmt.format(v), (r.get_x() + r.get_width() / 2, v), xytext=(0, pad),
                        textcoords="offset points", va="bottom", ha="center", fontsize=size, color=S.INK2)


# =============================================================================================== slide 1
def mem_latency():
    a = A.facts()
    rows = [("SMEM", a["lat_smem_cycles"], S.GREY), ("L1", a["lat_l1_cycles"], S.GREY),
            ("DSMEM (remote SM)", a["lat_dsmem_remote_cycles"], S.ORANGE),
            ("L2, near partition", a["lat_l2_near_cycles"], S.GREY),
            ("TMA load, L2", a["tma_chase_l2_cycles"], S.BLUE),
            ("L2, far partition", a["lat_l2_far_cycles"], S.GREY),
            ("HBM, near partition", a["lat_hbm_near_cycles"], S.GREY),
            ("TMA load, HBM", a["tma_chase_hbm_cycles"], S.BLUE),
            ("HBM, far partition", a["lat_hbm_far_cycles"], S.GREY)]
    f, ax = plt.subplots(figsize=S.HALF, constrained_layout=True)
    y = np.arange(len(rows))[::-1]
    rects = ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.68)
    bar_labels(ax, rects, horizontal=True)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlim(0, 780)
    ax.set_xlabel("Latency (cycles)")
    for r in rows:
        note(1, f"latency {r[0]}", round(r[1], 1))
    note(1, "plain ld.global chase L2 / HBM", (a["lat_l2_chase_cycles"], a["lat_hbm_chase_cycles"]))
    note(1, "TMA copy model T0, B/cycle", (a["tma_copy_T0_cycles"], a["tma_copy_bpc"]))
    note(1, "DSMEM / L2 near", round(a["lat_dsmem_remote_cycles"] / a["lat_l2_near_cycles"], 2))
    save(f, "mem_latency")


def mem_bandwidth():
    a, d = A.facts(), D.facts()
    s05 = data.pinned(data.read(data.res("05", "05__bandwidth__summary.csv")), data.PIN["05_summary"])
    smem_st = float(s05.set_index("level").loc["smem_st", "tbps"])
    gw = d["dsmem_gpu_wide_tbps"]
    dmin = min(min(v.values()) for v in gw.values())
    dmax = max(max(v.values()) for v in gw.values())
    f, (a1, a2) = plt.subplots(1, 2, figsize=S.FULL, constrained_layout=True, gridspec_kw={"width_ratios": [1.15, 1]})
    # (a) aggregate, TB/s, log x
    agg = [("SMEM load", a["bw_smem_tbps"], S.GREY), ("SMEM store", smem_st, S.GREY),
           ("L2 read", a["bw_l2_read_tbps"], S.GREY), ("L2 write", a["bw_l2_store_tbps"], S.GREY),
           ("HBM read", a["bw_hbm_read_tbps"], S.GREY), ("HBM write", a["bw_hbm_write_tbps"], S.GREY),
           ("DSMEM (CS 2–16)", float(np.median([v for d_ in gw.values() for v in d_.values()])), S.ORANGE)]
    y = np.arange(len(agg))[::-1]
    rects = a1.barh(y, [r[1] for r in agg], color=[r[2] for r in agg], height=0.68)
    a1.errorbar(agg[-1][1], y[-1], xerr=[[agg[-1][1] - dmin], [dmax - agg[-1][1]]], fmt="none",
                ecolor=S.INK, elinewidth=1.2, capsize=4)
    for r, (lab, v, _) in zip(rects, agg):
        txt = f"{dmin:.1f}–{dmax:.1f}" if lab.startswith("DSMEM") else f"{v:.1f}" if v >= 10 else f"{v:.2f}"
        x = dmax if lab.startswith("DSMEM") else v
        a1.annotate(txt, (x, r.get_y() + r.get_height() / 2), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=10.5, color=S.INK2)
    a1.set_yticks(y, [r[0] for r in agg])
    a1.tick_params(axis="y", length=0)
    a1.set_xscale("log")
    a1.set_xlim(1, 80)
    a1.set_xticks([1, 3, 10, 30], ["1", "3", "10", "30"])
    a1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a1.grid(axis="y", visible=False)
    a1.grid(axis="x", visible=True)
    a1.set_xlabel("Aggregate bandwidth (TB/s)")
    # (b) per SM, GB/s
    per = [("SMEM", S.GBS(a["bw_smem_bpc_per_sm"]), S.GREY),
           ("TMA load (one copy)", S.GBS(a["tma_copy_bpc"]), S.BLUE),
           ("L2 read", S.GBS(a["bw_l2_read_one_sm_bpc"]), S.GREY),
           ("HBM read", a["bw_hbm_read_one_sm_gbps"], S.GREY),
           ("DSMEM ejection", S.GBS(np.median([32.25, 32.75, 34.38, 32.16, 32.31, 32.09])), S.ORANGE),
           ("DSMEM injection", S.GBS(d["dsmem_send_limit_bpc"]), S.ORANGE)]
    fan_in = list(D.fig_dsmem_bandwidth()[1]["fan_in_st_total_bpc"].values())[1:] + \
        list(D.fig_dsmem_bandwidth()[1]["fan_in_tma_put_total_bpc"].values())[1:]
    plt.close("all")
    per[4] = ("DSMEM ejection", S.GBS(float(np.median(fan_in))), S.ORANGE)
    y = np.arange(len(per))[::-1]
    rects = a2.barh(y, [r[1] for r in per], color=[r[2] for r in per], height=0.68)
    bar_labels(a2, rects, horizontal=True)
    a2.set_yticks(y, [r[0] for r in per])
    a2.tick_params(axis="y", length=0)
    a2.grid(axis="y", visible=False)
    a2.grid(axis="x", visible=True)
    a2.set_xlim(0, 250)
    a2.set_xlabel("Bandwidth of one SM (GB/s)")
    for lab, v, _ in agg:
        note(1, f"aggregate {lab} TB/s", round(v, 2))
    note(1, "aggregate DSMEM TB/s range", (dmin, dmax))
    for lab, v, _ in per:
        note(1, f"per-SM {lab} GB/s", round(v, 1))
    note(1, "HBM read 90 % at SMs", a["bw_hbm_read_90pct_at_sms"])
    save(f, "mem_bandwidth")


# =============================================================================================== slide 2
def dsmem_latency_cs():
    i = D.fig_dsmem_latency_cs()[1]
    plt.close("all")
    cs = [2, 4, 8, 16]
    med = [i["per_cs"][c]["median"] for c in cs]
    lo = [i["per_cs"][c]["median"] - i["per_cs"][c]["p5"] for c in cs]
    hi = [i["per_cs"][c]["p95"] - i["per_cs"][c]["median"] for c in cs]
    f, ax = plt.subplots(figsize=S.HALF, constrained_layout=True)
    x = np.arange(len(cs))
    ax.bar(x, med, color=S.ORANGE, width=0.6, yerr=[lo, hi], capsize=4,
           error_kw={"elinewidth": 1.1, "ecolor": S.INK})
    l2 = i["refs"]["l2"]
    ax.axhline(l2, ls=":", color=S.INK2, lw=1.2)
    ax.set_xticks(x, [str(c) for c in cs])
    ax.set_xlabel("Cluster size")
    ax.set_ylabel("Latency (cycles)")
    ax.set_ylim(0, 300)
    ax.legend(handles=[Patch(color=S.ORANGE, label="DSMEM load, median (p5–p95)"),
                       Line2D([], [], ls=":", color=S.INK2, label=f"L2 hit ({l2:.0f})")],
              loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, fontsize=10)
    for c, m in zip(cs, med):
        note(2, f"DSMEM median CS {c}", m)
        note(2, f"DSMEM p5-p95 CS {c}", (i["per_cs"][c]["p5"], i["per_cs"][c]["p95"]))
    save(f, "dsmem_latency_cs")


def dsmem_heatmap(gpc=6):
    p = data.latest(data.read(data.res("10", "10__heatmap__pair_stats.csv")))
    p = p[(p.matrix == "chase") & (p.cs.astype(str) == "all") & (p.gpc == gpc)]
    sms = sorted(set(p.smid_src) | set(p.smid_dst))
    m = p.pivot(index="smid_src", columns="smid_dst", values="median").reindex(index=sms, columns=sms)
    f, ax = plt.subplots(figsize=(5.2, 4.3), constrained_layout=True)
    im = ax.imshow(m.values, cmap="Blues", vmin=185, vmax=220)
    ax.set_xticks(range(len(sms)), [str(s) for s in sms], rotation=90, fontsize=8.5)
    ax.set_yticks(range(len(sms)), [str(s) for s in sms], fontsize=8.5)
    ax.grid(False)
    ax.set_xlabel("Destination SM")
    ax.set_ylabel("Source SM")
    cb = f.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("Latency (cycles)")
    cb.outline.set_visible(False)
    allp = data.latest(data.read(data.res("10", "10__heatmap__pair_stats.csv")))
    allp = allp[(allp.matrix == "chase") & (allp.cs.astype(str) == "all")]
    note(2, "pair latency range / median (all 1876 pairs)", (allp["median"].min(), allp["median"].max(), allp["median"].median()))
    note(2, f"GPC {gpc} pair range", (float(np.nanmin(m.values)), float(np.nanmax(m.values))))
    save(f, "dsmem_heatmap")


# =============================================================================================== slide 3
def dsmem_bandwidth():
    i = D.fig_dsmem_bandwidth()[1]
    plt.close("all")
    f, (a1, a2) = plt.subplots(1, 2, figsize=S.FULL, constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.1]})
    # (a) per-SM injection / ejection, GB/s
    inj = [("1 peer", i["fan_out_st_total_bpc"][1]), ("15 peers", i["fan_out_st_total_bpc"][15])]
    ej = [(f"{k} peer" + ("s" if k > 1 else ""), i["fan_in_st_total_bpc"][k]) for k in (1, 3, 7, 15)]
    xs = [0, 1, 2.6, 3.6, 4.6, 5.6]
    vals = [S.GBS(v) for _, v in inj + ej]
    cols = [S.ORANGE] * 2 + [S.BLUE] * 4
    rects = a1.bar(xs, vals, color=cols, width=0.72)
    bar_labels(a1, rects)
    a1.set_xticks(xs, ["1", "15", "1", "3", "7", "15"])
    a1.set_xlabel("Peers")
    a1.set_ylabel("Bandwidth of one SM (GB/s)")
    a1.set_ylim(0, 68)
    a1.legend(handles=[Patch(color=S.ORANGE, label="injection (to k peers)"),
                       Patch(color=S.BLUE, label="ejection (from k peers)")],
              loc="upper left", ncol=1, fontsize=10)
    # (b) aggregate vs cluster size
    cs = [2, 4, 8, 16]
    ops = [("ld", "gpu_wide_ld_tbps", S.GREY), ("st", "gpu_wide_st_tbps", S.ORANGE), ("TMA put", "gpu_wide_tma_put_tbps", S.BLUE)]
    w = 0.26
    for k, (lab, key, col) in enumerate(ops):
        a2.bar(np.arange(4) + (k - 1) * w, [i[key][c] for c in cs], width=w, color=col, label=lab)
    a2.axhline(i["hbm_read_tbps_ref"], ls=":", color=S.INK2, lw=1.2)
    a2.set_xticks(range(4), [f"{c}\n({i['gpu_wide_active_sms'][c]} SMs)" for c in cs], fontsize=10)
    a2.set_xlabel("Cluster size (active SMs)")
    a2.set_ylabel("Aggregate DSMEM bandwidth (TB/s)")
    a2.set_ylim(0, 4.8)
    h = a2.get_legend_handles_labels()[0] + [Line2D([], [], ls=":", color=S.INK2, label=f"HBM read ({i['hbm_read_tbps_ref']:.2f})")]
    a2.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=4, fontsize=10, columnspacing=1.0)
    for lab, v in inj:
        note(3, f"injection {lab} GB/s", round(S.GBS(v), 1))
    for lab, v in ej:
        note(3, f"ejection {lab} GB/s", round(S.GBS(v), 1))
    for lab, key, _ in ops:
        note(3, f"aggregate {lab} TB/s by CS", i[key])
    note(3, "ld B/cycle by bytes in flight", i["ld_bpc_by_inflight"])
    note(3, "ring B/cycle per SM single vs full GPC", i["ring_bpc_per_sm"])
    save(f, "dsmem_bandwidth")


# =============================================================================================== slide 4
def sync_costs():
    s07 = data.latest(data.read(data.res("07", "07__sync__summary_v2.csv")))
    c = B.costs()

    def v(level, cfg, col="median"):
        return float(s07[(s07.level == level) & (s07.config == cfg)][col].iloc[0])

    def at256(level, cs=1, col="median"):
        return B._interp(v(level, f"cs {cs}, threads 128", col), v(level, f"cs {cs}, threads 512", col))
    rows = [("CTA", "__syncthreads()", c["syncthreads"]),
            ("CTA", "fence.proxy.async.shared::cta", c["fence_proxy_async"]),
            ("mbarrier", "mbarrier.arrive", v("k3/mbar_arrive/single", "cs 0, threads 32")),
            ("mbarrier", "mbarrier.arrive.expect_tx", c["expect_tx"]),
            ("mbarrier", "mbarrier.try_wait (phase complete)", v("k3/mbar_try_wait/complete", "cs 0, threads 1")),
            ("cluster", "barrier.cluster arrive.relaxed + wait",
             at256("k2/cluster_arrive_relaxed/single", 8) + v("k2/cluster_wait_acquire/all_arrived", "cs 8, threads 128")),
            ("cluster", "barrier.cluster (no prior store)", c[("barrier_nostore", 8)]),
            ("cluster", "barrier.cluster (after a store)", c[("barrier", 8)]),
            ("GPU", "fence.sc.gpu (no prior store)", at256("k1/fence_sc_gpu/single")),
            ("GPU", "fence.sc.gpu (after a store)", at256("k1/fence_sc_gpu/after_st_global")),
            ("GPU", "grid.sync()", v("k5/grid_sync/b132", "cs 0, threads 1024"))]
    col = {"CTA": S.GREY, "mbarrier": S.BLUE, "cluster": S.ORANGE, "GPU": S.MAGENTA}
    f, ax = plt.subplots(figsize=(5.6, 4.4), constrained_layout=True)
    y = np.arange(len(rows))[::-1]
    rects = ax.barh(y, [r[2] for r in rows], color=[col[r[0]] for r in rows], height=0.66)
    bar_labels(ax, rects, horizontal=True)
    ax.set_yticks(y, [r[1] for r in rows], fontsize=10.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xscale("log")
    ax.set_xlim(5, 4000)
    ax.set_xticks([10, 100, 1000], ["10", "100", "1000"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("Cost (cycles)")
    f.legend(handles=[Patch(color=col[k], label=k) for k in col], loc="outside upper center", ncol=4, fontsize=10.5)
    for g, lab, val in rows:
        note(4, f"{g}: {lab}", round(val, 1))
    save(f, "sync_costs")


def handoff():
    i = D.fig_handoff()[1]
    plt.close("all")
    rt = i["cs2_round_trip_cycles"]
    pick = lambda s: next(v for k, v in rt.items() if s in k)
    rows = [("st.async + mbarrier", pick("st.async"), S.BLUE),
            ("st + ld.volatile poll", pick("remote st + local"), S.GREY),
            ("remote mbarrier.arrive.release", pick("arrive (release)"), S.ORANGE),
            ("st.release / ld.acquire (SMEM)", pick("st.release"), S.ORANGE),
            ("st.release / ld.acquire (global)", pick("global flag"), S.ORANGE)]
    f, ax = plt.subplots(figsize=(5.4, 3.8), constrained_layout=True)
    y = np.arange(len(rows))[::-1]
    rects = ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.62)
    gf = i["global_flag_home_other_partition"]
    ax.plot([rows[-1][1], gf], [y[-1], y[-1]], color=S.INK, lw=1.2)
    ax.plot([gf, gf], [y[-1] - 0.16, y[-1] + 0.16], color=S.INK, lw=1.2)
    for r, (lab, val, _) in zip(rects, rows):
        txt = f"{val:.0f}–{gf:.0f}" if "global" in lab else f"{val:.0f}"
        x = gf if "global" in lab else val
        ax.annotate(txt, (x, r.get_y() + r.get_height() / 2), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=10.5, color=S.INK2)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=10.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlim(0, 2700)
    ax.set_xlabel("Round trip (cycles)")
    f.legend(handles=[Patch(color=S.BLUE, label="data + completion (no fence)"),
                      Patch(color=S.GREY, label="no ordering guarantee"),
                      Patch(color=S.ORANGE, label="release / acquire")],
             loc="outside upper center", ncol=1, fontsize=10.5)
    for lab, val, _ in rows:
        note(4, f"round trip {lab}", val)
    note(4, "global flag, home in other partition", gf)
    save(f, "handoff")


# =============================================================================================== slide 5
ALG = {  # file -> (algorithm, origin, rounds, transport)
    "allreduce__ring__nccl": ("Ring (RS + AG)", "NCCL", "2(N−1)", "bulk copy"),
    "allreduce__recursive_doubling__clusterfusion": ("Recursive doubling", "ClusterFusion", "log₂N", "bulk copy"),
    "allreduce__rabenseifner__mpich": ("Rabenseifner", "MPICH", "2·log₂N", "bulk copy"),
    "allreduce__reduce_bcast_binomial__mpich": ("Binomial reduce + broadcast", "MPICH", "2·log₂N", "bulk copy"),
    "allreduce__two_shot_push__trtllm": ("Two-shot push", "TensorRT-LLM", "2", "bulk copy"),
    "allreduce__one_shot_pull__trtllm": ("One-shot pull", "TensorRT-LLM", "1", "remote load"),
    "allreduce__direct_replicated_push__credit": ("Replicated direct push", "CREDIT", "1", "remote store"),
    "allreduce__pull_reduce_scatter_push_allgather__hpcops": ("Pull RS + push AG", "hpc-ops", "1", "remote load/store"),
    "allreduce__shift_accumulate_bulk__clusterfusion": ("Shift-accumulate (bulk)", "ClusterFusion", "N−2", "bulk copy"),
    "allreduce__shift_accumulate_atomic__clusterfusion": ("Shift-accumulate (atomic)", "ClusterFusion", "N−2", "remote atomic"),
    "allreduce__atomic_all_push__generic": ("Atomic all-push", "generic", "1", "remote atomic"),
    "allreduce__2d_mesh__generic": ("2-D mesh", "generic", "3", "bulk copy"),
    "allgather__ring__nccl": ("Ring", "NCCL", "N−1", "bulk copy"),
    "allgather__recursive_doubling__clusterfusion": ("Recursive doubling", "ClusterFusion", "log₂N", "bulk copy"),
    "allgather__direct_shift__clusterfusion": ("Direct shift", "ClusterFusion", "N−1", "bulk copy"),
    "allgather__bruck__mpich": ("Bruck", "MPICH", "⌈log₂N⌉", "bulk copy"),
    "allgather__gather_bcast_binomial__mpich": ("Binomial gather + broadcast", "MPICH", "2·log₂N", "bulk copy"),
    "allgather__direct_replicated_push__credit": ("Direct push", "CREDIT", "1", "remote store"),
    "allgather__direct_pull__generic": ("Direct pull", "generic", "1", "remote load"),
    "reducescatter__ring__nccl": ("Ring", "NCCL", "N−1", "bulk copy"),
    "reducescatter__recursive_halving__mpich": ("Recursive halving", "MPICH", "log₂N", "bulk copy"),
    "reducescatter__direct_push__generic": ("Direct push", "generic", "1", "bulk copy"),
    "reducescatter__direct_pull__generic": ("Direct pull", "generic", "1", "remote load"),
    "alltoall__pairwise_xor__mpich": ("Pairwise XOR", "MPICH", "N−1", "bulk copy"),
    "alltoall__bruck__mpich": ("Bruck", "MPICH", "log₂N", "bulk copy"),
    "alltoall__shift__generic": ("Shift", "generic", "N−1", "bulk copy"),
    "alltoall__direct_push__generic": ("Direct push", "generic", "1", "bulk copy"),
    "alltoall__direct_pull__generic": ("Direct pull", "generic", "1", "remote load"),
    "alltoall__pairwise_exchange__fa4": ("Pairwise exchange", "FlashAttention-4", "1 (N = 2 only)", "bulk copy"),
}
CLASS_OF = dict(zip(C.CF.STYLES, S.CLASSES))
OPS = ["allreduce", "allgather", "reducescatter", "alltoall"]
OPNAME = {"allreduce": "AllReduce", "allgather": "AllGather", "reducescatter": "ReduceScatter", "alltoall": "AlltoAll"}


def algorithm_table():
    rows = []
    for f, (alg, src, rounds, tr) in ALG.items():
        op = f.split("__")[0]
        rows.append({"Collective": OPNAME[op], "Algorithm": alg, "Origin": src,
                     "Synchronization": CLASS_OF[C.CF.sync_style(f)], "Rounds": rounds, "Transport": tr})
    t = pd.DataFrame(rows)
    order = {c: i for i, c in enumerate(S.CLASSES)}
    t["_o"] = t.Collective.map({OPNAME[o]: i for i, o in enumerate(OPS)}) * 10 + t.Synchronization.map(order)
    t = t.sort_values(["_o", "Algorithm"]).drop(columns="_o")
    t.to_csv(os.path.join(HERE, "collective_algorithms.csv"), index=False)
    return t


# =============================================================================================== slides 6, 7
SIZES = [256, 1024, 4096, 16384, 65536]
SIZE_LAB = ["256 B", "1 KiB", "4 KiB", "16 KiB", "64 KiB"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "p"]


def _ncu_bytes():
    def build():
        p = data.coll("ncu.csv")
        d = pd.read_csv(p, usecols=["run_id", "file", "N", "M", "kernel", "metric", "value"], low_memory=False)
        d = d[(d.metric == "l1tex__m_l1tex2xbar_write_bytes_mem_dshared.sum") & (d.kernel == "collective_kernel")]
        d = d[d.run_id == d.groupby(["file", "N", "M"]).run_id.transform("max")]
        d = d.assign(value=pd.to_numeric(d.value, errors="coerce"))
        return d.groupby(["file", "N", "M"]).value.mean().rename("dsmem_bytes").reset_index()
    return data.cached("coll_ncu_dsmem_bytes.csv", build)


def envelope(N=8):
    """Per collective, class and message size: the lowest-mean-latency algorithm of that class."""
    c = C.configs()
    c = c[(c.N == N) & c.payload.isin(SIZES)].copy()
    c["cls"] = c["style"].map(CLASS_OF)
    c["L_cyc"] = c.L_ns * S.CYC_PER_NS
    e = c.loc[c.groupby(["op", "cls", "payload"]).L_ns.idxmin()].copy()
    e = e.merge(_ncu_bytes(), on=["file", "N", "M"], how="left")
    e["gbs_per_sm"] = e.dsmem_bytes / e.N / e.L_ns
    e["util_pct"] = e.gbs_per_sm / S.GBS(64 / 3) * 100
    return e


def _glyphs(d):
    """Glyph per algorithm within one collective (panel), in class then size order."""
    files = list(dict.fromkeys(d.sort_values(["cls", "payload"], key=lambda c: c.map(
        {k: i for i, k in enumerate(S.CLASSES)}) if c.name == "cls" else c).file))
    return {f: MARKERS[i] for i, f in enumerate(files)}


def _coll_bars(e, col, ylab, name, log=False, ylim=None, ref=None):
    f, axs = plt.subplots(1, 4, figsize=(13, 5.6), constrained_layout=True, sharey=log)
    w = 0.26
    glyphs = {}
    for ax, op in zip(axs, OPS):
        d = e[e.op == op]
        gl = glyphs[op] = _glyphs(d)
        classes = [k for k in S.CLASSES if k in set(d.cls)]
        off = {k: (i - (len(classes) - 1) / 2) * w for i, k in enumerate(classes)}
        for k in classes:
            g = d[d.cls == k].set_index("payload").reindex(SIZES)
            x = np.arange(len(SIZES)) + off[k]
            ax.bar(x, g[col], width=w, color=S.CLASS_COL[k])
            for xi, v, fl in zip(x, g[col], g.file):
                if pd.notna(v):
                    yv = v * 1.22 if log else v + (ylim[1] if ylim else 100) * 0.035
                    ax.plot(xi, yv, marker=gl[fl], color=S.INK, ms=5.5, mfc="white", mew=1.0, ls="none", zorder=5)
        ax.set_xticks(range(len(SIZES)), SIZE_LAB, rotation=45, fontsize=10)
        ax.set_title(OPNAME[op], fontsize=12, color=S.INK)
        if log:
            ax.set_yscale("log")
        if ylim:
            ax.set_ylim(*ylim)
        if ref is not None:
            ax.axhline(ref, ls=":", color=S.INK2, lw=1.2)
        ax.set_xlabel("Message size")
        h = [Line2D([], [], marker=m, color=S.INK, mfc="white", ls="none", ms=6,
                    label=f"{ALG[fl][0]} ({ALG[fl][1]})") for fl, m in gl.items()]
        ax.legend(handles=h, loc="upper left", bbox_to_anchor=(-0.02, -0.3), ncol=1, fontsize=9.5,
                  handletextpad=0.3, borderaxespad=0)
    axs[0].set_ylabel(ylab)
    f.legend(handles=[Patch(color=S.CLASS_COL[k], label=k) for k in S.CLASSES], loc="outside upper center",
             ncol=3, fontsize=11)
    save(f, name)
    return glyphs


def coll_latency():
    e = envelope()
    _coll_bars(e, "L_cyc", "Latency (cycles)", "coll_latency", log=True, ylim=(300, 1e5))
    for _, r in e.sort_values(["op", "cls", "payload"]).iterrows():
        note(6, f"{r.op} {r.cls} {int(r.payload)} B -> {ALG[r.file][0]}", round(r.L_cyc))
    for op in ["allreduce", "reducescatter", "alltoall"]:
        d = e[e.op == op].pivot(index="payload", columns="cls", values="L_cyc")
        ratio = d[["single-exchange", "round-synchronous"]].min(axis=1) / d["mbarrier-only"]
        note(6, f"{op}: (fastest with cluster barriers) / mbarrier-only", ratio.round(2).to_dict())


def coll_utilization():
    e = envelope()
    _coll_bars(e, "util_pct", "Per-SM injection bandwidth used (%)", "coll_utilization", ylim=(0, 110), ref=100)
    for _, r in e.sort_values(["op", "cls", "payload"]).iterrows():
        note(7, f"{r.op} {r.cls} {int(r.payload)} B {ALG[r.file][0]}: GB/s per SM, %", (round(r.gbs_per_sm, 1), round(r.util_pct, 1)))


# =============================================================================================== slide 8
COMP = [("barrier", "cluster barrier", S.BLUE), ("handoff", "mbarrier hand-off", S.ORANGE),
        ("issue", "bulk-copy issue", S.AQUA), ("transfer", "data transfer", S.YELLOW),
        ("local", "local reduction", S.GREEN)]


def coll_breakdown():
    t = B.table()
    t = t.assign(local=t.local + t.loads)
    cols = [k for k, _, _ in COMP] + ["meas"]
    T = t.groupby(["alg", "op", "N", "payload_bytes"])[cols].sum().reset_index()
    algs = [("ring", "Ring (NCCL)"), ("recursive_doubling", "Recursive doubling (ClusterFusion)"),
            ("two_shot_push", "Two-shot push (TensorRT-LLM)")]
    f, axs = plt.subplots(2, 3, figsize=(12.5, 6.4), constrained_layout=True)
    for row, (by, fixed) in enumerate([("N", 4096), ("payload_bytes", 8)]):
        for ax, (alg, lab) in zip(axs[row], algs):
            g = T[(T.alg == alg) & (T.op == "allreduce")]
            if by == "N":
                g = g[g.payload_bytes == fixed].sort_values("N")
                xl = [str(n) for n in g.N]
            else:
                g = g[(g.N == fixed) & g.payload_bytes.isin(SIZES)].sort_values("payload_bytes")
                xl = [SIZE_LAB[SIZES.index(int(p))] for p in g.payload_bytes]
            for i, (_, r) in enumerate(g.iterrows()):
                bot = 0.0
                for k, _, c in COMP:
                    if r[k] > 0:
                        ax.bar(i, r[k] / 1e3, bottom=bot / 1e3, width=0.66, color=c, edgecolor="white", linewidth=0.5)
                        bot += r[k]
                if r.meas > bot:
                    ax.bar(i, (r.meas - bot) / 1e3, bottom=bot / 1e3, width=0.66, color="white",
                           edgecolor=S.LIGHT, hatch="////", linewidth=0)
                else:
                    ax.bar(i, (bot - r.meas) / 1e3, bottom=r.meas / 1e3, width=0.66, color="white", alpha=0.6, zorder=4)
                ax.plot([i - 0.33, i + 0.33], [r.meas / 1e3] * 2, color=S.INK, lw=1.4, zorder=5)
            ax.set_xticks(range(len(xl)), xl, fontsize=10.5)
            if row == 0:
                ax.set_title(lab, fontsize=12, color=S.INK)
            ax.set_xlabel("Cluster size (4 KiB messages)" if by == "N" else "Message size (N = 8)", fontsize=11)
    for ax in axs[:, 0]:
        ax.set_ylabel("Rank 0 time (kcycles)")
    h = [Patch(color=c, label=l) for _, l, c in COMP] + \
        [Patch(facecolor="white", edgecolor=S.LIGHT, hatch="////", label="unattributed (waiting)"),
         Line2D([], [], color=S.INK, lw=1.4, label="measured")]
    f.legend(handles=h, loc="outside upper center", ncol=7, fontsize=10.5, columnspacing=1.0)
    sh = C.share_table("N")
    note(8, "shares (%) vs N at 4 KiB", "\n" + sh.to_string())
    note(8, "shares (%) vs message size at N = 8", "\n" + C.share_table("payload").to_string())
    save(f, "coll_breakdown")


if __name__ == "__main__":
    for fn in [mem_latency, mem_bandwidth, dsmem_latency_cs, dsmem_heatmap, dsmem_bandwidth, sync_costs, handoff,
               coll_latency, coll_utilization, coll_breakdown]:
        fn()
        print("ok", fn.__name__)
    t = algorithm_table()
    with open(os.path.join(HERE, "numbers.txt"), "w") as fh:
        for s, w, v in NUM:
            fh.write(f"[slide {s}] {w}: {v}\n")
        fh.write("\n| " + " | ".join(t.columns) + " |\n|" + "---|" * len(t.columns) + "\n")
        for r in t.itertuples(index=False):
            fh.write("| " + " | ".join(map(str, r)) + " |\n")
    print(t.to_string(index=False))
