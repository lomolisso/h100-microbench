"""Section 5 of the report: DSMEM collectives (latency, effective synchronisation costs, per-SM cycle
breakdown, bandwidth). Data: the collectives performance sweep (one cluster per launch, 256 threads per CTA,
2 x 200 timed launches per configuration, SM clock locked at 1635 MHz) and the microbenchmark summaries.
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from . import data, style as S, breakdown as B

CF = data.import_file(data.os.path.join(data.COLL, "analysis", "collectives_figures.py"), "collectives_figures_r")
OPS = ["allreduce", "allgather", "reducescatter", "alltoall"]
OPNAME = {"allreduce": "AllReduce", "allgather": "AllGather", "reducescatter": "ReduceScatter",
          "alltoall": "AlltoAll"}
STYLE_COL = {CF.STYLES[0]: S.BLUE, CF.STYLES[1]: S.ORANGE, CF.STYLES[2]: S.AQUA}
STYLE_SHORT = {CF.STYLES[0]: "mbarrier only", CF.STYLES[1]: "one exchange + 2 barriers",
               CF.STYLES[2]: "barrier every round"}
COMP_COL = dict(zip([k for k, _ in B.COMPONENTS], [S.BLUE, S.ORANGE, S.AQUA, S.YELLOW, S.MAGENTA, S.GREEN]))
RESID = "#c9c8c3"
CYC = S.CYC_PER_NS
SEND_GBPS = 64 / 3 * 1.635          # per-SM send limit in GB/s at 1635 MHz


def _short(alg):
    return {"ring": "ring", "recursive_doubling": "recursive doubling", "two_shot_push": "two-shot push",
            "one_shot_pull": "one-shot pull", "direct_push": "direct push"}.get(alg, alg.replace("_", " "))


def configs():
    """Per configuration: mean collective latency L (first rank in -> last rank out) over all launches."""
    def build():
        d = pd.read_pickle(data.coll("derived/launch.pkl"))
        d = d[~d.clock_bad]
        g = d.groupby(["file", "op", "alg", "N", "M", "payload"]).L.agg(L_ns="mean", L_med_ns="median",
                                                                          n="size").reset_index()
        return g
    c = data.cached("coll_configs.csv", build)
    c["style"] = [CF.sync_style(f) for f in c.file]
    c["atomic"] = c.alg.str.contains("atomic")
    c["pow2"] = (c.payload & (c.payload - 1)) == 0      # sweep grid; the other points are per-file maxima
    c["algbw_GBps"] = c.payload / c.L_ns
    c["busbw_GBps"] = c.algbw_GBps * [CF.bus_factor(o, n) for o, n in zip(c.op, c.N)]
    return c


# ---------------------------------------------------------------------------------------------- latency ---
def fig_coll_latency(N=8):
    c = configs()
    c = c[(c.N == N) & ~c.atomic & c.pow2]
    f, axs = S.fig(4, 1, width=S.FULL, height=2.15, sharey=True)
    info = {}
    for ax, op, L in zip(axs, OPS, "abcd"):
        d = c[c.op == op]
        for fl, g in d.groupby("file"):
            g = g.sort_values("payload")
            ax.plot(g.payload, g.L_ns / 1e3, color=S.LIGHTGREY, lw=0.8, zorder=1)
        for st, g in d.groupby("style"):
            best = g.loc[g.groupby("payload").L_ns.idxmin()].sort_values("payload")
            ax.plot(best.payload, best.L_ns / 1e3, color=STYLE_COL[st], lw=1.6, marker="o", ms=2.5, zorder=3)
            info[(op, st)] = best.set_index("payload").L_ns.round(0).to_dict()
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(mpl.ticker.FixedLocator([256, 4096, 65536]))
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(S.kib_formatter))
        ax.set_xlabel("payload per CTA")
        ax.text(0.04, 0.96, OPNAME[op], transform=ax.transAxes, va="top", fontsize=8, color=S.INK)
        S.panel(ax, L)
    axs[0].set_ylabel(f"collective latency, N = {N} (µs)")
    axs[0].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    handles = [Line2D([], [], color=STYLE_COL[s], lw=1.6, marker="o", ms=2.5, label=STYLE_SHORT[s] + " (best)")
               for s in CF.STYLES] + [Line2D([], [], color=S.LIGHTGREY, lw=0.8, label="other algorithms")]
    f.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False)
    return f, info


def best_table(N=8, payloads=(256, 4096, 65536)):
    c = configs()
    c = c[(c.N == N) & ~c.atomic & c.payload.isin(payloads)]
    rows = []
    for (op, p), g in c.groupby(["op", "payload"]):
        b = g.loc[g.L_ns.idxmin()]
        w = g.loc[g.L_ns.idxmax()]
        rows.append({"collective": OPNAME[op], "payload": S.kib_formatter(p), "fastest": _short(b.alg),
                     "µs": round(b.L_ns / 1e3, 2), "cycles": int(round(b.L_ns * CYC)),
                     "slowest": _short(w.alg), "slowest / fastest": round(w.L_ns / b.L_ns, 1)})
    t = pd.DataFrame(rows)
    t["collective"] = pd.Categorical(t.collective, [OPNAME[o] for o in OPS])
    return t.sort_values(["collective"]).set_index(["collective", "payload"])


# --------------------------------------------------------------------- effective synchronisation costs ---
def sync_fit(payload=256):
    c = configs()
    c = c[(c.payload == payload) & ~c.atomic].copy()
    c[["nb", "nm"]] = [CF.SYNC[f](n) for f, n in zip(c.file, c.N)]
    X = np.c_[np.ones(len(c)), c.nb, c.nm]
    coef, *_ = np.linalg.lstsq(X, c.L_ns.values, rcond=None)
    pred = X @ coef
    r2 = 1 - ((c.L_ns - pred) ** 2).sum() / ((c.L_ns - c.L_ns.mean()) ** 2).sum()
    return {"t0": coef[0] * CYC, "t_barrier": coef[1] * CYC, "t_handoff": coef[2] * CYC, "r2": r2, "n": len(c)}


def fig_sync_effective():
    """(a) cost of one cluster barrier and one hand-off as fitted on the collectives vs measured in
    isolation; (b) the isolated hand-off ping-pong: round-trip distribution and its modes."""
    fit = sync_fit()
    c = B.costs()
    h = B.handoff_hist()
    f, (a, b) = S.fig(2, 1, width=S.FULL, height=2.2, gridspec_kw={"width_ratios": [1, 1.15]})
    rel = c[("barrier", 8)]
    med = c["handoff_rt_median_all_pairs"] / 2
    rows = [("cluster barrier", [("fitted inside collectives", fit["t_barrier"], S.BLUE),
                                 ("isolated, after a store", rel, S.GREY),
                                 ("isolated, no store", c[("barrier_nostore", 8)], S.LIGHTGREY)]),
            ("mbarrier hand-off\n(one way)", [("fitted inside collectives", fit["t_handoff"], S.BLUE),
                                   ("isolated, fast mode", c["handoff_one_way"], S.GREY),
                                   ("isolated, median", med, S.LIGHTGREY)])]
    y, ticks = 0, []
    for name, bars in rows:
        for lab, v, col in bars:
            a.barh(y, v, color=col, height=0.72)
            a.text(v + 8, y, f"{v:.0f}  {lab}", va="center", fontsize=7, color=S.INK2)
            y += 1
        ticks.append((y - (len(bars) + 1) / 2, name))
        y += 0.7
    a.set_yticks([t for t, _ in ticks], [n for _, n in ticks])
    a.invert_yaxis()
    a.set_xlim(0, 900)
    a.set_xlabel("cost (SM cycles)")
    a.grid(axis="y", visible=False)
    S.panel(a, "a")
    # (b) round-trip histogram of the isolated ping-pong
    hh = h.groupby((h.value // 4) * 4)["count"].sum()
    b.bar(hh.index, hh / hh.sum() * 100, width=4, color=S.GREY)
    rt_in = 2 * B.table().query("alg == 'two_shot_push' and N == 2 and payload_bytes == 256 and seg == 'round1'").meas.iloc[0]
    b.axvline(rt_in, ls=":", lw=1.0, color=S.BLUE)
    b.annotate(f"2 × one-way hand-off measured\ninside an N = 2 collective ({rt_in:.0f})", xy=(rt_in, 40),
               xytext=(548, 40), fontsize=6.5, color=S.BLUE, va="center")
    for x, lab in [(362, "both fast"), (531, "one slow"), (700, "both slow")]:
        v = hh[(hh.index >= x - 8) & (hh.index <= x + 8)].sum() / hh.sum() * 100
        b.text(x, v + 2, f"{x}\n{lab}", ha="center", va="bottom", fontsize=6.5, color=S.INK2)
    b.set_xlim(300, 780)
    b.set_ylim(0, 105)
    b.set_xlabel("isolated st.async + mbarrier round trip (SM cycles)")
    b.set_ylabel("share of round trips (%)")
    S.panel(b, "b")
    info = dict(fit, barrier_isolated=c[("barrier_nostore", 8)], barrier_after_store=rel, handoff_median=med, handoff_fast=c["handoff_one_way"],
                handoff_quantum=c["handoff_slow_step"], handoff_median_one_way=c["handoff_rt_median"] / 2,
                rt_in_collective=rt_in,
                share_fast=float(hh[hh.index < 450].sum() / hh.sum()),
                share_one_slow=float(hh[(hh.index >= 450) & (hh.index < 620)].sum() / hh.sum()))
    return f, info


# ------------------------------------------------------------------------------- per-SM cycle breakdown ---
def _stack(ax, x, row, width=0.72, total_label=False):
    """Stacked modelled components + unattributed remainder; the measured time is the bar's outline top."""
    bottom = 0.0
    for k, _ in B.COMPONENTS:
        v = row[k]
        if v > 0:
            ax.bar(x, v, bottom=bottom, width=width, color=COMP_COL[k], edgecolor=S.SURF, linewidth=0.4)
            bottom += v
    meas = row["meas"]
    if bottom > meas:          # modelled costs overlap in reality: fade the excess above the measurement
        ax.bar(x, bottom - meas, bottom=meas, width=width, color=S.SURF, alpha=0.65, linewidth=0, zorder=4)
    if meas > bottom:
        ax.bar(x, meas - bottom, bottom=bottom, width=width, color=S.SURF, edgecolor=RESID, hatch="/////",
               linewidth=0.0)
    ax.plot([x - width / 2, x + width / 2], [meas, meas], color=S.INK, lw=1.1, solid_capstyle="butt", zorder=5)


def _legend(f, ncol=4):
    handles = [Patch(color=COMP_COL[k], label=lab) for k, lab in B.COMPONENTS]
    handles += [Patch(facecolor=S.SURF, edgecolor=RESID, hatch="/////", label="unattributed: waiting for peers"),
                Line2D([], [], color=S.INK, lw=1.1, label="measured (rank 0)")]
    f.legend(handles=handles, loc="outside upper center", ncol=ncol, frameon=False, fontsize=7,
             handlelength=1.2, columnspacing=1.0)


def fig_breakdown_rounds(N=8, payload=4096):
    """Per-round view of rank 0 for ring AllReduce (left) and three one/two/three-round algorithms."""
    t = B.table()
    t = t[(t.N == N) & (t.payload_bytes == payload)]
    f, (a, b) = S.fig(2, 1, width=S.FULL, height=2.6, gridspec_kw={"width_ratios": [1.55, 1]})
    ring = t[t.alg.eq("ring") & t.op.eq("allreduce") & t.seg.ne("exit")].sort_values("idx")
    for i, (_, r) in enumerate(ring.iterrows()):
        _stack(a, i, r)
    a.set_xticks(range(len(ring)), [str(i + 1) for i in range(len(ring))])
    a.axvline(N - 1.5, color=S.INK2, lw=0.6, ls=":")
    a.text((N - 2) / 2, 1.0, "reduce-scatter rounds", transform=a.get_xaxis_transform(), ha="center",
           va="bottom", fontsize=7, color=S.INK2)
    a.text(N - 1 + (N - 2) / 2, 1.0, "all-gather rounds", transform=a.get_xaxis_transform(), ha="center",
           va="bottom", fontsize=7, color=S.INK2)
    a.set_xlabel(f"round of ring AllReduce (N = {N}, {S.kib_formatter(payload)})")
    a.set_ylabel("SM cycles (rank 0)")
    a.set_xlim(-0.6, len(ring) - 0.4)
    S.ns_axis(a)
    S.panel(a, "a", y=1.08)
    x, ticks = 0, []
    for alg, op, segs in [("recursive_doubling", "allreduce", None), ("two_shot_push", "allreduce", None),
                          ("one_shot_pull", "allreduce", ["round0", "exit"])]:
        g = t[t.alg.eq(alg) & t.op.eq(op)].sort_values("idx")
        g = g[g.seg.isin(segs)] if segs else g[g.seg.ne("exit")]
        start = x
        for i, (_, r) in enumerate(g.iterrows()):
            _stack(b, x, r)
            ticks.append((x, str(i + 1)))
            x += 1
        b.text((start + x - 1) / 2, -0.16, _short(alg).replace(" ", "\n", 1), transform=b.get_xaxis_transform(),
               ha="center", va="top", fontsize=7, color=S.INK, linespacing=1.0)
        x += 1.2
    b.set_xticks([p for p, _ in ticks], [l for _, l in ticks])
    b.set_xlabel("round", labelpad=16)
    b.set_ylabel("SM cycles (rank 0)")
    S.panel(b, "b", y=1.08)
    _legend(f, ncol=4)
    info = {"ring_rounds": ring[["seg", "meas", "model", "residual"]].round(0).to_dict("records"),
            "smid_rank0": int(ring.smid.iloc[0])}
    return f, info


ALGS = [("ring", "allreduce"), ("recursive_doubling", "allreduce"), ("two_shot_push", "allreduce"),
        ("one_shot_pull", "allreduce"), ("ring", "allgather"), ("direct_push", "alltoall")]


def _total(t):
    cols = [k for k, _ in B.COMPONENTS] + ["meas"]
    return t.groupby(["alg", "op", "N", "payload_bytes"])[cols].sum().reset_index()


def fig_breakdown_totals(by="N", N=8, payload=4096):
    """Whole-collective breakdown for rank 0: six algorithms, sweeping N at a fixed payload (by='N') or
    the payload at a fixed N (by='payload')."""
    T = _total(B.table())
    f, axs = S.fig(3, 2, width=S.FULL, height=4.2)
    for ax, (alg, op), L in zip(axs.flat, ALGS, "abcdef"):
        g = T[(T.alg == alg) & (T.op == op)]
        if by == "N":
            g = g[g.payload_bytes == payload].sort_values("N")
            xs = [str(n) for n in g.N]
        else:
            g = g[(g.N == N) & g.payload_bytes.isin([256, 1024, 4096, 16384, 65536])].sort_values("payload_bytes")
            xs = [S.kib_formatter(p).replace(" KiB", "K").replace(" B", "") for p in g.payload_bytes]
        for i, (_, r) in enumerate(g.iterrows()):
            _stack(ax, i, r)
        ax.set_xticks(range(len(xs)), xs)
        ax.text(0.03, 0.97, f"{OPNAME[op]}, {_short(alg)}", transform=ax.transAxes, va="top", fontsize=7.5,
                color=S.INK)
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v / 1e3:g}k" if v >= 1e3 else f"{v:g}"))
        ax.set_ylim(0, g.meas.max() * 1.22)
        S.panel(ax, L)
    for ax in axs[:, 0]:
        ax.set_ylabel("SM cycles (rank 0)")
    lab = "cluster size N" if by == "N" else "payload per CTA (bytes)"
    for ax in axs[-1, :]:
        ax.set_xlabel(lab)
    _legend(f, ncol=4)
    return f, {}


def share_table(by="N", N=8, payload=4096):
    """Share of rank 0's time per component (%), same selection as fig_breakdown_totals."""
    T = _total(B.table())
    sel = (T.payload_bytes == payload) if by == "N" else ((T.N == N) & T.payload_bytes.isin([256, 4096, 65536]))
    T = T[sel].copy()
    keys = [k for k, _ in B.COMPONENTS]
    T["unattributed"] = (T.meas - T[keys].sum(axis=1)).clip(lower=0)
    out = T[keys + ["unattributed"]].div(T[keys].sum(axis=1) + T.unattributed, axis=0) * 100
    out.columns = [dict(B.COMPONENTS).get(k, k).split(" (")[0].split(":")[0] for k in keys] + ["unattributed"]
    out.insert(0, "cycles", T.meas.round(0).astype(int))
    out.index = pd.MultiIndex.from_arrays([[f"{OPNAME[o]} {_short(a)}" for o, a in zip(T.op, T.alg)],
                                           T.N if by == "N" else [S.kib_formatter(p) for p in T.payload_bytes]],
                                          names=["collective", "N" if by == "N" else "payload"])
    return out.round(0).astype(int)


# --------------------------------------------------------------------------------------------- bandwidth ---
def fig_coll_bandwidth(N=8):
    c = configs()
    c = c[(c.N == N) & ~c.atomic & c.pow2]
    f, ax = S.fig(1, 1, width=S.COL, height=2.4)
    info = {}
    for op, col in zip(OPS, S.PAL):
        g = c[c.op == op]
        best = g.loc[g.groupby("payload").busbw_GBps.idxmax()].sort_values("payload")
        ax.plot(best.payload, best.busbw_GBps / SEND_GBPS * 100, color=col, marker="o", ms=2.5,
                label=OPNAME[op])
        info[op] = {int(p): (round(v, 1), a) for p, v, a in zip(best.payload, best.busbw_GBps, best.alg)}
    S.ref_line(ax, 100, f"per-SM send limit (21.3 B/cycle = {SEND_GBPS:.1f} GB/s)", pos=0.98, va="bottom")
    S.bytes_axis(ax)
    ax.set_ylim(0, 115)
    ax.set_xlabel("payload per CTA")
    ax.set_ylabel(f"bus bandwidth per SM, N = {N}\n(% of the send limit)")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.62), fontsize=7)
    return f, info


if __name__ == "__main__":
    import os
    out = os.path.join(data.CACHE, "preview")
    os.makedirs(out, exist_ok=True)
    for name, fn in [("coll_latency", fig_coll_latency), ("sync_effective", fig_sync_effective),
                     ("breakdown_rounds", fig_breakdown_rounds),
                     ("breakdown_N", lambda: fig_breakdown_totals("N")),
                     ("breakdown_payload", lambda: fig_breakdown_totals("payload")),
                     ("coll_bandwidth", fig_coll_bandwidth)]:
        fig, info = fn()
        fig.savefig(os.path.join(out, name + ".png"))
        plt.close(fig)
        print(name, {k: v for k, v in list(info.items())[:8]})
    print(best_table())
    print(share_table("N"))
    print(share_table("payload"))
