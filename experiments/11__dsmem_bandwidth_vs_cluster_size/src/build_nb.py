"""Builds 11__dsmem_bandwidth_vs_cluster_size.ipynb (unexecuted) from the cells below. Run from the experiment directory:
    python3 src/build_nb.py && jupyter nbconvert --execute --to notebook --inplace 11__dsmem_bandwidth_vs_cluster_size.ipynb
The markdown text lives in src/nb_text.py (one string per section)."""
import sys

import nbformat as nbf

sys.path.insert(0, "src")
import nb_text as T  # noqa: E402

cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


md(T.TITLE)
md(T.BACKGROUND)
md("## 3. Experimental setup")
code(r'''
%matplotlib inline
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

sys.path.insert(0, "src")
from nb_data import ASYNC, OPLAB, OPS, RES, affine, analysis_run, bw_side, gmem_reference, read_ids, runs, sm_map, sources

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]  # fixed categorical order
OPC = dict(zip(OPS, C))
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.prop_cycle": plt.cycler(color=C),
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6})
pd.set_option("display.width", 200)
pd.set_option("display.precision", 2)

SRC = sources()
ANA = analysis_run()
REG = runs()
UUID = REG[REG.run_id == SRC["A"]].gpu_uuid.iloc[0]
FITS = read_ids("11__dsmem_bw__fits.csv", [ANA])
GW = read_ids("11__dsmem_bw__gpuwide.csv", [ANA])
SA = read_ids("11__dsmem_bw__samples.csv", [SRC[s] for s in "ASBCDEF"])
SA["bpc"] = SA.bytes_moved / SA.cycles
LA = read_ids("11__dsmem_bw__launches.csv", [SRC[s] for s in ["A", "S", "B", "C", "D", "E", "F", "P1"]])
VI = read_ids("11__dsmem_bw__victim.csv", [SRC["K4"]])
IN = read_ids("11__dsmem_bw__interference.csv", [SRC["K5"]])
NC = read_ids("11__dsmem_bw__ncu_v2.csv", [SRC["ncu"]])
MBB = read_ids("11__dsmem_bw__p1_blocks.csv", [SRC["P1"]])
M = sm_map()
GPC = M.set_index("smid").gpc.to_dict()
MHZ = float(SA.sm_clock_mhz.median())
REF = gmem_reference(UUID)
BW = FITS[FITS.side == FITS.stream.map(bw_side)].copy()   # the bandwidth side of every configuration


def shared_total(step, op, pat, k, cs, mode="single"):
    """B/cycle at the shared SM of a broadcast pattern: all bytes of a cluster's sample / the last completion (affine slope)."""
    side = "target" if op in ASYNC else "issuer"
    d = SA[(SA.step == step) & (SA.kernel == op) & (SA.pattern == pat) & (SA.pattern_k == k) & (SA.cs == cs) & (SA.launch_mode == mode) &
           (SA.side == side)]
    last = d.groupby(["payload_bytes", "launch_idx", "sample", "cluster_id"]).agg(c=("cycles", "max"), b=("bytes_moved", "sum")).reset_index()
    lm = last.groupby("payload_bytes").agg(c=("c", "median"), b=("b", "median"))
    return 1 / affine(lm.b, lm.c)[1]


def fit(step, kernel, pattern="ring", k=1, cs=8, mode="single", stream=None, **kw):
    d = BW[(BW.step == step) & (BW.kernel == kernel) & (BW.pattern == pattern) & (BW.pattern_k == k) & (BW.cs == cs) &
           (BW.launch_mode == mode) & (BW.stream == (stream or kernel))]
    for c, v in kw.items():
        d = d[d[c] == v]
    assert len(d) == 1, (step, kernel, pattern, k, cs, mode, kw, len(d))
    return d.iloc[0]
''')
code(r'''
r0 = SA.iloc[0]
ncu_mhz = NC["sm__cycles_elapsed.avg.per_second"].astype(float).median() / 1e6
display(pd.DataFrame({"": [r0.gpu_name, r0.sm_count, f"{MHZ:.0f} MHz (NVML after every launch, median; locked at 1635 MHz)",
                           f"{ncu_mhz:.0f} MHz (ncu, profiled launches; most SMs idle in the single-cluster ones)",
                           f"{r0.mem_clock_mhz} MHz", "400 W", "enabled / disabled", r0.cuda_version, r0.driver_version,
                           "-arch=sm_90a -O3 -lineinfo"]},
                     index=["GPU", "SMs", "SM clock", "SM clock under ncu", "memory clock", "power limit", "ECC / MIG", "CUDA", "driver", "nvcc"]))
import os
dpath = f"{RES}/11__dsmem_bw__discards.csv"
disc = pd.read_csv(dpath) if os.path.getsize(dpath) else pd.DataFrame(columns=["run_id"])
inv = []
for s, rid in SRC.items():
    la = LA[LA.run_id == rid]
    n = len(la) if len(la) else (VI[VI.run_id == rid].launch_idx.nunique() if s == "K4" else IN[IN.run_id == rid].launch_idx.nunique() if s == "K5" else len(NC))
    nconf = la.groupby(["kernel", "pattern", "pattern_k", "cs", "policy", "threads", "ilp", "chunk_bytes", "payload_bytes", "passes", "launch_mode"]).ngroups if len(la) else np.nan
    inv.append(["many-block" if s == "P1" else s, rid, nconf, n, int((disc.run_id == rid).sum())])
display(pd.DataFrame(inv, columns=["step", "run_id", "configurations", "recorded launches", "discarded launches"]).set_index("step"))
''')
md(T.SETUP)
code(r'''
display(Markdown(f"GMEM reference lines (this card, same lock): **L2 read {REF['l2_read_tbps']:.2f} TB/s** "
                 f"(16 MiB, all SMs, {REF['n_l2']} launches, run {REF['l2_run']}), **HBM read {REF['hbm_read_tbps']:.2f} TB/s** "
                 f"({REF['n_hbm']} launches, run {REF['hbm_run']}); event floor removed. Same card as the DSMEM runs: {REF['same_card']}."))
''')
md("## 4. Results\n\n### 4.1 Aggregate SM-to-SM bandwidth vs cluster size (Fig. 1)")
code(r'''
S = GW[GW.source == "S"].copy()
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axs[0]
w = 0.16
css = [2, 4, 8, 16]
for i, op in enumerate(OPS):
    d = S[S.kernel == op].set_index("cs").reindex(css)
    x = np.arange(len(css)) + (i - 2) * w
    ax.bar(x, d.tbps_slope, w, color=OPC[op], label=OPLAB[op])
    for xx, v in zip(x, d.tbps_slope):
        ax.text(xx, v + 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, rotation=90)
ax.axhline(REF["hbm_read_tbps"], color=INK, ls=":", lw=1.5)
ax.text(3.45, REF["hbm_read_tbps"] + 0.05, f"HBM read {REF['hbm_read_tbps']:.2f}", ha="right", fontsize=8)
ax.axhline(REF["l2_read_tbps"], color=INK2, ls=":", lw=1.5)
ax.text(3.45, REF["l2_read_tbps"] - 0.32, f"L2 read {REF['l2_read_tbps']:.2f}", ha="right", fontsize=8)
act = S[S.kernel == "ld"].set_index("cs").active_sms.reindex(css)
ax.set_xticks(range(len(css)), [f"CS {c}\n({int(a)} SMs)" for c, a in zip(css, act)])
ax.set_ylabel("aggregate TB/s (whole GPU)")
ax.set_ylim(0, REF["l2_read_tbps"] * 1.08)
ax.set_title("ring (+1), every co-resident cluster at once, slope of event time over bytes", fontsize=9)
ax.legend(fontsize=7.5, loc="upper center", ncol=2)
ax = axs[1]
for op in OPS:
    d = S[S.kernel == op].set_index("cs").reindex(css)
    ax.plot(css, d.tbps_slope * 1e12 / (d.active_sms * MHZ * 1e6), "o-", color=OPC[op], label=OPLAB[op])
ax.set_xscale("log", base=2); ax.set_xticks(css, [str(c) for c in css])
ax.set_xlabel("cluster size"); ax.set_ylabel("B/cycle per active SM (= aggregate / SMs / clock)")
ax.set_title("the same per SM", fontsize=9)
fig.suptitle("Fig. 1 — SM-to-SM aggregate bandwidth per cluster size; dotted: this GPU's HBM and L2 read bandwidth", fontsize=10)
plt.tight_layout(); plt.show()
t = S.pivot(index="cs", columns="kernel", values="tbps_slope")[OPS]
t.columns = [OPLAB[c] for c in t.columns]
t.insert(0, "active SMs", S[S.kernel == "ld"].set_index("cs").active_sms.astype(int))
display(t.style.format(precision=3).set_caption("Aggregate TB/s (event-time slope over 4–32 passes of 96 KiB per SM, full chip)"))
chk = S.assign(tb_from_cta=S.bpc_cta_median * S.active_sms * MHZ * 1e6 / 1e12)[["kernel", "cs", "tbps_slope", "tb_from_cta"]]
display(chk.pivot(index="cs", columns="kernel", values="tb_from_cta")[OPS].style.format(precision=3)
        .set_caption("Cross-check: per-CTA median B/cycle at 32 passes × active SMs × clock (TB/s)"))
''')
md(T.R41)
md("### 4.2 Per-SM limits: one direction, both directions, one cluster vs the whole GPU, per GPC")
code(r'''
rows = []
for op in OPS:
    for pat, k in [("pair_uni", 1), ("pair_bi", 1), ("ring", 1)]:
        for cs in [2, 4, 8, 16]:
            st = "C" if cs == 8 or pat != "ring" else "A"
            try:
                f1, f2 = fit("C" if pat != "ring" else "A", op, pat, k, cs, "single"), None
            except AssertionError:
                continue
            try:
                f2 = fit("C" if pat != "ring" else "A", op, pat, k, cs, "full")
            except AssertionError:
                pass
            rows.append([OPLAB[op], pat, cs, f1.bpc_per_sm, f1.alpha_cycles, f1.rmse_pct, f2.bpc_per_sm if f2 is not None else np.nan,
                         f2.rmse_pct if f2 is not None else np.nan])
T42 = pd.DataFrame(rows, columns=["operation", "pattern", "CS", "single: B/cycle/SM", "single: alpha (cycles)", "single: RMSE %",
                                  "full: B/cycle/SM", "full: RMSE %"])
display(T42.set_index(["operation", "pattern", "CS"]).style.format(precision=2)
        .set_caption("Affine fits over the payload sweep 16–96 KiB (bandwidth side: issuer for ld/st, target for st.async/TMA)"))
''')
code(r'''
# per-SM rate vs number of active SMs of the same GPC (ring, 96 KiB): single clusters (CS 2/4/8/16) and full waves
d = SA[(SA.step == "A") & (SA.side == SA.stream.map(bw_side))].copy()
d["gpc"] = d.smid.map(GPC)
act = d.groupby(["run_id", "launch_idx", "gpc"]).smid.nunique().rename("active_in_gpc")
d = d.join(act, on=["run_id", "launch_idx", "gpc"])
g = d.groupby(["kernel", "cs", "launch_mode", "gpc", "active_in_gpc", "payload_bytes"]).cycles.median().reset_index()
pg = []
for key, x in g.groupby(["kernel", "cs", "launch_mode", "gpc", "active_in_gpc"]):
    a, b = affine(x.payload_bytes, x.cycles)
    pg.append(list(key) + [1 / b])
PG = pd.DataFrame(pg, columns=["kernel", "cs", "mode", "gpc", "active_in_gpc", "bpc"])
fig, axs = plt.subplots(1, 5, figsize=(14, 3.4), sharey=True)
for ax, op in zip(axs, OPS):
    x = PG[PG.kernel == op]
    for cs, mk in zip([2, 4, 8, 16], "osD^"):
        for mode, fc in [("single", "none"), ("full", None)]:
            y = x[(x.cs == cs) & (x["mode"] == mode)]
            ax.scatter(y.active_in_gpc, y.bpc, marker=mk, s=28, edgecolor=OPC[op], facecolor=fc or OPC[op], label=f"CS {cs} {mode}")
    ax.set_title(OPLAB[op], fontsize=8.5); ax.set_xlabel("active SMs in the GPC")
axs[0].set_ylabel("B/cycle per SM (fit, ring +1)")
axs[0].legend(fontsize=6.5, ncol=2, loc="lower left")
fig.suptitle("Per-SM ring bandwidth vs the number of SMs of the same GPC that take part (one point per GPC and configuration; open: single cluster)", fontsize=9.5)
plt.tight_layout(); plt.show()
display(PG.pivot_table(index=["kernel", "cs", "mode"], columns="gpc", values="bpc").reindex(OPS, level=0).style.format(precision=2, na_rep="")
        .set_caption("Per-GPC fits, ring (+1): B/cycle per SM (columns: GPC)"))
pr = SA[(SA.step == "A") & (SA.cs == 16) & (SA.payload_bytes == 98304) & (SA.side == SA.stream.map(bw_side))]
display(pr.groupby(["kernel", "launch_mode", "rank"]).bpc.median().unstack().reindex(OPS, level=0).style.format(precision=1)
        .set_caption("ring (+1) at CS 16, 96 KiB: B/cycle of each rank (columns; includes the fixed cost), median over clusters, samples, launches"))
''')
md(T.R42)
md("### 4.3 Threads and loads in flight (Fig. 2)")
code(r'''
B = BW[BW.step == "B"]
fig, axs = plt.subplots(1, 5, figsize=(14, 3.5), sharey=True)
panels = [("ld", "pair_uni"), ("ld", "ring"), ("st", "pair_uni"), ("st", "ring"), ("st_async", "pair_uni")]
for ax, (op, pat) in zip(axs, panels):
    x = B[(B.kernel == op) & (B.pattern == pat)]
    for i, ilp in enumerate(sorted(x.ilp.unique())):
        y = x[x.ilp == ilp]
        s = y.groupby("threads").bpc_per_sm.agg(["median", "min", "max"])
        ax.errorbar(s.index, s["median"], yerr=[s["median"] - s["min"], s["max"] - s["median"]], fmt="o-", color=C[i], capsize=2,
                    label=f"ILP {ilp}")
    if op == "st_async":
        y = B[(B.kernel == op) & (B.pattern == "ring")].groupby("threads").bpc_per_sm.agg(["median", "min", "max"])
        ax.errorbar(y.index, y["median"], yerr=[y["median"] - y["min"], y["max"] - y["median"]], fmt="s--", color=C[1], capsize=2, label="ILP 4, ring")
    ax.set_xscale("log", base=2); ax.set_xticks([128, 256, 512, 1024], ["128", "256", "512", "1024"])
    ax.set_title(f"{OPLAB[op]}, {pat}", fontsize=8.5); ax.set_xlabel("threads per CTA")
    ax.legend(fontsize=7)
axs[0].set_ylabel("B/cycle per SM (fit; median over CS 2/4/8/16, bars: min–max)")
fig.suptitle("Fig. 2 — per-SM bandwidth vs threads × ILP (one cluster; each CTA moves 16–96 KiB once)", fontsize=10)
plt.tight_layout(); plt.show()
t = B[B.kernel.isin(["ld", "st"])].pivot_table(index=["kernel", "pattern", "threads"], columns=["ilp", "cs"], values="bpc_per_sm")
display(t.style.format(precision=1).set_caption("B/cycle per SM by threads (rows) and (ILP, CS) (columns)"))
''')
md(T.R43)
md("### 4.4 Traffic patterns at CS 8 (Fig. 3)")
code(r'''
PATS = [("pair_uni", 1), ("pair_bi", 1), ("ring", 1), ("ring", 4), ("packed_pairs", 1), ("packed_pairs", 4), ("broadcast_in", 7),
        ("broadcast_out", 7), ("all_to_all", 1), ("all_to_all_full", 1), ("randperm", 1), ("uniform_random", 1)]
NAMES = {("ring", 1): "ring +1", ("ring", 4): "ring +4 (= pairs 4 apart, both ways)", ("packed_pairs", 1): "packed pairs (TPC mates)",
         ("packed_pairs", 4): "packed pairs 4 apart", ("broadcast_in", 7): "7 ↔ rank 0 (total at rank 0)", ("broadcast_out", 7): "rank 0 → 7 (total at rank 0)"}
C8 = BW[(BW.step == "C") & (BW.cs == 8)]
def val(op, pat, k, mode):
    d = C8[(C8.kernel == op) & (C8.pattern == pat) & (C8.pattern_k == k) & (C8.launch_mode == mode) & (C8.stream == op)]
    if not len(d):
        return np.nan, np.nan, np.nan
    r = d.iloc[0]
    if pat.startswith("broadcast"):   # the shared SM: everything it serves (ld), receives (st/TMA in) or sends (out)
        return shared_total("C", op, pat, k, 8, mode), 0.0, np.nan
    return r.bpc_per_sm, r.rmse_pct, r.bpc_at_max_payload
rows = []
for pat, k in PATS:
    for op in ["ld", "st", "tma_put"]:
        for mode in ["single", "full"]:
            b, rm, bm = val(op, pat, k, mode)
            rows.append([NAMES.get((pat, k), pat.replace("_", " ")), op, mode, b, rm, bm])
PT3 = pd.DataFrame(rows, columns=["pattern", "op", "mode", "bpc", "rmse", "bpc_max"])
mix = []
for op, second in [("mixed_ld_st", "st"), ("mixed_ld_tma", "tma_put")]:
    for pat in ["mixed_same_dir", "mixed_opposite"]:
        for mode in ["single", "full"]:
            d = C8[(C8.kernel == op) & (C8.pattern == pat) & (C8.launch_mode == mode)].set_index("stream")
            mix.append([f"{pat.replace('_', ' ')} (ld + {second})", mode, d.loc["ld"].bpc_per_sm, d.loc[second].bpc_per_sm,
                        d.loc["ld"].rmse_pct, d.loc[second].rmse_pct])
MX = pd.DataFrame(mix, columns=["pattern", "mode", "ld stream", "second stream", "ld RMSE %", "second RMSE %"])
fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [2.2, 1]})
ax = axs[0]
labels = list(dict.fromkeys(PT3.pattern))
y0 = np.arange(len(labels))
for i, op in enumerate(["ld", "st", "tma_put"]):
    for mode, alpha in [("single", 1.0), ("full", 0.45)]:
        d = PT3[(PT3.op == op) & (PT3["mode"] == mode)].set_index("pattern").reindex(labels)
        v = d.bpc.where(d.rmse <= 2, d.bpc_max)  # not affine: rate at 96 KiB
        ax.barh(y0 + (i - 1) * 0.27 + (0.12 if mode == "full" else -0.0), v, 0.12, color=OPC[op], alpha=alpha,
                label=f"{OPLAB[op]} ({mode})", hatch=None)
        for yy, vv, rm in zip(y0 + (i - 1) * 0.27, v, d.rmse):
            if rm > 2 and mode == "single":
                ax.text(vv + 0.3, yy, "*", fontsize=8, va="center")
ax.set_yticks(y0, labels); ax.invert_yaxis()
ax.set_xlabel("B/cycle per issuing SM (ld, st) or per receiving SM (TMA); fit over 16–96 KiB")
ax.axvline(64 / 3, color=INK2, ls=":", lw=1); ax.text(64 / 3, -0.8, "21.33", fontsize=7.5, ha="center")
ax.legend(fontsize=7, loc="lower right")
ax.set_title("single patterns (light: 15 clusters at once; *: not affine, rate at 96 KiB)", fontsize=9)
ax = axs[1]
yl = list(dict.fromkeys(MX.pattern))
for j, mode in enumerate(["single", "full"]):
    d = MX[MX["mode"] == mode].set_index("pattern").reindex(yl)
    yy = np.arange(len(yl)) + j * 0.38
    ax.barh(yy, d["ld stream"], 0.36, color=OPC["ld"], alpha=1 if mode == "single" else 0.45, label=f"ld stream ({mode})")
    ax.barh(yy, d["second stream"], 0.36, left=d["ld stream"], color=OPC["st"], alpha=1 if mode == "single" else 0.45,
            label=f"st / TMA stream ({mode})")
ax.set_yticks(np.arange(len(yl)) + 0.19, yl); ax.invert_yaxis()
ax.set_xlabel("B/cycle (two streams of 512 threads, stacked)")
ax.legend(fontsize=7, loc="lower right"); ax.set_title("rank 0 and rank 1 streaming at the same time", fontsize=9)
fig.suptitle("Fig. 3 — traffic patterns inside clusters of 8 CTAs (ld, st: 1024 threads, ILP 4; TMA: 16 KiB commands)", fontsize=10)
plt.tight_layout(); plt.show()
display(PT3.pivot_table(index="pattern", columns=["op", "mode"], values="bpc", sort=False).style.format(precision=2, na_rep="")
        .set_caption("Fig. 3 values (affine fits, B/cycle per SM)"))
display(MX.set_index(["pattern", "mode"]).style.format(precision=2))
''')
md(T.R44)
md("### 4.5 One SM serving or feeding many (Fig. 4)")
code(r'''
D = SA[SA.step == "D"]
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4))
ns = np.arange(1, 16)
series = [("ld", "broadcast_in", "issuer", "ld: N readers of rank 0"), ("st", "broadcast_out", "issuer", "st: rank 0 writes the payload to N"),
          ("tma_put", "broadcast_out", "target", "TMA put: rank 0 writes the payload to N")]
TOT, PER = {}, {}
for op, pat, side, lab in series:
    d = D[(D.kernel == op) & (D.pattern == pat) & (D.side == side)]
    # total rate of the shared SM: all bytes of a sample / the completion of the last reader (target)
    last = d.groupby(["pattern_k", "payload_bytes", "launch_idx", "sample"]).agg(c=("cycles", "max"), b=("bytes_moved", "sum")).reset_index()
    lm = last.groupby(["pattern_k", "payload_bytes"]).agg(c=("c", "median"), b=("b", "median")).reset_index()
    TOT[lab] = pd.Series({k: 1 / affine(x.b, x.c)[1] for k, x in lm.groupby("pattern_k")})
    if op == "st":   # one issuer covers every target: per-target rate = its rate / N
        PER[lab] = pd.DataFrame({"min": TOT[lab] / TOT[lab].index, "median": TOT[lab] / TOT[lab].index, "max": TOT[lab] / TOT[lab].index})
    else:            # per reader (target): its own affine fit
        pr = d.groupby(["pattern_k", "rank", "payload_bytes"]).cycles.median().reset_index()
        f = pd.Series({(k, r): 1 / affine(x.payload_bytes, x.cycles)[1] for (k, r), x in pr.groupby(["pattern_k", "rank"])})
        PER[lab] = f.groupby(level=0).agg(["min", "median", "max"])
    axs[0].plot(TOT[lab].index, TOT[lab], "o-", color=OPC[op], label=lab)
    p = PER[lab]
    axs[1].errorbar(p.index + {"ld": -0.15, "st": 0, "tma_put": 0.15}[op], p["median"], yerr=[p["median"] - p["min"], p["max"] - p["median"]],
                    fmt="o", color=OPC[op], capsize=2, label=lab + " (median, min–max over N)")
cfit = {lab: float(np.mean(TOT[lab])) for lab in TOT}
for lab, (op, *_ ) in zip(TOT, series):
    axs[1].plot(ns, cfit[lab] / ns, "-", color=OPC[op], lw=1, alpha=0.6)
axs[0].axhline(64 / 3, color=INK2, ls=":", lw=1)
axs[0].set_ylim(0, 26); axs[0].set_xlabel("N"); axs[0].set_ylabel("B/cycle out of the shared SM (all N together)")
axs[0].set_title("total: N × payload / completion of the last of the N", fontsize=9); axs[0].legend(fontsize=7.5, loc="lower right")
axs[1].set_yscale("log"); axs[1].set_xlabel("N"); axs[1].set_ylabel("B/cycle per reader / per target"); axs[1].legend(fontsize=7, loc="upper right")
axs[1].set_title("per reader / target (lines: total / N)", fontsize=9)
for ax in axs:
    ax.set_xticks(ns)
fig.suptitle("Fig. 4 — one CS-16 cluster: one SM shared by N others", fontsize=10)
plt.tight_layout(); plt.show()
t = pd.DataFrame({lab: TOT[lab] for lab in TOT})
for lab in PER:
    if not lab.startswith("st"):
        t[lab + ": per-reader min / max"] = [f"{a:.2f} / {b:.2f}" for a, b in zip(PER[lab]["min"], PER[lab]["max"])]
display(t.style.format(precision=2, subset=list(TOT)).set_caption("Total B/cycle out of the shared SM, and the spread of the per-reader (per-target) rates"))
pr = D[(D.kernel == "ld") & (D.pattern == "broadcast_in") & (D.side == "issuer")].groupby(["pattern_k", "rank", "payload_bytes"]).cycles.median().reset_index()
fr = pd.Series({(k, r): 1 / affine(x.payload_bytes, x.cycles)[1] for (k, r), x in pr.groupby(["pattern_k", "rank"])}).unstack()
display(fr.style.format(precision=2, na_rep="").set_caption("ld broadcast: B/cycle of each reader rank (columns) for N readers (rows); ranks 2k, 2k+1 share a TPC, rank 1 is rank 0's TPC mate"))
r = [[OPLAB[op], cs, fit("C", op, "broadcast_in", cs - 1, cs, "single", stream=op).bpc_per_sm * (1 if op in ASYNC else cs - 1)]
     for op in ["st", "tma_put"] for cs in [2, 4, 8, 16]]
display(pd.DataFrame(r, columns=["operation", "CS", "B/cycle into rank 0 (all CS−1 writers together)"]).set_index(["operation", "CS"])
        .style.format(precision=2).set_caption("broadcast_in with stores: one SM receiving from CS − 1 writers"))
''')
md(T.R45)
md("### 4.6 TMA command size (Fig. 5)")
code(r'''
E = FITS[FITS.step == "E"]
fig, axs = plt.subplots(1, 2, figsize=(12, 3.9))
for i, (op, pat) in enumerate([("tma_put", "pair_uni"), ("tma_put", "ring"), ("tma_red", "pair_uni"), ("tma_red", "ring")]):
    d = E[(E.kernel == op) & (E.pattern == pat) & (E.side == "target")].sort_values("chunk_bytes")
    axs[0].plot(d.chunk_bytes, d.bpc_per_sm, "o-" if pat == "pair_uni" else "s--", color=OPC[op], label=f"{OPLAB[op]}, {pat}")
    axs[1].plot(d.chunk_bytes, d.chunk_bytes / d.bpc_per_sm, "o-" if pat == "pair_uni" else "s--", color=OPC[op], label=f"{OPLAB[op]}, {pat}")
c = np.array(sorted(E.chunk_bytes.unique()))
axs[1].plot(c, c / (64 / 3), ":", color=INK2, lw=1, label="bytes / 21.33")
for ax in axs:
    ax.set_xscale("log", base=2); ax.set_xticks(c, [f"{int(x)}" if x < 1024 else f"{int(x) // 1024}K" for x in c]); ax.set_xlabel("bytes per command")
axs[0].set_ylabel("B/cycle delivered to the target (fit)"); axs[0].legend(fontsize=7.5)
axs[1].set_ylabel("cycles per command (= size / rate)"); axs[1].set_yscale("log"); axs[1].legend(fontsize=7.5)
fig.suptitle("Fig. 5 — TMA shared→shared commands of 128 B – 16 KiB, one issuing thread (CS 8, single cluster)", fontsize=10)
plt.tight_layout(); plt.show()
t = E[E.side == "target"].pivot_table(index="chunk_bytes", columns=["kernel", "pattern"], values="bpc_per_sm")
t2 = E[E.side == "target"].pivot_table(index="chunk_bytes", columns=["kernel", "pattern"], values="alpha_cycles")
display(pd.concat({"B/cycle": t, "alpha (cycles)": t2}, axis=1).style.format(precision=2))
# issue side: time of the issuing thread per command, from the issuer records at 96 KiB
s = SA[(SA.step == "E") & (SA.side == "issuer") & (SA.payload_bytes == 98304)]
iss = s.groupby(["kernel", "pattern", "chunk_bytes"]).cycles.median().reset_index()
iss["commands"] = 98304 // iss.chunk_bytes
iss["issue cycles per command"] = iss.cycles / iss.commands
display(iss.pivot_table(index="chunk_bytes", columns=["kernel", "pattern"], values="issue cycles per command").style.format(precision=1)
        .set_caption("Issuing thread: cycles from the barrier to the last command issued, per command (96 KiB)"))
c8 = SA[(SA.step == "C") & (SA.kernel == "tma_put") & (SA.side == "issuer") & (SA.launch_mode == "single") &
        SA.pattern.isin(["ring", "broadcast_out", "all_to_all"])]
ci = c8.groupby(["pattern", "cs", "payload_bytes"]).cycles.median().unstack()
display(ci.style.format(precision=0).set_caption("Issuing thread, cycles until the last command (16 KiB commands): ring = 1 target, "
                                                  "broadcast_out = CS−1 targets × full payload, all_to_all = CS−1 targets × 1/(CS−1)"))
''')
md(T.R46)
md("### 4.7 Scheduling policy")
code(r'''
rows = []
for op in ["ld", "st", "tma_put"]:
    for cs in [2, 4, 8, 16]:
        for mode in ["single", "full"]:
            r = [OPLAB[op], cs, mode, fit("A", op, "ring", 1, cs, mode).bpc_per_sm]
            for pol in ["Spread", "LoadBalancing"]:
                r.append(fit("F", op, "ring", 1, cs, mode, policy=pol).bpc_per_sm)
            rows.append(r)
display(pd.DataFrame(rows, columns=["operation", "CS", "mode", "Default", "Spread", "LoadBalancing"]).set_index(["operation", "CS", "mode"])
        .style.format(precision=2).set_caption("ring (+1), B/cycle per SM by scheduling policy"))
same = SA[(SA.step.isin(["A", "F"])) & (SA.launch_mode == "single") & (SA["sample"] == 0) & (SA.side == "issuer") & (SA.kernel == "ld")]
pl = same.groupby(["policy", "cs"]).smid.apply(lambda s: ",".join(map(str, sorted(s.unique())))).unstack(0)
display(Markdown("SMs used by the single cluster (ld ring, every launch, every policy):"))
display(pl)
''')
md(T.R47)
md("### 4.8 A many-block ring launch")
code(r'''
P = LA[LA.step == "P1"].copy()
P["tbps"] = P.total_bytes / P.event_ns * 1e-3
t = P.groupby(["kernel", "cs", "passes"]).tbps.median().unstack()
g = GW[GW.source == "P1"].set_index(["kernel", "cs"]).tbps_slope
t["slope (TB/s)"] = g.reindex(t.index)
display(t.style.format(precision=3).set_caption("50 160 blocks × 1024 threads, 16 KiB smem, 2 blocks per SM, ring (+1); "
                                                "columns: passes over the peer's 16 KiB, TB/s = bytes / event time"))
h = MBB.groupby(["kernel", "cs", "passes", "launch_idx"]).blocks.agg(["min", "max", "sum"]).reset_index()
display(Markdown(f"Blocks per SM over all many-block launches: {h['min'].min()}–{h['max'].max()} (every launch: {h['sum'].min()} blocks)."))
''')
md(T.R48)
md("### 4.9 Local shared memory of an SM that is being read remotely (Fig. 6)")
code(r'''
V = VI.copy()
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
for ax, mode in zip(axs, ["single", "full"]):
    x = V[V.launch_mode == mode]
    for (kern, rt, on), d in x[x.side == "victim"].groupby(["kernel", "reader_threads", "reader_on"]):
        s = d.groupby("victim_warps").bpc.median()
        lab = "victim alone" if not on else f"while rank 0 {'reads' if kern == 'ld' else 'stores into'} it ({rt} threads)"
        ax.plot(s.index, s, "o-" if on else "--", color=None if on else INK, label=lab)
    ax.set_xlabel("victim warps streaming their own shared memory (ld.shared.v4)"); ax.set_xticks([4, 8, 16, 32])
    ax.set_title(f"{mode}: {'one CS-2 cluster' if mode == 'single' else '66 CS-2 clusters (every SM)'}", fontsize=9)
    ax.legend(fontsize=7)
axs[0].set_ylabel("victim local B/cycle (median)")
fig.suptitle("Fig. 6 — local shared-memory bandwidth of an SM while its partner streams 3 MiB out of (or into) it", fontsize=10)
plt.tight_layout(); plt.show()
t = V.groupby(["launch_mode", "side", "kernel", "reader_threads", "victim_warps"]).bpc.median().unstack()
display(t.style.format(precision=2, na_rep="").set_caption("Median B/cycle (columns: victim warps)"))
''')
md(T.R49)
md("### 4.10 SM-to-SM traffic next to SM→L2 traffic in the same GPC (Fig. 7)")
code(r'''
I = IN.copy()
I["combo"] = I.role_map + " / " + I.dsmem_dest
combos = list(dict.fromkeys(I.combo))
fig, axs = plt.subplots(2, len(combos), figsize=(15, 5.6), sharey="row")
for j, cb in enumerate(combos):
    x = I[I.combo == cb]
    ax = axs[0, j]
    a = x[(x.group == "l2") & (x.alone_or_both == "alone")].groupby("ilp_l2").bpc.median()
    b = x[(x.group == "l2") & (x.alone_or_both == "both") & (x.ilp_dsmem == 8)].groupby("ilp_l2").bpc.median()
    ax.plot(a.index, a, "o-", color=C[0], label="L2 alone"); ax.plot(b.index, b, "x--", color=C[1], label="L2 with DSMEM (ILP 8)")
    ax.set_title(cb, fontsize=8.5); ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    ax = axs[1, j]
    a = x[(x.group == "dsmem") & (x.alone_or_both == "alone")].groupby("ilp_dsmem").bpc.median()
    b = x[(x.group == "dsmem") & (x.alone_or_both == "both") & (x.ilp_l2 == 8)].groupby("ilp_dsmem").bpc.median()
    ax.plot(a.index, a, "o-", color=C[2], label="DSMEM alone"); ax.plot(b.index, b, "x--", color=C[3], label="DSMEM with L2 (ILP 8)")
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"]); ax.set_xlabel("ILP of the swept group")
axs[0, 0].set_ylabel("L2 streamer B/cycle"); axs[1, 0].set_ylabel("DSMEM source B/cycle")
axs[0, 0].legend(fontsize=7); axs[1, 0].legend(fontsize=7)
fig.suptitle("Fig. 7 — one CS-16 cluster, roles by TPC index (groups A = {0,3,6}, B = {1,4,7}, C = {2,5}); per-CTA medians over 30 launches", fontsize=10)
plt.tight_layout(); plt.show()
k = ["combo", "group", "ilp_l2", "ilp_dsmem", "alone_or_both"]
t = I.groupby(k).agg(ctas=("smid", "nunique"), bpc=("bpc", "median"), lat=("loaded_latency_cycles", "median")).reset_index()
alone_l2 = t[(t.group == "l2") & (t.alone_or_both == "alone")].set_index(["combo", "ilp_l2"]).bpc
alone_ds = t[(t.group == "dsmem") & (t.alone_or_both == "alone")].set_index(["combo", "ilp_dsmem"]).bpc
both = t[t.alone_or_both == "both"].copy()
both["alone"] = [alone_l2[(c, il)] if g == "l2" else alone_ds[(c, idm)] for c, g, il, idm in zip(both.combo, both.group, both.ilp_l2, both.ilp_dsmem)]
both["change %"] = 100 * (both.bpc / both.alone - 1)
display(both.set_index(k).style.format(precision=2).set_caption("Both groups running: per-CTA B/cycle, group alone at the same ILP, change"))
lat = t.pivot_table(index=["combo", "group", "alone_or_both"], columns=["ilp_l2", "ilp_dsmem"], values="lat")
display(lat.style.format(precision=0, na_rep="").set_caption("Mean loaded latency per ILP group (cycles; columns: L2 ILP, DSMEM ILP)"))
''')
md(T.R410)
md("### 4.11 Hardware counters (Nsight Compute)")
code(r'''
N = NC.copy()
for c in N.columns:
    if c.startswith(("l1tex__", "sm__", "gpu__")):
        N[c] = pd.to_numeric(N[c], errors="coerce")
N["out/issuer"] = N["l1tex__m_l1tex2xbar_write_bytes_mem_dshared.sum"] / N.issuer_bytes
N["in/issuer"] = N["l1tex__m_xbar2l1tex_read_bytes_mem_dshared.sum"] / N.issuer_bytes
N["out max/avg"] = N["l1tex__m_l1tex2xbar_write_bytes_mem_dshared.max"] / N["l1tex__m_l1tex2xbar_write_bytes_mem_dshared.avg"]
N["in max/avg"] = N["l1tex__m_xbar2l1tex_read_bytes_mem_dshared.max"] / N["l1tex__m_xbar2l1tex_read_bytes_mem_dshared.avg"]
N["out max (B)"] = N["l1tex__m_l1tex2xbar_write_bytes_mem_dshared.max"]
N["in max (B)"] = N["l1tex__m_xbar2l1tex_read_bytes_mem_dshared.max"]
inst = ["sm__sass_inst_executed_op_dshared_ld.sum", "sm__sass_inst_executed_op_dshared_st.sum", "sm__sass_inst_executed_op_dshared_stas.sum",
        "sm__sass_inst_executed_op_dshared_tma_st.sum", "sm__sass_inst_executed_op_dshared_tma_red.sum"]
show = ["kernel", "pattern", "pattern_k", "launch_mode", "issuer_bytes", "out/issuer", "in/issuer", "out max (B)", "in max (B)", "out max/avg", "in max/avg"] + inst
t = N[show].rename(columns={c: c.split("op_dshared_")[1].replace(".sum", "") + " instr" for c in inst})
display(t.set_index(["kernel", "pattern", "pattern_k", "launch_mode"]).style.format(precision=2))
pk = [c for c in N.columns if c.endswith(".peak_sustained")]
display(N[pk].drop_duplicates().T.rename(columns=lambda c: "value").style.format(precision=2).set_caption("Rated peaks per SM per cycle reported by ncu"))
''')
md(T.R411)
md(T.INSIGHTS)

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "11__dsmem_bandwidth_vs_cluster_size.ipynb")
print(f"{len(cells)} cells")
