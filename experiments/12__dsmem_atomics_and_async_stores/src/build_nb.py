"""Builds 12__dsmem_atomics_and_async_stores.ipynb (unexecuted) from the cells below. Run from the experiment directory:
    python3 src/build_nb.py && jupyter nbconvert --execute --to notebook --inplace 12__dsmem_atomics_and_async_stores.ipynb
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
md(T.SETUP)
code(r'''
%matplotlib inline
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

sys.path.insert(0, "src")
from nb_data import (EVENT_FLOOR_MS, TARGET_LABEL, analysis_run, fix_2ld, kmeans_elbow, read_ids, runs, sm_partition, sources, wquantile)
from sass_forms import forms

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]  # fixed categorical order
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.prop_cycle": plt.cycler(color=C),
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 5})
pd.set_option("display.width", 220)
pd.set_option("display.precision", 1)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 400)
pd.set_option("display.max_colwidth", 70)

SRC = sources()
ANA = analysis_run()
REG = runs()
LS = read_ids("12__atomics__lat_summary.csv", [ANA], dtype={"pass": str})
PAIRS = read_ids("12__atomics__pairs.csv", [ANA])
SUMM = read_ids("12__atomics__summary.csv", [ANA])
LAT = read_ids("12__atomics__latency.csv", [SRC[m] for m in ["smem", "dsmem", "pairs", "l2", "hbm", "sys", "calib"]], dtype={"pass": str})
PART = sm_partition()
MHZ = 1635.0
first = LAT.iloc[0]
dev = pd.DataFrame({"": ["GPU", "SMs", "SM clock (locked; NVML samples)", "memory clock", "CUDA runtime", "driver"],
                    "value": [first.gpu_name, int(first.sm_count), f"{int(LAT.sm_clock_mhz.min())}–{int(LAT.sm_clock_mhz.max())} MHz",
                              f"{int(first.mem_clock_mhz)} MHz", first.cuda_version, first.driver_version]})
display(dev.set_index(""))
step_runs = pd.DataFrame(sorted(SRC.items()), columns=["step", "run_id"]).merge(REG[["run_id", "cmdline"]], on="run_id")
disc = pd.read_csv("results/12__atomics__discards.csv")
disc = disc[disc.run_id.isin(SRC.values())]
step_runs["discarded launches"] = step_runs.run_id.map(disc.groupby("run_id").size()).fillna(0).astype(int)
display(step_runs.set_index("step"))
print(f"analysis run {ANA}; discarded launches in total: {len(disc)}")
''')
code(r'''
# K0a: L2 partition of every SM on this card (timer pair included in the cycles)
k0 = PART.groupby("gpc").agg(SMs=("smid", "size"), partition=("partition", lambda s: "/".join(map(str, sorted(set(s))))),
                             near_cycles=("near_cycles", "median"), far_cycles=("far_cycles", "median"))
display(k0)
hr = pd.read_csv("results/12__atomics__homes_runs.csv")
hr = hr[hr.run_id.isin(SRC.values())]
print(f"page labels in {len(hr)} binary runs: pages per run {hr.pages.iloc[0]}, ambiguous pages per run: max {hr.pages_ambiguous.max()}, "
      f"4 KiB pages with lines of both homes: {hr.prefix_mixed_pages.max()} (of {int((hr.prefix_lines // 32).max())} checked), "
      f"lines that disagree with their page: {hr.prefix_line_page_disagree.max()}; near / far threshold {hr.thr_cycles.min():.0f}–{hr.thr_cycles.max():.0f} cycles")
''')

md(r"""
## 4. Results

### 4.1 Dependent latency on every path

Figure 1 shows the latency of one dependent atomic on each path, for the operations that compile to one hardware atomic and for representatives of the compare-and-swap loops. The plain load of the same path, measured with the same chain in the same binary, is the black marker. Table 1 lists every operation.
""")
code(r'''
def lat(target, op, kind="atom", sem="default", scope="default", cs=None, p=None, variant=None):
    d = LS[(LS.target == target) & (LS.op == op) & (LS.kind == kind) & (LS["sem"] == sem) & (LS.scope == scope)]
    if cs is not None:
        d = d[d.cs == cs]
    if p is not None:
        d = d[d["pass"] == p]
    if variant is not None:
        d = d[d.variant == variant]
    return d

ROWS = [("smem", 1, "0", "local smem"), ("smem_dsmem_if", 2, "0", "own smem\nvia window"), ("dsmem", 2, "0", "peer smem\nCS 2"),
        ("dsmem", 16, "0", "peer smem\nCS 16 (median)"), ("l2_near", 1, "1", "L2 near"), ("l2_far", 1, "1", "L2 far\npass 1"),
        ("l2_far", 1, "2", "L2 far\npass 2"), ("hbm", 1, "1_near", "HBM\nnear home"), ("hbm", 1, "1_far", "HBM\nfar home"),
        ("sysmem", 1, "1", "host\nmemory")]
OPS1 = [("add.u32", "atom"), ("cas.b32", "atom"), ("exch.b32", "atom"), ("add.u32", "red+ld"), ("add.u64", "atom"), ("add.f32", "atom"),
        ("add.f16x2", "atom")]
fig, ax = plt.subplots(figsize=(11, 4.2))
w = 0.8 / len(OPS1)
for j, (op, kind) in enumerate(OPS1):
    ys = []
    for t, cs, p, _ in ROWS:
        d = lat(t, op, kind, cs=cs, p=p)
        d = d[d.variant != "plus1"]
        ys.append(d.lat_cycles.median() if len(d) else np.nan)
    ax.bar(np.arange(len(ROWS)) + (j - len(OPS1) / 2 + 0.5) * w, ys, w, label=f"{op} {kind}", color=C[j % len(C)])
ref = []
for t, cs, p, _ in ROWS:
    d = lat(t, "add.u32", "ld", cs=cs, p=p)
    d = d[d.variant == "identity"]
    ref.append(d.lat_cycles.median() if len(d) else np.nan)
ax.scatter(np.arange(len(ROWS)), ref, marker="_", s=900, color=INK, zorder=5, label="plain load, same path")
ax.set_xticks(np.arange(len(ROWS)), [r[3] for r in ROWS])
ax.set_yscale("log")
ax.set_ylabel("dependent latency (SM cycles)")
ax.set_ylim(20, 4000)
sec = ax.secondary_yaxis("right", functions=(lambda c: c / MHZ * 1e3, lambda n: n * MHZ / 1e3))
sec.set_ylabel("ns at 1635 MHz")
ax.legend(ncol=4, fontsize=8, loc="upper left")
ax.set_title("Figure 1 — Latency of one dependent atomic per path (default semantics; black: plain load of the same chain)")
plt.tight_layout()
plt.show()
''')
code(r'''
# Table 1: latency (cycles) of every op at the default semantics; ld = same-chain plain load (same word format)
cols = [(t, cs, p, lab.replace("\n", " ")) for t, cs, p, lab in ROWS]
ops = ["add.u32", "add.u64", "inc.u32", "min.s32", "max.u32", "and.b32", "exch.b32", "cas.b32", "cas.b64", "add.f32", "add.f64", "add.f16x2",
       "add.bf16x2", "add.v2.f32", "add.v4.f32", "add.v2.f16x2", "add.v4.f16x2"]
tab = {}
for op in ops:
    for kind in ["atom", "red+ld"]:
        row = {}
        for t, cs, p, lab in cols:
            d = lat(t, op, kind, cs=cs, p=p)
            d = d[d.variant.isin(["identity", "cas_fail", "reverse_chain", "host_native_atomic_0"])]
            row[lab] = d.lat_cycles.median() if len(d) else np.nan
        if not all(np.isnan(v) for v in row.values()):
            tab[(op, kind)] = row
for op, var, lab in [("cas.b32", "same_word_success", "cas.b32 succeeding (one word)"), ("cas.b64", "same_word_success", "cas.b64 succeeding (one word)"),
                     ("inc.u32", "same_word", "inc.u32 (one word)"), ("add.u32", "same_word", "add.u32 (one word)")]:
    row = {}
    for t, cs, p, l in cols:
        d = lat(t, op, "same", cs=cs, p=p, variant=var)
        row[l] = d.lat_cycles.median() if len(d) else np.nan
    tab[(lab, "atom")] = row
for fmt, lab in [("add.u32", "ld (u32 words)"), ("add.f16x2", "ld (f16x2 words)")]:
    row = {}
    for t, cs, p, l in cols:
        d = lat(t, fmt, "ld", cs=cs, p=p)
        d = d[d.variant.isin(["identity", "host_native_atomic_0"])]
        row[l] = d.lat_cycles.median() if len(d) else np.nan
    tab[(lab, "reference")] = row
T1 = pd.DataFrame(tab).T
T1.index.names = ["op", "kind"]
display(Markdown("**Table 1 — Dependent latency (SM cycles, median over SMs)**; `red+ld` = red followed by a load of the same word; HBM: median of the per-access samples (timer included)"))
display(T1.round(1))
''')

md(r"""
### 4.2 Global atomics execute at the line's home partition

The L2 protocol walks the same lines three times after cleaning the L2. The lines were last written by an SM of their home partition, so they live only at their home (table 2, median over the 8 issuing SMs, one per GPC):
- **Pass 1** is the first access: near for lines homed in the issuer's partition, far for the others.
- **Pass 2** repeats the walk without cleaning. A plain load becomes near, because pass 1 left a copy in the reader's partition.
- **Pass 2ld** reads the lines with plain loads after the atomic passes.

Figure 2 shows the per-access distributions (one sample per operation, timer included) of the same protocol and of the HBM walk.
""")
code(r'''
L2 = LS[LS.target.isin(["l2_near", "l2_far"]) & (LS["sem"] == "default")]
rows = []
for (op, kind) in [("add.u32", "atom"), ("cas.b32", "atom"), ("exch.b32", "atom"), ("add.u32", "red+ld"), ("add.f32", "atom"), ("add.u64", "atom"),
                   ("add.v4.f32", "atom"), ("add.u32", "st+ld"), ("add.u32", "ld")]:
    r = {"op": op, "kind": kind}
    for t in ["l2_near", "l2_far"]:
        for p in ["1", "2"]:
            d = L2[(L2.target == t) & (L2.op == op) & (L2.kind == kind) & (L2["pass"] == p) & (L2.variant != "plus1")]
            r[f"{t} pass {p}"] = d.lat_cycles.median()
        d = L2[(L2.target == t) & (L2.op == op) & (L2.kind == "ld") & (L2["pass"] == "2ld") & (L2.variant == f"after_{kind}")]
        r[f"{t} 2ld"] = d.lat_cycles.median() if kind != "ld" else np.nan
    rows.append(r)
T2 = pd.DataFrame(rows).set_index(["op", "kind"])
display(Markdown("**Table 2 — L2 protocol (SM cycles, median of the 8 issuing SMs)**"))
display(T2.round(1))
per_sm = fix_2ld(LAT[(LAT.run_id == SRC["l2"])])
per_sm = per_sm[(per_sm["pass"] != "2ld") | (per_sm.after == "atom")]  # 2ld column: the plain load after the atomic passes
g = per_sm.groupby(["smid_src", "target", "op", "kind", "pass", "hops"]).total_cycles.median().unstack("hops")
g = ((g[16384] - g[4096]) / 12288).rename("lat").reset_index()
g = g[(g.op == "add.u32") & (g.kind.isin(["atom", "ld"]))].pivot_table(index="smid_src", columns=["target", "kind", "pass"], values="lat")
g.index = [f"SM {s} (GPC {PART.set_index('smid').gpc[s]}, partition {PART.set_index('smid').partition[s]})" for s in g.index]
display(Markdown("**Per issuing SM** (add.u32 atom and plain load, slopes)"))
display(g.round(1))
''')
code(r'''
FINE = fix_2ld(read_ids("12__atomics__fine.csv", [SRC["l2fine"], SRC["hbm"]], dtype={"pass": str}))
HIST = read_ids("12__atomics__fine_hist.csv", [SRC["hbm"]], dtype={"pass": str})
pmap = PART.set_index("smid").partition
fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
f = FINE[(FINE.run_id == SRC["l2fine"])]
bins = np.arange(200, 560, 2)
for ax, (op, kind) in zip(axs[:2], [("add.u32", "atom"), ("add.u32", "ld")]):
    for i, (t, p) in enumerate([("l2_near", "1"), ("l2_far", "1"), ("l2_far", "2"), ("l2_far", "2ld")]):
        if kind == "ld" and p == "2ld":
            continue
        d = f[(f.op == op) & (f.kind == kind) & (f.target == t) & (f["pass"] == p)] if p != "2ld" else \
            f[(f.op == op) & (f.kind == "ld") & (f.target == t) & (f["pass"] == "2ld") & (f.after == kind)]
        if kind == "atom" and p == "2ld":
            lab = "far lines, plain load after the atomics"
        else:
            lab = f"{t.replace('l2_', '')} home, pass {p}"
        ax.hist(d.cycles, bins=bins, histtype="step", lw=1.6, color=C[i], label=lab, density=True)
    ax.set_title(f"L2 protocol, {op} {kind}")
    ax.set_xlabel("cycles per access (timer included)")
    ax.legend(fontsize=7)
ax = axs[2]
h = FINE[(FINE.run_id == SRC["hbm"])]
h = h.assign(rel=np.where(h.home_partition == h.smid.map(pmap), "near", "far"))
bins = np.arange(420, 1100, 4)
centres = {}
for i, rel in enumerate(["near", "far"]):
    d = h[h.rel == rel]
    ax.hist(d.cycles, bins=bins, histtype="step", lw=1.6, color=C[i], density=True, label=f"add.u32 atom, {rel} home")
    k, (c, b, sse) = kmeans_elbow(d.cycles.clip(upper=1500).values[::7])
    centres[rel] = (k, c)
    for x in c:
        ax.axvline(x, color=C[i], ls=":", lw=1)
hh = HIST[(HIST.op == "add.u32") & (HIST.kind == "ld")]
hh = hh.assign(rel=np.where(hh.home_partition == hh.smid.map(pmap), "near", "far"))
for i, rel in enumerate(["near", "far"]):
    d = hh[hh.rel == rel]
    w, _ = np.histogram(d.cycles, bins=bins, weights=d["count"])
    ax.step(bins[:-1], w / d["count"].sum() / np.diff(bins)[0], where="post", color=C[i + 2], lw=1, label=f"ld, {rel} home")
over = (h.cycles > bins[-1]).mean()
ax.set_xlim(bins[0], bins[-1])
ax.set_xlabel("cycles per access (timer included)")
ax.set_title("HBM (L2 miss), dotted: k-means centres")
ax.legend(fontsize=7)
fig.suptitle("Figure 2 — Per-access latency of global atomics: L2 near / far, passes 1 / 2 / plain load after the atomics; HBM by home")
plt.tight_layout()
plt.show()
for rel, (k, c) in centres.items():
    print(f"HBM add.u32 atom, {rel}-homed lines: k = {k} classes, centres {np.round(c, 1)} cycles")
print(f"HBM add.u32 atom samples above {bins[-1]} cycles (not drawn): {100 * over:.3f} % (max {h.cycles.max()} cycles)")
''')

md(r"""
### 4.3 Memory-ordering semantics and scope

Table 3a lists the latency of the same dependent chains with explicit semantics and scope (add.u32 `atom`, add.u32 `red`+ld, cas.b32). On global targets, the L2 values are from SMs 0 and 2 (one per partition).
""")
code(r'''
S = LS[(LS.op.isin(["add.u32", "cas.b32"])) & (LS.kind.isin(["atom", "red+ld"])) & (LS.variant.isin(["identity", "cas_fail"]))]
S = S[S["pass"].isin(["0", "1"]) & ((S.target != "dsmem") | (S.cs == 2))]
S = S.assign(semantics=S["sem"] + "." + S.scope)
order = ["default.default", "relaxed.cta", "relaxed.cluster", "relaxed.gpu", "relaxed.sys", "acq_rel.cta", "release.cta", "acq_rel.cluster",
         "release.cluster", "acq_rel.gpu", "release.gpu", "acq_rel.sys", "release.sys"]
T3a = S.pivot_table(index=["op", "kind", "semantics"], columns="target", values="lat_cycles", aggfunc="median")
T3a = T3a.reindex(columns=[c for c in ["smem", "smem_dsmem_if", "dsmem", "l2_near", "l2_far", "sysmem"] if c in T3a.columns])
T3a = T3a.reset_index()
T3a["o"] = T3a.semantics.map({s: i for i, s in enumerate(order)})
T3a = T3a.sort_values(["op", "kind", "o"]).drop(columns="o").set_index(["op", "kind", "semantics"])
display(Markdown("**Table 3a — Latency (SM cycles) by semantics and scope**; dsmem = cluster of 2"))
display(T3a.round(1))
F = forms()
T3b = F.pivot_table(index=["op", "kind"], columns="space", values="sass", aggfunc="first")[["shared::cta", "shared::cluster", "global"]]
display(Markdown("**Table 3b — SASS of each operation** (default semantics; `CAS` / `CAST.SPIN` = compare-and-swap loop; a `.shared::cluster` op has one form per run-time case: own CTA, peer CTA, generic global)"))
with pd.option_context("display.max_colwidth", 80):
    display(T3b)
''')

md(r"""
### 4.4 Remote (DSMEM) atomics per SM pair

Every directed pair of SMs that can share a cluster (full waves of clusters of 16, 8, 4 and 2) chased the other SM's chain with add.u32 `atom`, cas.b32, add.u32 `red`+ld and a plain load (4096 hops). Figure 3a shows the distribution of the per-pair difference to the plain load of the same pair in the same run. Figure 3b compares the per-pair load and atomic latency with the latency matrix of the same pairs from an independent load-only measurement of this card.
""")
code(r'''
pp = PAIRS.pivot_table(index=["smid_src", "smid_dst"], columns=["op", "kind"], values="cycles_per_op", aggfunc="first")
pp.columns = [f"{a} {b}" for a, b in pp.columns]
ld10 = PAIRS.groupby(["smid_src", "smid_dst"]).ld10_cycles.first()
fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
for i, c in enumerate(["add.u32 atom", "cas.b32 atom", "add.u32 red+ld"]):
    dd = (pp[c] - pp["add.u32 ld"]).dropna()
    axs[0].hist(dd, bins=np.arange(6, 17, 0.05), color=C[i], label=f"{c}: median {dd.median():.2f}, IQR {dd.quantile(.25):.2f}–{dd.quantile(.75):.2f} ({len(dd)} pairs)")
axs[0].set_xlabel("atomic − plain load of the same pair (cycles)")
axs[0].set_ylabel("SM pairs")
axs[0].legend(fontsize=7)
axs[0].set_title("Figure 3a — Atomic minus load, per SM pair")
j = pp.join(ld10)
axs[1].scatter(j.ld10_cycles, j["add.u32 ld"], s=4, color=C[0], label="plain load (this run)")
axs[1].scatter(j.ld10_cycles, j["add.u32 atom"], s=4, color=C[1], label="add.u32 atom")
axs[1].plot([180, 230], [180, 230], color=INK2, ls=":", lw=1)
axs[1].set_xlabel("load latency of the pair, independent load-only matrix (cycles)")
axs[1].set_ylabel("this run (cycles)")
axs[1].legend(fontsize=7)
axs[1].set_title("Figure 3b — Per pair: atomic and load vs the load matrix")
plt.tight_layout()
plt.show()
print(f"{len(pp)} directed pairs; this run's plain load − independent matrix: median {(j['add.u32 ld'] - j.ld10_cycles).median():+.3f}, "
      f"max |Δ| {(j['add.u32 ld'] - j.ld10_cycles).abs().max():.2f} cycles")
lone = LS[(LS.target == "dsmem") & (LS["sem"] == "default") & (LS.variant.isin(["identity", "cas_fail", "reverse_chain"]))]
display(Markdown("**Lone clusters** (rank 0 → each peer; median over peers, cycles)"))
display(lone.pivot_table(index=["op", "kind"], columns="cs", values="lat_cycles", aggfunc="median").round(1))
''')

md(r"""
### 4.5 Throughput of one SM

Figure 4 shows the operations per cycle of one CTA (one SM) versus threads, one line per number of independent operations per thread and step, for add.u32 `red` and `atom`. The payload rate (operations × 4 B) is on the right axis; for local shared memory the 128 B/cycle of a conflict-free shared-memory access is dotted. Table 4 lists the best configuration of every target, operation and pattern at 1024 threads, and the same with a CTA on every SM at once.
""")
code(r'''
K2 = read_ids("12__atomics__throughput_sm.csv", [SRC["k2"], SRC["k2all"], SRC["k2x"]])
K2 = K2[K2.discard.isna()]
k2 = K2[K2.run_id == SRC["k2"]]
tg = ["smem", "smem_dsmem_if", "dsmem", "l2_near", "l2_far", "hbm"]
pats = ["coal", "bank", "sector", "line", "warp1", "block1"]
for kind in ["red", "atom"]:
    fig, axs = plt.subplots(len(tg), len(pats), figsize=(13, 13), sharex=True)
    for i, t in enumerate(tg):
        for j, p in enumerate(pats):
            ax = axs[i, j]
            d = k2[(k2.target == t) & (k2.pattern == p) & (k2.op == "add.u32") & (k2.kind == kind)]
            if d.empty:
                ax.axis("off")
                continue
            m = d.groupby(["ilp", "threads"]).ops_per_cycle.median().unstack("ilp")
            for c, ilp in enumerate(m.columns):
                ax.plot(m.index, m[ilp], marker="o", color=C[c], label=f"ILP {ilp}")
            if t == "smem":
                ax.axhline(128 / 4, color=INK2, ls=":", lw=1)
                if kind == "red":  # control: the same red with a run-time operand (plain ATOMS.ADD), 1024 threads
                    xr = K2[(K2.run_id == SRC["k2x"]) & (K2.placement == "1sm") & (K2.pattern == p)].groupby("ilp").ops_per_cycle.median()
                    if len(xr):
                        ax.scatter([1024] * len(xr), xr.values, marker="x", color=INK, zorder=5, label="operand not a constant")
            ax.set_xscale("log", base=2)
            ax.set_xticks([128, 256, 512, 1024], ["128", "256", "512", "1024"])
            ax.set_ylim(bottom=0)
            ax.set_title(p, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{TARGET_LABEL.get(t, t)}\nops / cycle / SM")
            if i == len(tg) - 1:
                ax.set_xlabel("threads per CTA")
            ax.secondary_yaxis("right", functions=(lambda o: o * 4, lambda b: b / 4))
    axs[0, 0].legend(fontsize=7)
    fig.suptitle(f"Figure 4{'a' if kind == 'red' else 'b'} — add.u32 {kind}: operations per cycle of one SM (left axes; right axes: payload B/cycle; "
                 f"dotted: 128 B/cycle){'; black x: local red with a non-constant operand' if kind == 'red' else ''}", y=1.0)
    plt.tight_layout()
    plt.show()
''')
code(r'''
best = K2[(K2.threads == 1024)].groupby(["placement", "target", "op", "kind", "pattern", "ilp"]).ops_per_cycle.median().reset_index()
best = best.loc[best.groupby(["placement", "target", "op", "kind", "pattern"]).ops_per_cycle.idxmax()]
best["placement"] = best.placement.map({"1sm": "1 SM", "132sm": "132 SMs (median SM)"})
T4 = best.pivot_table(index=["target", "op", "kind"], columns=["placement", "pattern"], values="ops_per_cycle")
display(Markdown("**Table 4 — Best operations per cycle per SM at 1024 threads** (over ILP 1–8); ld / st = plain loads / stores of the same addresses; "
                 "red_rt = local red.add.u32 whose operand is not a compile-time constant"))
display(T4.round(2))
''')

md(r"""
### 4.6 GPU-wide throughput of global atomics

Figure 5 shows the GPU-wide operation rate (net of the cudaEvent floor) versus footprint, for `red` and `atom` of add.u32, add.f32 and add.u64. The same kernel's plain loads and stores of the same addresses are dotted. `local` / `remote` means each partition's SMs work on lines homed in their own / the other partition.
""")
code(r'''
K3 = read_ids("12__atomics__throughput_gpu.csv", [SRC[m] for m in ["k3a", "k3b", "k3c", "k3d"]])
K3 = K3[K3.discard.isna()].copy()
K3["gops"] = K3.ops / ((K3.ms_event - EVENT_FLOOR_MS) * 1e-3) / 1e9
K3["bytes_op"] = np.where(K3.op == "add.u64", 8, 4)
K3["tbps"] = K3.gops * K3.bytes_op / 1e3
a = K3[K3.step == "k3a"]
fig, axs = plt.subplots(2, 3, figsize=(13, 6.5), sharey="row")
for i, pat in enumerate(["coal", "sector"]):
    for j, var in enumerate(["mixed", "local", "remote"]):
        ax = axs[i, j]
        d = a[(a.pattern == pat) & (a.variant == var)]
        for c, (op, kind) in enumerate([("add.u32", "red"), ("add.u32", "atom"), ("add.f32", "red"), ("add.f32", "atom"), ("add.u64", "red"),
                                        ("add.u64", "atom")]):
            m = d[(d.op == op) & (d.kind == kind)].groupby("footprint_bytes").gops.median()
            ax.plot(m.index / 2**20, m.values, marker="o", color=C[c], label=f"{op} {kind}")
        for c, kind in enumerate(["ld", "st"]):
            m = d[(d.op == "add.u32") & (d.kind == kind)].groupby("footprint_bytes").gops.median()
            ax.plot(m.index / 2**20, m.values, ls=":", color=[INK, INK2][c], label=f"plain {kind} (u32)")
        ax.set_xscale("log", base=2)
        ax.set_title(f"{pat}, {var}")
        if j == 0:
            ax.set_ylabel("Gops / s (GPU)")
        if i == 1:
            ax.set_xlabel("footprint (MiB)")
axs[0, 0].legend(fontsize=7, ncol=2)
fig.suptitle("Figure 5 — GPU-wide throughput vs footprint (2 × 132 CTAs × 1024 threads, 4 operations per thread and step)")
plt.tight_layout()
plt.show()
t5 = a[a.footprint_bytes.isin([16 << 20, 256 << 20])].pivot_table(index=["pattern", "op", "kind"], columns=["footprint_bytes", "variant"], values="tbps")
t5.columns = [f"{fp >> 20} MiB {v}" for fp, v in t5.columns]
display(Markdown("**Table 5 — GPU-wide payload rate (TB/s) at 16 and 256 MiB**"))
display(t5.round(2))
''')
code(r'''
b = K3[K3.step == "k3b"]
gp = PART.groupby("gpc").partition.first()
def label(mask):
    gs = [g for g in range(8) if mask >> g & 1]
    return "all" if len(gs) == 8 else "G" + ",".join(map(str, gs))
b = b.assign(gpcs=b.reader_gpc_mask.map(lambda m: bin(m).count("1")), part=b.reader_gpc_mask.map(lambda m: "both" if m == 0xFF else
                 f"partition {gp[[g for g in range(8) if m >> g & 1][0]]}"))
fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
for i, p in enumerate(["partition 0", "partition 1"]):
    ax = axs[i]
    for c, kind in enumerate(["red", "atom", "st", "ld"]):
        m = b[(b.part == p) & (b.kind == kind)].groupby("gpcs").tbps.median()
        ax.plot(m.index, m.values, marker="o", color=C[c] if kind in ("red", "atom") else [INK, INK2][kind == "ld"],
                ls="-" if kind in ("red", "atom") else ":", label=f"{kind} (add.u32 / u32)")
    both = b[(b.part == "both")].groupby("kind").tbps.median()
    ax.set_title(f"remote lines, CTAs on 1–4 GPCs of {p} (all GPCs: red {both.get('red', np.nan):.2f}, atom {both.get('atom', np.nan):.2f}, "
                 f"st {both.get('st', np.nan):.2f} TB/s)", fontsize=8)
    ax.set_xlabel("GPCs issuing")
    ax.set_ylabel("payload TB/s")
    ax.set_xticks([1, 2, 3, 4])
axs[0].legend(fontsize=7)
fig.suptitle("Figure 6 — Remote-homed atomics through the partition link (16 MiB; dotted: plain stores / loads of the same lines)")
plt.tight_layout()
plt.show()
d = K3[K3.step == "k3d"].assign(gpc=lambda x: x.reader_gpc_mask.map(lambda m: int(np.log2(m))))
display(Markdown("**Table 6 — One GPC alone, local lines, 16 MiB** (payload TB/s)"))
display(d.pivot_table(index="gpc", columns="kind", values="tbps", aggfunc="median").round(3))
''')

md(r"""
### 4.7 Hot spots

Figure 7a: every thread of the GPU (2 × 132 CTAs × 1024 threads) on one word (`hot1`, plotted at N = 0.5), or each warp instruction on one of N hot lines (32 consecutive words per instruction). Figure 7b: one word / one line / 32 lines versus the number of SMs issuing (one CTA of 1024 threads per SM). The counter table checks the L2 requests and sectors against the operations issued.
""")
code(r'''
c = K3[K3.step == "k3c"]
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
for i, (op, kind) in enumerate([("add.u32", "red"), ("add.u32", "atom"), ("add.f32", "red"), ("add.f32", "atom")]):
    d = c[(c.op == op) & (c.kind == kind)]
    m = d[d.pattern == "hotN"].groupby("n_hot_lines").gops.median()
    one = d[d.pattern == "hot1"].gops.median()
    axs[0].plot(list(m.index), m.values, marker="o", color=C[i], label=f"{op} {kind}")
    axs[0].scatter([0.5], [one], marker="s", color=C[i])
axs[0].set_xscale("log", base=2)
axs[0].set_yscale("log")
axs[0].set_xlabel("hot lines N (square at 0.5: one word)")
axs[0].set_ylabel("Gops / s (GPU)")
axs[0].legend(fontsize=7)
axs[0].set_title("Figure 7a — Hot spots, all SMs")
s = K3[K3.step == "k3c_sms"]
for i, (pat, n, lab) in enumerate([("hot1", 0, "one word"), ("hotN", 1, "one line"), ("hotN", 32, "32 lines")]):
    for k, kind in enumerate(["red", "atom"]):
        m = s[(s.pattern == pat) & (s.n_hot_lines == n) & (s.op == "add.u32") & (s.kind == kind)].groupby("active_sms").gops.median()
        axs[1].plot(m.index, m.values, marker="o", ls="-" if kind == "red" else "--", color=C[i], label=f"{lab}, {kind}")
axs[1].set_xscale("log", base=2)
axs[1].set_yscale("log")
axs[1].set_xlabel("SMs issuing (one CTA of 1024 threads each)")
axs[1].set_ylabel("Gops / s")
axs[1].legend(fontsize=7, ncol=2)
axs[1].set_title("Figure 7b — Hot spots vs SMs (add.u32)")
plt.tight_layout()
plt.show()
''')
code(r'''
NC = pd.read_csv("results/12__atomics__ncu.csv")
NC = NC[NC.run_id.isin(set(NC.run_id))]
NC["role"] = NC.config.str.extract(r"role=(\w+)")[0]
NC["label"] = NC.config.str.extract(r"label=([^;]+)")[0].fillna(NC.config.str.extract(r"kernel=(\w+);op=([^;]+);pattern=(\w+)").fillna("").agg("/".join, axis=1))
want = ["lts__t_requests_srcunit_tex_op_red.sum", "lts__t_requests_srcunit_tex_op_atom.sum", "lts__t_sectors_op_red.sum", "lts__t_sectors_op_atom.sum",
        "lts__t_sectors_op_red.max", "lts__t_sectors_op_atom.max", "lts__t_sectors_srcunit_ltcfabric.sum", "smsp__inst_executed_op_global_red.sum",
        "l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum", "dram__bytes_read.sum"]
g = NC[NC.tag.str.startswith("tput") & NC.metric.isin(want)].copy()
g["value"] = pd.to_numeric(g.value, errors="coerce")
base = g[g.role == "baseline"].groupby(["tag", "metric"]).value.first()
prof = g[g.role == "profiled"].groupby(["tag", "metric"]).value.first()
net = (prof - base.reindex(prof.index).fillna(0)).unstack("metric")
labs = g[g.role == "profiled"].groupby("tag").label.first()
ops = NC[NC.role == "profiled"].config.str.extract(r"ops=(\d+)")[0].astype(float).groupby(NC[NC.role == "profiled"].tag).first()
net.insert(0, "ops issued", ops)
net.index = [labs[t] for t in net.index]
net = net[[c for c in ["ops issued"] + want if c in net.columns]]
display(Markdown("**Table 7 — Nsight Compute counters of profiled K2/K3 launches, minus a same-grid all-exit baseline** (`.max` = the busiest L2 slice; `.max == .sum` ⇔ one slice served everything)"))
with pd.option_context("display.float_format", "{:,.0f}".format):
    display(net.sort_index())
''')
code(r'''
l2n = NC[NC.tag == "l2"].copy()
l2n["value"] = pd.to_numeric(l2n.value, errors="coerce")
l2n["order"] = l2n.config.str.extract(r"order=(\d+)")[0].astype(int)
l2n["what"] = l2n.config.str.extract(r"target=([^;]+);op=([^;]+);kind=([^;]+);pass=([^;]+)").agg(" ".join, axis=1)
pv = l2n.pivot_table(index=["order", "what"], columns="metric", values="value", aggfunc="first")
b0 = pv.iloc[0]
cols = [c for c in ["lts__t_sectors_srcunit_ltcfabric.sum", "lts__t_sectors_op_atom.sum", "lts__t_sectors_op_red.sum", "lts__t_sectors_op_read.sum",
                    "lts__t_sectors_op_write.sum", "lts__t_requests_srcunit_tex_op_atom.sum", "lts__t_sectors_srcunit_tex_lookup_miss.sum"] if c in pv.columns]
per = (pv[cols].iloc[1:] - b0[cols]) / 4096
display(Markdown("**Table 8 — Counters per operation of the 4096-hop L2 protocols** (SM 0; minus a 16-hop baseline launch; per hop)"))
display(per.round(2))
''')

md(r"""
### 4.8 DSMEM: fire-and-forget throughput, contention, hand-off, store width

Figure 8a shows one CTA of 1024 threads issuing into a peer (cluster of 2), and the ring over every resident cluster (every rank into the next one). Figure 8b shows 1–15 CTAs of a 16-CTA cluster into one CTA, with that CTA idle or running its own atomics on the same words. Figure 8c shows the round trip of a hand-off per signalling primitive over all 120 pairs of GPC 1. Figure 8d shows the store width.
""")
code(r'''
K45 = read_ids("12__dsmem_atomics__throughput.csv", [SRC["k4"], SRC["k4ring"], SRC["k5"]])
k4 = K45[K45.kernel == "k4"]
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
ops4 = ["red.shared::cluster.add.u32", "atom.shared::cluster.add.u32", "red.async.add.u32", "st.async.b32", "st.async.v4.b32"]
x = np.arange(3)
for i, op in enumerate(ops4):
    src_ = k4[(k4.op == op) & (k4.role == "src")].groupby("pattern").bpc.median()
    dst = k4[(k4.op == op) & (k4.role == "dst")]
    vals = [src_.get(p, np.nan) for p in ["distinct", "per32", "one"]]
    axs[0].bar(x + (i - 2) * 0.16, vals, 0.16, color=C[i], label=op)
axs[0].set_xticks(x, ["distinct words", "32 lanes / word", "all threads / word"])
axs[0].set_ylabel("issuer payload B / cycle")
axs[0].set_yscale("log")
axs[0].legend(fontsize=7)
axs[0].set_title("Figure 8a — One CTA into a peer (cluster of 2)")
r = K45[(K45.kernel == "k4ring") & (K45.role == "src")]
for i, op in enumerate(ops4):
    m = r[r.op == op].groupby(["cs", "launch_idx"]).agg(ops=("ops", "sum"), ms=("ms_event", "first")).reset_index()
    m["gops"] = m.ops / ((m.ms - EVENT_FLOOR_MS) * 1e-3) / 1e9
    mm = m.groupby("cs").gops.median()
    axs[1].plot(mm.index, mm.values * (16 if "v4" in op else 4) / 1e3, marker="o", color=C[i], label=op)
axs[1].set_xscale("log", base=2)
axs[1].set_xticks([2, 4, 8, 16], ["2", "4", "8", "16"])
axs[1].set_xlabel("cluster size (ring: rank r → r + 1, every resident cluster)")
axs[1].set_ylabel("GPU payload TB/s")
axs[1].set_title("Ring over the whole GPU")
plt.tight_layout()
plt.show()
comp = k4[(k4.role == "dst") & k4.completion_cycles.notna()].groupby(["op", "pattern"]).completion_cycles.median()
iss = k4[(k4.role == "src")].groupby(["op", "pattern"]).agg(cycles=("cycles", "median"), ops=("ops", "median"))
t8 = iss.join(comp).assign(issue_ops_per_cycle=lambda d: d.ops / d.cycles, completion_ops_per_cycle=lambda d: d.ops / d.completion_cycles)
display(Markdown("**Table 9 — One CTA into a peer**: issue span (red / atom: until every thread's fence returned) and completion at the receiver (`.async`: last mbarrier phase)"))
display(t8.round(3))
''')
code(r'''
k5 = K45[K45.kernel == "k5"]
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), sharey=False)
for j, pat in enumerate(["distinct", "same_bank", "one"]):
    ax = axs[j]
    for i, op in enumerate(["red.shared::cluster.add.u32", "atom.shared::cluster.add.u32", "red.async.add.u32"]):
        for busy in [0, 1]:
            m = k5[(k5.op == op) & (k5.pattern == pat) & (k5.role == "dst") & (k5.dest_busy == busy)].groupby("n_sources").ops_per_cycle.median()
            ax.plot(m.index, m.values, marker="o", ls="-" if busy == 0 else "--", color=C[i], label=f"{op.split('.')[0]}{'.async' if 'async' in op else ''}, receiver {'busy' if busy else 'idle'}")
    loc = k5[(k5.role == "dst_local") & (k5.pattern == pat)].groupby("n_sources").ops_per_cycle.median()
    ax.plot(loc.index, loc.values, color=INK2, ls=":", marker="x", label="receiver's own local atomics (busy)")
    ax.set_xlabel("source CTAs")
    ax.set_ylabel("operations / cycle at the receiver")
    ax.set_title({"distinct": "spread words", "same_bank": "one bank", "one": "one word"}[pat])
axs[0].legend(fontsize=6)
fig.suptitle("Figure 8b — N sources into one CTA (cluster of 16, GPC 1)")
plt.tight_layout()
plt.show()
''')
code(r'''
V = read_ids("12__dsmem_atomics__visibility.csv", [SRC[m] for m in ["k6", "k6poll1", "flags"] if m in SRC])
vd = V[V.scope == "cluster"]
prims = ["st_plain", "st_release", "st_async_mbarrier", "red_async_mbarrier", "red_cluster", "atom_cluster"]
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
for k, var in enumerate(["nogt", "nogt_poll1"]):
    d = vd[vd.variant == var]
    if d.empty:
        continue
    data = [d[d.primitive == p].round_trip_cycles.dropna().values for p in prims]
    axs[0].boxplot(data, positions=np.arange(len(prims)) + (k - 0.5) * 0.35, widths=0.3, showfliers=False,
                   patch_artist=True, boxprops=dict(facecolor=C[k], alpha=0.5), medianprops=dict(color=INK))
axs[0].set_xticks(np.arange(len(prims)), [p.replace("_mbarrier", "\n+mbarrier").replace("_", "\n", 1) for p in prims], fontsize=7)
axs[0].set_ylabel("round trip (SM cycles)")
axs[0].set_title("Figure 8c — Hand-off round trip, 120 pairs of GPC 1\n(blue: poll loop unrolled by the compiler; orange: one poll per loop iteration)", fontsize=8)
d = vd[vd.variant == "gt"]
ow = d.groupby("primitive").one_way_ns.median().reindex(prims)
owb = d.groupby("primitive").one_way_back_ns.median().reindex(prims)
axs[1].bar(np.arange(len(prims)) - 0.2, ow.values, 0.4, color=C[2], label="a → b")
axs[1].bar(np.arange(len(prims)) + 0.2, owb.values, 0.4, color=C[3], label="b → a")
axs[1].set_xticks(np.arange(len(prims)), [p.replace("_mbarrier", "\n+mbarrier").replace("_", "\n", 1) for p in prims], fontsize=7)
axs[1].set_ylabel("one way, median (ns; 32 ns ticks)")
axs[1].legend(fontsize=7)
axs[1].set_title("One-way time (%globaltimer stamps on both sides)")
plt.tight_layout()
plt.show()
t = vd.groupby(["primitive", "variant"]).agg(rt_median=("round_trip_cycles", "median"), rt_p5=("round_trip_cycles", lambda s: s.quantile(.05)),
                                             rt_p95=("round_trip_cycles", lambda s: s.quantile(.95)), one_way_ns=("one_way_ns", "median"),
                                             n=("trial", "size"))
display(Markdown("**Table 10 — DSMEM hand-off per primitive** (round trip without timer reads in the loop; one way from the `gt` variant)"))
display(t.round(1))
vg = V[V.scope == "gpu"]
part = PART.set_index("smid").partition
vg = vg.assign(pair=vg.smid_src.astype(str) + "–" + vg.smid_dst.astype(str) +
               np.where(vg.smid_src.map(part) == vg.smid_dst.map(part), " (same partition)", " (other partition)"))
tg = vg.groupby(["pair", "variant", "primitive"]).agg(rt=("round_trip_cycles", "median"), ow=("one_way_ns", "median")).reset_index()
tg["homes"] = tg.variant.str.replace("_gt", "")
tg = tg.groupby(["pair", "homes", "primitive"]).agg(round_trip_cycles=("rt", "max"), one_way_ns=("ow", "max")).unstack("primitive")
display(Markdown("**Table 11 — Global-flag hand-off** (two pinned warps; flag lines homed as stated; round trip median, one way median)"))
display(tg.round(0))
''')
code(r'''
W = read_ids("12__dsmem_atomics__width.csv", [SRC["k7"]])
fig, ax = plt.subplots(figsize=(6, 3.6))
for i, (fam, side) in enumerate([("st.shared::cluster", "issuer"), ("st.async", "issuer"), ("st.async", "dest")]):
    d = W[W.primitive.str.startswith(fam) & (W.side == side)].groupby("width_bytes").cycles.median()
    ax.plot(d.index, d.values, marker="o", color=C[i], label=f"{fam} ({'completion at the receiver' if side == 'dest' else 'issuer'})")
x = np.array([1, 16])
ax.plot(x, W[(W.primitive == "st.shared::cluster.u8")].cycles.median() / x, ls=":", color=INK2, label="time ∝ 1 / width")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
ax.set_xlabel("bytes per store instruction")
ax.set_ylabel("cycles for 256 KB (32 threads)")
ax.legend(fontsize=7)
ax.set_title("Figure 8d — Remote-store width")
plt.tight_layout()
plt.show()
display(W.groupby(["primitive", "side"]).agg(width=("width_bytes", "first"), cycles=("cycles", "median"), bpc=("bpc", "median")).sort_values("width").round(2))
''')
code(r'''
display(Markdown("**Summary table** (default semantics; latency = Table 1; per SM = K2 best ILP at 1024 threads, coalesced; GPU = K3 at 16 MiB (L2) / 256 MiB (HBM))"))
main = ["add.u32", "cas.b32", "exch.b32", "add.u64", "add.f32", "add.f16x2"]
sm_ = SUMM[SUMM.op.isin(main) & SUMM.kind.isin(["atom", "red+ld"])]
display(sm_[["target", "cs", "op", "kind", "latency_cycles", "ld_ref_cycles", "sm_ops_per_cycle", "sm_bpc", "gpu_gops", "gpu_tbps_payload"]]
        .set_index(["target", "cs", "op", "kind"]).round(2))
''')
md(T.INSIGHTS)

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
nbf.write(nb, "12__dsmem_atomics_and_async_stores.ipynb")
print(f"wrote 12__dsmem_atomics_and_async_stores.ipynb ({len(cells)} cells)")
