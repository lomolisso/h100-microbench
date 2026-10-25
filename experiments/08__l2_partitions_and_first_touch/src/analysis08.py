"""Analysis of experiment 08 shared by src/summarize.py and the notebook (paths relative to the experiment directory).

  gpc_map()                    smid → GPC from the latest run of 03's map
  load_k5(rid)                 K5 rows with first-read and repeat (samples 1-63) medians; per-sample arrays on request
  sm_signature(k5, tp)         per-SM partition, sub-node, near/far/copy latencies (timer pair tp subtracted):
     partition = the home label (0: SM 0's partition, 1: SM 2's) whose lines the SM reads fastest on first read;
     sub-node: for each partition, the matrix X[SM, line] of first-read medians of the lines homed there, read by the
     SMs of that partition, is double-centred; the signs of its first singular vectors split SMs and lines into two
     groups each. An SM's "same sub-node" lines are the line group its SM group reads faster.
     Sub-node ids: 2·partition + (0 for the SM group that contains the partition's smallest smid, else 1).
  line_groups(...)             the same split for any (SM set, line set, value) matrix (used for far and copy reads)
  share_null(...)              permutation null of line_groups' first-component share (is there SM-group × line-group
                               structure at all?)
  line_subnode(k5, sig, groups) home sub-node of every K5 line (the sub-node of its home partition that reads it fastest)
  quad_table(k5, sig, groups, tp) first reads labelled same/other partition × same/other sub-node position (sub-nodes
                               paired across partitions by the fastest far reads)
  k2_bits(hb)                  per bit: share of (base, base ^ 2^b) pairs whose home differs; linear coefficients
  k2_linearity(hb, coef)       agreement of pair / random-mask home changes with the XOR of the single-bit coefficients
  k2_first_reads(rid)          sample-0 rows of the K2 run (probe_k2.csv, read in chunks)
  k2_page_terms(f0, coef)      per-2-MiB-page term of the K2 buffer and the page-pair flip table (bits 21..30)
  k2vmm(pages)                 page term per handle / mapping and page-pair flip shares inside every allocation (k2vmm run)
  gmm1d(values, counts, k)     weighted 1-D Gaussian mixture fitted by EM (multi-start) on a (value, count) histogram;
                               boundaries at equal posterior between neighbouring components (K4 access classes)
  fabric1()                    ncu_fabric1 counters per line with the all-exit baseline subtracted
"""
import numpy as np
import pandas as pd

from common08 import read_run

MAP03 = "../03__sm_gpc_topology/results/03__sm_gpc_topology__map.csv"


def gpc_map():
    g = pd.read_csv(MAP03)
    g = g[g.run_id == g.run_id.max()]
    return g.set_index("smid").gpc.astype(int)


def load_k5(rid, samples=False):
    d = read_run("08__l2__sm_signature_samples.csv", rid)
    seq = np.array([np.array(s.split(";"), dtype=np.int32) for s in d.cycles_seq])
    d["repeat_median"] = np.median(seq[:, 1:], axis=1)
    d = d.drop(columns="cycles_seq")
    return (d, seq) if samples else d


def _sm_line_matrix(d, sms, lines, value):
    x = d[d.probe_smid.isin(sms) & d.line_offset.isin(lines)].groupby(["probe_smid", "line_offset"])[value].median().unstack()
    return x.dropna(axis=1)


def _first_share(v):
    """Share of the double-centred variance of matrix v carried by its first singular component (and the SVD)."""
    c = v - v.mean(axis=1, keepdims=True) - v.mean(axis=0, keepdims=True) + v.mean()
    u, s, vt = np.linalg.svd(c, full_matrices=False)
    return float(s[0] ** 2 / (s ** 2).sum()), u, vt


def line_groups(d, sms, lines, value):
    """Rank-1 split of the matrix value[SM, line] (medians over reps): returns (sm_group Series, line_group Series, share
    of the centred variance explained by the first component)."""
    x = _sm_line_matrix(d, sms, lines, value)
    share, u, vt = _first_share(x.values)
    sg = pd.Series((u[:, 0] > 0).astype(int), index=x.index)
    lg = pd.Series((vt[0] > 0).astype(int), index=x.columns)
    if sg.loc[min(sg.index)] == 1:  # group 0 = the group of the smallest smid
        sg, lg = 1 - sg, 1 - lg
    return sg, lg, share


def share_null(d, sms, lines, value, n_perm=1000, seed=0, axis="row"):
    """Permutation null for line_groups' first-component share of the SM × line matrix X (medians over reps).

    The question is whether X has SM-group × line-group (interaction) structure beyond an SM term and a line term, which
    double-centring removes anyway. axis="row": every SM row is permuted independently across the lines. This keeps each
    SM's own set of values (its mix of fast and slow reads) but breaks any link between an SM and particular lines, so
    every interaction is destroyed; the line term is scrambled into noise, which the statistic (a variance share) does
    not reward. axis="col": every line column is permuted across the SMs (the mirror image; keeps each line's values).
    Returns (observed share, array of n_perm null shares); seed fixed (numpy default_rng)."""
    v = _sm_line_matrix(d, sms, lines, value).values
    obs = _first_share(v)[0]
    rng = np.random.default_rng(seed)
    a = 1 if axis == "row" else 0
    null = np.array([_first_share(rng.permuted(v, axis=a))[0] for _ in range(n_perm)])
    return obs, null


def sm_signature(k5, tp):
    d = k5.copy()
    d["lat_first"] = d.first_cycles - tp
    d["lat_repeat"] = d.repeat_median - tp
    gpc = gpc_map()
    med = d.groupby(["probe_smid", "line_home"]).first_cycles.median().unstack()
    part = (med[1] < med[0]).astype(int)  # faster on home-1 lines → partition 1
    rows, groups = [], {}
    for p in (0, 1):
        sms = part.index[part == p]
        near_lines = d.loc[d.line_home == p, "line_offset"].unique()
        far_lines = d.loc[d.line_home != p, "line_offset"].unique()
        sg, lg, ev = line_groups(d, sms, near_lines, "lat_first")
        fsg, flg, fev = line_groups(d, sms, far_lines, "lat_first")
        csg, clg, cev = line_groups(d, sms, far_lines, "lat_repeat")
        groups[p] = dict(sm=sg, line=lg, ev=ev, far_sm=fsg, far_line=flg, far_ev=fev, copy_sm=csg, copy_line=clg, copy_ev=cev)
        for s in sms:
            x = d[d.probe_smid == s]
            near, far = x[x.line_home == p], x[x.line_home != p]
            nf = near.assign(g=near.line_offset.map(lg))
            # which line group this SM's group reads faster
            gm = nf.groupby("g").lat_first.median()
            same = int(gm.idxmin())
            ff = far.assign(g=far.line_offset.map(flg)).groupby("g").lat_first.median()
            rows.append(dict(
                smid=s, gpc=int(gpc[s]), partition=p, subnode=2 * p + int(sg[s]),
                near_median=near.lat_first.median(), far_median=far.lat_first.median(),
                near_same_subnode_median=gm[same], near_other_subnode_median=gm[1 - same],
                subnode_offset=gm[1 - same] - gm[same],
                far_lo_median=ff.min(), far_hi_median=ff.max(),
                copy_median=far.lat_repeat.median(), home_repeat_median=near.lat_repeat.median(),
                n_first_near=len(near), n_first_far=len(far)))
    return pd.DataFrame(rows).sort_values("smid").reset_index(drop=True), groups


def k2_bits(hb):
    s = hb[hb.set == "single"]
    frac = s.groupby("bit").flipped_class_changed.mean()
    coef = {int(b): int(f > 0.5) for b, f in frac.items() if f >= 0.9 or f <= 0.1}
    return frac, coef


def k2_linearity(hb, coef):
    """Pairs and random masks restricted to bits 7..20: observed home change vs parity of the participating bits."""
    out = {}
    for set_ in ("pair", "mask"):
        x = hb[hb.set == set_].copy()
        m = x.mask_hex.map(lambda h: int(h, 16))
        M = sum(1 << b for b, c in coef.items() if c and b <= 20)
        x["pred"] = (m & M).map(lambda v: bin(v).count("1") & 1)
        out[set_] = (float((x.pred == x.flipped_class_changed).mean()), len(x))
    return out


def line_subnode(k5, sig, groups):
    """Home sub-node of every K5 line: the sub-node whose SMs read it fastest among the SMs of its home partition."""
    s = sig.set_index("smid")
    out = {}
    for p in (0, 1):
        sg, lg = groups[p]["sm"], groups[p]["line"]
        x = k5[(k5.line_home == p) & k5.probe_smid.isin(sg.index)]
        t = x.assign(g=x.probe_smid.map(sg), l=x.line_offset.map(lg)).groupby(["g", "l"]).first_cycles.median().unstack()
        sub_of_g = {g: int(s.subnode[sg.index[sg == g][0]]) for g in (0, 1)}
        for g in (0, 1):
            fast = int(t.loc[g].idxmin())
            for o in lg.index[lg == fast]:
                out[o] = sub_of_g[g]
    return pd.Series(out, name="line_subnode")


def quad_table(k5, sig, groups, tp):
    """First-read latency by (same / other partition) × (same / other sub-node position), where sub-nodes are paired
    across the partitions by the far-read pattern: the far sub-node a sub-node reads fastest is its partner."""
    s = sig.set_index("smid")
    ls = line_subnode(k5, sig, groups)
    d = k5.assign(c=k5.first_cycles - tp, sp=k5.probe_smid.map(s.partition), sn=k5.probe_smid.map(s.subnode),
                  ln=k5.line_offset.map(ls)).dropna(subset=["ln"])
    far = d[d.line_home != d.sp]
    partner = far.groupby(["sn", "ln"]).c.median().unstack().idxmin(axis=1).astype(int).to_dict()
    d["partition"] = np.where(d.line_home == d.sp, "same partition", "other partition")
    d["position"] = [("same" if (ln == sn or partner[sn] == ln) else "other") + " sub-node position" for sn, ln in zip(d.sn, d.ln)]
    return d, partner


def k2_first_reads(rid, res="results"):
    """First reads (sample 0) of every K2 line: probe_smid, line_offset, cycles, rep, line_home."""
    parts = []
    for ch in pd.read_csv(f"{res}/08__l2__probe_k2.csv", chunksize=2_000_000,
                          usecols=["run_id", "probe_smid", "line_offset", "sample_idx", "cycles", "rep", "line_home"]):
        parts.append(ch[(ch.run_id == rid) & (ch.sample_idx == 0)])
    return pd.concat(parts, ignore_index=True)


def k2_page_terms(f0, coef, page=2 << 20):
    """Page term of the K2 buffer: g(page) = home XOR parity of the in-page hash bits (coef, bits 7..20) at the fixed in-page
    offsets probed in every page (the 'page' line set: offsets present in ≥ 512 pages). Returns dict(lines_stable (share of
    lines with one home in every rep and direction), n_lines, first_reads_per_line, pages_consistent (share of pages whose
    offsets all give the same g), n_pages, offsets_per_page, g1 (share of pages with g = 1), flips: DataFrame per bit
    21 + k of the page index: page pairs (both orders) and the share whose g differs).
    The flip shares are properties of how this allocation's pages lie in physical memory, not hash coefficients."""
    cons = f0.groupby("line_offset").line_home.agg(["mean", "size"])
    home = cons["mean"].round().astype(int)
    M = sum(1 << b for b in range(7, 21) if coef.get(b) == 1)
    inpage = pd.Series(home.index % page).value_counts()
    scan = inpage[inpage >= 512].index
    pg = pd.DataFrame({"off": home.index, "home": home.values})
    pg = pg[(pg.off % page).isin(scan)].assign(page=lambda d: d.off // page)
    pg["g"] = [h ^ (bin(int(o % page) & M).count("1") & 1) for h, o in zip(pg.home, pg.off)]
    gp = pg.groupby("page").g.agg(["mean", "size"])
    G = gp["mean"].round().astype(int)
    rows = []
    for k in range(int(np.log2(len(G)))):
        pp = G.index.values
        q = pp ^ (1 << k)
        ok = np.isin(q, G.index.values)
        rows.append({"bit": 21 + k, "page pairs": int(ok.sum()), "g differs [%]": 100 * (G.loc[pp[ok]].values != G.loc[q[ok]].values).mean()})
    return dict(lines_stable=float(((cons["mean"] == 0) | (cons["mean"] == 1)).mean()), n_lines=len(cons),
                first_reads_per_line=int(cons["size"].median()), pages_consistent=float(((gp["mean"] == 0) | (gp["mean"] == 1)).mean()),
                n_pages=len(gp), offsets_per_page=int(gp["size"].median()), g1=float(G.mean()), flips=pd.DataFrame(rows).set_index("bit"))


def k2vmm(pages):
    """Page term G of the k2vmm run (rows of k2vmm_pages.csv for one run_id).

    Returns a dict with
      consistency  per mapping: pages, share of (page, rep, direction) rows whose 8 lines all agree, share of pages with
                   the same G in every rep and direction, share of pages with G = 1, base address and its alignment;
      handles      per 2 MiB handle (mappings A1, A2, A3): G in each mapping, and the agreement of A1 with A2 and A3;
      flips        per allocation (B*, C, and A1 in slot = handle-creation order) and bit 21 + k inside it: the share of
                   page pairs (j, j XOR 2^k) whose G differs, and the number of pairs;
      chunks       per allocation: G of the first page of every aligned group of 4 pages (8 MiB) and the share of groups
                   whose 4 pages follow G, ¬G, ¬G, G (bits 21 and 22 both flip G)."""
    G = pages.groupby(["mapping", "page_slot"]).page_term.agg(["mean", "size"])
    stable = (G["mean"] == 0) | (G["mean"] == 1)
    g = G["mean"].round().astype(int)
    first = pages.groupby("mapping").first()
    cons = pd.DataFrame({
        "kind": first.alloc_kind, "MiB": first.alloc_bytes // 2**20, "pages": g.groupby(level=0).size(),
        "rows with all 8 lines agreeing [%]": 100 * pages.groupby("mapping").lines_agree.apply(lambda x: (x == 8).mean()),
        "pages with one G in every rep and direction [%]": 100 * stable.groupby(level=0).mean(),
        "G = 1 [%]": 100 * g.groupby(level=0).mean(),
        "base": first.alloc_base_hex,
        "base alignment [MiB]": [((b & -b) >> 20) for b in first.alloc_base_hex.map(lambda h: int(h, 16))]})
    a = pages[pages.mapping.str.startswith("A")].groupby(["mapping", "page_slot"]).handle.first()
    h = pd.DataFrame({"handle": a.values, "G": g.loc[a.index].values, "mapping": a.index.get_level_values(0)})
    handles = h.pivot_table(index="handle", columns="mapping", values="G")
    flips, chunks = [], {}
    for m in [m for m in cons.index if not m.startswith("A")] + ["A1"]:
        x = g.loc[m].values
        name = m if m != "A1" else "A1 (handles in creation order)"
        for k in range(int(np.log2(len(x)))):
            i = np.arange(len(x))
            j = i ^ (1 << k)
            ok = j > i
            flips.append({"allocation": name, "bit": 21 + k, "G differs [%]": 100 * (x[i[ok]] != x[j[ok]]).mean(), "pairs": int(ok.sum())})
        c = x.reshape(-1, 4)
        chunks[name] = dict(first_G="".join(map(str, c[:, 0])), pattern_ok=float(((c ^ c[:, :1]) == [0, 1, 1, 0]).all(axis=1).mean()))
    flips = pd.DataFrame(flips).pivot_table(index="allocation", columns="bit", values="G differs [%]")
    return dict(consistency=cons, handles=handles, flips=flips, chunks=chunks,
                agree={b: float((handles["A1"] == handles[b]).mean()) for b in ("A2", "A3")})


def _em1d(x, w, mu, sd, max_iter, tol):
    from scipy.stats import norm

    W = w.sum()
    pi = np.full(len(mu), 1.0 / len(mu))
    ll0 = -np.inf
    for it in range(1, max_iter + 1):
        lp = np.log(pi)[None, :] + norm.logpdf(x[:, None], mu[None, :], sd[None, :])
        m = lp.max(axis=1, keepdims=True)
        lse = m + np.log(np.exp(lp - m).sum(axis=1, keepdims=True))
        ll = float((lse[:, 0] * w).sum() / W)
        r = np.exp(lp - lse) * w[:, None]
        nk = np.maximum(r.sum(axis=0), 1e-300)
        pi = np.maximum(nk / W, 1e-300)
        mu = (r * x[:, None]).sum(axis=0) / nk
        sd = np.maximum(np.sqrt((r * (x[:, None] - mu) ** 2).sum(axis=0) / nk), 0.5)
        if ll - ll0 < tol:
            break
        ll0 = ll
    return mu, sd, pi, ll, it


def gmm1d(values, counts, k=4, n_starts=50, seed=0, max_iter=10000, tol=1e-12):
    """Weighted 1-D Gaussian mixture of k components fitted by expectation-maximisation on a histogram.

    values, counts: the (value, count) pairs of a histogram; every sample enters with its exact count (no binning).
    EM is started n_starts times (numpy default_rng(seed)): start means = k distinct values drawn from the histogram with
    probability ∝ count, start SDs 20, equal weights. Each start iterates until the mean log-likelihood per sample improves
    by < tol (or max_iter); SDs are floored at 0.5 (the values are integers). The start with the highest log-likelihood
    is kept. Returns dict(mean, sd, weight (sums to 1), loglik, iterations, starts, starts_at_best, boundaries):
    components in ascending mean order; boundaries[j] = the value between mean[j] and mean[j+1] at which the posterior of
    component j+1 first exceeds that of component j (equal posterior, located on a 0.01 grid)."""
    from scipy.stats import norm

    x = np.asarray(values, float)
    w = np.asarray(counts, float)
    rng = np.random.default_rng(seed)
    fits = []
    for _ in range(n_starts):
        mu0 = np.sort(rng.choice(x, k, replace=False, p=w / w.sum()))
        fits.append(_em1d(x, w, mu0, np.full(k, 20.0), max_iter, tol))
    lls = np.array([f[3] for f in fits])
    mu, sd, pi, ll, it = fits[int(np.argmax(lls))]
    o = np.argsort(mu)
    mu, sd, pi = mu[o], sd[o], pi[o]
    bounds = []
    for j in range(k - 1):
        grid = np.arange(mu[j], mu[j + 1], 0.01)
        diff = (np.log(pi[j + 1]) + norm.logpdf(grid, mu[j + 1], sd[j + 1])) - (np.log(pi[j]) + norm.logpdf(grid, mu[j], sd[j]))
        bounds.append(float(grid[np.argmax(diff > 0)]))
    return dict(mean=mu, sd=sd, weight=pi, loglik=ll, iterations=it, starts=n_starts,
                starts_at_best=int((lls > lls.max() - 1e-9).sum()), boundaries=np.array(bounds))


def fabric1(res="results"):
    """ncu_fabric1: per profiled kernel (latest row per ncu_order of each of the three exports; each export row is one
    binary run that profiled exactly one launch), the counter minus the all-exit baseline (ncu_order 0), per probed line.
    Returns (per-line table, raw table, run_ids used)."""
    names = {"lookup_miss": "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum", "lookup_hit": "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
             "fabric": "lts__t_sectors_srcunit_ltcfabric.sum"}
    label = {"lookup_miss": "L2 lookup misses", "lookup_hit": "L2 lookup hits", "fabric": "partition-link sectors"}
    raw, rids = [], []
    for t, met in names.items():
        d = pd.read_csv(f"{res}/08__l2__ncu_fabric1_{t}.csv")
        d = d.sort_values("run_id").groupby("ncu_order").tail(1).set_index("what")
        rids += list(d.run_id)
        raw.append(pd.DataFrame({label[t]: d[met], f"HBM bytes ({label[t]} run)": d["dram__bytes_read.sum"],
                                 f"duration ns ({label[t]} run)": d["gpu__time_duration.sum"], f"first read ({label[t]} run)": d.first_median,
                                 "lines": d.lines, f"passes ({label[t]} run)": d.profiler__replayer_passes}))
    raw = pd.concat(raw, axis=1)
    raw = raw.loc[:, ~raw.columns.duplicated()]
    whats = ["far_first", "near_first", "far_second"]
    per = pd.DataFrame(index=whats)
    for c in label.values():
        per[c] = (raw.loc[whats, c] - raw.loc["all_exit", c]) / raw.loc[whats, "lines"]
    hb = [c for c in raw.columns if c.startswith("HBM bytes")]
    per["HBM bytes (min–max of the 3 runs)"] = [
        f"{min((raw.loc[w, c] - raw.loc['all_exit', c]) / raw.loc[w, 'lines'] for c in hb):.1f}–"
        f"{max((raw.loc[w, c] - raw.loc['all_exit', c]) / raw.loc[w, 'lines'] for c in hb):.1f}" for w in whats]
    du = [c for c in raw.columns if c.startswith("duration")]
    per["kernel time per line [ns]"] = [np.median([raw.loc[w, c] / raw.loc[w, "lines"] for c in du]) for w in whats]
    fr = [c for c in raw.columns if c.startswith("first read")]
    per["median first read in the profiled kernel [cycles, raw]"] = [np.median([raw.loc[w, c] for c in fr]) for w in whats]
    per["lines"] = raw.loc[whats, "lines"].astype(int)
    return per, raw, sorted(rids)
