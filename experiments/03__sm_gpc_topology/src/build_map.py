"""Build a new run of results/03__sm_gpc_topology__map.csv from the latest complete cosched and clockdom runs.

1. Co-scheduling domains: every pair of SMs that hosted CTAs of the same cluster in one launch is joined
   (union-find over all CS and all launches). A component larger than 18 SMs would mean the probe merged records
   of different launches (all CTAs of a cluster share one GPC) and is reported as an error.
2. Clock domains: SMs i and j share a clock domain iff |sm_mhz_i − sm_mhz_j| < TOL_MHZ in every clockdom launch.
   Each domain must be a clique under that rule (checked), and the closest distinct domains are printed.
3. Physical GPC: a clock domain that contains exactly one co-scheduling domain able to host clusters of ≥ 4 CTAs
   becomes one GPC together with the smaller co-scheduling domains (TPCs restricted to clusters of ≤ 2) that share
   its clock. If any check fails, gpc falls back to the co-scheduling domain and a warning is printed.
GPC and domain ids are numbered by their smallest %smid.

Output: one new map run (132 rows) with its own run_id (UTC ISO timestamp, same format as the harness), registered
in results/03__sm_gpc_topology__runs.csv as `started` before and `complete` after the rows are appended.
Columns: mandatory (values of the cosched run, new run_id, launch_idx = -1), then smid, gpc, gpc_size, hosts_cs16
(the GPC hosted a cluster of 16), tpc_mate_hyp (= smid ^ 1), partition, subnode, notes, cosched_domain,
cosched_domain_size, max_cs_observed, clock_domain, clockdom_run_id.
- partition / subnode are not measured here: they are copied per smid from the latest earlier map run that has them
  filled for every SM (written by the L2 experiment 08), together with its "partition/subnode from …" note; empty
  if no such run exists.
- notes: "card serial <S>" (the board's `Serial Number` line of results/…card.txt, not the chassis serial);
  "only in clusters of <= N; GPC from clock domain" only for SMs outside every co-scheduling domain that hosts
  clusters of >= 4 (SMs 124–131 on this card), i.e. the SMs whose GPC comes from the clock domain alone.
Appended, never rewritten. Run from experiments/03__sm_gpc_topology (src/run.sh does).
"""
import csv
import fcntl
import io
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

R = "results/03__sm_gpc_topology__"
TOL_MHZ = 0.02
MANDATORY = ["gpu_name", "sm_count", "sm_clock_mhz", "mem_clock_mhz", "cuda_version", "driver_version", "run_id", "launch_idx"]


def latest_complete(sub):
    runs = pd.read_csv(R + "runs.csv")
    started = runs[runs.status == "started"]
    done = set(runs[runs.status == "complete"].run_id)
    ids = [r for r, c in zip(started.run_id, started.cmdline) if r in done and re.search(rf"gpc_probe {sub}(\s|$)", c)]
    if not ids:
        raise SystemExit(f"no complete {sub} run in {R}runs.csv")
    return ids[-1]


def components(n, pairs):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = [find(i) for i in range(n)]
    order = {r: k for k, r in enumerate(sorted(set(roots)))}  # roots are the smallest member → numbered by smallest smid
    return np.array([order[r] for r in roots])


def append_rows(path, header, rows):
    with open(path, "a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0, os.SEEK_END)
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        if f.tell() == 0:
            w.writerow(header)
        else:
            f.seek(0)
            if next(csv.reader(f)) != header:
                raise SystemExit(f"header mismatch in {path}")
            f.seek(0, os.SEEK_END)
        w.writerows(rows)
        f.write(buf.getvalue())
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def main():
    rid_e, rid_c = latest_complete("cosched"), latest_complete("clockdom")
    e = pd.read_csv(R + "edges.csv", usecols=MANDATORY + ["cs", "cluster_id", "smid"])
    e = e[e.run_id == rid_e]
    n = int(e.sm_count.iloc[0])
    print(f"cosched run {rid_e}: {e.launch_idx.nunique()} launches, {len(e)} CTAs; clockdom run {rid_c}")

    # 1. co-scheduling domains
    e = e.sort_values(["launch_idx", "cluster_id", "smid"], kind="stable")
    same = (e.launch_idx.values[1:] == e.launch_idx.values[:-1]) & (e.cluster_id.values[1:] == e.cluster_id.values[:-1])
    a, b = e.smid.values[:-1][same], e.smid.values[1:][same]
    pairs = set(zip(a[a != b].tolist(), b[a != b].tolist()))
    dom = components(n, pairs)
    dom_size = np.bincount(dom)[dom]
    max_cs = e.groupby("smid").cs.max().reindex(range(n)).fillna(0).astype(int).values
    sizes = sorted(np.bincount(dom).tolist(), reverse=True)
    print(f"co-scheduling domains ({len(sizes)}): sizes {sizes}, sum {sum(sizes)}")
    ok = sum(sizes) == n
    if max(sizes) > 18:
        print("ERROR: a co-scheduling domain has more than 18 SMs (records of different launches merged?)")
        ok = False

    # 2. clock domains
    c = pd.read_csv(R + "clock_domains.csv")
    c = c[c.run_id == rid_c]
    rate = c.pivot(index="launch_idx", columns="smid", values="sm_mhz").reindex(columns=range(n)).values  # launches × SMs
    diff = np.nanmax(np.abs(rate[:, :, None] - rate[:, None, :]), axis=0)
    clk = components(n, [(i, j) for i in range(n) for j in range(i + 1, n) if diff[i, j] < TOL_MHZ])
    for d in np.unique(clk):
        m = clk == d
        if diff[np.ix_(m, m)].max() >= TOL_MHZ:
            print(f"WARNING: clock domain {d} is not a clique (max pairwise difference {diff[np.ix_(m, m)].max():.4f} MHz)")
            ok = False
    within = max(diff[np.ix_(clk == d, clk == d)].max() for d in np.unique(clk))
    between = min(diff[np.ix_(clk == d1, clk == d2)].max() for d1 in np.unique(clk) for d2 in np.unique(clk) if d1 < d2)
    print(f"clock domains ({clk.max() + 1}): sizes {sorted(np.bincount(clk).tolist(), reverse=True)}; "
          f"max difference within a domain {within:.4f} MHz, closest pair of domains differs by up to {between:.4f} MHz")

    # 3. physical GPCs
    big = {d for d in np.unique(dom) if max_cs[dom == d].max() >= 4}
    for d in big:
        if len(set(clk[dom == d])) != 1:
            print(f"WARNING: co-scheduling domain {d} spans several clock domains")
            ok = False
    for k in np.unique(clk):
        nb = len({d for d in set(dom[clk == k]) if d in big})
        if nb != 1:
            print(f"WARNING: clock domain {k} contains {nb} co-scheduling domains able to host clusters of >= 4")
            ok = False
    if ok:
        gpc = components(n, [(i, j) for i in range(n) for j in range(i + 1, n) if clk[i] == clk[j]])
    else:
        print("WARNING: falling back to gpc = co-scheduling domain")
        gpc = dom
    gpc_size = np.bincount(gpc)[gpc]
    cs16 = set(e[e.cs == 16].smid)
    hosts16 = {g for g in np.unique(gpc) if any(s in cs16 for s in np.where(gpc == g)[0])}

    serial = ""
    if os.path.exists(R + "card.txt"):
        m = re.findall(r"^\s*Serial Number\s*:\s*(\S+)", open(R + "card.txt").read(), flags=re.M)  # not "Chassis Serial Number"
        serial = m[-1] if m else ""

    # partition / subnode: copied from the latest earlier map run that has them for every SM
    header = MANDATORY + ["smid", "gpc", "gpc_size", "hosts_cs16", "tpc_mate_hyp", "partition", "subnode", "notes",
                          "cosched_domain", "cosched_domain_size", "max_cs_observed", "clock_domain", "clockdom_run_id"]
    part, sub, part_note, rid_p = {}, {}, "", ""
    if os.path.exists(R + "map.csv"):
        old = pd.read_csv(R + "map.csv", dtype=str, keep_default_na=False)
        full = [r for r, g in old.groupby("run_id") if (g.partition != "").all() and (g.subnode != "").all()
                and set(g.smid.astype(int)) == set(range(n))]
        if full:
            rid_p = max(full)
            g = old[old.run_id == rid_p].set_index(old[old.run_id == rid_p].smid.astype(int))
            part, sub = g.partition.to_dict(), g.subnode.to_dict()
            frag = sorted({f for t in g.notes for f in re.findall(r"partition/subnode from [^;]*", t)})
            part_note = "; ".join(frag)
            moved = [s for s in range(n) if int(g.gpc[s]) != gpc[s]]
            print(f"partition/subnode copied from map run {rid_p} ({part_note or 'no source note'})"
                  + (f"; WARNING: gpc differs from that run for smid {moved}" if moved else ""))
        else:
            print("no earlier map run with partition/subnode for every SM: left empty")

    first = e.iloc[0]
    rid = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    rows = []
    for s in range(n):
        notes = [f"card serial {serial}"] if serial else []
        if dom[s] not in big:  # GPC known only from the clock domain (SMs never in clusters of >= 4)
            notes.append(f"only in clusters of <= {max_cs[s]}; GPC from clock domain")
        if part_note:
            notes.append(part_note)
        rows.append([first[k] for k in MANDATORY[:-2]] + [rid, -1, s, gpc[s], gpc_size[s], int(gpc[s] in hosts16), s ^ 1,
                                                         part.get(s, ""), sub.get(s, ""), "; ".join(notes), dom[s],
                                                         dom_size[s], max_cs[s], clk[s], rid_c])

    # register the map run (started → rows → complete) so an interrupted write shows up as partial
    reg_hdr = ["run_id", "started_iso", "finished_iso", "status", "gpu_uuid", "cmdline"]
    runs = pd.read_csv(R + "runs.csv", dtype=str, keep_default_na=False)
    uuid = runs[runs.run_id == rid_e].gpu_uuid.iloc[0]
    cmd = (f"src/build_map.py (from cosched run {rid_e} and clockdom run {rid_c}"
           + (f"; partition/subnode copied from map run {rid_p}: {part_note})" if rid_p else ")"))
    append_rows(R + "runs.csv", reg_hdr, [[rid, rid, "", "started", uuid, cmd]])
    append_rows(R + "map.csv", header, rows)
    append_rows(R + "runs.csv", reg_hdr, [[rid, "", datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                                           "complete", uuid, cmd]])
    for g in np.unique(gpc):
        sm = np.where(gpc == g)[0]
        print(f"G{g}: {len(sm):2d} SMs, hosts CS16 {int(g in hosts16)}, cluster-capable {int((max_cs[sm] >= 4).sum()):2d}: "
              + " ".join(map(str, sm)))
    print(f"GPCs: {len(np.unique(gpc))}, sizes {sorted(np.bincount(gpc).tolist(), reverse=True)}, "
          f"able to host CS 16: {len(hosts16)}; map run {rid} appended to {R}map.csv")


if __name__ == "__main__":
    main()
