"""Per-SM view of a DSMEM collective: measured time per round of one rank, and its split into the
costs of the primitives the round executes.

Measured part (`rounds()`): every rank's thread 0 writes a %globaltimer stamp at entry (after the entry
cluster barrier), at the end of each round (after the round's last synchronisation) and at exit. A
segment is the difference of consecutive stamps of the same rank and launch. In these runs the timer of
each SM advanced in 256-ns steps (every per-rank difference is a multiple of 256 ns; the step phase differs
between SMs and launches), so one launch cannot resolve a segment. The phase of a launch relative to the
work is random, hence the plain mean over the 400 launches of a configuration (two sweeps x 200) is an
unbiased estimate (standard error ~ 6 ns per segment); a median or trimmed mean would stay biased towards
the 256-ns grid. Launches in which the rank's total time exceeds 3x its median (rare stalls) are dropped
and counted. The reduction runs once over the stamp files and is cached.
"""
import numpy as np
import pandas as pd
from . import data

OPS = ["allreduce", "allgather", "reducescatter", "alltoall"]
STAMP_COLS = ["run_id", "file", "N", "M", "rep", "rank", "smid", "stamp", "t_ns", "clock_bad"]
STEP_NS = 256


def _reduce_file(path):
    df = pd.read_csv(path, usecols=STAMP_COLS, dtype={"t_ns": "int64"}, low_memory=False)
    last = df.groupby(["file", "N", "M"]).run_id.transform("max")
    df = df[(df.run_id == last) & (df.clock_bad == 0)]
    order = {"entry": -1, "exit": 10_000}
    df = df.assign(idx=df.stamp.map(lambda s: order[s] if s in order else int(s[5:])))
    df = df.sort_values(["file", "N", "M", "rep", "rank", "idx"])
    df = df.assign(dt=df.groupby(["file", "N", "M", "rep", "rank"], sort=False).t_ns.diff(), seg=df.stamp)
    df = df[df.dt.notna()]     # the entry stamp starts each rank's chain
    # 'exit' coincides with the last round stamp in files without work after it (segment ~ 0)
    return df[["file", "N", "M", "rep", "rank", "smid", "seg", "idx", "dt"]]


def _build():
    parts = []
    for op in OPS:
        for s in (1, 2):
            d = _reduce_file(data.coll(f"stamps_{op}_s{s}.csv"))
            d["sweep"] = s
            parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    key = ["file", "N", "M", "rank"]
    tot = df.groupby(key + ["sweep", "rep"]).dt.sum().rename("L").reset_index()
    tot["stall"] = tot.L > 3 * tot.groupby(key).L.transform("median")
    df = df.merge(tot, on=key + ["sweep", "rep"])
    nstall = tot.groupby(key).stall.sum().rename("n_stall").reset_index()
    df, tot = df[~df.stall], tot[~tot.stall]
    out = (df.groupby(key + ["seg", "idx"])
             .agg(smid=("smid", "first"), n=("dt", "size"), dt_ns=("dt", "mean"), dt_sd_ns=("dt", "std"))
             .reset_index())
    out["dt_se_ns"] = out.dt_sd_ns / np.sqrt(out.n)
    L = tot.groupby(key).L.agg(L_rank_ns="mean", L_rank_med_ns="median").reset_index()
    return out.merge(L, on=key).merge(nstall, on=key)


def rounds():
    """Per (file, N, M, rank, segment): mean duration in ns over all valid launches (dt_se_ns: its s.e.).
    seg = 'round<k>' (from the previous stamp to the end of round k) or 'exit'."""
    df = data.cached("coll_rank_rounds.csv", _build)
    meta = data.read(data.coll("summary_latency.csv"))[["file", "op", "alg", "N", "M", "payload_bytes",
                                                         "L_med_ns"]]
    return df.merge(meta, on=["file", "N", "M"], how="left")


# ------------------------------------------------------------------------------------------------------
# Modelled part: the primitives each round executes on rank 0's critical path, priced with the costs
# measured in isolation (SM cycles at 1635 MHz). The collectives run 256 threads per CTA; costs measured at
# 128 and 512 threads are interpolated linearly to 256.

CYC = 1.635                                  # cycles per ns
COMPONENTS = [                               # stacking order (bottom -> top) and legend text
    ("barrier", "cluster barrier"),
    ("handoff", "hand-off: arm, 1st copy, flight, completion seen"),
    ("issue", "further copy commands"),
    ("transfer", "payload at 21.3 B/cycle"),
    ("loads", "remote loads (own chain)"),
    ("local", "local reduce/copy + proxy fence"),
]


def _summ(nn, name):
    return data.latest(data.read(data.res(nn, name)))


def _interp(lo, hi, t_lo=128, t_hi=512, t=256):
    return lo + (hi - lo) * (t - t_lo) / (t_hi - t_lo)


def costs():
    """Primitive costs (cycles) with their provenance, read from the microbenchmark summaries."""
    s07 = _summ("07", "07__sync__summary_v2.csv")
    s06 = _summ("06", "06__tma__summary.csv")
    s09 = _summ("09", "09__dsmem_latency__summary.csv")
    s04 = _summ("04", "04__latency__summary_v2.csv")

    def v07(level, cs, thr, col="median"):
        r = s07[(s07.level == level) & (s07.config == f"cs {cs}, threads {thr}")]
        assert len(r) == 1, (level, cs, thr)
        return float(r[col].iloc[0])

    c = {}
    for cs in (2, 4, 8, 16):
        c[("barrier_nostore", cs)] = _interp(v07("k2/cluster_sync/single", cs, 128),
                                             v07("k2/cluster_sync/single", cs, 512))
        # every round of a collective issues stores (bulk copies, reduce writes) before its barrier, so the
        # release arrive pays the "after a store" fence; + the wait once everybody has arrived
        c[("barrier", cs)] = _interp(v07("k2/cluster_arrive_release/after_st_global", cs, 128),
                                     v07("k2/cluster_arrive_release/after_st_global", cs, 512)) + \
            v07("k2/cluster_wait_acquire/all_arrived", cs, 128)
        r = s09[(s09.level == "k1/ld_cluster/stride1/Default/remote") & (s09.config == f"cs {cs}")]
        c[("remote_ld", cs)] = float(r["median"].iloc[0])
    c["self_ld"] = float(s09[(s09.level == "k1/ld_cluster/stride1/Default/self") & (s09.config == "cs 2")]["median"].iloc[0])
    c["expect_tx"] = v07("k3/mbar_arrive_expect_tx/single", 0, 1)
    c["fence_proxy_async"] = _interp(v07("k1/fence_proxy_async/single", 1, 128), v07("k1/fence_proxy_async/single", 1, 512))
    c["syncthreads"] = _interp(v07("k1/bar_sync0/single", 1, 128, "net_cycles"), v07("k1/bar_sync0/single", 1, 512, "net_cycles"))
    c["handoff_rt_median"] = v07("k4/st_async_mbarrier/push_r01/round_trip_cycles", 2, 1)
    c["handoff_one_way"], c["handoff_slow_step"] = handoff_modes()
    h = handoff_hist()
    cum = h["count"].cumsum() / h["count"].sum()
    c["handoff_rt_median_all_pairs"] = float(h.value[cum >= 0.5].iloc[0])
    c["issue_cmd"] = 65.0      # per further shared->shared bulk command from one thread (11: 64-66 cycles
                               # per command for commands <= 1 KiB; only given in that notebook's text)
    c["lds"] = float(s04[s04.level == "smem"]["median_cycles_at_1635"].iloc[0])
    c["send_bpc"], c["recv_bpc"], c["smem_bpc"] = 64 / 3, 32.0, 128.0
    return c


def handoff_modes():
    """One-way st.async + mbarrier hand-off from the ping-pong over every SM pair (round trips without
    timer reads): the round trip has modes at 2h, 2h + q and 2h + 2q (each direction is either fast, h,
    or one quantum q slower). Returns (h, q)."""
    def build():
        p = data.res("10", "10__heatmap__visibility_hist.csv")
        d = pd.read_csv(p, usecols=["method", "value", "count"])
        d = d[d.method == "st_async_mbarrier_nogt"]
        return d.groupby("value")["count"].sum().reset_index()
    h = data.cached("handoff_rt_hist.csv", build).set_index("value")["count"]
    fast = h[h.index < 450]
    mid = h[(h.index >= 450) & (h.index < 620)]
    rt_fast = (fast.index * fast).sum() / fast.sum()
    rt_mid = (mid.index * mid).sum() / mid.sum()
    return rt_fast / 2, rt_mid - rt_fast


def handoff_hist():
    """Round-trip histogram (value, count) of the isolated st.async + mbarrier ping-pong."""
    handoff_modes()
    return data.read(data.CACHE + "/handoff_rt_hist.csv")


def _local(c, elems, reads, writes, threads=256):
    """Local smem pass over `elems` 4-B elements by 256 threads: one dependent load-add-store chain per
    iteration plus the smem bandwidth for the bytes moved."""
    iters = -(-elems // threads)
    return iters * (c["lds"] + 10) + elems * 4 * (reads + writes) / c["smem_bpc"]


def model(file, N, M, c=None):
    """List of (segment, {component: cycles}) for rank 0, in stamp order. Bytes are 4-B elements."""
    c = c or costs()
    bar, rld = c[("barrier", N)], c[("remote_ld", N)]
    hop = c["handoff_one_way"]
    extra = lambda k: (k - 1) * c["issue_cmd"]
    snd = lambda b: b / c["send_bpc"]
    fence = c["fence_proxy_async"]
    alg = file.split("__")[1]
    op = file.split("__")[0]
    rounds = []
    if op == "allreduce" and alg == "ring":
        C = M // N
        for s in range(N - 1):
            rounds.append({"barrier": bar, "handoff": hop, "transfer": snd(4 * C),
                           "local": _local(c, C, 2, 1) + fence})
        for s in range(N - 1):
            rounds.append({"barrier": bar, "handoff": hop, "transfer": snd(4 * C)})
    elif op == "allreduce" and alg == "recursive_doubling":
        for s in range(N.bit_length() - 1):
            rounds.append({"barrier": bar, "handoff": hop, "transfer": snd(4 * M),
                           "local": _local(c, M, 2, 1) + fence})
    elif op == "allgather" and alg == "ring":
        for s in range(N - 1):
            rounds.append({"barrier": bar, "handoff": hop, "transfer": snd(4 * M)})
    elif op == "allreduce" and alg == "two_shot_push":
        C = M // N
        # the last of the N-1 peers to serve rank 0 issues N-1 commands and sends (N-1)*C*4 B first
        # The last peer to serve rank 0 issues N-1 commands; the payload streams while later commands
        # are issued, so only the transfer time not hidden behind issue is on the critical path.
        k, b = N - 1, 4 * C * (N - 1)
        push = {"handoff": hop, "issue": extra(k), "transfer": max(0.0, snd(b) - extra(k))}
        rounds.append(dict(push, local=_local(c, C, N, 1) + fence + c["syncthreads"]))
        rounds.append(dict(push))
    elif op == "alltoall" and alg == "direct_push":
        k, b = N - 1, 4 * M * (N - 1)
        push = {"handoff": hop, "issue": extra(k), "transfer": max(0.0, snd(b) - extra(k))}
        # thread 0 copies its own chunk locally before waiting: only the part longer than the wait counts
        push["local"] = max(0.0, _local(c, M, 1, 1) - sum(push.values()))
        rounds.append(push)
    elif op == "allreduce" and alg == "one_shot_pull":
        iters = -(-M // 4 // 256)                          # 16-B vectors per thread
        rounds.append({"barrier": bar, "loads": iters * ((N - 1) * rld + c["self_ld"])})
        rounds.append({"barrier": bar})                   # trailing barrier (stamp 'exit')
    else:
        raise KeyError(file)
    return rounds


BREAKDOWN_FILES = ["allreduce__ring__nccl", "allreduce__recursive_doubling__clusterfusion",
                   "allreduce__two_shot_push__trtllm", "allreduce__one_shot_pull__trtllm",
                   "allgather__ring__nccl", "alltoall__direct_push__generic"]


def table(rank=0, files=BREAKDOWN_FILES):
    """Measured vs modelled cycles per (file, N, M, segment) for one rank."""
    c = costs()
    meas = rounds()
    meas = meas[(meas["rank"] == rank) & meas.file.isin(files)]
    rows = []
    for (f, N, M), g in meas.groupby(["file", "N", "M"]):
        g = g.sort_values("idx")
        segs = list(g.seg)
        mod = model(f, N, M, c)
        if f.startswith("allreduce__one_shot_pull"):
            pairs = list(zip(["round0", "exit"], mod))
        else:
            pairs = list(zip([s for s in segs if s != "exit"], mod)) + [("exit", {})]
        for seg, comp in pairs:
            r = g[g.seg == seg].iloc[0]
            row = dict(file=f, op=r.op, alg=r.alg, N=N, M=M, payload_bytes=r.payload_bytes, seg=seg,
                       idx=r.idx, smid=r.smid, meas=r.dt_ns * CYC, meas_se=r.dt_se_ns * CYC,
                       L_rank=r.L_rank_ns * CYC)
            row.update({k: comp.get(k, 0.0) for k, _ in COMPONENTS})
            rows.append(row)
    t = pd.DataFrame(rows)
    t["model"] = t[[k for k, _ in COMPONENTS]].sum(axis=1)
    t["residual"] = t.meas - t.model
    return t
