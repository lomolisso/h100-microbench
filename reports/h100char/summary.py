"""Device table and the one-page cheat sheet of the report (values read from the data)."""
import pandas as pd
from . import data, breakdown as B
from . import figs_mem_tma as A, figs_dsmem_sync as D

CARD_A = ["01", "02", "03", "04", "05", "06", "07", "08"]
CARD_B = ["09", "10", "11", "12"]


def device_table():
    """Run configuration as recorded in the mandatory CSV columns, for the two cards."""
    rows = []
    for label, nn, f in [("card A", "04", "04__latency__summary_v2.csv"),
                         ("card B", "11", "11__dsmem_bw__summary.csv")]:
        r = data.latest(data.read(data.res(nn, f))).iloc[0]
        rows.append({"": label, "GPU": r.gpu_name, "SMs": int(r.sm_count), "SM clock (MHz, locked)": int(r.sm_clock_mhz),
                     "HBM clock (MHz)": int(r.mem_clock_mhz), "CUDA": r.cuda_version, "driver": r.driver_version})
    t = pd.DataFrame(rows).set_index("")
    cd = data.read(data.res("01", "01__timers__clock_domains.csv"))
    cd = cd[cd.clock_control == "none"]
    t["L2 clock / SM clock"] = [f"{cd.lts_over_sm.min():.3f}–{cd.lts_over_sm.max():.3f}", "not re-measured"]
    t["used for"] = ["topology, memory hierarchy, TMA, synchronisation",
                     "DSMEM latency, heatmap, bandwidth, atomics, collectives"]
    return t.T


def gpc_clock_spread():
    m = data.pinned(data.read(data.res("03", "03__sm_gpc_topology__map.csv")), data.PIN["03_map"])
    c = data.pinned(data.read(data.res("03", "03__sm_gpc_topology__clock_domains.csv")), data.PIN["03_clockdom"])
    g = c.merge(m[["smid", "gpc"]], on="smid").groupby("gpc").sm_mhz.median()
    return g.min(), g.max()


def cheat_sheet():
    a, d, c = A.facts(), D.facts(), B.costs()
    gw = d["dsmem_gpu_wide_tbps"]
    lo = min(min(v.values()) for v in gw.values())
    hi = max(max(v.values()) for v in gw.values())
    R = []

    def add(group, what, val, unit, sec):
        R.append((group, what, val, unit, sec))
    add("latency", "local shared memory load (ld.shared)", f"{a['lat_smem_cycles']:.0f}", "cycles", "1")
    add("latency", "L1 hit", f"{a['lat_l1_cycles']:.0f}", "cycles", "1")
    add("latency", "remote DSMEM load (any cluster size 2–16)",
        f"{a['lat_dsmem_remote_cycles']:.0f} ({d['dsmem_pair_latency_range'][0]:.0f}–{d['dsmem_pair_latency_range'][1]:.0f} by SM pair)",
        "cycles", "2")
    add("latency", "L2 hit, near / far partition", f"{a['lat_l2_near_cycles']:.0f} / {a['lat_l2_far_cycles']:.0f}", "cycles", "1")
    add("latency", "HBM, near / far partition", f"{a['lat_hbm_near_cycles']:.0f} / {a['lat_hbm_far_cycles']:.0f}", "cycles", "1")
    add("latency", "one TMA bulk copy global → shared (issue → mbarrier completion seen)",
        f"{a['tma_copy_T0_cycles']:.0f} + (bytes ÷ {a['tma_copy_bpc']:.0f})", "cycles", "3")
    add("latency", "DSMEM atomic add.u32 (native)",
        f"remote load + {d['dsmem_atom_minus_load']:.0f} ≈ {a['lat_dsmem_remote_cycles'] + d['dsmem_atom_minus_load']:.0f}", "cycles", "2")
    add("latency", "DSMEM atomic add.f32 (compiled to a compare-and-swap loop)", f"{d['dsmem_f32_add_cas']:.0f}", "cycles", "2")
    vis = data.latest(data.read(data.res("10", "10__heatmap__summary.csv"))).set_index("level")["median"]
    add("latency", "remote store → visible at the peer, one way: plain st + poll / st.async + mbarrier",
        f"{vis['vis_st_plain/one_way_cycles']:.0f} / {vis['vis_st_async/one_way_cycles']:.0f}", "cycles", "4")
    add("bandwidth", "shared memory / L1, per SM (loads and stores)", f"{a['bw_smem_bpc_per_sm']:.0f}", "B/cycle", "1")
    add("bandwidth", "DSMEM send limit: total out of one SM to 1 or many SMs (ld replies, st, st.async, TMA put)",
        f"{d['dsmem_send_limit_bpc']:.2f}", "B/cycle", "2")
    add("bandwidth", "DSMEM receive, one SM fed by ≥ 3 peers", f"32–{d['dsmem_receive_max_bpc']:.0f}", "B/cycle", "2")
    add("bandwidth", "L2 read, one SM alone / all SMs", f"{a['bw_l2_read_one_sm_bpc']:.0f} / {a['bw_l2_read_bpc_per_sm']:.0f}",
        "B/cycle per SM", "1")
    add("bandwidth", "GPU-wide: L2 read / HBM read / DSMEM", f"{a['bw_l2_read_tbps']:.1f} / {a['bw_hbm_read_tbps']:.2f} / {lo:.1f}–{hi:.1f}",
        "TB/s", "1, 2")
    add("bandwidth", "TMA engine per SM (global → shared)", "128", "B/cycle", "3")
    add("issue", "TMA copy global → shared, back to back: one warp / four warps",
        f"{a['tma_issue_1warp_cycles_per_copy']:.0f} / {a['tma_issue_4warps_cycles_per_copy']:.0f}", "cycles per copy", "3")
    add("issue", "TMA put shared → shared (other CTA), back to back: fixed cost per command",
        f"{a['tma_s2s_issue_bound_cycles_per_command']:.0f} (rate = bytes ÷ max({a['tma_s2s_issue_bound_cycles_per_command']:.0f}, bytes ÷ 21.3))",
        "cycles per command", "3")
    add("sync", "__syncthreads (1 / 8 / 32 warps)",
        f"{d['syncthreads_net'][1]:.0f} / {c['syncthreads']:.0f} / {d['syncthreads_net'][32]:.0f}", "cycles", "4")
    add("sync", "mbarrier arrive.expect_tx / try_wait on a completed phase", f"{c['expect_tx']:.0f} / 39", "cycles", "4")
    add("sync", "fence.proxy.async.shared::cta / unscoped fence.proxy.async (256 threads)",
        f"{c['fence_proxy_async']:.0f} / {c['fence_proxy_async_all']:.0f}", "cycles", "4")
    add("sync", "cluster barrier, 256 threads: no prior store / after a store",
        f"{c[('barrier_nostore', 8)]:.0f} / {c[('barrier', 8)]:.0f}", "cycles", "4, 5")
    add("sync", "one-way hand-off SM A → SM B, st.async + mbarrier: fast mode / median",
        f"{c['handoff_one_way']:.0f} / {B.handoff_hist().pipe(lambda h: h.value[(h['count'].cumsum() / h['count'].sum()) >= 0.5].iloc[0]) / 2:.0f}",
        "cycles", "4, 5")
    add("sync", "round trip A → B → A: st.async + mbarrier / release-acquire flag in shared memory / "
        "release-acquire flag in global memory (home near–far)",
        f"{d['handoff_st_async_rt']:.0f} / {d['handoff_release_acquire_rt']:.0f} / {d['handoff_global_flag_rt'][0]:.0f}–{d['handoff_global_flag_rt'][1]:.0f}",
        "cycles", "4")
    add("placement", "max active clusters at 1 CTA/SM, CS 2 / 4 / 8 / 16",
        " / ".join(str(v) for v in d["max_active_clusters_1cta"].values()) + "  (idle SMs " +
        " / ".join(str(v) for v in d["idle_sms_1cta"].values()) + ")", "", "2")
    t = pd.DataFrame(R, columns=["kind", "quantity", "value", "unit", "section"])
    return t.set_index(["kind", "quantity"])
