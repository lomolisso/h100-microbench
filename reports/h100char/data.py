"""Paths and readers shared by the report's figure modules.

All data are local CSVs written by the experiments (raw data never leave this machine). Readers select
the run explicitly: `latest(df)` for derived summaries (one analysis run per file), or a pinned run_id
where the latest run of a file belongs to a different measurement campaign.
"""
import os
import importlib.util
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXP = os.path.join(ROOT, "experiments")
COLL = os.path.abspath(os.path.join(ROOT, "..", "h100-dsmem-collectives", "collectives"))
CACHE = os.path.join(ROOT, "reports", "cache")
os.makedirs(CACHE, exist_ok=True)

EXPDIR = {
    "01": "01__timers_and_harness", "02": "02__cluster_launch_and_occupancy", "03": "03__sm_gpc_topology",
    "04": "04__memory_hierarchy_latency", "05": "05__memory_hierarchy_bandwidth",
    "06": "06__tma_latency_and_issue", "07": "07__synchronization_primitives",
    "08": "08__l2_partitions_and_first_touch", "09": "09__dsmem_latency_vs_cluster_size",
    "10": "10__dsmem_latency_heatmap", "11": "11__dsmem_bandwidth_vs_cluster_size",
    "12": "12__dsmem_atomics_and_async_stores",
}

# Runs pinned because a later run of the same file belongs to the other card or another campaign.
PIN = {
    "03_map": "2026-10-25T01:12:52.246",        # card A map with L2 partition/sub-node columns filled
    "03_clockdom": "2026-10-24T00:09:39.951",   # card A per-SM clock under the lock
    "05_summary": "2026-10-25T04:51:55.475",
}

MANDATORY = ["gpu_name", "sm_count", "sm_clock_mhz", "mem_clock_mhz", "cuda_version", "driver_version",
             "run_id", "launch_idx"]


def res(nn, name=""):
    """Absolute path of experiment NN's results directory (or of one file in it)."""
    return os.path.join(EXP, EXPDIR[nn], "results", name)


def coll(name=""):
    return os.path.join(COLL, "results", name)


def read(path, **kw):
    return pd.read_csv(path, low_memory=False, **kw)


def latest(df):
    return df[df.run_id == df.run_id.max()].copy()


def pinned(df, run_prefix):
    out = df[df.run_id.astype(str).str.startswith(run_prefix)].copy()
    assert len(out), f"pinned run {run_prefix} not found"
    return out


def read_run(path, run_prefix=None, usecols=None, chunksize=1_000_000, query=None, dtype=None):
    """Chunked read of a large raw CSV keeping one run (prefix match on run_id; None = all) and rows
    matching `query` (a pandas query string)."""
    parts = []
    cols = None if usecols is None else list(dict.fromkeys(["run_id"] + list(usecols)))
    for ch in pd.read_csv(path, usecols=cols, chunksize=chunksize, low_memory=False, dtype=dtype):
        if run_prefix is not None:
            ch = ch[ch.run_id.astype(str).str.startswith(run_prefix)]
        if query is not None and len(ch):
            ch = ch.query(query)
        if len(ch):
            parts.append(ch)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def runs(nn):
    """Run registry of experiment NN (run_id, status, gpu_uuid, cmdline)."""
    return read(res(nn, f"{EXPDIR[nn]}__runs.csv"))


def cached(name, build):
    """Return a cached reduction (CSV under reports/cache), building it once with `build()`."""
    p = os.path.join(CACHE, name)
    if os.path.exists(p):
        return pd.read_csv(p, low_memory=False)
    df = build()
    df.to_csv(p, index=False)
    return df


def import_file(path, name):
    """Import a helper module from another directory under a distinct name (no chdir, no sys.path)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
