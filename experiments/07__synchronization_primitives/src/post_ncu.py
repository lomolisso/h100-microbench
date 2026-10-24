"""Turn the ncu exports of one `bin/sync_* ncu` invocation into result rows with the mandatory columns.

The binary writes a manifest (bin/ncu_manifest_<tag>.csv): one row per profiled launch, in launch order, with the
mandatory columns (run_id, NVML clocks sampled after the launch) and the configuration (kernel group, primitive, cs,
block_threads, variant). ncu profiles every launch of the ncu mode, so its kernels are joined to the manifest by order.
Inputs (bin/, from src/run.sh):
  ncu_<tag>_raw.csv     ncu -i … --page raw --csv --print-units base                 (one row per kernel)
  ncu_<tag>_opc.csv     same with --print-metric-instances details --metrics sass__inst_executed_per_opcode
  ncu_<tag>_source.csv  ncu -i … --page source --csv --print-source sass             (per-PC block per kernel)
Outputs (append only):
  results/07__sync__ncu_metrics_<tag>.csv  manifest + kernel name + kept raw metrics (stall ratios, pc-sampling totals,
                                    instruction counts, clocks, launch facts, replay passes)
  results/07__sync__ncu_opcodes.csv manifest + kernel name + opcode + executed warp-level instructions
  results/07__sync__stalls.csv      primitive, pc_opcode, stall_reason, samples (+ configuration, kernel, PC offset,
                                    SASS text): one row per (profiled launch, PC, stall reason with samples > 0),
                                    from the "All Samples" stall columns of the source page
  results/07__sync__ncu_pcs.csv     every PC of every profiled kernel (configuration, kernel, PC offset, SASS, total
                                    samples), so that stalls can be attributed to the instruction window they belong to
--pcs-only writes only the last file (back-fill for exports processed before it existed).
"""
import argparse
import csv
import re

from common07 import MANDATORY, append_rows

KEEP_PREFIX = ("smsp__average_warps_issue_stalled_", "smsp__pcsamp_warps_issue_stalled_", "smsp__pcsamp_sample_count",
               "sm__cycles_elapsed.avg.per_second", "gpu__time_duration.sum", "smsp__inst_executed.sum",
               "smsp__cycles_active.avg", "sm__warps_active", "launch__", "profiler__replayer_passes")


def quoted_rows(path):
    with open(path, newline="") as f:
        return list(csv.reader([l for l in f if l.startswith('"')]))  # drop ==PROF== messages


def raw_table(path):
    rows = quoted_rows(path)
    header = rows[0]
    data = [r for r in rows[1:] if r and r[0].strip().isdigit()]  # numeric ID: drops the units row
    return header, data


def source_blocks(path):
    """[(kernel name, header, rows)], one per "Launch Index" marker (first block after it; repeats are skipped)."""
    out, cur, take = [], None, False
    for r in quoted_rows(path):
        if r and r[0] == "Launch Index":
            take, cur = True, None
        elif r and r[0] == "Kernel Name":
            cur = [r[1], None, []] if take else None
            if take:
                out.append(cur)
            take = False
        elif cur is not None and cur[1] is None:
            cur[1] = r
        elif cur is not None and r:
            cur[2].append(r)
    return out


def opcode(sass):
    toks = sass.split()
    if toks and toks[0].startswith("@"):
        toks = toks[1:]
    return toks[0].rstrip(";") if toks else ""


ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)  # k1 … k5
ap.add_argument("--res", default="results")  # output directory (tests: a scratch directory)
ap.add_argument("--pcs-only", action="store_true")
a = ap.parse_args()
t = a.tag
with open(f"bin/ncu_manifest_{t}.csv", newline="") as f:
    rd = csv.reader(f)
    man_header = next(rd)
    man = list(rd)
cfg = man_header[len(MANDATORY):]

n = len(man)
if not a.pcs_only:
    header, data = raw_table(f"bin/ncu_{t}_raw.csv")
    keep = [i for i, c in enumerate(header) if c.startswith(KEEP_PREFIX)]
    keep.sort(key=lambda i: header[i])
    kn = header.index("Kernel Name")
    if len(data) != len(man):
        print(f"WARNING {t}: {len(data)} profiled kernels, manifest has {len(man)}; joining the first {min(len(data), len(man))}")
    n = min(len(data), len(man))
    append_rows(f"{a.res}/07__sync__ncu_metrics_{t}.csv", man_header + ["kernel_name"] + [header[i] for i in keep],
                [man[j] + [data[j][kn]] + [data[j][i] for i in keep] for j in range(n)])

    oh, od = raw_table(f"bin/ncu_{t}_opc.csv")
    oc, okn = oh.index("sass__inst_executed_per_opcode"), oh.index("Kernel Name")
    opc_rows = []
    for j in range(min(len(od), len(man))):
        m = re.search(r"\((.*)\)\s*$", od[j][oc])
        for item in (m.group(1).split(";") if m else []):
            op, _, cnt = item.strip().partition(":")
            opc_rows.append(man[j] + [od[j][okn], op.strip(), cnt.strip()])
    append_rows(f"{a.res}/07__sync__ncu_opcodes.csv", man_header + ["kernel_name", "opcode", "inst_executed"], opc_rows)

blocks = source_blocks(f"bin/ncu_{t}_source.csv")
if len(blocks) != len(man):
    print(f"WARNING {t}: {len(blocks)} source blocks, manifest has {len(man)}")
st_rows, pc_rows = [], []
for j, (kname, sh, rows) in enumerate(blocks[: len(man)]):
    ia, isrc = sh.index("Address"), sh.index("Source")
    reasons = [(i, c[len("stall_"):]) for i, c in enumerate(sh) if c.startswith("stall_") and "(Not Issued)" not in c]
    base = int(rows[0][ia], 16) if rows else 0
    mrow = man[j]
    conf = dict(zip(cfg, mrow[len(MANDATORY):]))
    isam = sh.index("Warp Stall Sampling (All Samples)")
    for r in rows:
        sass = r[isrc].strip()
        pc_rows.append(mrow[: len(MANDATORY)] + [conf.get("kernel", ""), conf.get("primitive", ""), conf.get("cs", ""),
                       conf.get("block_threads", ""), conf.get("variant", ""), kname, hex(int(r[ia], 16) - base), sass,
                       r[isam].strip()])
        for i, reason in reasons:
            v = r[i].strip()
            if v and v != "0" and v != "-":
                st_rows.append(mrow[: len(MANDATORY)] + [conf.get("primitive", ""), opcode(sass), reason, v,
                               conf.get("kernel", ""), conf.get("cs", ""), conf.get("block_threads", ""),
                               conf.get("variant", ""), kname, hex(int(r[ia], 16) - base), sass])
append_rows(f"{a.res}/07__sync__ncu_pcs.csv",
            MANDATORY + ["kernel", "primitive", "cs", "block_threads", "variant", "kernel_name", "pc_offset", "sass",
                         "samples"], pc_rows)
if a.pcs_only:
    print(f"ncu {t}: {len(pc_rows)} PC rows")
    raise SystemExit(0)
append_rows(f"{a.res}/07__sync__stalls.csv",
            MANDATORY + ["primitive", "pc_opcode", "stall_reason", "samples", "kernel", "cs", "block_threads", "variant",
                         "kernel_name", "pc_offset", "sass"], st_rows)
print(f"ncu {t}: {n} launches, {len(opc_rows)} opcode rows, {len(st_rows)} stall rows, {len(pc_rows)} PC rows")
