#!/usr/bin/env python3
"""Plain-text results summary of every experiment that has an executed notebook, for readers that see the repository
only through git (raw results/ stay on this machine).

For each experiments/NN__name/NN__name.ipynb writes experiments/NN__name/NN__name__results_summary.txt with:
  1. the experiment's row of the AGENTS.md status table (status, key results);
  2. the notebook as text: markdown cells verbatim, code outputs (printed text, tables rendered as aligned text,
     figures as "[figure k]"); code itself is omitted;
  3. the latest run_id of the experiment's derived tables, if present locally: results/*summary*.csv and the per-SM
     maps (03 map, 08 sm_signature / sm_map), mandatory columns other than run_id dropped.
Usage (repo root): h100 run --no-gpu -- python3 tools/export_results_summary.py [NN ...]
"""
import glob
import html
import json
import os
import re
import sys
from html.parser import HTMLParser

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANDATORY = ["gpu_name", "sm_count", "sm_clock_mhz", "mem_clock_mhz", "cuda_version", "driver_version", "launch_idx"]
EXTRA = ["*sm_gpc_topology__map.csv", "*__sm_signature.csv"]


class Tables(HTMLParser):
    """Collects <table> contents (caption + rows of cell texts); ignores <style>."""
    def __init__(self):
        super().__init__()
        self.tables, self.cur, self.row, self.cell, self.caption, self.skip = [], None, None, None, None, 0
        self.span, self.carry = 1, {}  # rowspan of the open cell; column → (text, rows left) carried into later rows

    def handle_starttag(self, tag, attrs):
        if tag == "style": self.skip += 1
        elif tag == "table": self.cur, self.carry = {"caption": "", "rows": []}, {}
        elif tag == "caption": self.caption = ""
        elif tag == "tr" and self.cur is not None: self.row = []; self.fill()
        elif tag in ("td", "th") and self.row is not None:
            a = dict(attrs)
            self.cell, self.span, self.cspan = "", int(a.get("rowspan", 1) or 1), int(a.get("colspan", 1) or 1)

    def fill(self):
        """Insert cells carried down by a rowspan from earlier rows at the current position."""
        while len(self.row) in self.carry:
            text, left = self.carry.pop(len(self.row))
            self.row.append(text)
            if left > 1: self.carry[len(self.row) - 1] = (text, left - 1)

    def handle_endtag(self, tag):
        if tag == "style": self.skip -= 1
        elif tag == "caption" and self.cur is not None:
            self.cur["caption"], self.caption = self.caption.strip(), None
        elif tag in ("td", "th") and self.cell is not None:
            text = " ".join(self.cell.split())
            if self.span > 1: self.carry[len(self.row)] = (text, self.span - 1)
            self.row.append(text); self.row += [""] * (self.cspan - 1); self.cell = None; self.fill()
        elif tag == "tr" and self.row is not None:
            if any(self.row): self.cur["rows"].append(self.row)
            self.row = None
        elif tag == "table" and self.cur is not None:
            self.tables.append(self.cur); self.cur = None

    def handle_data(self, d):
        if self.skip: return
        if self.cell is not None: self.cell += d
        elif self.caption is not None: self.caption += d


def table_text(rows):
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    w = [max(len(r[i]) for r in rows) for i in range(n)]
    return "\n".join("  ".join(c.ljust(w[i]) for i, c in enumerate(r)).rstrip() for r in rows)


def notebook_text(path):
    nb = json.load(open(path))
    out, fig = [], 0
    for c in nb["cells"]:
        if c["cell_type"] == "markdown":
            out.append("".join(c["source"]).strip())
            continue
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream":
                out.append("".join(o["text"]).rstrip())
            elif "data" in o:
                d = o["data"]
                if "image/png" in d:
                    fig += 1
                    out.append(f"[figure {fig}]")
                elif "text/html" in d:
                    p = Tables()
                    p.feed("".join(d["text/html"]))
                    for t in p.tables:
                        out.append(((t["caption"] + "\n") if t["caption"] else "") + table_text(t["rows"]))
                elif "text/markdown" in d:
                    out.append("".join(d["text/markdown"]).strip())
                elif "text/plain" in d:
                    out.append("".join(d["text/plain"]).rstrip())
            elif o.get("output_type") == "error":
                out.append(f"[error: {o.get('ename')}: {o.get('evalue')}]")
    return "\n\n".join(x for x in out if x)


def status_row(nn):
    for line in open(os.path.join(ROOT, "AGENTS.md")):
        if line.startswith(f"| {nn} |"):
            cells = [x.strip() for x in line.strip().strip("|").split("|")]
            return f"status: {cells[2]}\n\nkey results (AGENTS.md):\n{cells[3]}"
    return "status: (no row in AGENTS.md)"


def derived_tables(d):
    files = sorted(set(glob.glob(os.path.join(d, "results", "*summary*.csv"))
                       + [f for pat in EXTRA for f in glob.glob(os.path.join(d, "results", pat))]))
    out = []
    for f in files:
        try:
            t = pd.read_csv(f, dtype=str, keep_default_na=False)
        except Exception as e:  # noqa: BLE001
            out.append(f"--- {os.path.basename(f)}: unreadable ({e})")
            continue
        if "run_id" in t:
            t = t[t.run_id == t.run_id.max()]
        t = t.drop(columns=[c for c in MANDATORY if c in t])
        rows = [list(t.columns)] + t.values.tolist()
        out.append(f"--- {os.path.basename(f)} (latest run_id, {len(t)} rows)\n" + table_text([[str(x) for x in r] for r in rows]))
    return "\n\n".join(out)


def main():
    want = set(sys.argv[1:])
    for d in sorted(glob.glob(os.path.join(ROOT, "experiments", "[0-9][0-9]__*"))):
        name = os.path.basename(d)
        nn = name[:2]
        nbp = os.path.join(d, name + ".ipynb")
        if (want and nn not in want) or not os.path.exists(nbp):
            continue
        parts = [f"RESULTS SUMMARY — {name}",
                 "Generated by tools/export_results_summary.py from the executed notebook and the local derived tables. "
                 "Raw per-sample data stays on the measurement machine; this file is the text form of the published results.",
                 "=" * 100, status_row(nn), "=" * 100, "NOTEBOOK (text; figures omitted)", "=" * 100, notebook_text(nbp)]
        dt = derived_tables(d)
        if dt:
            parts += ["=" * 100, "DERIVED TABLES (latest run of each file)", "=" * 100, dt]
        outp = os.path.join(d, f"{name}__results_summary.txt")
        with open(outp, "w") as f:
            f.write("\n\n".join(parts) + "\n")
        print(f"{outp}: {os.path.getsize(outp) / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
