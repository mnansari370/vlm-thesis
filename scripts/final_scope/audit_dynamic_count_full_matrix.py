"""
audit_dynamic_count_full_matrix.py — completeness audit of the Dynamic-COUNT matrix (CPU).

Per cell (2 models × 4 datasets): probe p15/p25 present + reproduction-gate OK, calibration config
present, DC-D composed, DC-C (rule) present. Writes
results/final_scope/tables/dynamic_count_full_matrix_audit.{csv,md} and prints counts.
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.final_scope import dense_pilot as dp
from src.pruning import dynamic_count as dc

TABLES = os.path.join("results", "final_scope", "tables")
CFG_DIR = os.path.join("results", "final_scope", "dynamic_count_configs")
CSV = os.path.join(TABLES, "dynamic_count_full_matrix_audit.csv")
MD = os.path.join(TABLES, "dynamic_count_full_matrix_audit.md")
COLS = ["model", "dataset", "probe_p15", "probe_p15_equiv", "probe_p25", "probe_p25_equiv",
        "textsim_probes", "config", "dcd", "dcc_rule", "dcc_ridge", "cow_dcd", "cow_dcc", "status"]


def _agg(path):
    try:
        return json.load(open(path))
    except (OSError, json.JSONDecodeError):
        return None


def audit_cell(m, d):
    cell_dir = os.path.join(dp.OUT_ROOT, m, d)
    sel = "cls_attn" if m == "llava15" else "norm"
    row = {c: "" for c in COLS}
    row.update(model=m, dataset=d)
    for pct, jkey, ekey in ((15, "probe_p15", "probe_p15_equiv"), (25, "probe_p25", "probe_p25_equiv")):
        p = os.path.join(cell_dir, dc.probe_basename(d, sel, pct) + ".json")
        a = _agg(p) if os.path.exists(p) else None
        row[jkey] = a is not None
        row[ekey] = bool(a and a.get("equivalence_ok"))
    row["textsim_probes"] = len(glob.glob(os.path.join(cell_dir, "dynamic_count_probe_textsim_*.json")))
    row["config"] = os.path.exists(os.path.join(CFG_DIR, f"{m}_{d}.json"))
    # PURE Dynamic-COUNT (locked static selector) drives the status; COUNT-on-WHICH counted apart
    row["dcd"] = len(glob.glob(os.path.join(cell_dir, f"dynamic_count_dcd_{sel}_*.json"))) > 0
    row["dcc_rule"] = len(glob.glob(os.path.join(cell_dir, f"dynamic_count_dcc_rule_{sel}_*.json"))) > 0
    row["dcc_ridge"] = len(glob.glob(os.path.join(cell_dir, f"dynamic_count_dcc_ridge_{sel}_*.json"))) > 0
    row["cow_dcd"] = len(glob.glob(os.path.join(cell_dir, "dynamic_count_dcd_textsim_*.json")))
    row["cow_dcc"] = len(glob.glob(os.path.join(cell_dir, "dynamic_count_dcc_*_textsim_*.json")))
    if row["dcc_rule"]:
        row["status"] = "dcc_complete"
    elif row["dcd"]:
        row["status"] = "dcd_only"
    elif row["probe_p15"] or row["probe_p25"]:
        row["status"] = "probe_only"
    else:
        row["status"] = "missing"
    return row


def main() -> int:
    rows = [audit_cell(m, d) for m in dp.MODELS for d in dp.DATASETS]
    os.makedirs(TABLES, exist_ok=True)
    with open(CSV, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join([",".join(COLS)] + [",".join(str(r[c]) for c in COLS) for r in rows]) + "\n")
    md = ["# Dynamic-COUNT full-matrix audit", "",
          "Status reflects the PURE Dynamic-COUNT path (locked static selector); COUNT-on-WHICH "
          "(textsim, qwen×textvqa) is tracked in its own columns.", "",
          "| Model | Dataset | p15 probe | equiv | p25 probe | equiv | textsim | cfg | DC-D | "
          "DC-C rule | DC-C ridge | CoW DC-D | CoW DC-C | Status |",
          "|---|---|:--:|:--:|:--:|:--:|--:|:--:|:--:|:--:|:--:|--:|--:|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['dataset']} | {'Y' if r['probe_p15'] else '·'} | "
                  f"{'Y' if r['probe_p15_equiv'] else '·'} | {'Y' if r['probe_p25'] else '·'} | "
                  f"{'Y' if r['probe_p25_equiv'] else '·'} | {r['textsim_probes']} | "
                  f"{'Y' if r['config'] else '·'} | {'Y' if r['dcd'] else '·'} | "
                  f"{'Y' if r['dcc_rule'] else '·'} | {'Y' if r['dcc_ridge'] else '·'} | "
                  f"{r['cow_dcd']} | {r['cow_dcc']} | {r['status']} |")
    with open(MD, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md) + "\n")
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    equiv_fail = [f"{r['model']}/{r['dataset']}" for r in rows
                  if (r["probe_p15"] and not r["probe_p15_equiv"]) or
                     (r["probe_p25"] and not r["probe_p25_equiv"])]
    print(f"[written] {CSV}\n[written] {MD}\n")
    print(f"TOTAL_CELLS={len(rows)}")
    for k in ("dcc_complete", "dcd_only", "probe_only", "missing"):
        print(f"{k.upper()}={counts.get(k, 0)}")
    print(f"EQUIVALENCE_FAILURES={len(equiv_fail)}" + (f"  ({', '.join(equiv_fail)})" if equiv_fail else ""))
    return 0 if not equiv_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
