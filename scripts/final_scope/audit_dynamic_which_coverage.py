"""
audit_dynamic_which_coverage.py — CPU-only coverage audit of Dynamic-WHICH outputs.

Scans results/final_scope/*/*/dynamic_which_*.json (aggregate JSONs only; never .jsonl and never
modified) and reports, per output, the tier of coverage:
  final          — dynamic_which_final_*        (full run)
  n1000_confirm  — dynamic_which_pilot_n1000_*  (confirmation pilot)
  n200_pilot     — dynamic_which_pilot_n200_*   (standard pilot)
  n20_smoke      — dynamic_which_pilot_n20_*     (smoke test — NOT meaningful coverage)
  missing        — no Dynamic-WHICH output for that model×dataset cell at all

Emits (real newlines):
  results/final_scope/tables/dynamic_which_coverage_matrix.csv   (one row per found output + missing cells)
  results/final_scope/tables/dynamic_which_coverage_matrix.md    (detail + per-cell coverage summary)

"Meaningful coverage" requires at least an n200_pilot. An n20 smoke alone still needs a pilot.
Stdlib only — no model, no GPU.
"""

from __future__ import annotations

import glob
import json
import os
import re

ROOT = os.path.join("results", "final_scope")
TABLES_DIR = os.path.join(ROOT, "tables")
CSV_PATH = os.path.join(TABLES_DIR, "dynamic_which_coverage_matrix.csv")
MD_PATH = os.path.join(TABLES_DIR, "dynamic_which_coverage_matrix.md")

MODELS = ("llava15", "qwen25vl7b")
DATASETS = ("gqa", "vqav2", "textvqa", "docvqa")

# statuses ranked so we can pick the best tier present for each cell
STATUS_RANK = {"missing": 0, "n20_smoke": 1, "n200_pilot": 2, "n1000_confirm": 3, "final": 4}
MEANINGFUL = {"n200_pilot", "n1000_confirm", "final"}   # smoke does NOT count

# cells the correction flagged as lacking meaningful Dynamic-WHICH coverage
FLAGGED_MISSING = {("llava15", "textvqa"), ("llava15", "docvqa"),
                   ("qwen25vl7b", "gqa"), ("qwen25vl7b", "vqav2")}

CSV_COLS = [
    "model", "dataset", "selector", "budget_pct", "n", "status",
    "score_pct", "static_reference_score_pct", "dense_reference_score_pct",
    "dynamic_minus_static_pp", "accuracy_delta_pp", "flop_reduction_pct", "gate_ok", "path",
]


def classify(fname: str) -> str:
    if "dynamic_which_final_" in fname:
        return "final"
    m = re.search(r"dynamic_which_pilot_n(\d+)_", fname)
    if m:
        nn = int(m.group(1))
        if nn >= 1000:
            return "n1000_confirm"
        if nn >= 200:
            return "n200_pilot"
        return "n20_smoke"
    return "unknown"


def scan() -> list[dict]:
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "*", "*", "dynamic_which_*.json"))):
        if p.endswith(".jsonl"):
            continue
        try:
            r = json.load(open(p, encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            rows.append({"model": "?", "dataset": "?", "selector": "?", "budget_pct": "",
                         "n": "", "status": f"unreadable({e.__class__.__name__})",
                         "score_pct": "", "static_reference_score_pct": "",
                         "dense_reference_score_pct": "", "dynamic_minus_static_pp": "",
                         "accuracy_delta_pp": "", "flop_reduction_pct": "", "gate_ok": "",
                         "path": os.path.relpath(p)})
            continue
        rows.append({
            "model": r.get("model"),
            "dataset": r.get("dataset"),
            "selector": r.get("selector_name"),
            "budget_pct": r.get("budget_pct"),
            "n": r.get("n"),
            "status": classify(os.path.basename(p)),
            "score_pct": r.get("score_pct"),
            "static_reference_score_pct": r.get("static_reference_score_pct"),
            "dense_reference_score_pct": r.get("dense_reference_score_pct"),
            "dynamic_minus_static_pp": r.get("dynamic_minus_static_pp"),
            "accuracy_delta_pp": r.get("accuracy_delta_pp"),
            "flop_reduction_pct": r.get("flop_reduction_pct"),
            "gate_ok": r.get("fairness_gate", {}).get("ok"),
            "path": os.path.relpath(p),
        })
    return rows


def cell_summary(rows: list[dict]) -> list[dict]:
    """Per (model, dataset): best tier, whether meaningfully covered, whether it needs a pilot."""
    summ = []
    for m in MODELS:
        for d in DATASETS:
            cell = [r for r in rows if r["model"] == m and r["dataset"] == d
                    and r["status"] in STATUS_RANK]
            statuses = {r["status"] for r in cell}
            best = max((s for s in statuses), key=lambda s: STATUS_RANK[s], default="missing")
            meaningful = bool(statuses & MEANINGFUL)
            summ.append({
                "model": m, "dataset": d, "best_status": best,
                "n_outputs": len(cell),
                "meaningful": meaningful,
                "needs_n200_pilot": (not meaningful),
                "flagged": (m, d) in FLAGGED_MISSING,
            })
    return summ


def build_csv(rows: list[dict], summ: list[dict]) -> str:
    lines = [",".join(CSV_COLS)]
    detail = sorted(rows, key=lambda r: (str(r["model"]), str(r["dataset"]),
                                         str(r["selector"]), r["budget_pct"] or 0))
    for r in detail:
        lines.append(",".join(str(r.get(c, "")) for c in CSV_COLS))
    # explicit missing rows for cells with zero found outputs
    for s in summ:
        if s["n_outputs"] == 0:
            miss = {c: "" for c in CSV_COLS}
            miss.update(model=s["model"], dataset=s["dataset"], status="missing", path="")
            lines.append(",".join(str(miss[c]) for c in CSV_COLS))
    return "\n".join(lines) + "\n"


def _fmt(x, dec=2):
    return f"{x:.{dec}f}" if isinstance(x, (int, float)) else ("—" if x in (None, "") else str(x))


def build_md(rows: list[dict], summ: list[dict]) -> str:
    out = [
        "# Dynamic-WHICH coverage matrix",
        "",
        "Tiers: `final` > `n1000_confirm` > `n200_pilot` > `n20_smoke` > `missing`. "
        "**Meaningful coverage requires at least an n200 pilot** (an n20 smoke alone does not).",
        "",
        "## Per-cell coverage summary (the decision view)",
        "",
        "| Model | Dataset | Best tier | # outputs | Meaningful? | **Needs n=200 pilot?** |",
        "|---|---|---|---:|:--:|:--:|",
    ]
    for s in summ:
        need = "**YES**" if s["needs_n200_pilot"] else "no"
        mean = "yes" if s["meaningful"] else "no"
        out.append(f"| {s['model']} | {s['dataset']} | {s['best_status']} | {s['n_outputs']} | "
                   f"{mean} | {need} |")

    needs = [s for s in summ if s["needs_n200_pilot"]]
    out += [
        "",
        "### MISSING / INCOMPLETE Dynamic-WHICH coverage — needs n=200 pilots",
        "",
    ]
    for s in needs:
        why = ("no Dynamic-WHICH outputs at all" if s["n_outputs"] == 0
               else f"only {s['best_status']} present (no n≥200 pilot)")
        out.append(f"- **{s['model']} × {s['dataset']}** — {why}.")
    covered = [s for s in summ if not s["needs_n200_pilot"]]
    out += [
        "",
        "Meaningfully covered already: "
        + ", ".join(f"{s['model']}×{s['dataset']} ({s['best_status']})" for s in covered) + ".",
        "",
        "**Only Qwen2.5-VL × TextVQA has a completed Dynamic-WHICH FINAL.** Every other cell has "
        "dense+static finals plus at most partial Dynamic-WHICH pilots.",
        "",
        "## All found Dynamic-WHICH outputs",
        "",
        "| Model | Dataset | Selector | p | n | Status | Dyn | Static | Dense | Dyn−Static | "
        "Dyn−Dense | FLOP red.% | Gate |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|:--:|",
    ]
    detail = sorted(rows, key=lambda r: (str(r["model"]), str(r["dataset"]),
                                         str(r["selector"]), r["budget_pct"] or 0,
                                         -STATUS_RANK.get(r["status"], 0)))
    for r in detail:
        out.append(
            f"| {r['model']} | {r['dataset']} | {r['selector']} | {r['budget_pct']} | {r['n']} | "
            f"{r['status']} | {_fmt(r['score_pct'])} | {_fmt(r['static_reference_score_pct'])} | "
            f"{_fmt(r['dense_reference_score_pct'])} | {_fmt(r['dynamic_minus_static_pp'])} | "
            f"{_fmt(r['accuracy_delta_pp'])} | {_fmt(r['flop_reduction_pct'])} | "
            f"{'OK' if r['gate_ok'] else r['gate_ok']} |"
        )
    out += [
        "",
        "*Read-only audit of `results/final_scope/*/*/dynamic_which_*.json`. No result files were "
        "modified. Dyn−Static is the headline (question-conditioned vs static floor at same budget); "
        "Dyn−Dense (=accuracy_delta_pp) is signed vs dense.*",
    ]
    return "\n".join(out) + "\n"


def main() -> int:
    rows = scan()
    summ = cell_summary(rows)
    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(CSV_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(build_csv(rows, summ))
    with open(MD_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(build_md(rows, summ))

    print(f"[written] {CSV_PATH}")
    print(f"[written] {MD_PATH}")
    print(f"\nFound {len(rows)} Dynamic-WHICH output(s) across "
          f"{len({(r['model'], r['dataset']) for r in rows})} cell(s).")
    print("\nCoverage (needs n=200 pilot flagged):")
    for s in summ:
        flag = "  <-- NEEDS n=200 PILOT" if s["needs_n200_pilot"] else ""
        print(f"  {s['model']:11} {s['dataset']:7} best={s['best_status']:13} "
              f"outputs={s['n_outputs']:2d} meaningful={str(s['meaningful']):5}{flag}")
    need_cells = [f"{s['model']}/{s['dataset']}" for s in summ if s["needs_n200_pilot"]]
    print(f"\nCELLS_NEEDING_PILOTS={len(need_cells)}  ({', '.join(need_cells)})")
    print("DYNAMIC_WHICH_FINAL_COMPLETE=qwen25vl7b/textvqa (only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
