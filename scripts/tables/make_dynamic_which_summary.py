"""
make_dynamic_which_textsim_full_final_summary.py — summarize all COMPLETED full-final Dynamic-WHICH
textsim runs across the final-scope matrix (2 models × 4 datasets × 5 budgets).

Reads (read-only) every `dynamic_which_final_textsim_p{b}[variant].json` aggregate that exists and
passed its fairness gate, and writes:
  results/tables/dynamic_which_textsim_full_final_summary.csv
  results/tables/dynamic_which_textsim_full_final_summary.md

Only complete, gate-OK textsim finals are included. Missing cells are noted (run the launcher).
Stdlib + torch-free helpers. No GPU, no result JSON/JSONL modified.
"""

from __future__ import annotations

import json
import os

from src.dynamic_which import evaluate_dynamic_which as dw

MODELS = ("llava15", "qwen25vl7b")
DATASETS = ("gqa", "vqav2", "textvqa", "docvqa")
BUDGETS = (15, 25, 35, 50, 75)
SELECTOR = "textsim"

ROOT = os.path.join("results", "runs")
TABLES_DIR = os.path.join("results", "tables")
CSV_PATH = os.path.join(TABLES_DIR, "dynamic_which_textsim_full_final_summary.csv")
MD_PATH = os.path.join(TABLES_DIR, "dynamic_which_textsim_full_final_summary.md")

CSV_COLS = [
    "model", "dataset", "budget_pct", "dynamic_score_pct", "static_score_pct", "dense_score_pct",
    "dynamic_minus_static_pp", "dynamic_minus_dense_pp", "token_reduction_pct", "flop_reduction_pct",
    "gate_ok", "n", "path",
]


def final_json(model, dataset, budget):
    base = dw.dynamic_basename(dataset, 0, SELECTOR, budget, use_ocr=True, instruction=True, full=True)
    return os.path.join(ROOT, model, dataset, f"{base}.json")


def collect():
    rows, missing = [], []
    for m in MODELS:
        for d in DATASETS:
            for b in BUDGETS:
                p = final_json(m, d, b)
                if not os.path.exists(p):
                    missing.append((m, d, b))
                    continue
                r = json.load(open(p, encoding="utf-8"))
                gate = r.get("fairness_gate", {}).get("ok")
                if not (r.get("selector_name") == SELECTOR and gate is True):
                    missing.append((m, d, b))     # exists but not a valid complete textsim final
                    continue
                rows.append({
                    "model": m, "dataset": d, "budget_pct": b,
                    "dynamic_score_pct": r.get("score_pct"),
                    "static_score_pct": r.get("static_reference_score_pct"),
                    "dense_score_pct": r.get("dense_reference_score_pct"),
                    "dynamic_minus_static_pp": r.get("dynamic_minus_static_pp"),
                    "dynamic_minus_dense_pp": r.get("accuracy_delta_pp"),   # score − dense (signed)
                    "token_reduction_pct": r.get("token_reduction_pct"),
                    "flop_reduction_pct": r.get("flop_reduction_pct"),
                    "gate_ok": gate, "n": r.get("n"), "path": os.path.relpath(p),
                })
    return rows, missing


def _f(x):
    return f"{x:.2f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else ("—" if x is None else str(x))


def _s(x):
    return f"{x:+.2f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else ("—" if x is None else str(x))


def build_md(rows, missing):
    out = [
        "# Dynamic-WHICH textsim — full-final summary (complete matrix)",
        "",
        f"Completed full textsim finals: **{len(rows)}** / {len(MODELS)*len(DATASETS)*len(BUDGETS)} "
        f"expected. Missing: **{len(missing)}**.",
        "",
        "| Model | Dataset | p | Dynamic | Static | Dense | Dyn−Static | Dyn−Dense | Tok red.% | "
        "FLOP red.% | Gate | n |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|:--:|--:|",
    ]
    for r in sorted(rows, key=lambda r: (r["model"], r["dataset"], r["budget_pct"])):
        out.append(
            f"| {r['model']} | {r['dataset']} | {r['budget_pct']} | {_f(r['dynamic_score_pct'])} | "
            f"{_f(r['static_score_pct'])} | {_f(r['dense_score_pct'])} | "
            f"{_s(r['dynamic_minus_static_pp'])} | {_s(r['dynamic_minus_dense_pp'])} | "
            f"{_f(r['token_reduction_pct'])} | {_f(r['flop_reduction_pct'])} | "
            f"{'OK' if r['gate_ok'] else 'FAIL'} | {r['n']} |"
        )
    if missing:
        out += ["", "## Not yet complete", ""]
        for m, d, b in missing:
            out.append(f"- {m} × {d} p{b}")

    # win/loss framing over completed non-TextVQA-Qwen cells
    beats = [r for r in rows if isinstance(r["dynamic_minus_static_pp"], (int, float))
             and r["dynamic_minus_static_pp"] > 0]
    out += [
        "",
        "## Interpretation",
        "",
        "- **Qwen2.5-VL × TextVQA is the only already-validated full success** (Dyn−Static positive at "
        "every budget; independently reproduced by the clean-room `textsim_ref` at n=200).",
        "- The other full runs are added for a **fair, complete comparison** against dense and static "
        "on the same footing — including datasets where the n=200 pilot evidence was negative. This is "
        "a complete evaluation matrix, **not cherry-picking**; negative full results are reported as-is.",
        "- Headline reading: **Dynamic-WHICH textsim is strong for Qwen×TextVQA but is NOT universal** "
        "— it under-performs or loses to the static floor on most other datasets. The single primary "
        "method (textsim) is held fixed everywhere so the comparison is clean.",
        f"- Cells (of the completed {len(rows)}) where textsim beats static: "
        f"**{len(beats)}** ({', '.join(sorted({r['model']+'/'+r['dataset'] for r in beats})) or 'none'}).",
    ]
    return "\n".join(out) + "\n"


def main() -> int:
    rows, missing = collect()
    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(CSV_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join([",".join(CSV_COLS)]
                          + [",".join(str(r[c]) for c in CSV_COLS) for r in
                             sorted(rows, key=lambda r: (r["model"], r["dataset"], r["budget_pct"]))]) + "\n")
    with open(MD_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(build_md(rows, missing))
    print(f"[written] {CSV_PATH}")
    print(f"[written] {MD_PATH}")
    print(f"\nCOMPLETED_FULL_TEXTSIM_FINALS={len(rows)}  MISSING={len(missing)}")
    for r in sorted(rows, key=lambda r: (r["model"], r["dataset"], r["budget_pct"])):
        print(f"  {r['model']:11} {r['dataset']:7} p{r['budget_pct']:<2} "
              f"dyn={_f(r['dynamic_score_pct']):>6} static={_f(r['static_score_pct']):>6} "
              f"dyn−static={_s(r['dynamic_minus_static_pp']):>7} n={r['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
