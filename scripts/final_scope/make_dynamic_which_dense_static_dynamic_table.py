"""
make_dynamic_which_dense_static_dynamic_table.py — thesis-ready dense/static/dynamic comparison
tables from the COMPLETE full-final Dynamic-WHICH textsim matrix (40 cells).

Reads (read-only) every `dynamic_which_final_textsim_p{b}[variant].json` aggregate — each already
carries its dense_reference_score_pct and static_reference_score_pct computed over the same
manifest — and writes:

  results/final_scope/tables/dynamic_which_dense_static_dynamic_comparison.csv
  results/final_scope/tables/dynamic_which_dense_static_dynamic_comparison.md
      one row per model × dataset × budget, with an interpretation_label:
        dynamic_win   if dynamic_minus_static_pp >  0
        near_tie      if -0.50 <= dynamic_minus_static_pp <= 0
        dynamic_loss  if dynamic_minus_static_pp < -0.50

  results/final_scope/tables/dynamic_which_compact_by_dataset.md
      per-dataset sections comparing LLaVA and Qwen across budgets.

Stdlib + torch-free helpers. No GPU, no result JSON/JSONL modified.
"""

from __future__ import annotations

import json
import os

from src.final_scope import dynamic_which_eval as dw

MODELS = ("llava15", "qwen25vl7b")
DATASETS = ("gqa", "vqav2", "textvqa", "docvqa")
BUDGETS = (15, 25, 35, 50, 75)
SELECTOR = "textsim"

ROOT = os.path.join("results", "final_scope")
TABLES_DIR = os.path.join(ROOT, "tables")
CSV_PATH = os.path.join(TABLES_DIR, "dynamic_which_dense_static_dynamic_comparison.csv")
MD_PATH = os.path.join(TABLES_DIR, "dynamic_which_dense_static_dynamic_comparison.md")
COMPACT_PATH = os.path.join(TABLES_DIR, "dynamic_which_compact_by_dataset.md")

METRIC = {"gqa": "GQA exact-match", "vqav2": "VQAv2 consensus", "textvqa": "TextVQA M4C soft-acc",
          "docvqa": "DocVQA ANLS"}

CSV_COLS = [
    "model", "dataset", "budget_pct", "dense_score_pct", "static_score_pct", "dynamic_score_pct",
    "dynamic_minus_static_pp", "dynamic_minus_dense_pp", "token_reduction_pct",
    "flop_reduction_pct", "gate_ok", "n", "interpretation_label",
]


def interpretation_label(dms: float) -> str:
    if dms > 0:
        return "dynamic_win"
    if -0.50 <= dms <= 0:
        return "near_tie"
    return "dynamic_loss"


def collect() -> list[dict]:
    rows = []
    for m in MODELS:
        for d in DATASETS:
            for b in BUDGETS:
                base = dw.dynamic_basename(d, 0, SELECTOR, b, use_ocr=True, instruction=True, full=True)
                p = os.path.join(ROOT, m, d, f"{base}.json")
                if not os.path.exists(p):
                    raise FileNotFoundError(f"full final missing (matrix should be complete): {p}")
                r = json.load(open(p, encoding="utf-8"))
                dms = float(r["dynamic_minus_static_pp"])
                rows.append({
                    "model": m, "dataset": d, "budget_pct": b,
                    "dense_score_pct": r["dense_reference_score_pct"],
                    "static_score_pct": r["static_reference_score_pct"],
                    "dynamic_score_pct": r["score_pct"],
                    "dynamic_minus_static_pp": dms,
                    "dynamic_minus_dense_pp": r["accuracy_delta_pp"],
                    "token_reduction_pct": r["token_reduction_pct"],
                    "flop_reduction_pct": r["flop_reduction_pct"],
                    "gate_ok": r["fairness_gate"]["ok"], "n": r["n"],
                    "interpretation_label": interpretation_label(dms),
                })
    return rows


def _f(x):
    return f"{x:.2f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x)


def _s(x):
    return f"{x:+.2f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x)


def build_comparison_md(rows: list[dict]) -> str:
    counts = {}
    for r in rows:
        counts[r["interpretation_label"]] = counts.get(r["interpretation_label"], 0) + 1
    out = [
        "# Dense / Static / Dynamic-WHICH (textsim) — full-final comparison (40 cells)",
        "",
        "One row per model × dataset × budget; all runs on the FULL locked manifest with gates OK. "
        "Dynamic = current Dynamic-WHICH textsim; Static = the primary static floor at the same "
        "budget (LLaVA cls_attn, Qwen norm); Dense = the keep-all reference.",
        "",
        f"Labels: dynamic_win={counts.get('dynamic_win', 0)}, near_tie={counts.get('near_tie', 0)}, "
        f"dynamic_loss={counts.get('dynamic_loss', 0)}  "
        "(win: Dyn−Static>0; near_tie: −0.50≤Dyn−Static≤0; loss: Dyn−Static<−0.50).",
        "",
        "| Model | Dataset | p | Dense | Static | Dynamic | Dyn−Static | Dyn−Dense | Tok red.% | "
        "FLOP red.% | Gate | n | Label |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|:--:|--:|---|",
    ]
    for r in rows:
        out.append(
            f"| {r['model']} | {r['dataset']} | {r['budget_pct']} | {_f(r['dense_score_pct'])} | "
            f"{_f(r['static_score_pct'])} | {_f(r['dynamic_score_pct'])} | "
            f"{_s(r['dynamic_minus_static_pp'])} | {_s(r['dynamic_minus_dense_pp'])} | "
            f"{_f(r['token_reduction_pct'])} | {_f(r['flop_reduction_pct'])} | "
            f"{'OK' if r['gate_ok'] else 'FAIL'} | {r['n']} | {r['interpretation_label']} |"
        )
    return "\n".join(out) + "\n"


def build_compact_md(rows: list[dict]) -> str:
    by = {(r["model"], r["dataset"], r["budget_pct"]): r for r in rows}
    out = [
        "# Dynamic-WHICH (textsim) — compact per-dataset comparison",
        "",
        "Full-final results; LLaVA-1.5-7B and Qwen2.5-VL-7B side by side per dataset. "
        "Dyn−Static > 0 means question-conditioned selection beats the image-only static floor "
        "at the same budget.",
    ]
    for d in DATASETS:
        out += [
            "",
            f"## {d.upper()}  ({METRIC[d]})",
            "",
            "| Model | Budget | Dense | Static | Dynamic | Dyn−Static | Dyn−Dense | FLOP red.% |",
            "|---|--:|--:|--:|--:|--:|--:|--:|",
        ]
        for m in MODELS:
            for b in BUDGETS:
                r = by[(m, d, b)]
                out.append(
                    f"| {r['model']} | p{b} | {_f(r['dense_score_pct'])} | "
                    f"{_f(r['static_score_pct'])} | {_f(r['dynamic_score_pct'])} | "
                    f"{_s(r['dynamic_minus_static_pp'])} | {_s(r['dynamic_minus_dense_pp'])} | "
                    f"{_f(r['flop_reduction_pct'])} |"
                )
    return "\n".join(out) + "\n"


def main() -> int:
    rows = collect()
    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(CSV_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join([",".join(CSV_COLS)]
                          + [",".join(str(r[c]) for c in CSV_COLS) for r in rows]) + "\n")
    with open(MD_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(build_comparison_md(rows))
    with open(COMPACT_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(build_compact_md(rows))

    counts = {}
    for r in rows:
        counts[r["interpretation_label"]] = counts.get(r["interpretation_label"], 0) + 1
    print(f"[written] {CSV_PATH}")
    print(f"[written] {MD_PATH}")
    print(f"[written] {COMPACT_PATH}")
    print(f"\nROWS={len(rows)}  dynamic_win={counts.get('dynamic_win', 0)}  "
          f"near_tie={counts.get('near_tie', 0)}  dynamic_loss={counts.get('dynamic_loss', 0)}")
    for r in rows:
        if r["interpretation_label"] != "dynamic_loss":
            print(f"  [{r['interpretation_label']:11}] {r['model']:11} {r['dataset']:7} "
                  f"p{r['budget_pct']:<2} Dyn−Static={_s(r['dynamic_minus_static_pp'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
