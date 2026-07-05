"""
make_dynamic_which_final_tables.py — regenerate the Qwen×TextVQA Dynamic-WHICH final summary
tables (CSV + Markdown) from the five accepted final aggregate JSONs.

Reads (read-only; never modifies result JSON/JSONL):
  results/final_scope/qwen25vl7b/textvqa/dynamic_which_final_textsim_p{15,25,35,50,75}_ocr_on.json
Writes (real newline characters, LF):
  results/final_scope/tables/dynamic_which_qwen_textvqa_final_summary.csv
  results/final_scope/tables/dynamic_which_qwen_textvqa_final_summary.md

Fixes the earlier artifact where the Markdown was a single line containing literal "\\n".
Stdlib only — CPU, no model, no GPU.
"""

from __future__ import annotations

import json
import os

MODEL = "qwen25vl7b"
DATASET = "textvqa"
SELECTOR = "textsim"
VARIANT = "_ocr_on"
BUDGETS = (15, 25, 35, 50, 75)

RESULT_DIR = os.path.join("results", "final_scope", MODEL, DATASET)
TABLES_DIR = os.path.join("results", "final_scope", "tables")
CSV_PATH = os.path.join(TABLES_DIR, "dynamic_which_qwen_textvqa_final_summary.csv")
MD_PATH = os.path.join(TABLES_DIR, "dynamic_which_qwen_textvqa_final_summary.md")

CSV_HEADER = [
    "model", "dataset", "method", "selector", "budget_pct",
    "dynamic_score_pct", "static_score_pct", "dense_score_pct",
    "dynamic_minus_static_pp", "dynamic_minus_dense_pp",
    "token_reduction_pct", "flop_reduction_pct", "visual_tokens_avg", "n", "gate_ok",
]


def agg_path(budget: int) -> str:
    return os.path.join(RESULT_DIR, f"dynamic_which_final_{SELECTOR}_p{budget}{VARIANT}.json")


def load_rows() -> list[dict]:
    rows = []
    for b in BUDGETS:
        p = agg_path(b)
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing final aggregate: {p}")
        r = json.load(open(p, encoding="utf-8"))
        rows.append({
            "model": r["model"],
            "dataset": r["dataset"],
            "method": r["method"],
            "selector": r["selector_name"],
            "budget_pct": r["budget_pct"],
            "dynamic_score_pct": r["score_pct"],
            "static_score_pct": r["static_reference_score_pct"],
            "dense_score_pct": r["dense_reference_score_pct"],
            "dynamic_minus_static_pp": r["dynamic_minus_static_pp"],
            "dynamic_minus_dense_pp": r["accuracy_delta_pp"],   # score - dense (signed)
            "token_reduction_pct": r["token_reduction_pct"],
            "flop_reduction_pct": r["flop_reduction_pct"],
            "visual_tokens_avg": r["visual_tokens"]["avg"],
            "n": r["n"],
            "gate_ok": r["fairness_gate"]["ok"],
        })
    return rows


def build_csv(rows: list[dict]) -> str:
    lines = [",".join(CSV_HEADER)]
    for r in rows:
        lines.append(",".join(str(r[c]) for c in CSV_HEADER))
    return "\n".join(lines) + "\n"


def build_md(rows: list[dict]) -> str:
    dense = rows[0]["dense_score_pct"]
    out = [
        "# Dynamic-WHICH final — Qwen2.5-VL-7B × TextVQA (selector=textsim, OCR-on)",
        "",
        f"Metric: M4C soft-accuracy. Dense reference = {dense}. n = {rows[0]['n']} (full val).",
        "Dynamic-WHICH = fixed-budget, question-conditioned selection (same K as static; different WHICH).",
        "",
        "| Budget | Dynamic | Static | Dense | Dyn−Static | Dyn−Dense | Token red.% | FLOP red.% | Vis tok | Gate | n |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|---:|",
    ]
    for r in rows:
        out.append(
            f"| p{r['budget_pct']} | {r['dynamic_score_pct']:.2f} | {r['static_score_pct']:.2f} | "
            f"{r['dense_score_pct']:.2f} | {r['dynamic_minus_static_pp']:+.2f} | "
            f"{r['dynamic_minus_dense_pp']:+.2f} | {r['token_reduction_pct']:.2f} | "
            f"{r['flop_reduction_pct']:.2f} | {r['visual_tokens_avg']:.1f} | "
            f"{'OK' if r['gate_ok'] else 'FAIL'} | {r['n']} |"
        )
    out += [
        "",
        "**Reading:** Dyn−Static is the headline Dynamic-WHICH gain (question-conditioned selection "
        "vs the static `norm` floor at the same budget); it is positive at every budget and largest "
        "at the tightest budgets (p15 +7.50, p25 +8.36). Dyn−Dense stays negative because dense keeps "
        "all ~964 visual tokens; the trade is accuracy for large token/FLOP reductions.",
    ]
    return "\n".join(out) + "\n"


def main() -> int:
    rows = load_rows()
    os.makedirs(TABLES_DIR, exist_ok=True)
    csv_text = build_csv(rows)
    md_text = build_md(rows)
    with open(CSV_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(csv_text)
    with open(MD_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(md_text)
    print(f"[written] {CSV_PATH}")
    print(f"[written] {MD_PATH}")
    print("\n----- Markdown preview -----")
    print(md_text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
