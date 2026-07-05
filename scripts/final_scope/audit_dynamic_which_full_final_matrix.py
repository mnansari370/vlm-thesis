"""
audit_dynamic_which_full_final_matrix.py — audit the FULL-FINAL Dynamic-WHICH textsim matrix.

Scope (the fair, complete comparison the thesis wants):
  models   = {llava15, qwen25vl7b}
  datasets = {gqa, vqav2, textvqa, docvqa}
  budgets  = {15, 25, 35, 50, 75}
  selector = textsim  (the single primary method everywhere)
  => 2 × 4 × 5 = 40 expected full-final cells.

Per cell it resolves the expected final basename (variant-aware: TextVQA _ocr_on, DocVQA
_instruction_on) and classifies:
  complete — both {json, jsonl} exist, method==dynamic_which, selector_name==textsim,
             fairness_gate.ok, n == manifest length, and jsonl line count == n.
  missing  — the json and/or jsonl does not exist.
  invalid  — files exist but a completeness check fails (gate not ok / wrong n / line-count mismatch
             / wrong selector). (These are flagged for human review; the launcher will still SKIP any
             cell whose json+jsonl both exist, never overwriting.)

Qwen×TextVQA textsim finals already exist for all five budgets and are marked complete.

Writes (read-only w.r.t. result JSON/JSONL):
  results/final_scope/tables/dynamic_which_full_final_matrix_audit.csv
  results/final_scope/tables/dynamic_which_full_final_matrix_audit.md
Stdlib + torch-free final-scope helpers only. No GPU.
"""

from __future__ import annotations

import json
import os

from src.final_scope import dynamic_which_eval as dw
from src.final_scope.sample_ids import load_manifest

MODELS = ("llava15", "qwen25vl7b")
DATASETS = ("gqa", "vqav2", "textvqa", "docvqa")
BUDGETS = (15, 25, 35, 50, 75)
SELECTOR = "textsim"

ROOT = os.path.join("results", "final_scope")
TABLES_DIR = os.path.join(ROOT, "tables")
CSV_PATH = os.path.join(TABLES_DIR, "dynamic_which_full_final_matrix_audit.csv")
MD_PATH = os.path.join(TABLES_DIR, "dynamic_which_full_final_matrix_audit.md")
MANIFEST_DIR = os.path.join("configs", "final_scope", "sample_ids")

CSV_COLS = [
    "model", "dataset", "budget_pct", "selector", "expected_basename",
    "exists_final_json", "exists_final_jsonl", "status", "n", "gate_ok", "score_pct",
    "static_reference_score_pct", "dense_reference_score_pct",
    "dynamic_minus_static_pp", "accuracy_delta_pp", "token_reduction_pct", "flop_reduction_pct",
    "path_json", "path_jsonl",
]

_MANIFEST_N: dict[str, int] = {}


def manifest_len(dataset: str) -> int:
    if dataset not in _MANIFEST_N:
        _MANIFEST_N[dataset] = len(load_manifest(os.path.join(MANIFEST_DIR, f"{dataset}.json")).ids)
    return _MANIFEST_N[dataset]


def _count_lines(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def expected_basename(dataset: str, budget: int) -> str:
    return dw.dynamic_basename(dataset, 0, SELECTOR, budget, use_ocr=True, instruction=True, full=True)


def audit_cell(model: str, dataset: str, budget: int) -> dict:
    base = expected_basename(dataset, budget)
    jpath = os.path.join(ROOT, model, dataset, f"{base}.json")
    lpath = os.path.join(ROOT, model, dataset, f"{base}.jsonl")
    row = {c: "" for c in CSV_COLS}
    row.update(model=model, dataset=dataset, budget_pct=budget, selector=SELECTOR,
               expected_basename=base, exists_final_json=os.path.exists(jpath),
               exists_final_jsonl=os.path.exists(lpath),
               path_json=os.path.relpath(jpath), path_jsonl=os.path.relpath(lpath))

    if not (row["exists_final_json"] and row["exists_final_jsonl"]):
        row["status"] = "missing"
        return row

    try:
        agg = json.load(open(jpath, encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        row["status"] = "invalid"
        return row

    exp_n = manifest_len(dataset)
    lines = _count_lines(lpath)
    gate_ok = agg.get("fairness_gate", {}).get("ok")
    row.update(
        n=agg.get("n"), gate_ok=gate_ok, score_pct=agg.get("score_pct"),
        static_reference_score_pct=agg.get("static_reference_score_pct"),
        dense_reference_score_pct=agg.get("dense_reference_score_pct"),
        dynamic_minus_static_pp=agg.get("dynamic_minus_static_pp"),
        accuracy_delta_pp=agg.get("accuracy_delta_pp"),
        token_reduction_pct=agg.get("token_reduction_pct"),
        flop_reduction_pct=agg.get("flop_reduction_pct"),
    )
    ok = (agg.get("method") == "dynamic_which" and agg.get("selector_name") == SELECTOR
          and gate_ok is True and agg.get("n") == exp_n and lines == exp_n)
    row["status"] = "complete" if ok else "invalid"
    return row


def build_rows() -> list[dict]:
    return [audit_cell(m, d, b) for m in MODELS for d in DATASETS for b in BUDGETS]


def _f(x):
    return f"{x:+.2f}" if isinstance(x, float) else ("—" if x in ("", None) else str(x))


def build_md(rows: list[dict]) -> str:
    comp = [r for r in rows if r["status"] == "complete"]
    miss = [r for r in rows if r["status"] == "missing"]
    inval = [r for r in rows if r["status"] == "invalid"]
    out = [
        "# Dynamic-WHICH full-final textsim matrix — completeness audit",
        "",
        f"Expected cells: **{len(rows)}** (2 models × 4 datasets × 5 budgets, selector=textsim). "
        f"complete=**{len(comp)}**, missing=**{len(miss)}**, invalid=**{len(inval)}**.",
        "",
        "| Model | Dataset | p | Status | json | jsonl | n | gate | Dyn | Static | Dense | "
        "Dyn−Static | Dyn−Dense | Tok red.% | FLOP red.% |",
        "|---|---|--:|---|:--:|:--:|--:|:--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        out.append(
            f"| {r['model']} | {r['dataset']} | {r['budget_pct']} | {r['status']} | "
            f"{'Y' if r['exists_final_json'] else '·'} | {'Y' if r['exists_final_jsonl'] else '·'} | "
            f"{r['n'] or '—'} | {r['gate_ok'] if r['gate_ok'] != '' else '—'} | "
            f"{_f(r['score_pct'])} | {_f(r['static_reference_score_pct'])} | "
            f"{_f(r['dense_reference_score_pct'])} | {_f(r['dynamic_minus_static_pp'])} | "
            f"{_f(r['accuracy_delta_pp'])} | {_f(r['token_reduction_pct'])} | "
            f"{_f(r['flop_reduction_pct'])} |"
        )
    out += ["", "## Missing jobs (to launch)", ""]
    if miss:
        for r in miss:
            out.append(f"- {r['model']} × {r['dataset']} p{r['budget_pct']}  →  {r['expected_basename']}")
    else:
        out.append("- none — the full-final textsim matrix is complete.")
    if inval:
        out += ["", "## Invalid cells (review — NOT auto-overwritten)", ""]
        for r in inval:
            out.append(f"- {r['model']} × {r['dataset']} p{r['budget_pct']} "
                       f"(n={r['n']}, gate={r['gate_ok']}) — {r['expected_basename']}")
    out += [
        "",
        "## Framing",
        "",
        "This is a **complete** evaluation matrix, not cherry-picking. Qwen×TextVQA textsim is the "
        "already-validated full success; the remaining full finals are added so Dynamic-WHICH is "
        "compared against dense and static on the SAME footing everywhere — including datasets where "
        "the pilot evidence is negative. Negative full results are reported, not hidden.",
    ]
    return "\n".join(out) + "\n"


def main() -> int:
    rows = build_rows()
    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(CSV_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join([",".join(CSV_COLS)]
                          + [",".join(str(r[c]) for c in CSV_COLS) for r in rows]) + "\n")
    with open(MD_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(build_md(rows))

    comp = [r for r in rows if r["status"] == "complete"]
    miss = [r for r in rows if r["status"] == "missing"]
    inval = [r for r in rows if r["status"] == "invalid"]
    print(f"[written] {CSV_PATH}")
    print(f"[written] {MD_PATH}")
    print(f"\nTOTAL_EXPECTED_ROWS={len(rows)}")
    print(f"COMPLETED_ROWS={len(comp)}")
    print(f"MISSING_ROWS={len(miss)}")
    print(f"INVALID_ROWS={len(inval)}")
    print(f"MISSING_JOB_COUNT={len(miss)}")
    print("MISSING_JOBS:")
    for r in miss:
        print(f"  - {r['model']} {r['dataset']} p{r['budget_pct']} ({r['expected_basename']})")
    if inval:
        print("INVALID_CELLS:")
        for r in inval:
            print(f"  - {r['model']} {r['dataset']} p{r['budget_pct']} (n={r['n']}, gate={r['gate_ok']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
