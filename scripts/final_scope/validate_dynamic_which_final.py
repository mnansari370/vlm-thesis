"""
validate_dynamic_which_final.py — validate the Qwen2.5-VL-7B × TextVQA Dynamic-WHICH FINAL run.

Read-only. Checks, for budgets {15,25,35,50,75}:
  - the aggregate JSON exists, and its matching per-sample JSONL exists;
  - n == 5000 and the JSONL has exactly 5000 lines;
  - fairness_gate.ok == True;
  - method == dynamic_which, selector_name == textsim, dataset == textvqa, model == qwen25vl7b.
Prints the final table and a single machine-readable line:  ALL_FINAL_VALID=True|False
Stdlib only — CPU, no GPU.
"""

from __future__ import annotations

import json
import os

MODEL = "qwen25vl7b"
DATASET = "textvqa"
SELECTOR = "textsim"
METHOD = "dynamic_which"
VARIANT = "_ocr_on"
BUDGETS = (15, 25, 35, 50, 75)
EXPECTED_N = 5000

RESULT_DIR = os.path.join("results", "final_scope", MODEL, DATASET)


def basename(budget: int) -> str:
    return f"dynamic_which_final_{SELECTOR}_p{budget}{VARIANT}"


def count_lines(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def check_budget(budget: int) -> tuple[list[str], dict]:
    """Return (issues, row). row carries display fields even when partially missing."""
    issues: list[str] = []
    base = basename(budget)
    jpath = os.path.join(RESULT_DIR, f"{base}.json")
    lpath = os.path.join(RESULT_DIR, f"{base}.jsonl")
    row = {"budget": budget, "dynamic": None, "static": None, "dense": None,
           "dyn_static": None, "n": None, "lines": None, "gate": None}

    if not os.path.exists(jpath):
        issues.append(f"p{budget}: aggregate JSON missing ({jpath})")
        return issues, row
    r = json.load(open(jpath, encoding="utf-8"))

    if not os.path.exists(lpath):
        issues.append(f"p{budget}: per-sample JSONL missing ({lpath})")
    else:
        row["lines"] = count_lines(lpath)
        if row["lines"] != EXPECTED_N:
            issues.append(f"p{budget}: JSONL line count {row['lines']} != {EXPECTED_N}")

    row["dynamic"] = r.get("score_pct")
    row["static"] = r.get("static_reference_score_pct")
    row["dense"] = r.get("dense_reference_score_pct")
    row["dyn_static"] = r.get("dynamic_minus_static_pp")
    row["n"] = r.get("n")
    row["gate"] = r.get("fairness_gate", {}).get("ok")

    if r.get("model") != MODEL:
        issues.append(f"p{budget}: model={r.get('model')} != {MODEL}")
    if r.get("dataset") != DATASET:
        issues.append(f"p{budget}: dataset={r.get('dataset')} != {DATASET}")
    if r.get("method") != METHOD:
        issues.append(f"p{budget}: method={r.get('method')} != {METHOD}")
    if r.get("selector_name") != SELECTOR:
        issues.append(f"p{budget}: selector_name={r.get('selector_name')} != {SELECTOR}")
    if r.get("budget_pct") != budget:
        issues.append(f"p{budget}: budget_pct={r.get('budget_pct')} != {budget}")
    if r.get("n") != EXPECTED_N:
        issues.append(f"p{budget}: n={r.get('n')} != {EXPECTED_N}")
    if row["gate"] is not True:
        issues.append(f"p{budget}: fairness_gate.ok={row['gate']} (expected True)")
    return issues, row


def main() -> int:
    all_issues: list[str] = []
    rows = []
    for b in BUDGETS:
        issues, row = check_budget(b)
        all_issues += issues
        rows.append(row)

    budgets_found = sorted(r["budget"] for r in rows if r["dynamic"] is not None)
    if budgets_found != list(BUDGETS):
        all_issues.append(f"budgets found {budgets_found} != {list(BUDGETS)}")

    print(f"Dynamic-WHICH FINAL — {MODEL} × {DATASET} (selector={SELECTOR})")
    print(f"{'Budget':>6} | {'Dynamic':>7} | {'Static':>6} | {'Dense':>6} | "
          f"{'Dyn-Static':>10} | {'n':>5} | {'lines':>5} | Gate")
    print("-" * 72)
    for r in rows:
        def f(x, w, dec=2):
            return (f"{x:>{w}.{dec}f}" if isinstance(x, (int, float)) else f"{str(x):>{w}}")
        print(f"{('p'+str(r['budget'])):>6} | {f(r['dynamic'],7)} | {f(r['static'],6)} | "
              f"{f(r['dense'],6)} | {f(r['dyn_static'],10)} | {str(r['n']):>5} | "
              f"{str(r['lines']):>5} | {r['gate']}")

    n_valid = len([r for r in rows if r["dynamic"] is not None])
    print(f"\nN_QWEN_TEXTVQA_DYNAMIC_FINAL={n_valid}")
    print(f"BUDGETS_FOUND={budgets_found}")
    ok = (not all_issues)
    if all_issues:
        print("\nISSUES:")
        for p in all_issues:
            print(f"  - {p}")
    print(f"\nALL_FINAL_VALID={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
