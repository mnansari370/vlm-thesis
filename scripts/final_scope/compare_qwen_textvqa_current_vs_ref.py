"""
compare_qwen_textvqa_current_vs_ref.py — validate the HEADLINE Qwen2.5-VL × TextVQA Dynamic-WHICH
win against the independent clean-room reference (textsim_ref), per budget, at n=200.

Read-only; writes only summary tables (no result JSON/JSONL touched). The comparison set is the
FIRST 200 manifest sample IDs (exactly what the ref n=200 pilots use). Per budget:
  * ref     = results/final_scope/qwen25vl7b/textvqa/dynamic_which_ref_pilot_n200_textsim_ref_p{b}_ocr_on
  * current = the n=200 textsim pilot if it exists (p25/p35/p50); otherwise the FIRST 200 rows of the
              full final JSONL (p15/p75).
All scores (current, ref, static, dense, dyn−static) are recomputed over the SAME 200 IDs, so the
comparison is apples-to-apples even when the current source is a 5000-sample final.

Reports per budget: current/ref paths, same IDs?, same order?, pred_raw exact match, score-diff
count, current/ref/static/dense scores, current/ref dyn−static, visual & text token averages, and a
metadata match/mismatch summary. If a ref output is not present yet, that budget is marked
REF-MISSING (run scripts/final_scope/run_qwen_textvqa_ref_validation.sh first).

Writes:
  results/final_scope/tables/qwen_textvqa_current_vs_ref_validation.csv
  results/final_scope/tables/qwen_textvqa_current_vs_ref_validation.md
Stdlib only — CPU, no GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.final_scope.sample_ids import load_manifest

MODEL, DATASET = "qwen25vl7b", "textvqa"
BUDGETS = (15, 25, 35, 50, 75)
N = 200
ROOT = os.path.join("results", "final_scope")
CELL_DIR = os.path.join(ROOT, MODEL, DATASET)
MANIFEST = os.path.join("configs", "final_scope", "sample_ids", f"{DATASET}.json")
TABLES_DIR = os.path.join(ROOT, "tables")
CSV_PATH = os.path.join(TABLES_DIR, "qwen_textvqa_current_vs_ref_validation.csv")
MD_PATH = os.path.join(TABLES_DIR, "qwen_textvqa_current_vs_ref_validation.md")

META_FIELDS = ("selector_text_source", "prompt_template", "model_instruction_suffix",
               "max_new_tokens", "static_reference_path", "dense_reference_path")

CSV_COLS = [
    "budget_pct", "current_source", "current_path", "ref_path", "ref_present",
    "same_ids", "same_order", "n_compared", "pred_raw_match", "score_diff_count",
    "current_score", "ref_score", "current_dyn_static", "ref_dyn_static",
    "static_score", "dense_score",
    "current_visual_avg", "ref_visual_avg", "current_text_avg", "ref_text_avg",
    "metadata_mismatches",
]


def _load_jsonl(path):
    by_id, order = {}, []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                sid = str(r["sample_id"])
                by_id[sid] = r
                order.append(sid)
    return by_id, order


def _agg(jsonl_path):
    j = jsonl_path[:-1]  # .jsonl -> .json
    return json.load(open(j, encoding="utf-8")) if os.path.exists(j) else {}


def _ex(agg):
    return agg.get("extra_dynamic_which") or agg.get("extra_dynamic_which_ref") or {}


def _meta(agg):
    ex = _ex(agg)
    return {
        "selector_text_source": ex.get("selector_text_source"),
        "prompt_template": agg.get("prompt_template"),
        "model_instruction_suffix": agg.get("model_instruction_suffix"),
        "max_new_tokens": agg.get("max_new_tokens"),
        "static_reference_path": agg.get("static_reference_path"),
        "dense_reference_path": agg.get("dense_reference_path"),
    }


def current_source(budget):
    """(kind, jsonl_path). n=200 pilot if present, else the full final (first 200 rows)."""
    pilot = os.path.join(CELL_DIR, f"dynamic_which_pilot_n200_textsim_p{budget}_ocr_on.jsonl")
    final = os.path.join(CELL_DIR, f"dynamic_which_final_textsim_p{budget}_ocr_on.jsonl")
    if os.path.exists(pilot):
        return "n200_pilot", pilot
    if os.path.exists(final):
        return "final_first200", final
    return "missing", ""


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def analyze(budget, ids):
    ref_path = os.path.join(CELL_DIR, f"dynamic_which_ref_pilot_n200_textsim_ref_p{budget}_ocr_on.jsonl")
    kind, cur_path = current_source(budget)
    row = {c: "" for c in CSV_COLS}
    row.update(budget_pct=budget, current_source=kind,
               current_path=os.path.basename(cur_path) if cur_path else "MISSING",
               ref_path=os.path.basename(ref_path), ref_present=os.path.exists(ref_path),
               n_compared=len(ids))
    if kind == "missing":
        row["metadata_mismatches"] = "current source missing"
        return row

    cur_by, cur_order = _load_jsonl(cur_path)
    cur_agg = _agg(cur_path)
    # static + dense per-sample over the same ids (paths declared by the current aggregate)
    static_path = cur_agg.get("static_reference_path")
    dense_path = cur_agg.get("dense_reference_path")
    static_by = _load_jsonl(static_path)[0] if static_path and os.path.exists(static_path) else {}
    dense_by = _load_jsonl(dense_path)[0] if dense_path and os.path.exists(dense_path) else {}

    def col(by, f):
        return [float(by[s][f]) for s in ids if s in by]

    cur_score = mean([float(cur_by[s]["per_sample_score"]) for s in ids if s in cur_by]) * 100
    static_score = mean(col(static_by, "per_sample_score")) * 100 if static_by else float("nan")
    dense_score = mean(col(dense_by, "per_sample_score")) * 100 if dense_by else float("nan")
    row["current_score"] = round(cur_score, 2)
    row["static_score"] = round(static_score, 2)
    row["dense_score"] = round(dense_score, 2)
    row["current_dyn_static"] = round(cur_score - static_score, 2)
    row["current_visual_avg"] = round(mean([float(cur_by[s]["n_visual_tokens"]) for s in ids if s in cur_by]), 2)
    row["current_text_avg"] = round(mean([float(cur_by[s]["n_text_tokens"]) for s in ids if s in cur_by]), 2)

    if not os.path.exists(ref_path):
        row["metadata_mismatches"] = "REF-MISSING (run run_qwen_textvqa_ref_validation.sh)"
        return row

    ref_by, ref_order = _load_jsonl(ref_path)
    ref_agg = _agg(ref_path)
    ref_ids = ref_order
    # same IDs / order (ref vs the canonical first-200 comparison set)
    row["same_ids"] = set(ref_ids) == set(ids)
    cur_restr = [s for s in cur_order if s in set(ids)][:len(ids)]
    row["same_order"] = (ref_ids == ids) and (cur_restr == ids)

    common = [s for s in ids if s in ref_by and s in cur_by]
    row["n_compared"] = len(common)
    row["pred_raw_match"] = sum(1 for s in common
                                if str(cur_by[s]["pred_raw"]) == str(ref_by[s]["pred_raw"]))
    row["score_diff_count"] = sum(1 for s in common
                                  if float(cur_by[s]["per_sample_score"]) != float(ref_by[s]["per_sample_score"]))
    ref_score = mean([float(ref_by[s]["per_sample_score"]) for s in common]) * 100
    row["ref_score"] = round(ref_score, 2)
    row["ref_dyn_static"] = round(ref_score - static_score, 2)
    row["ref_visual_avg"] = round(mean([float(ref_by[s]["n_visual_tokens"]) for s in common]), 2)
    row["ref_text_avg"] = round(mean([float(ref_by[s]["n_text_tokens"]) for s in common]), 2)

    # metadata compare (ignore intentional method/selector/n differences)
    cm, rm = _meta(cur_agg), _meta(ref_agg)
    mism = [f for f in META_FIELDS if cm.get(f) != rm.get(f)]
    row["metadata_mismatches"] = "none" if not mism else ";".join(mism)
    return row


def _fmt(x):
    return f"{x:+.2f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x)


def build_md(rows):
    out = [
        "# Qwen2.5-VL × TextVQA — current textsim vs clean-room textsim_ref (n=200 validation)",
        "",
        "Independent validation of the headline Dynamic-WHICH win. Comparison set = first 200 manifest "
        "IDs (the ref n=200 sample set). p25/p35/p50 compare against the current n=200 pilot; p15/p75 "
        "against the first 200 rows of the current full final. All scores recomputed over the same 200.",
        "",
        "| p | current source | same IDs | same order | pred match | score diff | cur score | ref score | "
        "cur dyn−static | ref dyn−static | static | dense | cur vis | ref vis | meta |",
        "|--:|---|:--:|:--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|",
    ]
    for r in rows:
        if not r["ref_present"]:
            out.append(f"| p{r['budget_pct']} | {r['current_source']} | — | — | — | — | "
                       f"{r['current_score']} | REF-MISSING | {r['current_dyn_static']} | — | "
                       f"{r['static_score']} | {r['dense_score']} | {r['current_visual_avg']} | — | "
                       f"{r['metadata_mismatches']} |")
            continue
        out.append(
            f"| p{r['budget_pct']} | {r['current_source']} | {r['same_ids']} | {r['same_order']} | "
            f"{r['pred_raw_match']}/{r['n_compared']} | {r['score_diff_count']} | {r['current_score']} | "
            f"{r['ref_score']} | {_fmt(r['current_dyn_static'])} | {_fmt(r['ref_dyn_static'])} | "
            f"{r['static_score']} | {r['dense_score']} | {r['current_visual_avg']} | {r['ref_visual_avg']} | "
            f"{r['metadata_mismatches']} |")
    any_ref = any(r["ref_present"] for r in rows)
    out += ["", "## Interpretation", ""]
    if not any_ref:
        out += ["> Ref outputs not present yet. Run "
                "`scripts/final_scope/run_qwen_textvqa_ref_validation.sh` (GPU 1, qwen_env), then re-run "
                "this script. The current baseline is shown above; ref columns fill in after the run."]
    else:
        allmatch = all((not r["ref_present"]) or
                       (r["pred_raw_match"] == r["n_compared"] and r["score_diff_count"] == 0)
                       for r in rows)
        out += [
            "The headline claim is validated when, for every budget, the clean-room `textsim_ref` "
            "reproduces the current `textsim` predictions (pred match = n, 0 score diffs) and the "
            "ref dyn−static ≈ current dyn−static. "
            + ("**All compared budgets match exactly → the Qwen×TextVQA win is implementation-robust.**"
               if allmatch else
               "Some budgets differ — inspect with debug_textsim_current_vs_ref.py before claiming robustness."),
        ]
    return "\n".join(out) + "\n"


def main() -> int:
    argparse.ArgumentParser(
        description="Validate Qwen×TextVQA Dynamic-WHICH: current textsim vs clean-room textsim_ref "
                    "(n=200; writes only summary tables). No arguments — compares all budgets.").parse_args()
    manifest = load_manifest(MANIFEST)
    ids = [str(x) for x in manifest.ids[:N]]
    rows = [analyze(b, ids) for b in BUDGETS]
    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(CSV_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join([",".join(CSV_COLS)]
                          + [",".join(str(r[c]) for c in CSV_COLS) for r in rows]) + "\n")
    with open(MD_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(build_md(rows))
    print(f"[written] {CSV_PATH}")
    print(f"[written] {MD_PATH}")
    for r in rows:
        if r["ref_present"]:
            print(f"  p{r['budget_pct']:<2} [{r['current_source']:14}] pred_match={r['pred_raw_match']}/{r['n_compared']} "
                  f"score_diff={r['score_diff_count']} cur_dynstatic={_fmt(r['current_dyn_static'])} "
                  f"ref_dynstatic={_fmt(r['ref_dyn_static'])} meta={r['metadata_mismatches']}")
        else:
            print(f"  p{r['budget_pct']:<2} [{r['current_source']:14}] REF-MISSING "
                  f"(cur dyn−static={_fmt(r['current_dyn_static'])}; run the launcher)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
