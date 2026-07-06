"""
analyze_dynamic_count_oracle.py — CPU-only ORACLE upper bound for Dynamic-COUNT (adaptive budget).

>>> THIS IS AN ORACLE / UPPER BOUND, NOT A DEPLOYABLE METHOD. <<<
It peeks at the per-sample CORRECTNESS (per_sample_score, computed against gold) to choose a budget,
so it cannot be run at inference time. It answers ONE question for the thesis: if a *perfect*
per-sample router could pick a budget from {15,25,35,50,75}% while reusing the EXISTING static
selectors (LLaVA→cls_attn, Qwen→norm), how much accuracy/compute could adaptive budgeting buy over
a single fixed budget or dense? No training. No new labels are produced. No GPU. No full runs.

Inputs (read-only): the accepted dense + 5 static FINAL per-sample JSONLs for each model×dataset:
  results/runs/{model}/{dataset}/{dense_final|static_final_{sel}_p{b}}[variant].jsonl
Uses only fields sample_id, per_sample_score, n_visual_tokens.

Two oracle policies per cell:
  * BEST-SCORE oracle : per sample take max static score over the 5 budgets, charging the SMALLEST
                        budget that attains that max. Upper bound on adaptive-budget accuracy.
  * MATCH-DENSE oracle: per sample take the SMALLEST budget whose static score ≥ the dense score
                        (else fall back to dense). A "no worse than dense per sample" router's cost.

Outputs (real newlines):
  results/tables/dynamic_count_oracle_summary.csv
  results/tables/dynamic_count_oracle_summary.md
Stdlib only.
"""

from __future__ import annotations

import json
import os

BUDGETS = (15, 25, 35, 50, 75)
MODELS = ("llava15", "qwen25vl7b")
DATASETS = ("gqa", "vqav2", "textvqa", "docvqa")
STATIC_SEL = {"llava15": "cls_attn", "qwen25vl7b": "norm"}
VARIANT = {"gqa": "", "vqav2": "", "textvqa": "_ocr_on", "docvqa": "_instruction_on"}
DENSE_NAME = {"gqa": "dense_final", "vqav2": "dense_final",
              "textvqa": "dense_final_ocr_on", "docvqa": "dense_final_instruction_on"}
METRIC = {"gqa": "gqa_exact", "vqav2": "vqa_consensus", "textvqa": "m4c_soft", "docvqa": "anls"}

TABLES_DIR = os.path.join("results", "tables")
CSV_PATH = os.path.join(TABLES_DIR, "dynamic_count_oracle_summary.csv")
MD_PATH = os.path.join(TABLES_DIR, "dynamic_count_oracle_summary.md")
EPS = 1e-9


def _root(model: str, dataset: str) -> str:
    return os.path.join("results", "runs", model, dataset)


def _load(path: str) -> dict:
    """sample_id -> (per_sample_score, n_visual_tokens)."""
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                d[str(r["sample_id"])] = (float(r["per_sample_score"]), int(r["n_visual_tokens"]))
    return d


def cell_paths(model: str, dataset: str):
    sel, v = STATIC_SEL[model], VARIANT[dataset]
    dense = os.path.join(_root(model, dataset), f"{DENSE_NAME[dataset]}.jsonl")
    static = {b: os.path.join(_root(model, dataset), f"static_final_{sel}_p{b}{v}.jsonl")
              for b in BUDGETS}
    return dense, static


def analyze_cell(model: str, dataset: str):
    dense_path, static_paths = cell_paths(model, dataset)
    if not os.path.exists(dense_path) or not all(os.path.exists(p) for p in static_paths.values()):
        return None
    dense = _load(dense_path)
    stat = {b: _load(static_paths[b]) for b in BUDGETS}
    ids = [sid for sid in dense if all(sid in stat[b] for b in BUDGETS)]
    n = len(ids)
    if n == 0:
        return None

    dense_score_sum = dense_vis_sum = 0.0
    static_score_sum = {b: 0.0 for b in BUDGETS}
    oracle_score_sum = oracle_vis_sum = 0.0
    solved = {b: 0 for b in BUDGETS}
    unsolved = 0
    md_score_sum = md_vis_sum = 0.0
    md_bucket = {b: 0 for b in BUDGETS}
    needs_dense = 0

    for sid in ids:
        ds, dv = dense[sid]
        dense_score_sum += ds
        dense_vis_sum += dv
        scores = [stat[b][sid][0] for b in BUDGETS]
        vis = [stat[b][sid][1] for b in BUDGETS]
        for j, b in enumerate(BUDGETS):
            static_score_sum[b] += scores[j]

        # BEST-SCORE oracle: max score, smallest budget attaining it
        mx = max(scores)
        j_best = next(j for j, s in enumerate(scores) if s >= mx - EPS)
        oracle_score_sum += mx
        oracle_vis_sum += vis[j_best]
        if mx > EPS:
            solved[BUDGETS[j_best]] += 1
        else:
            unsolved += 1

        # MATCH-DENSE oracle: smallest budget with score >= dense score, else dense
        j_md = next((j for j, s in enumerate(scores) if s >= ds - EPS), None)
        if j_md is None:
            md_score_sum += ds
            md_vis_sum += dv
            needs_dense += 1
        else:
            md_score_sum += scores[j_md]
            md_vis_sum += vis[j_md]
            md_bucket[BUDGETS[j_md]] += 1

    dense_acc = dense_score_sum / n * 100
    static_acc = {b: static_score_sum[b] / n * 100 for b in BUDGETS}
    best_fixed_budget = max(BUDGETS, key=lambda b: static_acc[b])
    best_fixed_acc = static_acc[best_fixed_budget]
    oracle_acc = oracle_score_sum / n * 100
    dense_mean_vis = dense_vis_sum / n
    oracle_mean_vis = oracle_vis_sum / n
    md_acc = md_score_sum / n * 100
    md_mean_vis = md_vis_sum / n

    def red(mean_vis):
        return 100.0 * (1.0 - mean_vis / dense_mean_vis) if dense_mean_vis > 0 else 0.0

    return {
        "model": model, "dataset": dataset, "metric": METRIC[dataset], "n": n,
        "dense_acc": round(dense_acc, 2),
        "static_acc": {b: round(static_acc[b], 2) for b in BUDGETS},
        "best_fixed_budget": best_fixed_budget, "best_fixed_acc": round(best_fixed_acc, 2),
        "oracle_acc": round(oracle_acc, 2),
        "oracle_vs_dense_pp": round(oracle_acc - dense_acc, 2),
        "oracle_vs_bestfixed_pp": round(oracle_acc - best_fixed_acc, 2),
        "oracle_mean_visual": round(oracle_mean_vis, 1),
        "dense_mean_visual": round(dense_mean_vis, 1),
        "oracle_visual_reduction_pct": round(red(oracle_mean_vis), 2),
        "solved_pct": {b: round(solved[b] / n * 100, 1) for b in BUDGETS},
        "unsolved_pct": round(unsolved / n * 100, 1),
        "matchdense_acc": round(md_acc, 2),
        "matchdense_mean_visual": round(md_mean_vis, 1),
        "matchdense_visual_reduction_pct": round(red(md_mean_vis), 2),
        "matchdense_needs_dense_pct": round(needs_dense / n * 100, 1),
        "matchdense_bucket_pct": {b: round(md_bucket[b] / n * 100, 1) for b in BUDGETS},
    }


CSV_COLS = [
    "model", "dataset", "metric", "n", "dense_acc",
    "static_p15", "static_p25", "static_p35", "static_p50", "static_p75",
    "best_fixed_budget", "best_fixed_acc",
    "oracle_acc", "oracle_vs_dense_pp", "oracle_vs_bestfixed_pp",
    "oracle_mean_visual", "dense_mean_visual", "oracle_visual_reduction_pct",
    "matchdense_acc", "matchdense_mean_visual", "matchdense_visual_reduction_pct",
    "matchdense_needs_dense_pct",
]


def csv_row(c: dict) -> str:
    vals = [c["model"], c["dataset"], c["metric"], c["n"], c["dense_acc"]]
    vals += [c["static_acc"][b] for b in BUDGETS]
    vals += [c["best_fixed_budget"], c["best_fixed_acc"],
             c["oracle_acc"], c["oracle_vs_dense_pp"], c["oracle_vs_bestfixed_pp"],
             c["oracle_mean_visual"], c["dense_mean_visual"], c["oracle_visual_reduction_pct"],
             c["matchdense_acc"], c["matchdense_mean_visual"],
             c["matchdense_visual_reduction_pct"], c["matchdense_needs_dense_pct"]]
    return ",".join(str(v) for v in vals)


def build_md(cells: list[dict]) -> str:
    out = [
        "# Dynamic-COUNT ORACLE — adaptive-budget upper bound (NOT a deployable method)",
        "",
        "> **ORACLE / UPPER BOUND.** For each sample this peeks at the per-sample correctness "
        "(`per_sample_score` vs gold) to pick a budget from {15,25,35,50,75}% while reusing the "
        "EXISTING static selectors (LLaVA→`cls_attn`, Qwen→`norm`). It is **not runnable at "
        "inference** and is **not** a claimed controller — it bounds what a perfect per-sample "
        "budget router could achieve. No training, no new labels, no GPU.",
        "",
        "Two policies:",
        "- **Best-score oracle** — per sample, max static score over the 5 budgets, charging the "
        "*smallest* budget that attains it. Upper bound on adaptive-budget accuracy + its token cost.",
        "- **Match-dense oracle** — per sample, the *smallest* budget whose static score ≥ dense "
        "(else fall back to dense). A 'no worse than dense per sample' router and its token cost.",
        "",
        "Reductions are **visual-token** reductions vs dense (mean kept visual tokens / dense mean).",
        "",
        "## Best-score oracle",
        "",
        "| Model | Dataset | Metric | Dense | Best fixed (budget) | **Oracle** | Oracle−Dense | "
        "Oracle−BestFixed | Oracle vis tok | Dense vis tok | Vis red.% |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in cells:
        out.append(
            f"| {c['model']} | {c['dataset']} | {c['metric']} | {c['dense_acc']:.2f} | "
            f"{c['best_fixed_acc']:.2f} (p{c['best_fixed_budget']}) | **{c['oracle_acc']:.2f}** | "
            f"{c['oracle_vs_dense_pp']:+.2f} | {c['oracle_vs_bestfixed_pp']:+.2f} | "
            f"{c['oracle_mean_visual']:.0f} | {c['dense_mean_visual']:.0f} | "
            f"{c['oracle_visual_reduction_pct']:.1f} |"
        )
    out += [
        "",
        "### Per-budget static accuracy + where the oracle 'solves' each sample first",
        "",
        "| Model | Dataset | s@p15 | s@p25 | s@p35 | s@p50 | s@p75 | first@p15 | first@p25 | "
        "first@p35 | first@p50 | first@p75 | unsolved |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in cells:
        sa = c["static_acc"]
        sp = c["solved_pct"]
        out.append(
            f"| {c['model']} | {c['dataset']} | {sa[15]:.1f} | {sa[25]:.1f} | {sa[35]:.1f} | "
            f"{sa[50]:.1f} | {sa[75]:.1f} | {sp[15]:.1f} | {sp[25]:.1f} | {sp[35]:.1f} | "
            f"{sp[50]:.1f} | {sp[75]:.1f} | {c['unsolved_pct']:.1f} |"
        )
    out += [
        "",
        "*first@pX = % of samples whose cheapest max-scoring static budget is pX; unsolved = % where "
        "no budget scored > 0. Reading the distribution: mass at p15 means many samples are already "
        "solved at the tightest budget (adaptive routing can be cheap); mass at p75/unsolved means "
        "the task needs many tokens regardless.*",
        "",
        "## Match-dense oracle (cheapest per-sample budget that is ≥ dense)",
        "",
        "| Model | Dataset | Acc (≥dense) | Mean vis tok | Vis red.% vs dense | needs-dense% |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for c in cells:
        out.append(
            f"| {c['model']} | {c['dataset']} | {c['matchdense_acc']:.2f} | "
            f"{c['matchdense_mean_visual']:.0f} | {c['matchdense_visual_reduction_pct']:.1f} | "
            f"{c['matchdense_needs_dense_pct']:.1f} |"
        )
    out += [
        "",
        "*needs-dense% = share of samples where NO static budget matches the dense score (must keep "
        "dense). A low needs-dense% with a high vis-reduction% means most samples are dense-lossless "
        "at a small budget — the strongest motivation for Dynamic-COUNT.*",
        "",
        "## How to read this for the thesis",
        "",
        "1. **Oracle−BestFixed** is the accuracy headroom a perfect budget router could add over the "
        "single best static budget. If it is large, adaptive *count* is worth pursuing; if small, "
        "one fixed budget is already near-optimal and the interesting axis stays *which* tokens.",
        "2. **Match-dense vis-reduction% at low needs-dense%** quantifies free compute: how many "
        "visual tokens could be dropped, per sample, with no accuracy loss vs dense — the clean, "
        "honest motivation for Dynamic-COUNT.",
        "3. These are **ceilings**. A real controller must predict the budget from the input alone "
        "(image/question), with NO access to correctness — so realized gains will be strictly lower. "
        "This script exists to decide whether that gap is worth closing, not to claim it.",
    ]
    return "\n".join(out) + "\n"


def main() -> int:
    cells = []
    for m in MODELS:
        for d in DATASETS:
            c = analyze_cell(m, d)
            if c is not None:
                cells.append(c)
            else:
                print(f"[skip] {m} × {d}: dense + 5 static finals not all present")

    os.makedirs(TABLES_DIR, exist_ok=True)
    csv_text = "\n".join([",".join(CSV_COLS)] + [csv_row(c) for c in cells]) + "\n"
    md_text = build_md(cells)
    with open(CSV_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(csv_text)
    with open(MD_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(md_text)

    print(f"[written] {CSV_PATH}")
    print(f"[written] {MD_PATH}")
    print(f"\nAnalyzed {len(cells)} oracle-ready cells (ORACLE / upper bound; not deployable).")
    print("\n----- Best-score oracle (compact) -----")
    print(f"{'model':11} {'dataset':7} {'dense':>6} {'bestfix':>7} {'oracle':>6} "
          f"{'o-dense':>7} {'o-bestfix':>9} {'visRed%':>7}")
    for c in cells:
        print(f"{c['model']:11} {c['dataset']:7} {c['dense_acc']:6.2f} {c['best_fixed_acc']:7.2f} "
              f"{c['oracle_acc']:6.2f} {c['oracle_vs_dense_pp']:+7.2f} "
              f"{c['oracle_vs_bestfixed_pp']:+9.2f} {c['oracle_visual_reduction_pct']:7.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
