# TODO — Next Runs (clean final-scope experiment plan)

*Locked scope (2026-06-27). Dependency-ordered. Each phase lists exact commands/intent. No GPU work has been
run by this reorganization — this is the plan to populate `results/final_scope/`.*

Budget levels = {15, 25, 35, 50, 75, 100}% of dense. LLaVA-1.5 K-map: 86 / 144 / 202 / 288 / 432 / 576.
Qwen: per-sample K = round(pct × dense_tokens_sample).

---

## Phase 0 — No-GPU checks (do first, minutes)
1. `python -m compileall src scripts` → expect exit 0 (already verified post-archive).
2. Import smoke: `python -c "import sys; sys.path.insert(0,'.'); import src.metrics, src.utils, src.analysis.flops, src.models.static.static"`.
3. Confirm the FLOPs utilities cover both models: `python -m src.analysis.flops` (LLaVA-1.5) and inspect
   `src/analysis/qwen_flops.py` (set per-dataset `N_TEXT` for GQA/TextVQA/VQAv2 — currently hardcoded to a
   DocVQA estimate of 40).
4. Verify datasets resolve: `data/{vqav2,gqa,textvqa}` present; DocVQA hub cache reachable.

## Phase 1 — Dense baselines (the ceilings / reproduction anchors)
Goal: fill the 4 dense cells that are TODO, re-confirm the 4 that are DONE.
- **LLaVA-1.5 × {GQA, TextVQA, VQAv2}** — DONE (lift 61.42 / 57.65 / 76.44 from `results/thesis_main`).
- **LLaVA-1.5 × DocVQA** — **TODO**: build a DocVQA loader path into the LLaVA-1.5 generation harness (or run
  `StaticPrunedLlava` method=`none` with the DocVQA images + ANLS scorer). Expect low absolute (576 tokens
  can't read dense docs) — informative.
- **Qwen-2.5-VL-7B × DocVQA** — DONE (97.19; optionally rerun full-val to replace n=200).
- **Qwen-2.5-VL-7B × {GQA, TextVQA, VQAv2}** — **TODO**: requires a new Qwen dataset adapter (the pruner only
  loads DocVQA/ChartQA today). Add `load_bench` branches + wire GQA/TextVQA/VQAv2 scorers.

## Phase 2 — Static fixed-budget frontier (15/25/35/50/75%)
Goal: a clean static frontier at the 6 budget levels for every model×dataset.
- **LLaVA-1.5:** relax the K constraint to allow K=86 (15%) and K=202 (35%) (the `cls_attn` path uses `topk`,
  so no precomputed index is needed — just widen `SUPPORTED_K`/argparse choices in the *active* static code).
  Run K∈{86,144,202,288,432,576} on GQA/TextVQA/VQAv2 (and DocVQA once Phase-1 loader exists).
- **Qwen-2.5-VL-7B:** implement **%-of-dense** budgeting (per-sample K) and a **fair CLS-equivalent blind
  baseline** (Qwen ViT/merger attention) — the current `uniform`/`norm` floors are too weak for an honest
  comparison. Run the 6 levels on all four datasets.

## Phase 3 — Dynamic WHICH (question-conditioned selection at fixed %)
Goal: the headline — QC selection vs the static/blind baseline at **matched % (matched FLOPs)**.
- **Qwen-2.5-VL-7B:** run `textattn` (L16 teacher) at the 6 levels on DocVQA (re-express existing K-runs as %,
  extend to full-val), then GQA/TextVQA/VQAv2. Always pair with the fair blind baseline from Phase 2.
- **LLaVA-1.5:** run the QC selection at the 6 levels on GQA/TextVQA/VQAv2/DocVQA. Expect it to **lose/tie**
  CLS-attention (consistent with the K=64 probe negatives) — report as the regime-specific negative.
- Keep the **mismatched-question control** (real vs mismatched vs blind) at one budget per dataset to show the
  gain is question-driven.

## Phase 4 — Dynamic COUNT (fresh redesign)
Goal: a new per-sample budget mechanism, tested **against static at matched average FLOPs**.
- Design the new mechanism (not the old BudgetController / Qwen MLP — those stay as evidence/methodology).
- Reuse the **monotone oracle-noise correction** (`src/pruning/dynamic_budget/qwen_oracle*.py`) to bound the
  headroom before training anything.
- Acceptance test: does the learned per-sample budget beat the **best fixed %** at the **same average tokens**?
  (The prior answer was ~+0.05pp / +0.09pp — the redesign must clear that honestly or be reported as a
  confirmed negative.)

## Phase 5 — FLOPs / efficiency extraction (no GPU)
- For every populated cell, emit token-reduction (measured avg tokens) **and** FLOP-reduction (analytical
  prefill) **separately** — never conflate. Use `flops.py`/`flops_vqav2.py` (LLaVA-1.5) and `qwen_flops.py`
  (Qwen, after setting per-dataset `N_TEXT`).
- State the selector cost caveat: the QC teacher needs a full forward (excluded from the FLOP numbers); the
  deployable path is the distilled student.

## Output convention
All new runs write to `results/final_scope/{llava15,qwen25vl7b}/{gqa,textvqa,docvqa,vqav2}/` with a filename
encoding `{method}_{pct}.json` so the matrix in `CLEAN_EXPERIMENT_MATRIX.md` can be auto-filled.

## First concrete action
**Phase 0 + Phase 2 (LLaVA-1.5 static at 15%/35%)** — lowest cost, fills real matrix cells, needs only a
one-line K-constraint relaxation in the active static code (no new harness). Then the Qwen dataset-adapter
build (gates all Qwen rows).
