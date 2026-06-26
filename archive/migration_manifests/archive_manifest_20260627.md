# Archive Manifest — Final Scope Lock (2026-06-27)

*Read-only-safe reorganization. **Nothing was deleted.** Out-of-scope files were **moved** (preserved on
disk) into `archive/` and recorded here. Tracked code/config/script files now show as `D` (deleted) in
`git status` — they are preserved on disk under `archive/` (which is git-ignored) and remain in git history.
Git-ignored data/result trees were moved with no git-status change. `compileall src scripts` = exit 0;
in-scope import smoke = 0 src-errors after the move.*

**Locked scope:** models = {LLaVA-1.5-7B, Qwen-2.5-VL-7B}; datasets = {GQA, TextVQA, DocVQA, VQAv2};
methods = {dense, static, dynamic-WHICH (question-conditioned fixed budget), dynamic-COUNT (to be
redesigned)}; static/fixed budgets = {15, 25, 35, 50, 75, 100}% of dense tokens.

**Restore rule:** every archived path mirrors its original location under the bucket, so restoring is
`mv archive/<bucket>/<original/relative/path> <original/relative/path>`.

---

## A. Moved — CODE / CONFIG / SCRIPT (tracked; now `D` in git, preserved on disk)

### A.1 Elastic LoRA Stage-1 cluster (the only non-frozen track; excluded from thesis)
| Old path | New archive path | Reason | Git | Safe to ignore for thesis? |
|---|---|---|---|---|
| `src/models/elastic/` (4 files) | `archive/legacy_models/src/models/elastic/` | elastic LoRA backbone wrapper + tests; trains the backbone (out of scope: all thesis wins are frozen) | tracked | Yes |
| `src/training/train_stage1.py` | `archive/legacy_experiments/src/training/train_stage1.py` | elastic Stage-1 trainer | tracked | Yes |
| `src/training/test_train_stage1.py` | `archive/legacy_experiments/src/training/test_train_stage1.py` | elastic trainer smoke test | tracked | Yes |
| `src/evaluation/gqa/evaluate_gqa.py` | `archive/legacy_experiments/src/evaluation/gqa/evaluate_gqa.py` | imports `ElasticPrunedLlava`; NOT the canonical GQA driver (that's `run_*_testdev.py`) | tracked | Yes |
| `src/evaluation/textvqa/evaluate_textvqa.py` | `archive/legacy_experiments/src/evaluation/textvqa/evaluate_textvqa.py` | imports `ElasticPrunedLlava`; NOT the canonical TextVQA driver (`run_textvqa.py`) | tracked | Yes |
| `src/evaluation/test_generate.py` | `archive/legacy_experiments/src/evaluation/test_generate.py` | imports `ElasticPrunedLlava` | tracked | Yes |
| `src/data/llava_mix.py` | `archive/legacy_datasets/src/data/llava_mix.py` | LLaVA-665K mix loader — only consumed by elastic Stage-1 | tracked | Yes |
| `src/data/test_llava_mix.py` | `archive/legacy_datasets/src/data/test_llava_mix.py` | mix loader unit test | tracked | Yes |
| `configs/stage1_elastic.yaml` | `archive/legacy_experiments/configs/stage1_elastic.yaml` | elastic Stage-1 config | tracked | Yes |
| `scripts/training/launch_stage1_full.sh` | `archive/legacy_scripts/scripts/training/launch_stage1_full.sh` | elastic launcher | tracked | Yes |

### A.2 LLaVA-1.6 high-res bridge cluster (out-of-scope model)
| Old path | New archive path | Reason | Git | Safe? |
|---|---|---|---|---|
| `src/evaluation/textvqa/evaluate_textvqa_highres.py` | `archive/legacy_experiments/src/evaluation/textvqa/...` | LLaVA-1.6 AnyRes TextVQA | tracked | Yes |
| `src/evaluation/textvqa/evaluate_textvqa_highres_kcurve.py` | `archive/legacy_experiments/...` | `HighResPruner` (LLaVA-1.6) K-curve | tracked | Yes |
| `src/evaluation/textvqa/evaluate_textvqa_highres_spread.py` | `archive/legacy_experiments/...` | LLaVA-1.6 spread eval | tracked | Yes |
| `src/evaluation/textvqa/textattn_layer_sweep.py` | `archive/legacy_experiments/...` | LLaVA-1.6 layer sweep | tracked | Yes |
| `src/evaluation/gqa/evaluate_gqa_highres.py` | `archive/legacy_experiments/...` | LLaVA-1.6 GQA | tracked | Yes |
| `src/evaluation/docvqa/evaluate_highres_docchart.py` | `archive/legacy_experiments/...` | LLaVA-1.6 DocVQA/ChartQA | tracked | Yes |
| `src/evaluation/docvqa/evaluate_highres_spread_docchart.py` | `archive/legacy_experiments/...` | LLaVA-1.6 spread DocVQA/ChartQA | tracked | Yes |
| `src/evaluation/docvqa/control_question_conditioning.py` | `archive/legacy_experiments/...` | imports `HighResPruner` (LLaVA-1.6 control) | tracked | Yes |
| `src/analysis/llava_latency.py` | `archive/legacy_experiments/src/analysis/...` | LLaVA-1.6 latency (the 3.31× number; **kept only as a number in the ledger, code archived**) | tracked | Yes (cite number, not code) |
| `src/analysis/oracle_decomposition.py` | `archive/legacy_experiments/...` | reads LLaVA-1.6 spread files | tracked | Yes |
| `src/analysis/make_figures_highres.py` | `archive/legacy_experiments/...` | LLaVA-1.6 figure maker | tracked | Yes |
| `src/analysis/flops_frontier.py` | `archive/legacy_experiments/...` | LLaVA-1.6 FLOPs frontier (hardcoded high-res numbers) | tracked | Yes |
| `src/pruning/dynamic_budget/llava_budget_data.py` | `archive/legacy_experiments/...` | cross-family budget data on LLaVA-1.6 (imports `HighResPruner`) | tracked | Yes |

### A.3 POPE + ScienceQA evaluation (out-of-scope datasets)
| Old path | New archive path | Reason | Git | Safe? |
|---|---|---|---|---|
| `src/evaluation/pope/` (`run_pope.py`, `run_pope_speculative.py`, `__init__.py`) | `archive/legacy_experiments/src/evaluation/pope/` | POPE eval (out of scope); one-way import of `StaticPrunedLlava` (kept) | tracked | Yes |
| `src/evaluation/scienceqa/` (`run_sqa.py`, `__init__.py`) | `archive/legacy_experiments/src/evaluation/scienceqa/` | ScienceQA eval (out of scope) | tracked | Yes |

### A.4 Qwen 3B/32B + LLaVA-1.6/mix download scripts
| Old path | New archive path | Reason | Git | Safe? |
|---|---|---|---|---|
| `scripts/data/dl_qwen3b.py` | `archive/legacy_scripts/scripts/data/dl_qwen3b.py` | downloads Qwen-2.5-VL-3B (out of scope) | tracked | Yes |
| `scripts/data/dl_qwen32b.py` | `archive/legacy_scripts/...` | downloads Qwen-2.5-VL-32B (out of scope) | tracked | Yes |
| `scripts/data/download_llava16.py` | `archive/legacy_scripts/...` | downloads LLaVA-1.6 (out of scope) | tracked | Yes |
| `scripts/data/download_mix.sh` | `archive/legacy_scripts/...` | downloads LLaVA-665K mix (elastic only) | tracked | Yes |

## B. Moved — DATA / RESULTS (git-ignored; no git-status change)
| Old path | New archive path | Reason | Git | Safe? |
|---|---|---|---|---|
| `data/budget_oracle/` (16 files, 24M) | `archive/legacy_datasets/data/budget_oracle/` | retired VQAv2 budget-predictor diagnostics; **generator code not in repo**; classification-era protocol; nothing imports it | ignored | Yes |
| `results/archived/stage1_full{,.log}`, `stage1_quickfix{,.log}` | `archive/legacy_results/stage1_elastic/` | elastic Stage-1 checkpoints + logs (out of scope) | ignored | Yes |

---

## C. Shared files to clean LATER — **NOT moved** (partly in-scope, partly legacy, or import-risky)

These contain out-of-scope branches/constants/lazy-imports but are still needed by in-scope code. Moving them
would break in-scope imports or lose in-scope evidence. **Leave in place; clean during the method-code rewrite.**

| File | Why kept | What is legacy in it |
|---|---|---|
| `src/models/dynamic_budget/` (whole package) | `generate_and_score.py` lazily imports `LlavaDynamicVQAModel`; it is the L2 budget-ties-static evidence (LLaVA-1.5 × VQAv2, **in scope**) | the OLD `BudgetController` is the dynamic-COUNT attempt being redesigned; `token_scorer.py` (`QuestionConditionedTokenScorer`) is conceptually dynamic-WHICH stranded here |
| `src/models/static/` (whole package) | holds `StaticPrunedLlava` (in scope, depended on by visionzip + QC probes) | also holds the retired classification `LlavaStaticVQAModel` + its answer head |
| `src/pruning/dynamic_budget/qwen_oracle.py`, `qwen_oracle_qc.py`, `qwen_budget_data.py`, `qwen_budget_eval.py`, `qwen_budget_robust.py` | Qwen-7B budget-mirage evidence on **DocVQA (in scope)**; the redesign will likely reuse the monotone-oracle methodology | each also has a `chartqa`/`infovqa` branch (out of scope) |
| `src/evaluation/docvqa/qwen_kcurve.py`, `qwen25_dense_eval.py`, `qwen_control.py`, `qwen_layer_sweep.py` | Qwen-7B DocVQA (in scope) | `qwen_kcurve`/`qwen25_dense_eval` also handle `chartqa` |
| `src/metrics/chartqa_score.py` | imported by the kept qwen oracle/dense files | ChartQA out of scope (scorer only) |
| `src/metrics/pope_score.py`, `eval_pope_official.py` | now **orphaned** (only consumer, POPE eval, was archived) — harmless, tiny | POPE out of scope; safe to archive in the later pass |
| `src/analysis/flops.py` | canonical FLOPs for GQA/TextVQA (in scope) | carries `N_TEXT_POPE`, `N_TEXT_SQA` constants (out of scope, harmless) |
| `src/analysis/cascade_sweep.py`, `src/evaluation/gqa/run_speculative_testdev.py` | GQA/TextVQA confidence cascade (in scope) | `cascade_sweep` also globs POPE/SQA result dirs (now absent → graceful skip) |
| `scripts/data/download_docchart.py` | downloads **DocVQA** (in scope) | also caches ChartQA |
| `scripts/data/budget_variance_gate.py` | dynamic-COUNT K-variance smoke on the in-scope BudgetController | the OLD dynamic-count attempt; revisit in the redesign |

---

## D. Out-of-scope but left in `data/` due to size (recommend delete later, not now)
| Path | Size | Note |
|---|---|---|
| `data/llava_mix/` | ~34G | LLaVA-665K mix images — only the (archived) elastic Stage-1 used it. Left in place to avoid 34G of same-disk churn; its loader is archived. **Recommend deleting after thesis submission**, not now. |

---

## E. Result trees deliberately NOT moved (ledger-pinned; indexed instead)

`results/thesis_main/{gqa,vqav2,highres}/` were **left in place** and catalogued in `results/legacy_index.md`.
Reason (a concrete code/path issue): the evidence ledger and several active reader scripts hardcode
`results/thesis_main/...` paths (e.g. `cascade_sweep.py`, the Qwen budget evals). `results/thesis_main/highres/`
in particular **mixes in-scope Qwen-7B/DocVQA results with out-of-scope LLaVA-1.6 / ChartQA / 3B / 32B
results in one folder**, so a bulk move would archive in-scope evidence and break pinned paths. The index
records which highres files are in-scope vs out-of-scope; physical separation is deferred to the run-folder
restandardization (`results/final_scope/`).

---

## F. Verification (this manifest's moves)
- `python -m compileall src scripts` → **exit 0**.
- In-scope import smoke (26 modules) → **0 src-errors** (env-only skips for torch/transformers absent).
- `git status --short` → **35 `D`** (tracked code/config/script moved out; preserved on disk in `archive/`).
- `find archive -type f | wc -l` → 38 files.
- Out-of-scope keywords remaining in the active tree are **textual/branch references, not imports** (§C),
  flagged for the later code-clean pass.

---

## G. Docs archived after evidence extraction (2026-06-27, doc-cleanup pass)

The still-useful, final-scope evidence from the docs below was **extracted into the new active
`docs/FINAL_EVIDENCE_LEDGER.md`** (numbers, status, caveats, risks, per-claim result files, source provenance)
**before** archiving. The originals were then moved to `archive/legacy_docs/docs/` (structure preserved;
markdown-only, so trackable per the §-policy). Nothing was deleted.

| Old path | Archive path | Reason | Evidence extracted into FINAL_EVIDENCE_LEDGER.md? |
|---|---|---|---|
| `docs/THESIS_EVIDENCE_LEDGER.md` | `archive/legacy_docs/docs/THESIS_EVIDENCE_LEDGER.md` | superseded by the scope-filtered ledger | **Yes** — claim numbers L1–L9/L11–L12/E1–E5 (in-scope rows), canonical values, reconciliations |
| `docs/THESIS_VERIFICATION.md` | `archive/legacy_docs/docs/THESIS_VERIFICATION.md` | deep audit; key corrections lifted | **Yes** — +7.50→+0.81pp reframe, +60pp baseline asymmetry, AUC-0.643 fix (→ §7 caveats) |
| `docs/THESIS_AUDIT.md` | `archive/legacy_docs/docs/THESIS_AUDIT.md` | LLaVA-1.5 result inventory; FLOPs conventions | **Yes** — static frontier numbers + FLOPs-convention risk (→ §5–6, §8) |
| `docs/THESIS_MASTER_PLAN.md` | `archive/legacy_docs/docs/THESIS_MASTER_PLAN.md` | story/scope superseded by `FINAL_SCOPE_LOCK.md` | **Yes (partial)** — thesis claim, frozen-regime framing (→ §1) |
| `docs/PAPER_PUBLICATION_PLAN.md` | `archive/legacy_docs/docs/PAPER_PUBLICATION_PLAN.md` | paper-readiness; gaps lifted | **Yes (partial)** — "not paper-ready" gaps + claims-to-avoid (→ §7, §8) |
| `docs/FULL_REPOSITORY_UNDERSTANDING_REPORT.md` | `archive/legacy_docs/docs/...` | my repo-wide map; superseded by `CLEAN_REPOSITORY_MAP.md` | **Yes (partial)** — risks/gaps (→ §8) |
| `docs/METHOD_DATASET_MODEL_CLARITY_REPORT.md` | `archive/legacy_docs/docs/...` | my method/dataset/model breakdown; superseded by ledger + matrix | **Yes** — all verified per-cell numbers (→ §5–6) |
| `docs/FINAL_REPOSITORY_CLEANUP_REPORT.md` | `archive/legacy_docs/docs/...` | prior (method-migration) cleanup record; historical | No (historical only) |
| `docs/REPOSITORY_CLEANUP_LOG.md` | `archive/legacy_docs/docs/...` | chronological cleanup log; historical | No (historical only) |
| `docs/literature/` | `archive/legacy_docs/docs/literature/` | related-work notes (paper-stage material) | No (kept intact for the background chapter) |
| `docs/source_findings/` | `archive/legacy_docs/docs/source_findings/` | original per-track FINDINGS/PAPER drafts | **Yes (partial)** — Qwen/distill headline numbers (→ §5–6) |
| `docs/migration_history/` | `archive/legacy_docs/docs/migration_history/` | superseded migration plans/reports | No (provenance only) |

**Active docs after this pass (kept in `docs/`):** `README.md` (rewritten as a clean index),
`FINAL_SCOPE_LOCK.md`, `FINAL_EVIDENCE_LEDGER.md`, `CLEAN_EXPERIMENT_MATRIX.md`, `CLEAN_REPOSITORY_MAP.md`,
`FINAL_CONFIG_INVENTORY.md`, `TODO_NEXT_RUNS.md`. `.gitignore` updated to track these (and `results/legacy_index.md`).
