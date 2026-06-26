# Final Repository Cleanup Report — method-based migration

*Date: 2026-06-24. Branch: `method-migration`. This completes the Phase-3E method-based migration: the
three historical track folders (`GQA/`, `VQA_V2/`, `v2/`) and `outputs/`, `literature/` no longer exist
at the top level; all code is now under `src/` by method, all results under `results/`.*

## 1. Final top-level structure (achieved)

```
README.md  requirements.txt
configs/  data/  docs/  experiments/  results/  scripts/  src/
```

No `GQA/`, `VQA_V2/`, `v2/`, `outputs/`, or `literature/` remain at the top level (verified).

## 2. What moved where

**Code → `src/` (by method).** 98 `.py` files migrated, imports rewritten to the `src.*` namespace:
- `src/data/` — gqa, vqav2/ (loaders + collate + answer norm), docvqa, llava_mix
- `src/metrics/` — official/GQA, M4C/TextVQA, POPE, DocVQA ANLS, ChartQA scorers (deduped, one copy)
- `src/models/` — dense/, static/, dynamic_budget/, distillation/, elastic/ (re-exporting `__init__`s preserved)
- `src/pruning/` — static/ (visionzip), dynamic_budget/ (budget oracle/eval), question_conditioned_selection/ (qwen_pruner, clip/qcond probes)
- `src/evaluation/` — gqa/, vqa/, textvqa/, docvqa/, pope/, scienceqa/ (per-task entrypoints)
- `src/analysis/` — flops (deduped GQA≡v2), flops_vqav2, flops_frontier, qwen_flops, latency, oracle_decomposition, cascade_sweep, figures
- `src/utils/` — config/seed/logger/io/checkpoint/device (single copy; from earlier batch 1)
- `src/training/` — train_cached, train_dynamic, train_student, cache_teacher, train_stage1

**Configs → `configs/`** (27 yaml): dense/ (2), static/ (21), dynamic_budget/ (3), stage1_elastic.yaml.
Output dirs inside configs repointed from `*/outputs/` to `results/` (prevents re-creating old folders).

**Launchers → `scripts/`** (38 `.sh` + 9 `.py`): `scripts/data/` (download/cache/vocab), `scripts/training/`.

**Literature → `docs/literature/`**; **v2 FINDINGS / PAPER drafts + old track READMEs → `docs/source_findings/`** (11 files, origin-tagged names to avoid collisions).

**Results → `results/`** (302 files, 681M, git-ignored), prefix-preserving so ledger paths and reader
scripts remap by a simple prefix:
- `outputs/` → `results/thesis_main/gqa/`
- `VQA_V2/outputs/` → `results/thesis_main/vqav2/`
- `v2/outputs/` → `results/thesis_main/highres/` (figures included)
- `qwen_budget_data_*.json` → `results/paper_candidates/` (L10 recompute inputs, preserved)
- elastic Stage-1 checkpoints → `results/archived/`

## 3. Documents updated (paths kept in lockstep)

`docs/THESIS_EVIDENCE_LEDGER.md`, `docs/THESIS_MASTER_PLAN.md`, `docs/PAPER_PUBLICATION_PLAN.md`,
`docs/REPOSITORY_CLEANUP_LOG.md` — all File:/code/draft path references rewritten to the new `src/`,
`results/`, `docs/` locations. The tracked scaffold READMEs (`src/`, `scripts/`, `experiments/`,
`configs/`, `results/`) and `results/INDEX.md` were rewritten from "scaffold only" to accurate
post-migration content. The root `README.md` was rewritten for the clean method-based structure.

## 4. Verification (all passing)

| Check | Result |
|---|---|
| `python -m compileall src scripts experiments configs` | exit 0 |
| stale `(GQA\|VQA_V2\|v2).` import statements in code | 0 |
| `src.*` import smoke (85 modules) | 77 import OK, 8 env/3rd-party skips, **0 `src.*` path errors** |
| top-level = `configs data docs experiments results scripts src` | yes; old folders gone |
| `__pycache__` dirs (excl. data) | 0 |
| ledger `results/`+`src/` paths resolve | 22/22 |
| `data/` untouched | yes (7 entries, `data/vqav2` present) |
| external archive backup intact | `/home/nafees/vlm-thesis-backups/archive-removed-from-repo-20260624-165537/archive` |
| results contents git-ignored (commit not bloated) | yes — 0 non-index `results/` files staged |

## 5. Honest residual notes

- **Main vs appendix** result classification is carried by the ledger's *Placement* column, not by
  physical folder; the per-track split under `results/thesis_main/` was chosen to keep ledger/reader
  paths mapping by a simple prefix (lower risk than a file-by-file scatter). The authority is the ledger.
- **`docs/`, `results/`, `data/`, `archive/`, `feature_cache/`, `logs/` are git-ignored** (private/large/
  local/runtime). The commits contain only tracked code/config/script/structure changes; doc edits are local.
- The **external archive** (failed experiments, retired code, redundant checkpoints, consolidated-docs
  backup) remains outside the repo and was not deleted.

## 6. Final polish pass (2026-06-24)

- **Empty placeholder folders removed:** `configs/{distillation,evaluation,question_conditioned_selection}`,
  all 7 empty `experiments/*` method subfolders, `scripts/{evaluation,analysis,migration}`,
  `src/evaluation/chartqa`, `src/models/question_conditioned_selection`, `results/appendix` — none held
  any file but a `.gitkeep`. All now-redundant `.gitkeep` files (in populated folders) were removed too.
- **Docs simplified.** Active set is exactly: `README.md`, `THESIS_MASTER_PLAN.md`,
  `THESIS_EVIDENCE_LEDGER.md`, `PAPER_PUBLICATION_PLAN.md`, `FINAL_REPOSITORY_CLEANUP_REPORT.md`,
  `REPOSITORY_CLEANUP_LOG.md` + `literature/` + `source_findings/`. Superseded planning/report docs
  (method-based migration plan, batch reports, docs/archive cleanup notes) moved to
  `docs/migration_history/`.
- **All stale old-folder references in active code eliminated → 0.** Fixed docstring CLI examples and
  config-path defaults (`VQA_V2/{static,dense,dynamic}/*.yaml` → `configs/…`,
  `v2/configs/stage1_elastic.yaml` → `configs/…`); repointed legacy `.sh` runtime dirs
  (`VQA_V2/feature_cache` → `feature_cache`, `VQA_V2/logs` → `logs`, added `feature_cache/` to
  `.gitignore`); updated file-location/scorer comments to `src/…`; cleaned former-filename docstring
  prefixes. (`GQA/POPE/SQA` in one comment was a dataset list, reworded to `GQA, POPE, SQA`.)
- **Generated clutter:** `__pycache__`/`*.pyc`/`.pytest_cache`/`.ipynb_checkpoints` = 0.

Polish verification: `compileall src scripts` exit 0; the 8 named packages (`src.utils, metrics, analysis,
data, evaluation, models, pruning, training`) import OK; submodules 77 OK / 8 env-skip / **0 `src.*`
errors**; active-code old-folder refs **0**; ledger paths **22/22**; results git-ignored (only
`README.md`/`INDEX.md` tracked); **0** `.pt`/data/docs staged.

## 7. What remains

**For thesis writing (repo is ready):** the repo is clean enough to write from now. Chapters 3–7 are
fully supported by saved artifacts today (`THESIS_MASTER_PLAN.md` §5). Optional cheap pre-work: the L10
generality recompute (raw data in `results/paper_candidates/`) to make that claim writable.

**For paper work (`PAPER_PUBLICATION_PLAN.md`):** not blocking the thesis. (1) full-validation reruns of
L5/L7/L9; (2) strong baselines at matched FLOPs + faithful layer-split FastV; (3) save the L10 per-model
result JSONs; (4) ChartQA/InfoVQA student distillation. Draft from `docs/source_findings/`.

## 8. Git

Migration committed on `method-migration` as `migration: complete method-based thesis repository cleanup`;
polish committed as `cleanup: final polish after method-based migration` (no push, no history rewrite).
