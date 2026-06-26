# Phase 3E — Migration Batch 2 Report (externalize archive + unify metrics)

*Branch `method-migration`. Executed 2026-06-24. Verified; reversible.*

## 1. Generated cache removed
- Removed 27 regenerated `__pycache__/` dirs + `.pyc` (git-ignored, regenerable). No `.pytest_cache`/
  `.ipynb_checkpoints` present.

## 2. Archive externalized (moved out of repo, NOT deleted)
- `archive/` (969 MB, 226 files) → `/home/nafees/vlm-thesis-backups/archive-removed-from-repo-20260624-165537/archive/`.
- Verified: top-level `archive/` gone; external copy exists; **226 == 226** file-count match.
- Also still present in the pre-migration backup `vlm-thesis-pre-method-migration-20260624-162623/archive/`
  (a second external copy). Details + restore command in `ARCHIVE_EXTERNALIZATION_REPORT.md`.
- `archive/` was git-ignored, so this is invisible to git (no tracked change).

## 3. Batch 2 — metrics unified into `src/metrics/`
Moved (canonical), and removed byte-identical duplicates:
- From `GQA/shared/` → `src/metrics/`: `official_score.py`, `m4c_evaluator.py`, `textvqa_score.py`,
  `pope_score.py`, `eval_pope_official.py`, `metrics.py`.
- From `v2/shared/` → `src/metrics/`: `docvqa_score.py`, `chartqa_score.py` (v2-only).
- **Removed** the 6 byte-identical `v2/shared/` scorer duplicates (verified identical before removal;
  preserved in git history + backups).
- Added `src/metrics/__init__.py`.

## 4. Duplicates removed
- 6 `v2/shared/{official_score,m4c_evaluator,textvqa_score,pope_score,eval_pope_official,metrics}.py` —
  each confirmed byte-identical to the GQA canonical via `diff` before removal.

## 5. Imports updated
- Rewrote `(GQA|v2).shared.<scorer>` → `src.metrics.<scorer>` in **36 files** (`.py` + `.sh`).
- **0** stale `(GQA|v2).shared.<scorer>` references remain.
- VQA_V2 imports none of these scorers (it uses its own `shared/datasets/vqav2_answers.py`), so VQA_V2
  importers were not affected.
- The internal `textvqa_score → m4c_evaluator` cross-import (a `sys.path` to its own dir) still resolves
  because both now live in `src/metrics/`.

## 6. Tests passed
- `python -m compileall GQA VQA_V2 v2 src scripts`: exit 0.
- `src.metrics.*` target existence: all resolve.
- Import smoke (conda env): all 8 `src.metrics` submodules import OK.
- Real importer-chain smoke: `GQA.dynamic.cascade_sweep` (uses `src.metrics.official_score` +
  `GQA.shared.flops`) and `VQA_V2.shared.evaluation.instance_headroom` import OK.
- Evidence spot-check 0 missing; `data/` untouched.

## 7. What still remains messy
- `GQA/shared/` still holds `static.py`, `flops.py`, `dataset.py`; `v2/shared/` still holds `flops.py`
  (identical dup — Batch 3). `GQA/`, `VQA_V2/`, `v2/`, `outputs/` still present (Batches 3–8).

## 8. Was `archive/` permanently deleted? **NO.**
Moved to an external folder (and present in the pre-migration backup). Permanent deletion remains a later
step — **only after thesis defense / paper submission, with explicit final confirmation.**

---

## Answers
1. **Is `archive/` gone from the repo top level?** Yes — moved out (not deleted).
2. **Where is it preserved externally?** `/home/nafees/vlm-thesis-backups/archive-removed-from-repo-20260624-165537/archive/` (plus the pre-migration backup copy).
3. **What did Batch 2 migrate?** The shared scorers into `src/metrics/` (8 modules), removing 6 identical duplicates; 36 import sites rewritten.
4. **Did any thesis evidence break?** No — all ledger evidence paths still resolve; no result/data/checkpoint moved.
5. **Which old top-level folders still remain?** `GQA/`, `VQA_V2/`, `v2/`, `outputs/`.
6. **Next batch:** Batch 3 — FLOPs/analysis → `src/analysis/` (dedup `GQA/shared/flops.py` == `v2/shared/flops.py`), then Batch 4 (data loaders → `src/data/`).
