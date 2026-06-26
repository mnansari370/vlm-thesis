# Phase 3E — Migration Batch 1 Report (utils unification)

*Branch `method-migration`. Executed 2026-06-24. First real code move. Verified; reversible.*

## 1. What was moved
- **`utils/` unified into `src/utils/`** (the dependency-graph leaf; identical across all three tracks).
  - `GQA/shared/utils/{__init__,config,seed,logger,io,device,checkpoint}.py` → `src/utils/` (canonical).
  - `VQA_V2/shared/utils/*` and `v2/shared/utils/*` removed (byte-identical duplicates; preserved in git
    history and in the backup `vlm-thesis-pre-method-migration-20260624-162623/`).
  - Added `src/__init__.py` so `src` is an importable package.
- **`.gitignore` fix:** `data/` → `/data/` (anchored), so `src/data/` and `scripts/data/` are no longer
  ignored (the scaffold `.gitkeep`s in them are now tracked).

## 2. What was NOT moved (still in place)
- All model/training/eval code in `GQA/`, `VQA_V2/`, `v2/` (only their `utils` import lines changed).
- All other `GQA/shared/` modules (`static.py`, scorers, `flops.py`, `dataset.py`) — Batches 2–4.
- All results (`outputs/`, `VQA_V2/outputs/`, `v2/outputs/`) — Batch 8.
- All datasets in `data/`. No `data/` content moved.

## 3. Imports updated
- Rewrote `(GQA|VQA_V2|v2).shared.utils` → `src.utils` in **20 files** (18 `.py` + 2 `.sh` heredocs):
  GQA dense/dynamic/eval_runners/static runners (11), `v2/training/train_stage1.py`,
  `VQA_V2/dynamic/train_dynamic.py`, `VQA_V2/shared/evaluation/{cascade_pass,generate_and_score}.py`,
  `VQA_V2/shared/scripts/{budget_variance_gate,cache_features}.py`,
  `VQA_V2/shared/training/cached/train_cached.py`, and `VQA_V2/shared/experiments/{cache_dense_443k_pipeline,run_gpu1_tasks}.sh`.
- **0** stale `*.shared.utils` references remain anywhere.

## 4. Tests / checks passed
- `python -m compileall src` and the changed `.py` files: **OK** (syntax).
- Target-existence: every `from src.utils.X` resolves to a real file.
- **Real import smoke** (project conda env, no GPU): `src.utils` + all 7 submodules import **OK**.
- Evidence spot-check: 6/6 ledger paths still resolve. No result/data/checkpoint touched.

## 5. What remains messy
- `GQA/`, `VQA_V2/`, `v2/`, `outputs/` are **still present** — only the `utils` layer has been unified.
  Their scorers, flops, data loaders, models, eval, training, configs, and results are not migrated yet
  (Batches 2–8). So the top level still shows the historical folders.

## 6. Can `GQA/ VQA_V2/ v2/ outputs/` be removed from the top level yet?
**No.** They still contain the large majority of the code and all results. They can only be removed after
Batches 2–8 migrate everything out and Batch 9 confirms they are empty of real files.

## 7. Can `archive/` be deleted yet?
**No.** It holds the only on-disk copy (besides the backup) of the retired early-proxy code, the redundant
checkpoints, the failed experiments, and the consolidated old docs. Deletion stays a post-defense /
post-paper step, per the deletion policy — and only after confirming each item is backed up and unneeded.

## 8. Exact next batch
**Batch 2 — scorers → `src/metrics/`** (`official_score`, `m4c_evaluator`, `textvqa/pope/docvqa/chartqa`
scorers, `metrics.py`), dedup the duplicated copies, rewrite importers. Then Batch 3 (`src/analysis/`
flops), Batch 4 (`src/data/`). See `PHASE3E_REAL_MIGRATION_MAP.md` for the full sequence.

---

## Answers
- **Is the repo cleaner now?** Marginally — the 3 duplicate `utils/` trees are now one `src/utils/`, and
  the `.gitignore` gotcha is fixed. The big visual cleanup (removing `GQA/ VQA_V2/ v2/ outputs/`) comes in
  later batches.
- **Which old top-level folders remain?** `GQA/`, `VQA_V2/`, `v2/`, `outputs/`, `archive/`.
- **Why do they remain?** They still hold most of the code (Batches 2–7) and all results (Batch 8);
  `archive/` is preserved history.
- **What next batch will remove them?** They are emptied progressively by Batches 2–8 and removed in
  Batch 9 (only once verified empty of real files).
- **Safe to permanently delete `archive/` now?** **No** — not until everything in it is confirmed backed
  up and unnecessary (post-defense/paper), with explicit approval.
