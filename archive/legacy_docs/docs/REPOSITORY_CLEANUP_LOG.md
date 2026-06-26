# Repository Cleanup Log

> **2026-06-24 — Phase 3E EXECUTED (method-based migration complete).** `GQA/`, `VQA_V2/`, `v2/`,
> `outputs/`, `literature/` are gone from the top level. Code → `src/` by method (98 files, imports
> rewritten to `src.*`); configs → `configs/`; launchers → `scripts/`; results → `results/`
> (prefix-preserving: `outputs/`→`thesis_main/gqa/`, `VQA_V2/outputs/`→`thesis_main/vqav2/`,
> `v2/outputs/`→`thesis_main/highres/`, `qwen_budget_data_*`→`paper_candidates/`); literature + v2
> FINDINGS/PAPER drafts → `docs/{literature,source_findings}/`. Ledger/master-plan/paper-plan paths
> updated in lockstep (22/22 resolve). Verified: compileall exit 0, 0 stale imports, 0 `src.*` path
> errors, results git-ignored. Full detail in `FINAL_REPOSITORY_CLEANUP_REPORT.md`. The Phase-3E plan
> text below is the *historical* pre-execution note (some path strings churned by the migration's
> prefix-rewrite; superseded by this entry and the final report).

*Compact history of repo cleanup + the plan for what remains. Detailed per-phase reports are preserved in
`archive/docs_consolidated_backup/` (BACKUP_REPORT, MIGRATION_REPORT, CLEANUP_MIGRATION_PLAN,
CURRENT_TREE_CLEANUP_AUDIT, SAFE_CLEANUP_REPORT, RETIRED_CODE_ARCHIVE_REPORT, PHASE2_RESULTS_MANIFEST,
PHASE3_ARCHITECTURE_PLAN, DOCS_CONSOLIDATION_AUDIT, SKIPPED_ITEMS_DECISION). Consolidated 2026-06-24.*

---

## Backup (do this before any further moves)
- Verified backup of all local-only (git-ignored) evidence:
  `/home/nafees/vlm-thesis-backups/vlm-thesis-evidence-20260624-111205/` — ~1.7 G, 6 folders
  (`outputs`, `results/thesis_main/vqav2`, `results/thesis_main/highres`, `VQA_V2_early_proxy/results_summary`, `docs`, `archive`),
  file-count parity verified. Refresh command if results change:
  `rsync -aR outputs results/thesis_main/vqav2 results/thesis_main/highres VQA_V2_early_proxy/results_summary docs archive <DEST>/`.

## What has been done (chronological)
| Phase | Action | Result |
|---|---|---|
| **Story/ledger lock** | wrote story lock, evidence ledger, FLOPs artifact | `THESIS_MASTER_PLAN.md`, `THESIS_EVIDENCE_LEDGER.md`, `results/thesis_main/highres/qwen_flops_summary.json` |
| **Phase 1** | docs consolidation (round 1) | 13 superseded notes → `docs/archive_docs/` + `archive/docs_old/` |
| **Phase 2A** | declutter | 35 files / ~923 MB → `archive/{checkpoints,debug_runs,failed_experiments}/` |
| **Phase 2A (skipped items)** | archive 10 reference-gated leftovers | → `archive/failed_experiments/` |
| **Phase 3A** | safe generated cleanup | 38 `__pycache__/` + 227 `.pyc` (~2.4 MB) deleted (regenerable) |
| **Phase 3C** | retire early-proxy code | `VQA_V2_early_proxy/` → `archive/retired_code/VQA_V2_early_proxy/` (109 files) |
| **Phase 3B** | docs consolidation (round 2) | 20 docs → 6 active; rest → `archive/docs_consolidated_backup/` |
| **Phase 3D** | thesis-facing root README | root `README.md` rewritten as a method map (committed `49eca3e`) |
| **Phase 3E-0/3E-1** | migration prep + empty scaffold | branch `method-migration`; backup refreshed; empty `src/configs/scripts/experiments/results/` scaffold + index files created; **no code/results/data moved** |
| **Phase 3E-2 (batch 1)** | first real move: `utils/`→`src/utils/` (dedup 3→1) + `.gitignore` `/data/` fix | 20 import sites rewritten; verified (compileall + import smoke); on `method-migration` |
| **Phase 3E-2 (batch 2)** | externalize `archive/` (969M→external backup) + unify scorers into `src/metrics/` (dedup) | 36 imports rewritten; verified; archive preserved externally (2 copies) |

## What is archived (preserved, not deleted) — under `archive/`
- `checkpoints/` (~0.9 G redundant Stage-1 / overfit-student `.pt`)
- `debug_runs/` (download/install logs)
- `failed_experiments/` (oracle-feature/phase-B JSONs, `*_head_v1`, `dense_v1*`, `*_zeroshot`, etc.)
- `retired_code/VQA_V2_early_proxy/` (the retired classification-head track + its `results_summary/`)
- `docs_old/` and `docs_consolidated_backup/` (superseded documentation + figs + PDFs)

## What was NOT touched
- All code in `GQA/`, `VQA_V2/`, `v2/` (no edits, no moves, no import changes).
- All datasets in `data/` (kept flat; git-ignored).
- All thesis-evidence results: `results/thesis_main/gqa/`, `results/thesis_main/vqav2/`, `results/thesis_main/highres/` (ledger paths intact).
- No file referenced by the evidence ledger was moved or deleted (22/22 L+E paths verified after each step).

## Current top-level structure
```
GQA/  VQA_V2/  v2/        # code (still v1/v2-named; method-based rename deferred)
results/thesis_main/gqa/                  # GQA-track results
data/                     # datasets (git-ignored, flat)
docs/                     # 6 active docs (see docs/README.md)
archive/                  # all archived material (git-ignored)
README.md  requirements.txt  .gitignore  .claude/
```

## What still looks messy (intentionally deferred)
- Top-level `GQA/`, `VQA_V2/`, `v2/` (confusing v1/v2 names) and `results/thesis_main/gqa/`, `results/thesis_main/vqav2/`,
  `results/thesis_main/highres/` (generic). These are **heavily code-referenced**, so renaming/moving them is high-risk.

## Remaining cleanup phases (risk-ordered)
- **Phase 3D — thesis-facing root map (low risk):** rewrite the root `README.md` so a reader understands
  the folders and that "v1/v2" are historical names while the thesis is method-based ("Selection over
  Budget"). *Note: root `README.md` is git-tracked/public — confirm exposure before editing.*
- **Phase 3E — full method-based code/results migration (high risk; after the thesis draft):** map
  `GQA/ VQA_V2/ v2/` → `src/{dense,static,dynamic_budget,question_conditioned_selection,distill,
  evaluation,analysis,utils}`; move `results/thesis_main/gqa/*` → `results/`; update imports, ~40 configs, `.sh` scripts,
  docs, and ledger paths in lockstep; run smoke tests (cascade_sweep reproduces its JSON; qwen_flops
  reproduces the FLOPs artifact; static K=576==dense; keep-all==stock). Or replace with a lighter reframe
  (root map + `results/INDEX.md`).

## What must NOT be deleted
- Any ledger evidence (L1–L12, E1–E5); the paper raw data `results/paper_candidates/qwen_budget_data_*.json`; the
  distill teachers/students; all code; the backup. Archive ≠ delete.

## What can be deleted later (only after backup + explicit approval)
- Redundant Stage-1 checkpoints in `archive/checkpoints/` (~0.9 G) — after thesis defense.
- `archive/failed_experiments/*` and `archive/retired_code/` — after the paper ships.

## Git status implications
- Tracked changes so far: `.gitignore` (adds `archive/`) **modified**, and the **109 early-proxy files
  show as deleted** (they live in the git-ignored `archive/`). Everything in `docs/`, `results/thesis_main/gqa/`,
  `results/thesis_main/highres/`, `data/` is git-ignored, so all the doc/result/checkpoint archiving is invisible to git.
- **Nothing has been committed.** All moves are reversible (`mv` back, or restore from the backup).
  Committing later would (a) ignore `archive/` and (b) remove the retired early-proxy track from the
  public repo — both intended.
