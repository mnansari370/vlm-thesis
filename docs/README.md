# `docs/` — clean final-scope documentation index

*Rebuilt 2026-06-27 after the scope lock. This folder now holds **only** the active final-scope planning
documents. The older thesis/audit/report docs were archived (with their evidence extracted first) to
`archive/legacy_docs/docs/`.*

**Locked scope:** models {LLaVA-1.5-7B, Qwen-2.5-VL-7B}; datasets {GQA, TextVQA, DocVQA, VQAv2}; methods
{dense, static, dynamic-WHICH, dynamic-COUNT}; static budgets {15, 25, 35, 50, 75, 100}% of dense.

## Active documents (read in this order)
| If you want… | Read |
|---|---|
| The locked scope (models/datasets/methods/budgets) + thesis sentence | **[`FINAL_SCOPE_LOCK.md`](FINAL_SCOPE_LOCK.md)** |
| Every final-scope number, status (DONE/PARTIAL/TODO), caveats, risks, evidence files | **[`FINAL_EVIDENCE_LEDGER.md`](FINAL_EVIDENCE_LEDGER.md)** ← source of truth |
| The model×dataset×method experiment grid with status per cell | **[`CLEAN_EXPERIMENT_MATRIX.md`](CLEAN_EXPERIMENT_MATRIX.md)** |
| The cleaned repo layout (active code/configs/data/results vs archive) | **[`CLEAN_REPOSITORY_MAP.md`](CLEAN_REPOSITORY_MAP.md)** |
| Which configs exist vs must be created for the final runs | **[`FINAL_CONFIG_INVENTORY.md`](FINAL_CONFIG_INVENTORY.md)** |
| The exact next experiment plan (no-GPU checks → dense → static → WHICH → COUNT → FLOPs) | **[`TODO_NEXT_RUNS.md`](TODO_NEXT_RUNS.md)** |
| Which legacy result files are in-scope vs out-of-scope | [`../results/legacy_index.md`](../results/legacy_index.md) |

**Rule:** every number written in the thesis must trace to a row in `FINAL_EVIDENCE_LEDGER.md`.

## Archived documentation (preserved, not deleted)
The previous docs were moved to **`archive/legacy_docs/docs/`** after their still-useful evidence was
extracted into `FINAL_EVIDENCE_LEDGER.md`. They remain on disk and in git history:
- `THESIS_EVIDENCE_LEDGER.md`, `THESIS_MASTER_PLAN.md`, `THESIS_AUDIT.md`, `THESIS_VERIFICATION.md`
- `PAPER_PUBLICATION_PLAN.md`, `FINAL_REPOSITORY_CLEANUP_REPORT.md`, `REPOSITORY_CLEANUP_LOG.md`
- `FULL_REPOSITORY_UNDERSTANDING_REPORT.md`, `METHOD_DATASET_MODEL_CLARITY_REPORT.md`
- `literature/`, `source_findings/`, `migration_history/`

The full move record (old path → archive path, reason, whether evidence was extracted) is in
[`../archive/migration_manifests/archive_manifest_20260627.md`](../archive/migration_manifests/archive_manifest_20260627.md).
