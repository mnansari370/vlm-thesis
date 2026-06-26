# Archive Externalization Report

> Moving the repo-local `archive/` out of the repository to an external backup folder, so the top level
> looks cleaner. **This is a move, not a deletion** — nothing is permanently deleted. Executed 2026-06-24
> on branch `method-migration`. `archive/` is git-ignored, so this is invisible to git.

## Source → destination
| | |
|---|---|
| Source | `archive/` (repo top level) |
| Destination | `/home/nafees/vlm-thesis-backups/archive-removed-from-repo-20260624-165537/archive/` |
| Size / files | **969 MB / 226 files** |
| Move type | `mv` (reversible) |

## Contents moved (preserved)
| Subdir | Size | Files | What |
|---|---|---|---|
| `checkpoints/` | 891 M | 8 | redundant Stage-1 / overfit-student `.pt` |
| `failed_experiments/` | 49 M | 60 | oracle-feature/phase-B JSONs, `*_head_v1`, `dense_v1*`, `*_zeroshot`, etc. |
| `debug_runs/` | 21 M | 8 | download/install logs |
| `retired_code/VQA_V2_early_proxy/` | 6.9 M | 109 | the retired classification-head track + its `results_summary/` |
| `docs_consolidated_backup/` | 2.6 M | 36 | superseded docs + figs + PDFs + old ledger |
| `docs_old/` | 36 K | 5 | superseded `outputs/`-era notes |

## Safety checks (all pass)
- **No thesis evidence in `archive/`:** the evidence ledger has **zero** L/E (`File:`) paths under
  `archive/`; its only `archive/` mention is a historical "prior maps" footer pointer (not evidence).
- **Redundant preservation:** the pre-migration backup
  `/home/nafees/vlm-thesis-backups/vlm-thesis-pre-method-migration-20260624-162623/archive/` already holds
  a full copy (226 files). So after this move there are **two** external copies of `archive/`.
- **Not permanent deletion:** the external folder is kept; restore with
  `mv /home/nafees/vlm-thesis-backups/archive-removed-from-repo-20260624-165537/archive ./archive`.

## Doc references that now point outside the repo (historical, acceptable)
A few docs mention `archive/...` paths (e.g. `THESIS_EVIDENCE_LEDGER.md` "prior maps" footer;
`APPENDIX_RESULTS.md` early-proxy note in `docs_consolidated_backup/`; `REPOSITORY_CLEANUP_LOG.md`).
These are **historical pointers**, not active evidence. After externalization they resolve under the
external archive folder above. No active thesis claim depends on them.

## Is `archive/` permanently deleted? **NO.**
It is moved to the external backup folder (and also exists in the pre-migration backup). Permanent
deletion remains a later step — only after thesis defense / paper submission, with explicit confirmation.
