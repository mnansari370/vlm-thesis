# `results/` — future home for experiment results (SCAFFOLD ONLY)

> **Empty scaffold. No results have been moved here.** All result files still live in their current
> locations and are referenced by `docs/THESIS_EVIDENCE_LEDGER.md`:
> `outputs/` (GQA-track), `VQA_V2/outputs/`, `v2/outputs/`.

Target layout (filled during Phase 3E migration, only **with** the matching code-path + ledger updates):

```
results/
  thesis_main/        # L1–L12, E1–E5 headline evidence
  appendix/           # ablations, frontier, probe detail
  paper_candidates/   # qwen_budget_data_*, raw inputs for the L10 recompute
  archived/           # superseded result sets (pointer to ../archive/ where applicable)
```

**Until migration runs, do not look here for evidence — use `docs/THESIS_EVIDENCE_LEDGER.md` paths.**
See `INDEX.md` and `docs/METHOD_BASED_MIGRATION_PLAN.md`.
