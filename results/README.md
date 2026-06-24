# `results/` — experiment results

All experiment results from the three historical tracks now live here (migrated 2026-06-24). The contents
are **git-ignored** (large, regenerable); only this `README.md` and `INDEX.md` are tracked. The
authoritative paths for every thesis/paper number are in `docs/THESIS_EVIDENCE_LEDGER.md`.

```
results/
  thesis_main/      L1–L12, E1–E5 headline evidence, by track:
    gqa/              frozen GQA-track results (was outputs/)
    vqav2/            frozen VQAv2 budget track (was VQA_V2/outputs/)
    highres/          high-res / Qwen selection + distillation (was v2/outputs/)
  paper_candidates/ qwen_budget_data_*.json — raw inputs for the L10 recompute
  archived/         superseded result sets (e.g. elastic Stage-1 checkpoints)
  appendix/         supporting/secondary result sets
```

The `thesis_main/{gqa,vqav2,highres}` split preserves each track's internal layout so the ledger paths and
the reader scripts in `src/` map by a simple prefix. **Main vs appendix classification is carried by the
ledger's Placement column, not by physical folder.** See `INDEX.md` and
`docs/FINAL_REPOSITORY_CLEANUP_REPORT.md`.
