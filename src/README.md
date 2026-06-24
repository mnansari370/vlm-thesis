# `src/` — future method-based source tree (SCAFFOLD ONLY)

> **Empty scaffold.** No code lives here yet. The active, runnable code still lives in the historical
> folders `GQA/`, `VQA_V2/`, and `v2/`. Do **not** treat anything under `src/` as source of truth until
> the method-based migration (Phase 3E) is actually executed.

This is the target layout for when `GQA/ VQA_V2/ v2/` are reorganized by method (see
`docs/METHOD_BASED_MIGRATION_PLAN.md` for the file-by-file mapping and batch plan):

```
src/
  data/        # dataset loaders (GQA, VQAv2, LLaVA-mix)
  metrics/     # scorers: official_score, m4c_evaluator, textvqa/pope/docvqa/chartqa
  models/
    dense/  static/  dynamic_budget/  question_conditioned_selection/  distillation/
  pruning/
    static/  dynamic_budget/  question_conditioned_selection/
  evaluation/
    vqa/  gqa/  textvqa/  docvqa/  chartqa/  pope/  scienceqa/
  analysis/    # FLOPs, latency, oracle decomposition, figures
  utils/       # config/seed/logger/io/checkpoint/device (one copy)
```

- Exact numbers/evidence remain in `docs/THESIS_EVIDENCE_LEDGER.md`.
- No results have been moved; result files still live in `outputs/`, `VQA_V2/outputs/`, `v2/outputs/`.
