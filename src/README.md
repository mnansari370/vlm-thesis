# `src/` — method-based source tree

All project code lives here, organized by **method**, run from the repo root as `python -m src.<...>`.

```
src/
  data/        dataset loaders — gqa, vqav2/, docvqa, llava_mix
  metrics/     scorers — official/GQA, M4C/TextVQA, POPE, DocVQA ANLS, ChartQA
  models/      dense/  static/  dynamic_budget/  distillation/  elastic/
  pruning/     static/  dynamic_budget/  question_conditioned_selection/
  evaluation/  per task — gqa/  vqa/  textvqa/  docvqa/  pope/  scienceqa/
  analysis/    FLOPs, latency, oracle decomposition, figures, cascade sweep
  utils/       config / seed / logger / io / checkpoint / device (one copy)
  training/    training entry points (cached heads, dynamic, student, stage1)
```

This tree replaces the historical `GQA/`, `VQA_V2/`, and `v2/` track folders (migrated 2026-06-24;
see `docs/FINAL_REPOSITORY_CLEANUP_REPORT.md`). Imports use the `src.*` namespace throughout. Exact
numbers/evidence remain in `docs/THESIS_EVIDENCE_LEDGER.md`; results live in `results/`.
