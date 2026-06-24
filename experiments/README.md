# `experiments/` — future per-method experiment entrypoints (SCAFFOLD ONLY)

> **Empty scaffold.** The active experiments still live inside the historical folders (`GQA/`, `VQA_V2/`,
> `v2/`). Nothing has been moved here yet.

Target layout (filled during Phase 3E migration):

```
experiments/
  diagnostic_foundation/             # the frozen low-res diagnostic (the "v1" work, by function)
  dense_baselines/  static_pruning/  dynamic_budget/
  question_conditioned_selection/    # frozen probes (negative) + mid-layer selector (positive)
  distillation/                      # the cheap student
  ablations/
```

`diagnostic_foundation/` is where the historical "v1" experiments land — named by their role, not "v1".
See `docs/METHOD_BASED_MIGRATION_PLAN.md`.
