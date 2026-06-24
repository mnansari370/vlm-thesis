# `configs/` — future method-based configs (SCAFFOLD ONLY)

> **Empty scaffold.** The active YAML configs still live with the historical code (`VQA_V2/{dense,dynamic,
> static}/*.yaml`, `v2/configs/stage1_elastic.yaml`). Nothing has been moved here yet.

Target layout (filled during Phase 3E migration):

```
configs/{dense, static, dynamic_budget, question_conditioned_selection, distillation, evaluation}/
```

Do not point any run at these folders until the migration is executed. Dataset paths (`data/...`) will
**not** change — `data/` stays flat. See `docs/METHOD_BASED_MIGRATION_PLAN.md`.
