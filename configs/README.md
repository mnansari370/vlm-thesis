# `configs/` — method-based YAML configs

```
configs/
  dense/            full-576-token baselines (LLaVA-1.5 VQAv2)
  static/           CLS-attention top-K pruning, per-K configs (the static frontier)
  dynamic_budget/   question-conditioned scorer + budget controller / gate
  stage1_elastic.yaml   elastic-LoRA Stage-1 backbone (high-res track)
```

Dataset paths inside the configs point at the git-ignored, flat `data/` tree (unchanged by the migration).
Output directories were repointed from the old `*/outputs/` to `results/`. These configs were moved out of
the historical `VQA_V2/{dense,dynamic,static}/` and `v2/configs/` folders (2026-06-24). See
`docs/FINAL_REPOSITORY_CLEANUP_REPORT.md`.
