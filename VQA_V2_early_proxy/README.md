# `VQA_V2_early_proxy/` — RETIRED classification-head proxy (do not cite for claims)

The project's first VQAv2 track (60k train / 10k val / top-3500 answer vocabulary / trained MLP
classification head). It was an **early efficiency proxy**, superseded by the generation-protocol
track in `../VQA_V2/`. Its headline numbers (dense 66.73%, static K288 61.44%, dynamic 62.35%,
"dynamic +0.91pp vs static K288") are **retired** — see the protocol note in
`../docs/vqav2_findings.md` for why (unmatched token budgets; dense ~12pp below published; the gap
reverses to a +0.05pp wash under generation eval).

## What it is still good for

1. **The exp1–5 ablation story** (the thesis's "why" chapter): raising question-weight α hurts;
   the 7-type budget controller hits targets precisely but doesn't raise accuracy; mean-pool beats
   last-token; spatial/complex-reasoning failures are a backbone ceiling, not a budget problem.
   The preserved numbers live in `results_summary/` (metrics.json + config_resolved.json of each of
   the 10 deleted 60k runs) and `../data/budget_oracle/` (per-type diagnostics).
2. **History/reference** for how the framework evolved.

## Layout

```
dense/ static/ dynamic/   model + training engine + config + HPC run script, merged per variant
shared/                   datasets, evaluation, utils, old diagnostic scripts (budget oracle, diag1/2)
results_summary/          preserved metrics/configs of the deleted 60k training runs (exp1-5 etc.)
```

Imports/paths were refreshed to this layout (all 59 modules verified to import); training would run as
`python -m VQA_V2_early_proxy.<variant>.train_vqa --config VQA_V2_early_proxy/<variant>/<cfg>.yaml`,
writing to `VQA_V2_early_proxy/outputs/` (git-ignored). Checkpoints were deleted 2026-06-11 — the run
numbers live in `results_summary/`. This track stays retired; keep it for the ablation story only.
