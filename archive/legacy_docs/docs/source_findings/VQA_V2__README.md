# `VQA_V2/` — canonical VQAv2 track (generation protocol)

The VQAv2 half of the thesis's two-track evidence (the GQA half is `../GQA/`). LLaVA-1.5-7B fully
frozen; pruning = CLS-attention top-K; the trained pieces are a lightweight answer head and the
`BudgetController` (dynamic). **All thesis/paper claims from this track use the GENERATION protocol**
(greedy decode + VQA consensus on val2014 10K, expand2square, bs=1).

**Headline result:** dynamic type-adaptive budgeting (avg K≈264) = **75.76%** vs static uniform K=265
= **75.71%** → **+0.05pp at matched cost — a wash**. Full findings: `../docs/vqav2_findings.md`.
Dense baseline 76.44% (≈ published; FasterVLM-calibrated −0.13pp).

## Layout

```
dense/      LlavaDenseVQAModel + answer head + configs (150k/443k fullvocab)
static/     CLS-attention top-K wrapper + selector + per-K configs (K=64..432)
dynamic/    token_scorer, budget_controller, selector, wrapper, train_dynamic.py + configs
shared/     datasets (VQAv2 loader), evaluation (generate_and_score, per_type_accuracy,
            instance_headroom, cascade_pass/analyze, make_figures), utils, scripts, experiments
outputs/    (git-ignored) the results that back ../docs/vqav2_findings.md:
            static_baseline_locked_expand2square.json   ← the locked static curve
            static_k*_{pertype,matched,fixed,pad}/      ← generation evals per K
            dynamic_150k_clsonly/                       ← trained dynamic model + eval + summary
            cascade/, figures/, gate_real/              ← cascade analysis, figs 1-3, budget-gate proof
```

## Status

**Runnable.** All imports/paths were refreshed to this layout (2026-06-11): every module is
`python -m VQA_V2.<...>` from the repo root (all 33 modules verified to import; the analysis
figures regenerate from saved outputs). Common entry points:

```bash
# canonical generation eval (any checkpoint type)
python -m VQA_V2.shared.evaluation.generate_and_score --config VQA_V2/dense/llava_dense_150k_10k_fullvocab.yaml \
    --checkpoint VQA_V2/outputs/<run>/best_model.pt --model-type dense --output-path <out>.json
# dynamic training        python -m VQA_V2.dynamic.train_dynamic --config VQA_V2/dynamic/<cfg>.yaml ...
# analyses (no GPU)       python -m VQA_V2.shared.evaluation.{instance_headroom,per_type_accuracy,cascade_analyze,make_figures}
```

The numbers are final and frozen; the GPU phase is closed. The retired classification-head sibling
track is `../VQA_V2_early_proxy/`.
