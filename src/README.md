# `src/` — shared evaluation core

> **For method-specific navigation, start from [`dense/`](../dense/), [`static/`](../static/),
> [`dynamic_which/`](../dynamic_which/), or [`dynamic_count/`](../dynamic_count/).** Each names the
> exact files below that implement its method. This tree is the shared, tested runtime.

All active code for the final thesis matrix (dense · static · Dynamic-WHICH · Dynamic-COUNT DC-D/DC-C
on LLaVA-1.5-7B + Qwen2.5-VL-7B × GQA/VQAv2/TextVQA/DocVQA), run from the repo root as
`python -m src.<...>`.

```
src/
  common/        the shared evaluation core (fairness toolkit): sample_ids (manifests+sha),
                 output_writer (unified schema), schema_validator (fairness gate),
                 token_flops (per-sample FLOPs), test_evaluation_core (CPU self-tests)
  dense/         evaluate_dense.py        — dense runner core (+ the adapters/prompts/scorers reused everywhere)
  static/        evaluate_static.py       — static runner core
  dynamic_which/ evaluate_dynamic_which.py — Dynamic-WHICH runner core
  dynamic_count/ evaluate_dynamic_count.py — Dynamic-COUNT runner core (DC-D, DC-C)
  pruning/
    dynamic_which/       textsim selectors (LLaVA + Qwen) — question-conditioned WHICH
    dynamic_which_ref/   independent clean-room re-implementation (validation only)
    dynamic_count/       probe/confidence/controllers + LLaVA/Qwen COUNT wrappers
    question_conditioned_selection/qwen_pruner.py   THE frozen Qwen engine
    static/visionzip.py  VisionZip baseline
  models/static/static.py   THE frozen LLaVA engine (physical prune-before-LLM; cls_attn etc.)
  data/          vqav2/vqav2_answers.py (active VQAv2 consensus normalizer);
                 gqa.py / docvqa.py are legacy loaders kept for reference (the runner-core
                 adapters in src/dense/evaluate_dense.py read the raw data directly)
  metrics/       canonical scorers: official_score (GQA exact), m4c_evaluator +
                 textvqa_score (M4C soft-acc), docvqa_score (ANLS), metrics.py;
                 pope/chartqa scorers are out-of-scope leftovers (unused)
  analysis/      flops.py (LLaVA) + qwen_flops.py — the FastV Eq.5 prefill formulas
                 that src/common/token_flops.py wraps
  utils/         small generic helpers (config/seed/logger/io/checkpoint/device)
  final_scope/   backward-compatibility shims only (see src/final_scope/README.md); the real
                 code lives in src/common and the src/{dense,static,dynamic_which,dynamic_count} packages
```

The two model engines are **frozen**: every method composes them; the Dynamic-COUNT probes are gated
on reproducing their outputs byte-for-byte. Do not modify them, and do not import from `archive/`.

Everything retired — the classification-head pipeline, the old BudgetController, the distillation
study, all pre-final evaluation harnesses (`src/evaluation/`), trainers (`src/training/`), the old
Qwen budget study, and out-of-scope models/datasets — was archived in cleanup Passes 1–2
(2026-07-05, commits `37d6e79` + `e9bc88b`); see `archive/migration_manifests/`.
