# `experiments/` — index only

This folder is a thin pointer, not a code location. Every final-scope experiment is a module under
`src/` launched by a script under `scripts/final_scope/`, for example:

```
python -m scripts.final_scope.run_dense_pilot        --model llava15    --dataset gqa     --full
python -m scripts.final_scope.run_static_eval        --model qwen25vl7b --dataset docvqa  --budget-pct 25 --full
python -m scripts.final_scope.run_dynamic_which_eval --model qwen25vl7b --dataset textvqa --budget-pct 25 --selector textsim --full
```

The experiment-to-thesis mapping and all final numbers live in `results/final_scope/tables/`
(start with `final_thesis_results_summary.md`); the method documentation lives in `docs/`
(local-only). Historical track folders and legacy harnesses are preserved under `archive/`.
