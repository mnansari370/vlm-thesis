# `experiments/` — index only

This folder is a thin pointer, not a code location. Every final-scope experiment is a module under
`src/` launched by a script under the `scripts/` method folders, for example:

```
python -m scripts.dense.run_dense        --model llava15    --dataset gqa     --full
python -m scripts.static.run_static        --model qwen25vl7b --dataset docvqa  --budget-pct 25 --full
python -m scripts.dynamic_which.run_dynamic_which --model qwen25vl7b --dataset textvqa --budget-pct 25 --selector textsim --full
```

The experiment-to-thesis mapping and all final numbers live in `results/tables/`
(start with `final_thesis_results_summary.md`); the method documentation lives in `docs/`
(local-only). Historical track folders and legacy harnesses are preserved under `archive/`.
