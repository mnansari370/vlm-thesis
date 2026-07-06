# Static — commands

All commands run from the repository root. The CPU commands are safe and read-only over results.

## CPU validation (safe, no GPU, no overwrite)

```bash
# method-facing wrapper: imports the core, runs the self-checks, prints where results live
bash static/scripts/validate_static.sh

# or directly:
python -m compileall -q src scripts
python -m src.final_scope.test_final_scope
python -m scripts.final_scope.audit_dynamic_which_full_final_matrix   # confirms static references resolve
```

## Table generation (CPU, from saved results)

```bash
python -m scripts.final_scope.make_final_thesis_tables
```

## GPU rerun — EXPENSIVE, NOT needed unless reproducing the experiment

Each static cell requires its **matching dense final** to already exist. The runners skip any cell
whose result is already saved, so they do not overwrite; move a file aside to force a fresh run.

```bash
# LLaVA-1.5 cls_attn (GPU 0, vlm_env): 4 datasets x 5 budgets
bash scripts/final_scope/run_llava_static_final.sh

# Qwen2.5-VL norm (GPU 1, qwen_env): 4 datasets x 5 budgets
bash scripts/final_scope/run_qwen_static_final.sh

# a single cell:
CUDA_VISIBLE_DEVICES=0 python -m scripts.final_scope.run_static_eval \
    --model llava15 --dataset gqa --budget-pct 25 --full
```
