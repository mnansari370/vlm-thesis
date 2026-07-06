# Static — commands

All commands run from the repository root. The CPU commands are safe and read-only over results.

## CPU validation (safe, no GPU, no overwrite)

```bash
# method-facing wrapper: imports the core, runs the self-checks, prints where results live
bash static/scripts/validate_static.sh

# or directly:
python -m compileall -q src scripts
python -m src.common.test_evaluation_core
python -m scripts.validation.audit_dynamic_which   # confirms static references resolve
```

## Table generation (CPU, from saved results)

```bash
python -m scripts.tables.make_final_thesis_tables
```

## GPU rerun — EXPENSIVE, NOT needed unless reproducing the experiment

Each static cell requires its **matching dense final** to already exist. The runners skip any cell
whose result is already saved, so they do not overwrite; move a file aside to force a fresh run.

```bash
# LLaVA-1.5 cls_attn (GPU 0, vlm_env): 4 datasets x 5 budgets
bash scripts/static/run_llava_static.sh

# Qwen2.5-VL norm (GPU 1, qwen_env): 4 datasets x 5 budgets
bash scripts/static/run_qwen_static.sh

# a single cell:
CUDA_VISIBLE_DEVICES=0 python -m scripts.static.run_static \
    --model llava15 --dataset gqa --budget-pct 25 --full
```
