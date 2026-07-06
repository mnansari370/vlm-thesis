# Dynamic-WHICH — commands

All commands run from the repository root. The CPU commands are safe and read-only over results.

## CPU validation (safe, no GPU, no overwrite)

```bash
# method-facing wrapper: self-checks + the WHICH validators/audits, prints where results live
bash dynamic_which/scripts/validate_dynamic_which.sh

# or directly:
python -m compileall -q src scripts
python -m src.final_scope.test_final_scope
python -m scripts.final_scope.validate_dynamic_which_final          # the 5 Qwen x TextVQA finals
python -m scripts.final_scope.audit_dynamic_which_full_final_matrix  # all 40 cells complete/valid
```

## Table generation (CPU, from saved results)

```bash
python -m scripts.final_scope.make_dynamic_which_textsim_full_final_summary
python -m scripts.final_scope.make_dynamic_which_dense_static_dynamic_table
python -m scripts.final_scope.make_final_thesis_tables
```

## GPU rerun — EXPENSIVE, NOT needed unless reproducing the experiment

Each WHICH cell needs its **matching dense final and same-budget static final** to already exist. The
launcher skips any cell whose result is saved (no silent overwrite).

```bash
# the full textsim matrix (35 remaining full finals; Qwen x TextVQA already complete):
bash scripts/final_scope/run_dynamic_which_textsim_full_missing.sh

# a single cell:
CUDA_VISIBLE_DEVICES=1 python -m scripts.final_scope.run_dynamic_which_eval \
    --model qwen25vl7b --dataset textvqa --budget-pct 25 --selector textsim --full

# independent clean-room validation of the headline win (n=200 pilots, no overwrite of finals):
bash scripts/final_scope/run_qwen_textvqa_ref_validation.sh
```
