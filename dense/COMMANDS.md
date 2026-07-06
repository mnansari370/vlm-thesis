# Dense — commands

All commands run from the repository root. The CPU commands are safe and read-only over results.

## CPU validation (safe, no GPU, no overwrite)

```bash
# method-facing wrapper: imports the core, runs the self-checks, prints where results live
bash dense/scripts/validate_dense.sh

# or the underlying checks directly:
python -m compileall -q src scripts
python -m src.final_scope.test_final_scope
```

The dense finals are the reference the other methods load; these audits confirm the references
resolve over the full matrix:

```bash
python -m scripts.final_scope.audit_dynamic_which_full_final_matrix
python -m scripts.final_scope.audit_dynamic_count_full_matrix
```

## Table generation (CPU, regenerates committed summaries from saved results)

```bash
python -m scripts.final_scope.make_final_thesis_tables
```

Writes to `results/final_scope/tables/` (rereads saved aggregates; does not touch per-sample data).

## GPU rerun — EXPENSIVE, NOT needed unless reproducing the experiment

The dense finals are already computed and saved. Rerunning requires the two model environments and
GPUs, and takes hours. The runners **skip any cell whose result already exists**, so they do not
silently overwrite; to force a fresh run, move the existing file aside first.

```bash
# LLaVA-1.5 (GPU 0, vlm_env): GQA, VQAv2, TextVQA, DocVQA dense finals
bash scripts/final_scope/run_llava_dense_final.sh

# Qwen2.5-VL (GPU 1, qwen_env): GQA, VQAv2, TextVQA, DocVQA dense finals
bash scripts/final_scope/run_qwen_dense_final.sh

# a single pilot cell (n=200) for a quick check:
CUDA_VISIBLE_DEVICES=0 python -m scripts.final_scope.run_dense_pilot --model llava15 --dataset gqa --n 200
```
