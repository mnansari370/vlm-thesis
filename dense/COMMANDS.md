# Dense — commands

All commands run from the repository root. The CPU commands are safe and read-only over results.

## CPU validation (safe, no GPU, no overwrite)

```bash
# method-facing wrapper: imports the core, runs the self-checks, prints where results live
bash dense/scripts/validate_dense.sh

# or the underlying checks directly:
python -m compileall -q src scripts
python -m src.common.test_evaluation_core
```

The dense finals are the reference the other methods load; these audits confirm the references
resolve over the full matrix:

```bash
python -m scripts.validation.audit_dynamic_which
python -m scripts.validation.audit_dynamic_count
```

## Table generation (CPU, regenerates committed summaries from saved results)

```bash
python -m scripts.tables.make_final_thesis_tables
```

Writes to `results/tables/` (rereads saved aggregates; does not touch per-sample data).

## GPU rerun — EXPENSIVE, NOT needed unless reproducing the experiment

The dense finals are already computed and saved. Rerunning requires the two model environments and
GPUs, and takes hours. The runners **skip any cell whose result already exists**, so they do not
silently overwrite; to force a fresh run, move the existing file aside first.

```bash
# LLaVA-1.5 (GPU 0, vlm_env): GQA, VQAv2, TextVQA, DocVQA dense finals
bash scripts/dense/run_llava_dense.sh

# Qwen2.5-VL (GPU 1, qwen_env): GQA, VQAv2, TextVQA, DocVQA dense finals
bash scripts/dense/run_qwen_dense.sh

# a single pilot cell (n=200) for a quick check:
CUDA_VISIBLE_DEVICES=0 python -m scripts.dense.run_dense --model llava15 --dataset gqa --n 200
```
