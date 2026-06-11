#!/usr/bin/env bash
# Runs cache_dense_val then cache_dense_train sequentially on GPU 1.
# Usage: nohup bash VQA_V2/shared/experiments/run_dense_sequential.sh > VQA_V2/logs/dense_sequential.log 2>&1 &
set -euo pipefail

PYTHON=/home/nafees/miniconda3/envs/vlm_env/bin/python
PROJECT_ROOT="${HOME}/vlm-thesis"
cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "=== Dense sequential pipeline ==="
echo "Start: $(date)"

echo ""
echo "--- Step 1: cache_dense_val (10K samples, ~1h) ---"
${PYTHON} -u VQA_V2/shared/scripts/cache_features.py \
    --model-type dense \
    --split val \
    --config VQA_V2/dense/llava_dense_150k_10k_fullvocab.yaml \
    --cache-dir VQA_V2/feature_cache \
    --log-every 100
echo "cache_dense_val done: $(date)"

echo ""
echo "--- Step 2: cache_dense_train (150K samples, ~12-18h) ---"
${PYTHON} -u VQA_V2/shared/scripts/cache_features.py \
    --model-type dense \
    --split train \
    --config VQA_V2/dense/llava_dense_150k_10k_fullvocab.yaml \
    --cache-dir VQA_V2/feature_cache \
    --log-every 500
echo "cache_dense_train done: $(date)"

echo ""
echo "=== All dense caches complete: $(date) ==="
