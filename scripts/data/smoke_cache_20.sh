#!/bin/bash
# Section 2 smoke test: 20-sample cache build for K=288 (both splits), NaN check.
# Usage: bash scripts/data/smoke_cache_20.sh [gpu_id]
# If clean, prints "SMOKE TEST PASSED — proceed to full K=288 cache."
set -e

GPU=${1:-0}
SMOKE_CACHE_DIR=feature_cache_smoke_k288
CONFIG=configs/static/llava_static_clsattn_150k_10k_fullvocab_k288.yaml

echo "[Smoke] GPU=${GPU}  cache_dir=${SMOKE_CACHE_DIR}"
echo "[Smoke] Using 20 samples per split"

# Clean any previous smoke run
rm -rf "$SMOKE_CACHE_DIR"

CUDA_VISIBLE_DEVICES=$GPU conda run -n vlm_env python scripts/data/cache_features.py \
    --model-type static \
    --keep-tokens 288 \
    --split train \
    --config "$CONFIG" \
    --cache-dir "$SMOKE_CACHE_DIR" \
    --max-samples 20 \
    --log-every 10

CUDA_VISIBLE_DEVICES=$GPU conda run -n vlm_env python scripts/data/cache_features.py \
    --model-type static \
    --keep-tokens 288 \
    --split val \
    --config "$CONFIG" \
    --cache-dir "$SMOKE_CACHE_DIR" \
    --max-samples 20 \
    --log-every 10

echo ""
echo "[Smoke] NaN check..."
conda run -n vlm_env python scripts/data/check_cache_nan.py \
    --cache-root "$SMOKE_CACHE_DIR"

echo ""
echo "SMOKE TEST PASSED — proceed to full K=288 cache."
