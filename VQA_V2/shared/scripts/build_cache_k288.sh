#!/bin/bash
# Section 2.2(b): Full K=288 cache (train + val) with expand2square.
# Run ONLY after smoke_cache_20.sh passes NaN check.
# Usage: bash VQA_V2/shared/scripts/build_cache_k288.sh [gpu_id]
set -e

GPU=${1:-0}
CACHE_DIR=VQA_V2/feature_cache
CONFIG=VQA_V2/static/llava_static_clsattn_150k_10k_fullvocab_k288.yaml
LOG=/tmp/cache_k288_full.log

echo "[Cache K=288] GPU=${GPU}  Estimated disk: ~2.5GB train, ~0.16GB val" | tee $LOG

CUDA_VISIBLE_DEVICES=$GPU conda run -n vlm_env python VQA_V2/shared/scripts/cache_features.py \
    --model-type static \
    --keep-tokens 288 \
    --split train \
    --config "$CONFIG" \
    --cache-dir "$CACHE_DIR" \
    2>&1 | tee -a $LOG

CUDA_VISIBLE_DEVICES=$GPU conda run -n vlm_env python VQA_V2/shared/scripts/cache_features.py \
    --model-type static \
    --keep-tokens 288 \
    --split val \
    --config "$CONFIG" \
    --cache-dir "$CACHE_DIR" \
    2>&1 | tee -a $LOG

echo "[Cache K=288] NaN check..." | tee -a $LOG
conda run -n vlm_env python VQA_V2/shared/scripts/check_cache_nan.py \
    --cache-root "${CACHE_DIR}/static_k288" \
    2>&1 | tee -a $LOG

echo "[Cache K=288] DONE — check $LOG" | tee -a $LOG
