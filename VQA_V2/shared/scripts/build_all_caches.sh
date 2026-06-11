#!/bin/bash
# Section 2.2(c): Build remaining caches (K=64,128,144,192,432 + dense) after K=288 is verified clean.
# Split across both GPUs. Run ONLY after K=288 cache passes NaN check.
# Usage: bash VQA_V2/shared/scripts/build_all_caches.sh
set -e

CACHE_DIR=VQA_V2/feature_cache
LOG0=/tmp/cache_gpu0_rest.log
LOG1=/tmp/cache_gpu1_rest.log

# GPU0: K=64, K=144, K=432, dense
build_gpu0() {
    for K in 64 144 432; do
        CONFIG="VQA_V2/static/llava_static_clsattn_150k_10k_fullvocab_k${K}.yaml"
        echo "[GPU0] Caching K=${K} train..." | tee -a $LOG0
        CUDA_VISIBLE_DEVICES=0 conda run -n vlm_env python VQA_V2/shared/scripts/cache_features.py \
            --model-type static --keep-tokens $K --split train --config "$CONFIG" \
            --cache-dir "$CACHE_DIR" 2>&1 | tee -a $LOG0
        echo "[GPU0] Caching K=${K} val..." | tee -a $LOG0
        CUDA_VISIBLE_DEVICES=0 conda run -n vlm_env python VQA_V2/shared/scripts/cache_features.py \
            --model-type static --keep-tokens $K --split val --config "$CONFIG" \
            --cache-dir "$CACHE_DIR" 2>&1 | tee -a $LOG0
    done
    echo "[GPU0] Caching dense train..." | tee -a $LOG0
    CUDA_VISIBLE_DEVICES=0 conda run -n vlm_env python VQA_V2/shared/scripts/cache_features.py \
        --model-type dense --split train \
        --config VQA_V2/dense/llava_dense_150k_10k_fullvocab.yaml \
        --cache-dir "$CACHE_DIR" 2>&1 | tee -a $LOG0
    echo "[GPU0] Caching dense val..." | tee -a $LOG0
    CUDA_VISIBLE_DEVICES=0 conda run -n vlm_env python VQA_V2/shared/scripts/cache_features.py \
        --model-type dense --split val \
        --config VQA_V2/dense/llava_dense_150k_10k_fullvocab.yaml \
        --cache-dir "$CACHE_DIR" 2>&1 | tee -a $LOG0
    echo "[GPU0] ALL DONE" | tee -a $LOG0
}

# GPU1: K=128, K=192
build_gpu1() {
    for K in 128 192; do
        CONFIG="VQA_V2/static/llava_static_clsattn_150k_10k_fullvocab_k${K}.yaml"
        echo "[GPU1] Caching K=${K} train..." | tee -a $LOG1
        CUDA_VISIBLE_DEVICES=1 conda run -n vlm_env python VQA_V2/shared/scripts/cache_features.py \
            --model-type static --keep-tokens $K --split train --config "$CONFIG" \
            --cache-dir "$CACHE_DIR" 2>&1 | tee -a $LOG1
        echo "[GPU1] Caching K=${K} val..." | tee -a $LOG1
        CUDA_VISIBLE_DEVICES=1 conda run -n vlm_env python VQA_V2/shared/scripts/cache_features.py \
            --model-type static --keep-tokens $K --split val --config "$CONFIG" \
            --cache-dir "$CACHE_DIR" 2>&1 | tee -a $LOG1
    done
    echo "[GPU1] ALL DONE" | tee -a $LOG1
}

export -f build_gpu0
export -f build_gpu1

build_gpu0 &
PID0=$!
build_gpu1 &
PID1=$!

wait $PID0
wait $PID1

echo ""
echo "All caches built. Running full NaN check..."
conda run -n vlm_env python VQA_V2/shared/scripts/check_cache_nan.py --cache-root "$CACHE_DIR"
