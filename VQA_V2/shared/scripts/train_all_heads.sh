#!/bin/bash
# Section 2: Train all 7 answer heads to convergence on GPU0.
# Uses early stopping (--patience 8) and runs up to 60 epochs.
# Dense head trains ~30 epochs; static heads converge faster.
# Outputs: VQA_V2/outputs/{dense,static_k*}_head_v1/
set -e

CACHE_DIR=/home/nafees/vlm-thesis/VQA_V2/feature_cache
OUT_DIR=/home/nafees/vlm-thesis/VQA_V2/outputs
LOG=/tmp/train_all_heads.log
cd /home/nafees/vlm-thesis

run_head() {
    local NAME=$1
    local TRAIN_CACHE=$2
    local VAL_CACHE=$3
    local CONFIG=$4
    local OUT=$5

    echo "" | tee -a $LOG
    echo "========== Training: $NAME ==========" | tee -a $LOG
    echo "  train cache: $TRAIN_CACHE" | tee -a $LOG
    echo "  output: $OUT" | tee -a $LOG

    CUDA_VISIBLE_DEVICES=0 conda run -n vlm_env python \
        -m VQA_V2.shared.training.cached.train_cached \
        --train-cache "$TRAIN_CACHE" \
        --val-cache   "$VAL_CACHE" \
        --config      "$CONFIG" \
        --output-dir  "$OUT" \
        --max-epochs  60 \
        --patience    8 \
        --log-every   500 \
        >> $LOG 2>&1

    echo "[Done] $NAME — see $OUT/metrics.json" | tee -a $LOG
}

echo "[train_all_heads] Started $(date)" | tee $LOG

# Dense
run_head "dense" \
    "$CACHE_DIR/dense/train" \
    "$CACHE_DIR/dense/val" \
    "VQA_V2/dense/llava_dense_150k_10k_fullvocab.yaml" \
    "$OUT_DIR/dense_head_v1"

# Static K=64
run_head "static_k64" \
    "$CACHE_DIR/static_k64/train" \
    "$CACHE_DIR/static_k64/val" \
    "VQA_V2/static/llava_static_clsattn_150k_10k_fullvocab_k64.yaml" \
    "$OUT_DIR/static_k64_head_v1"

# Static K=128
run_head "static_k128" \
    "$CACHE_DIR/static_k128/train" \
    "$CACHE_DIR/static_k128/val" \
    "VQA_V2/static/llava_static_clsattn_150k_10k_fullvocab_k128.yaml" \
    "$OUT_DIR/static_k128_head_v1"

# Static K=144
run_head "static_k144" \
    "$CACHE_DIR/static_k144/train" \
    "$CACHE_DIR/static_k144/val" \
    "VQA_V2/static/llava_static_clsattn_150k_10k_fullvocab_k144.yaml" \
    "$OUT_DIR/static_k144_head_v1"

# Static K=192
run_head "static_k192" \
    "$CACHE_DIR/static_k192/train" \
    "$CACHE_DIR/static_k192/val" \
    "VQA_V2/static/llava_static_clsattn_150k_10k_fullvocab_k192.yaml" \
    "$OUT_DIR/static_k192_head_v1"

# Static K=288
run_head "static_k288" \
    "$CACHE_DIR/static_k288/train" \
    "$CACHE_DIR/static_k288/val" \
    "VQA_V2/static/llava_static_clsattn_150k_10k_fullvocab_k288.yaml" \
    "$OUT_DIR/static_k288_head_v1"

# Static K=432
run_head "static_k432" \
    "$CACHE_DIR/static_k432/train" \
    "$CACHE_DIR/static_k432/val" \
    "VQA_V2/static/llava_static_clsattn_150k_10k_fullvocab_k432.yaml" \
    "$OUT_DIR/static_k432_head_v1"

echo "" | tee -a $LOG
echo "[train_all_heads] ALL DONE $(date)" | tee -a $LOG
echo "Results:" | tee -a $LOG
for DIR in dense_head_v1 static_k64_head_v1 static_k128_head_v1 static_k144_head_v1 \
           static_k192_head_v1 static_k288_head_v1 static_k432_head_v1; do
    if [ -f "$OUT_DIR/$DIR/metrics.json" ]; then
        ACC=$(python3 -c "import json; d=json.load(open('$OUT_DIR/$DIR/metrics.json')); print(f\"{d['best_val_vqa_accuracy']:.4f} @ ep{d['best_epoch']}\")" 2>/dev/null || echo "parse error")
        echo "  $DIR: $ACC" | tee -a $LOG
    fi
done
