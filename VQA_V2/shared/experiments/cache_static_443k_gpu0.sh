#!/usr/bin/env bash
# 443K static cache pipeline on GPU 0: K=128, K=144, K=432
# Run AFTER cache_static_gpu0.sh (150K) and cache_dense_443k_pipeline.sh complete.
# These are the FINAL PAPER numbers — same 443K training data as dense baseline.
set -euo pipefail

PYTHON=/home/nafees/miniconda3/envs/vlm_env/bin/python
PROJECT_ROOT="${HOME}/vlm-thesis"
cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

for K in 128 144 432; do
    CACHE_DIR="VQA_V2/feature_cache/static_443k_k${K}"
    CONFIG="VQA_V2/static/llava_static_clsattn_443k_10k_fullvocab_k${K}.yaml"
    OUTPUT_DIR="VQA_V2/outputs/static_443k_k${K}_v1"

    # ── Cache 443K train ───────────────────────────────────────────────────
    if [ -f "${CACHE_DIR}/train/pooled_features.npy" ]; then
        log "K=${K} 443K train cache exists — skipping"
    else
        log "=== Caching static K=${K} 443K train (GPU 0) ==="
        ${PYTHON} -u VQA_V2/shared/scripts/cache_features.py \
            --model-type static \
            --keep-tokens "${K}" \
            --split train \
            --config "${CONFIG}" \
            --cache-dir "${CACHE_DIR}" \
            --log-every 2000
        log "=== K=${K} 443K cache done ==="
    fi

    # ── MLP training (30 epochs) ───────────────────────────────────────────
    log "--- K=${K} MLP training (443K, 30ep) ---"
    mkdir -p "${OUTPUT_DIR}"
    ${PYTHON} -u -m VQA_V2.shared.training.cached.train_cached \
        --train-cache "${CACHE_DIR}/train" \
        --val-cache   "VQA_V2/feature_cache/static_k${K}/val" \
        --config      "${CONFIG}" \
        --output-dir  "${OUTPUT_DIR}" \
        --batch-size 128 --eval-batch-size 256 --log-every 500
    log "--- K=${K} MLP done ---"

    # ── Gen eval 1K ────────────────────────────────────────────────────────
    log "--- K=${K} gen eval 1K ---"
    ${PYTHON} -u -m VQA_V2.shared.evaluation.generate_and_score \
        --config      "${CONFIG}" \
        --checkpoint  "${OUTPUT_DIR}/best_model.pt" \
        --model-type  static \
        --output-path "${OUTPUT_DIR}/generation_eval_1k.json" \
        --max-samples 1000 --skip-classification --log-every 200

    ${PYTHON} -c "
import json
m = json.load(open('${OUTPUT_DIR}/metrics.json'))
g = json.load(open('${OUTPUT_DIR}/generation_eval_1k.json'))
print(f'  K=${K} 443K: cls={m[\"best_val_vqa_accuracy\"]*100:.2f}% | gen1K={g[\"generation\"][\"vqa_accuracy\"]*100:.2f}%')
" 2>/dev/null
    log "--- K=${K} done ---"
done

log "=== GPU 0 443K static pipeline COMPLETE ==="
