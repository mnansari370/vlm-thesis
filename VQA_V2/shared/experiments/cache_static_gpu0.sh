#!/usr/bin/env bash
# Static train caches on GPU 0: K=128, K=144, K=432
# Runs alongside the dense gen eval (memory verified safe: 30GB < 49GB)
set -euo pipefail
PYTHON=/home/nafees/miniconda3/envs/vlm_env/bin/python
PROJECT_ROOT="${HOME}/vlm-thesis"
cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

for K in 128 144 432; do
    CACHE_DIR="VQA_V2/feature_cache/static_k${K}/train"
    if [ -f "${CACHE_DIR}/pooled_features.npy" ]; then
        log "K=${K} train already cached — skipping"
        continue
    fi
    log "=== Caching static K=${K} train (GPU 0) ==="
    ${PYTHON} -u VQA_V2/shared/scripts/cache_features.py \
        --model-type static \
        --keep-tokens "${K}" \
        --split train \
        --config "VQA_V2/static/llava_static_clsattn_150k_10k_fullvocab_k${K}.yaml" \
        --cache-dir VQA_V2/feature_cache \
        --log-every 1000
    log "=== K=${K} train cache done ==="

    log "--- Running MLP training for K=${K} ---"
    ${PYTHON} -u -m VQA_V2.shared.training.cached.train_cached \
        --train-cache "VQA_V2/feature_cache/static_k${K}/train" \
        --val-cache   "VQA_V2/feature_cache/static_k${K}/val" \
        --config      "VQA_V2/static/llava_static_clsattn_150k_10k_fullvocab_k${K}.yaml" \
        --output-dir  "VQA_V2/outputs/static_k${K}_v1" \
        --batch-size 128 --eval-batch-size 256 --log-every 300
    log "--- K=${K} MLP done ---"

    log "--- Gen eval 1K for K=${K} ---"
    ${PYTHON} -u -m VQA_V2.shared.evaluation.generate_and_score \
        --config      "VQA_V2/static/llava_static_clsattn_150k_10k_fullvocab_k${K}.yaml" \
        --checkpoint  "VQA_V2/outputs/static_k${K}_v1/best_model.pt" \
        --model-type  static \
        --output-path "VQA_V2/outputs/static_k${K}_v1/generation_eval_1k.json" \
        --max-samples 1000 --skip-classification --log-every 200
    log "--- K=${K} gen eval done ---"

    ${PYTHON} -c "
import json
d = json.load(open('VQA_V2/outputs/static_k${K}_v1/generation_eval_1k.json'))
print(f'  K=${K} gen acc (1K): {d[\"generation\"][\"vqa_accuracy\"]*100:.2f}%')
" 2>/dev/null
done

log "=== GPU 0 static pipeline COMPLETE ==="
