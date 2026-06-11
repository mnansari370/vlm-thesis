#!/usr/bin/env bash
# Master vonasah pipeline — runs all short jobs (≤1h) on GPU 1.
#
# Phase A: Static val caches (runs now, ~1h each × 6 = ~6h total)
# Phase B: Dense MLP training + gen eval 1K (auto-triggers when dense/train cache finishes)
# Phase C: Static MLP training + gen eval 1K per K (auto-triggers as each HPC train cache lands)
#
# Run in background:
#   nohup bash VQA_V2/shared/experiments/vonasah_pipeline.sh \
#       > VQA_V2/logs/vonasah_pipeline.log 2>&1 &
#
# Monitor:
#   tail -f VQA_V2/logs/vonasah_pipeline.log

set -euo pipefail

PYTHON=/home/nafees/miniconda3/envs/vlm_env/bin/python
PROJECT_ROOT="${HOME}/vlm-thesis"
CACHE_DIR="${PROJECT_ROOT}/VQA_V2/feature_cache"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

cache_ready() {
    local dir="$1"
    [ -f "${dir}/pooled_features.npy" ] && [ -f "${dir}/answer_labels.npy" ]
}

# ── Phase A: Static val caches ──────────────────────────────────────────────

log "=== Phase A: Static val caches ==="
for K in 64 128 144 192 288 432; do
    if cache_ready "${CACHE_DIR}/static_k${K}/val"; then
        log "static_k${K}/val already cached — skipping"
        continue
    fi

    log "--- static_k${K}/val: starting ---"
    ${PYTHON} -u VQA_V2/shared/scripts/cache_features.py \
        --model-type static \
        --keep-tokens "${K}" \
        --split val \
        --config "VQA_V2/static/llava_static_clsattn_150k_10k_fullvocab_k${K}.yaml" \
        --cache-dir "${CACHE_DIR}" \
        --log-every 100
    log "--- static_k${K}/val: done ---"
done

log "=== Phase A complete: all 6 static val caches ready ==="

# ── Phase B: Dense MLP training + gen eval 1K ───────────────────────────────

log "=== Phase B: Waiting for dense/train cache to complete ==="
while ! cache_ready "${CACHE_DIR}/dense/train"; do
    log "dense/train not ready yet — sleeping 5 min..."
    sleep 300
done
log "dense/train cache ready."

log "--- Dense MLP training ---"
${PYTHON} -u -m VQA_V2.shared.training.cached.train_cached \
    --train-cache "${CACHE_DIR}/dense/train" \
    --val-cache   "${CACHE_DIR}/dense/val" \
    --config      "VQA_V2/dense/llava_dense_150k_10k_fullvocab.yaml" \
    --output-dir  "VQA_V2/outputs/dense_v1" \
    --batch-size 128 \
    --eval-batch-size 256 \
    --log-every 200
log "--- Dense MLP training done ---"

log "--- Dense generation eval (1K samples) ---"
${PYTHON} -u -m VQA_V2.shared.evaluation.generate_and_score \
    --config      "VQA_V2/dense/llava_dense_150k_10k_fullvocab.yaml" \
    --checkpoint  "VQA_V2/outputs/dense_v1/best_model.pt" \
    --model-type  dense \
    --output-path "VQA_V2/outputs/dense_v1/generation_eval_1k.json" \
    --max-samples 1000 \
    --skip-classification \
    --log-every 100
log "--- Dense generation eval done ---"
log "=== Phase B complete ==="

# ── Phase C: Static MLP training + gen eval 1K (per K) ─────────────────────

log "=== Phase C: Static MLP + gen eval (waiting for HPC train caches) ==="
for K in 64 128 144 192 288 432; do
    log "--- Waiting for static_k${K}/train cache ---"
    while ! cache_ready "${CACHE_DIR}/static_k${K}/train"; do
        log "static_k${K}/train not ready — sleeping 5 min..."
        sleep 300
    done
    log "static_k${K}/train cache ready."

    log "--- Static K=${K} MLP training ---"
    ${PYTHON} -u -m VQA_V2.shared.training.cached.train_cached \
        --train-cache "${CACHE_DIR}/static_k${K}/train" \
        --val-cache   "${CACHE_DIR}/static_k${K}/val" \
        --config      "VQA_V2/static/llava_static_clsattn_150k_10k_fullvocab_k${K}.yaml" \
        --output-dir  "VQA_V2/outputs/static_k${K}_v1" \
        --batch-size 128 \
        --eval-batch-size 256 \
        --log-every 200
    log "--- Static K=${K} MLP training done ---"

    log "--- Static K=${K} generation eval (1K samples) ---"
    ${PYTHON} -u -m VQA_V2.shared.evaluation.generate_and_score \
        --config      "VQA_V2/static/llava_static_clsattn_150k_10k_fullvocab_k${K}.yaml" \
        --checkpoint  "VQA_V2/outputs/static_k${K}_v1/best_model.pt" \
        --model-type  static \
        --output-path "VQA_V2/outputs/static_k${K}_v1/generation_eval_1k.json" \
        --max-samples 1000 \
        --skip-classification \
        --log-every 100
    log "--- Static K=${K} generation eval done ---"
done

log "=== Phase C complete ==="
log "=== vonasah_pipeline.sh DONE — all short jobs finished ==="
