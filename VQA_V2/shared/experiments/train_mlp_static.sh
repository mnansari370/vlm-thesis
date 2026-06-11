#!/usr/bin/env bash
# Usage: bash vqa_v2/experiments/train_mlp_static.sh 288
# Requires: feature cache at VQA_V2/feature_cache/static_k${K}/{train,val}/
K="${1:?Usage: $0 <K>}"

set -euo pipefail
PROJECT_ROOT="${HOME}/vlm-thesis"
cd "${PROJECT_ROOT}"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate vlm_env
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python -u -m VQA_V2.shared.training.cached.train_cached \
    --train-cache VQA_V2/feature_cache/static_k${K}/train \
    --val-cache   VQA_V2/feature_cache/static_k${K}/val \
    --config      VQA_V2/static/llava_static_clsattn_150k_10k_fullvocab_k${K}.yaml \
    --output-dir  VQA_V2/outputs/static_k${K}_v1 \
    --batch-size 128 \
    --eval-batch-size 256 \
    --log-every 200

echo "=== Done K=${K}: $(date) ==="
