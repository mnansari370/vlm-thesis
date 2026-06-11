#!/usr/bin/env bash
#SBATCH --job-name=vqa2_cache_static_k144_train
#SBATCH --partition=gpu
#SBATCH --account=students
#SBATCH --qos=low
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=VQA_V2/logs/%x_%j.out
#SBATCH --error=VQA_V2/logs/%x_%j.err

set -euo pipefail
PROJECT_ROOT="${HOME}/vlm-thesis"
cd "${PROJECT_ROOT}"
echo "=== cache_static_k144_train === | $(date)"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate vlm_env
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python -u VQA_V2/shared/scripts/cache_features.py \
    --model-type static \
    --keep-tokens 144 \
    --split train \
    --config VQA_V2/static/llava_static_clsattn_150k_10k_fullvocab_k144.yaml \
    --cache-dir VQA_V2/feature_cache \
    --log-every 500

echo "=== Done: $(date) ==="
