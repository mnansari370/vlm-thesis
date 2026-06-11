#!/usr/bin/env bash
#SBATCH --job-name=vqa2_cache
#SBATCH --partition=gpu
#SBATCH --account=students
#SBATCH --qos=low
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Usage:
#   sbatch --job-name=vqa2_cache_dense_train \
#          VQA_V2/shared/experiments/hpc/cache_features.sh dense train
#
#   sbatch --job-name=vqa2_cache_static_k288_train \
#          VQA_V2/shared/experiments/hpc/cache_features.sh static train 288

set -euo pipefail

MODEL_TYPE="${1:?Usage: $0 <model_type: dense|static> <split: train|val> [keep_tokens]}"
SPLIT="${2:?Usage: $0 <model_type> <split: train|val> [keep_tokens]}"
KEEP_TOKENS="${3:-}"

PROJECT_ROOT="${HOME}/vlm-thesis"

echo "============================================================"
echo "vqa_v2 cache_features — IRIS HPC"
echo "model_type: ${MODEL_TYPE}  split: ${SPLIT}  keep_tokens: ${KEEP_TOKENS:-N/A}"
echo "Hostname:   $(hostname)"
echo "Job ID:     ${SLURM_JOB_ID:-N/A}"
echo "Partition:  ${SLURM_JOB_PARTITION:-N/A}"
echo "Start:      $(date)"
echo "============================================================"

[ -d "${PROJECT_ROOT}" ] || { echo "Project root not found: ${PROJECT_ROOT}"; exit 1; }
cd "${PROJECT_ROOT}"

mkdir -p VQA_V2/logs VQA_V2/feature_cache

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate vlm_env

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

nvidia-smi || true

if [ "${MODEL_TYPE}" = "dense" ]; then
    CONFIG="VQA_V2/dense/llava_dense_150k_10k_fullvocab.yaml"
    python -u VQA_V2/shared/scripts/cache_features.py \
        --model-type dense \
        --split "${SPLIT}" \
        --config "${CONFIG}" \
        --cache-dir VQA_V2/feature_cache \
        --log-every 500

elif [ "${MODEL_TYPE}" = "static" ]; then
    [ -n "${KEEP_TOKENS}" ] || { echo "keep_tokens required for static"; exit 1; }
    CONFIG="VQA_V2/static/llava_static_clsattn_150k_10k_fullvocab_k${KEEP_TOKENS}.yaml"
    python -u VQA_V2/shared/scripts/cache_features.py \
        --model-type static \
        --keep-tokens "${KEEP_TOKENS}" \
        --split "${SPLIT}" \
        --config "${CONFIG}" \
        --cache-dir VQA_V2/feature_cache \
        --log-every 500
else
    echo "Unknown model_type: ${MODEL_TYPE}"; exit 1
fi

echo "============================================================"
echo "cache_features done: $(date)"
echo "============================================================"
