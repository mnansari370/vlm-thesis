#!/usr/bin/env bash
#SBATCH --job-name=vqa2_mlp
#SBATCH --partition=gpu
#SBATCH --account=students
#SBATCH --qos=low
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Usage:
#   sbatch --job-name=vqa2_mlp_dense \
#          vqa_v2/experiments/hpc/train_mlp_cached.sh dense
#
#   sbatch --job-name=vqa2_mlp_static_k288 \
#          vqa_v2/experiments/hpc/train_mlp_cached.sh static 288

set -euo pipefail

MODEL_TYPE="${1:?Usage: $0 <model_type: dense|static> [keep_tokens]}"
KEEP_TOKENS="${2:-}"

PROJECT_ROOT="${HOME}/vlm-thesis"
cd "${PROJECT_ROOT}"

echo "============================================================"
echo "vqa_v2 train_mlp_cached — ${MODEL_TYPE}${KEEP_TOKENS:+ K=${KEEP_TOKENS}}"
echo "Job ID: ${SLURM_JOB_ID:-N/A} | Start: $(date)"
echo "============================================================"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate vlm_env
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
nvidia-smi || true
mkdir -p VQA_V2/outputs

if [ "${MODEL_TYPE}" = "dense" ]; then
    TRAIN_CACHE="VQA_V2/feature_cache/dense/train"
    VAL_CACHE="VQA_V2/feature_cache/dense/val"
    CONFIG="VQA_V2/dense/llava_dense_150k_10k_fullvocab.yaml"
    OUTPUT_DIR="VQA_V2/outputs/dense_v1"
elif [ "${MODEL_TYPE}" = "static" ]; then
    [ -n "${KEEP_TOKENS}" ] || { echo "keep_tokens required for static"; exit 1; }
    TRAIN_CACHE="VQA_V2/feature_cache/static_k${KEEP_TOKENS}/train"
    VAL_CACHE="VQA_V2/feature_cache/static_k${KEEP_TOKENS}/val"
    CONFIG="VQA_V2/static/llava_static_clsattn_150k_10k_fullvocab_k${KEEP_TOKENS}.yaml"
    OUTPUT_DIR="VQA_V2/outputs/static_k${KEEP_TOKENS}_v1"
else
    echo "Unknown model_type: ${MODEL_TYPE}"; exit 1
fi

python -u -m VQA_V2.shared.training.cached.train_cached \
    --train-cache "${TRAIN_CACHE}" \
    --val-cache   "${VAL_CACHE}" \
    --config      "${CONFIG}" \
    --output-dir  "${OUTPUT_DIR}" \
    --batch-size 128 \
    --eval-batch-size 256 \
    --log-every 200

echo "============================================================"
echo "train_mlp_cached done: $(date)"
echo "============================================================"
