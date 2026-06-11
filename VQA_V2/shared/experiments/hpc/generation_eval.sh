#!/usr/bin/env bash
#SBATCH --job-name=vqa2_gen_eval
#SBATCH --partition=gpu
#SBATCH --account=students
#SBATCH --qos=low
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Usage:
#   sbatch --job-name=vqa2_gen_eval_dense \
#          VQA_V2/shared/experiments/hpc/generation_eval.sh dense \
#          VQA_V2/outputs/dense_v1/best_model.pt \
#          VQA_V2/outputs/dense_v1/generation_eval_full.json
#
#   sbatch --job-name=vqa2_gen_eval_static_k288 \
#          VQA_V2/shared/experiments/hpc/generation_eval.sh static \
#          VQA_V2/outputs/static_k288_v1/best_model.pt \
#          VQA_V2/outputs/static_k288_v1/generation_eval_full.json \
#          288

set -euo pipefail

MODEL_TYPE="${1:?Usage: $0 <model_type> <checkpoint> <output_path> [keep_tokens]}"
CHECKPOINT="${2:?}"
OUTPUT_PATH="${3:?}"
KEEP_TOKENS="${4:-}"

PROJECT_ROOT="${HOME}/vlm-thesis"
cd "${PROJECT_ROOT}"

echo "============================================================"
echo "vqa_v2 generation_eval — ${MODEL_TYPE}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Job ID: ${SLURM_JOB_ID:-N/A} | Start: $(date)"
echo "============================================================"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate vlm_env
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
nvidia-smi || true

if [ "${MODEL_TYPE}" = "dense" ]; then
    CONFIG="VQA_V2/dense/llava_dense_150k_10k_fullvocab.yaml"
elif [ "${MODEL_TYPE}" = "static" ]; then
    [ -n "${KEEP_TOKENS}" ] || { echo "keep_tokens required for static"; exit 1; }
    CONFIG="VQA_V2/static/llava_static_clsattn_150k_10k_fullvocab_k${KEEP_TOKENS}.yaml"
elif [ "${MODEL_TYPE}" = "dynamic" ]; then
    CONFIG="VQA_V2/dynamic/llava_dynamic_150k_10k_fullvocab.yaml"
else
    echo "Unknown model_type: ${MODEL_TYPE}"; exit 1
fi

python -u -m VQA_V2.shared.evaluation.generate_and_score \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --model-type "${MODEL_TYPE}" \
    --output-path "${OUTPUT_PATH}" \
    --skip-classification \
    --log-every 50

echo "============================================================"
echo "generation_eval done: $(date)"
echo "============================================================"
