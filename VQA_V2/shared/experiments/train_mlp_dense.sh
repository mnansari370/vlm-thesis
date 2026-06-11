#!/usr/bin/env bash
# Can run on dev GPU (~30 min) or Slurm short-queue.
# Requires: feature cache at VQA_V2/feature_cache/dense/{train,val}/
#
# Slurm header (uncomment to submit):
##SBATCH --job-name=vqa2_mlp_dense
##SBATCH --partition=gpu
##SBATCH --account=students
##SBATCH --qos=low
##SBATCH --gpus=1
##SBATCH --cpus-per-task=8
##SBATCH --time=0-02:00:00
##SBATCH --output=VQA_V2/logs/%x_%j.out
##SBATCH --error=VQA_V2/logs/%x_%j.err

set -euo pipefail
PROJECT_ROOT="${HOME}/vlm-thesis"
cd "${PROJECT_ROOT}"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate vlm_env
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

RUN_ID="${1:-dense_v1}"
python -u -m VQA_V2.shared.training.cached.train_cached \
    --train-cache VQA_V2/feature_cache/dense/train \
    --val-cache   VQA_V2/feature_cache/dense/val \
    --config      VQA_V2/dense/llava_dense_150k_10k_fullvocab.yaml \
    --output-dir  VQA_V2/outputs/${RUN_ID} \
    --batch-size 128 \
    --eval-batch-size 256 \
    --log-every 200

echo "=== Done: $(date) ==="
