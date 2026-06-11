#!/usr/bin/env bash
#SBATCH --job-name=llava_static
#SBATCH --partition=gpu
#SBATCH --account=students
#SBATCH --qos=low
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Default to the new static K288 experiment because it is usually the most stable fixed budget.
CONFIG="${1:-VQA_V2_early_proxy/static/llava_static_clsattn_60k_10k_top3500_k288.yaml}"
PROJECT_ROOT="${HOME}/vlm-thesis"

echo "============================================================"
echo "Starting SLURM static-pruning job"
echo "Hostname: $(hostname)"
echo "Config: ${CONFIG}"
echo "Project root: ${PROJECT_ROOT}"
echo "Start time: $(date)"
echo "SLURM job id: ${SLURM_JOB_ID:-N/A}"
echo "SLURM partition: ${SLURM_JOB_PARTITION:-N/A}"
echo "============================================================"

if [ ! -d "${PROJECT_ROOT}" ]; then
    echo "Project root not found: ${PROJECT_ROOT}"
    exit 1
fi

cd "${PROJECT_ROOT}"

if [ ! -f "${CONFIG}" ]; then
    echo "Config file not found: ${CONFIG}"
    exit 1
fi

mkdir -p logs
mkdir -p outputs

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate vlm_env

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "================ GPU INFO ================"
nvidia-smi || true
echo "=========================================="

# If fvcore is not already available in the environment,
# uncomment the next line once:
# pip install fvcore

python -u -m VQA_V2_early_proxy.static.train_vqa --config "${CONFIG}"

echo "============================================================"
echo "Static-pruning job finished"
echo "End time: $(date)"
echo "============================================================"