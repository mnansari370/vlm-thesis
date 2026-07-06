#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/vlm-thesis"

echo "===== LLaVA dense final started $(date) ====="

CUDA_VISIBLE_DEVICES=0 /home/nafees/miniconda3/envs/vlm_env/bin/python -m scripts.dense.run_dense --model llava15 --dataset gqa --full
CUDA_VISIBLE_DEVICES=0 /home/nafees/miniconda3/envs/vlm_env/bin/python -m scripts.dense.run_dense --model llava15 --dataset vqav2 --full
CUDA_VISIBLE_DEVICES=0 /home/nafees/miniconda3/envs/vlm_env/bin/python -m scripts.dense.run_dense --model llava15 --dataset textvqa --full
CUDA_VISIBLE_DEVICES=0 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 /home/nafees/miniconda3/envs/vlm_env/bin/python -m scripts.dense.run_dense --model llava15 --dataset docvqa --full

echo "===== LLaVA dense final finished $(date) ====="
