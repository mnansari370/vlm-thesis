#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/vlm-thesis"

echo "===== Qwen dense final started $(date) ====="

CUDA_VISIBLE_DEVICES=1 /home/nafees/miniconda3/envs/qwen_env/bin/python -m scripts.final_scope.run_dense_pilot --model qwen25vl7b --dataset gqa --full
CUDA_VISIBLE_DEVICES=1 /home/nafees/miniconda3/envs/qwen_env/bin/python -m scripts.final_scope.run_dense_pilot --model qwen25vl7b --dataset vqav2 --full
CUDA_VISIBLE_DEVICES=1 /home/nafees/miniconda3/envs/qwen_env/bin/python -m scripts.final_scope.run_dense_pilot --model qwen25vl7b --dataset textvqa --full
CUDA_VISIBLE_DEVICES=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 /home/nafees/miniconda3/envs/qwen_env/bin/python -m scripts.final_scope.run_dense_pilot --model qwen25vl7b --dataset docvqa --full

echo "===== Qwen dense final finished $(date) ====="
