#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/vlm-thesis"
mkdir -p logs/final_static

PY="/home/nafees/miniconda3/envs/vlm_env/bin/python"
MODEL="llava15"
GPU="0"

DATASETS=(gqa vqav2 textvqa docvqa)
BUDGETS=(15 25 35 50 75)

echo "===== LLaVA static final started $(date) ====="

for d in "${DATASETS[@]}"; do
  for p in "${BUDGETS[@]}"; do
    echo "[start] ${MODEL} ${d} p${p} $(date)"

    if [[ "$d" == "docvqa" ]]; then
      CUDA_VISIBLE_DEVICES="$GPU" HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
        "$PY" -m scripts.final_scope.run_static_eval \
          --model "$MODEL" \
          --dataset "$d" \
          --budget-pct "$p" \
          --full
    else
      CUDA_VISIBLE_DEVICES="$GPU" \
        "$PY" -m scripts.final_scope.run_static_eval \
          --model "$MODEL" \
          --dataset "$d" \
          --budget-pct "$p" \
          --full
    fi

    echo "[done] ${MODEL} ${d} p${p} $(date)"
  done
done

echo "===== LLaVA static final finished $(date) ====="
