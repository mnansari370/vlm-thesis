#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/vlm-thesis"
mkdir -p logs/final_static

PY="/home/nafees/miniconda3/envs/qwen_env/bin/python"
MODEL="qwen25vl7b"
GPU="1"
SELECTOR="norm"

DATASETS=(gqa vqav2 textvqa docvqa)
BUDGETS=(15 25 35 50 75)

echo "===== Qwen static final started $(date) ====="

for d in "${DATASETS[@]}"; do
  for p in "${BUDGETS[@]}"; do
    echo "[start] ${MODEL} ${d} ${SELECTOR} p${p} $(date)"

    if [[ "$d" == "docvqa" ]]; then
      CUDA_VISIBLE_DEVICES="$GPU" HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
        "$PY" -m scripts.final_scope.run_static_eval \
          --model "$MODEL" \
          --dataset "$d" \
          --budget-pct "$p" \
          --selector "$SELECTOR" \
          --full
    else
      CUDA_VISIBLE_DEVICES="$GPU" \
        "$PY" -m scripts.final_scope.run_static_eval \
          --model "$MODEL" \
          --dataset "$d" \
          --budget-pct "$p" \
          --selector "$SELECTOR" \
          --full
    fi

    echo "[done] ${MODEL} ${d} ${SELECTOR} p${p} $(date)"
  done
done

echo "===== Qwen static final finished $(date) ====="
