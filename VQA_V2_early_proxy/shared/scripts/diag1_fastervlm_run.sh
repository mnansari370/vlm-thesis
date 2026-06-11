#!/usr/bin/env bash
# Diagnostic 1 — Run FasterVLM on our val2014 subset and score.
#
# Usage (from repo root):
#   CUDA_VISIBLE_DEVICES=1 bash scripts/diag1_fastervlm_run.sh 128
#   CUDA_VISIBLE_DEVICES=1 bash scripts/diag1_fastervlm_run.sh 288
#   CUDA_VISIBLE_DEVICES=1 bash scripts/diag1_fastervlm_run.sh 576   # no pruning
#
# Requires: scripts/diag1_data/ created by diag1_convert_data.py
#           fastervlm_eval conda env created by diag1_fastervlm_setup.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

TOKEN="${1:-128}"
REPO="/home/nafees/FasterVLM"
DATA_DIR="scripts/diag1_data"
ANSWERS_DIR="${DATA_DIR}/answers/k${TOKEN}"
MODEL="liuhaotian/llava-v1.5-7b"   # downloads ~14GB on first run

mkdir -p "${ANSWERS_DIR}"

echo "=== FasterVLM eval: K=${TOKEN} on val2014 ==="
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
conda run -n fastervlm_eval \
    python -W ignore -m llava.eval.model_vqa_loader \
        --model-path "${MODEL}" \
        --question-file "${DATA_DIR}/val2014_questions_short.jsonl" \
        --image-folder "data/vqav2/val2014" \
        --answers-file "${ANSWERS_DIR}/answers.jsonl" \
        --num-chunks 1 \
        --chunk-idx 0 \
        --visual-token-num "${TOKEN}" \
        --temperature 0 \
        --conv-mode vicuna_v1

echo ""
echo "=== Scoring against val2014 GT ==="
conda run -n fastervlm_eval python VQA_V2_early_proxy/shared/scripts/diag1_score.py \
    --answers "${ANSWERS_DIR}/answers.jsonl" \
    --gt      "${DATA_DIR}/val2014_gt.json" \
    --k       "${TOKEN}" \
    --output  "${ANSWERS_DIR}/scores.json"
