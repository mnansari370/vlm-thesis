#!/usr/bin/env bash
# run_qwen_textvqa_ref_validation.sh
# -----------------------------------------------------------------------------
# Independent clean-room validation of the HEADLINE positive result:
#   Qwen2.5-VL-7B × TextVQA × Dynamic-WHICH.
# Runs the clean-room REFERENCE selector `textsim_ref` at n=200 for every budget, to check
# that the current-textsim win reproduces under an independent implementation.
#
#   *** n=200 PILOTS ONLY — NO FULL RUNS. GPU JOB. Do NOT run without approval. ***
#
# Outputs (cannot overwrite existing dynamic_which outputs — distinct `_ref_` names):
#   results/runs/qwen25vl7b/textvqa/dynamic_which_ref_pilot_n200_textsim_ref_p{b}_ocr_on.json[l]
#
# Env / GPU:  CUDA_VISIBLE_DEVICES=1  /home/nafees/miniconda3/envs/qwen_env/bin/python
# TextVQA defaults to OCR-on (no --no-ocr). Logs tee'd under logs/dynamic_which_ref_textvqa_validation/.
#
# Run:  bash scripts/dynamic_which/run_qwen_textvqa_ref_validation.sh
# -----------------------------------------------------------------------------
set -uo pipefail

REPO="/home/nafees/vlm-thesis"
cd "$REPO"

QWEN_PY="/home/nafees/miniconda3/envs/qwen_env/bin/python"
LOGDIR="logs/dynamic_which_ref_textvqa_validation"
OUTDIR="results/runs/qwen25vl7b/textvqa"
N=200
BUDGETS=(15 25 35 50 75)
mkdir -p "$LOGDIR"

# ── no-overwrite pre-check: refuse to start if ANY target output already exists ──
overwrite=0
for b in "${BUDGETS[@]}"; do
  base="dynamic_which_ref_pilot_n200_textsim_ref_p${b}_ocr_on"
  if [[ -e "${OUTDIR}/${base}.json" || -e "${OUTDIR}/${base}.jsonl" ]]; then
    echo "[abort] would overwrite existing output: ${OUTDIR}/${base}.*"
    overwrite=1
  fi
done
if [[ "${overwrite}" -ne 0 ]]; then
  echo "Refusing to run — existing ref outputs present. Move/remove them first."
  exit 3
fi

# run <log-tag> -- <cli args...>
run() {
  local tag="$1"; shift
  local log="${LOGDIR}/${tag}.log"
  echo "===== START ${tag} (GPU 1) $(date -Is) ====="
  # n=200 pilot only; the ref CLI has no --full flag.
  CUDA_VISIBLE_DEVICES=1 "${QWEN_PY}" \
      -m scripts.dynamic_which.run_dynamic_which_ref --n "${N}" "$@" 2>&1 | tee "${log}"
  local rc=${PIPESTATUS[0]}
  echo "===== DONE ${tag} rc=${rc} $(date -Is) ====="
  echo
}

echo "########## Qwen TextVQA clean-room ref validation (textsim_ref, n=${N}) — START $(date -Is) ##########"
echo "Logs -> ${LOGDIR}/"
echo

for b in "${BUDGETS[@]}"; do
  run "qwen25vl7b_textvqa_textsim_ref_p${b}" \
      --model qwen25vl7b --dataset textvqa --budget-pct "${b}" --selector textsim_ref
done

echo "########## Qwen TextVQA clean-room ref validation — ALL DONE $(date -Is) ##########"
echo "Next: python -m scripts.dynamic_which.compare_qwen_textvqa_current_vs_ref"
