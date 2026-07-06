#!/usr/bin/env bash
# run_dynamic_which_missing_pilots.sh
# -----------------------------------------------------------------------------
# Launch the MISSING n=200 Dynamic-WHICH pilots so every model×dataset cell has
# at least meaningful (n=200) coverage before deciding on Dynamic-COUNT.
#
#   *** n=200 PILOTS ONLY — NO FULL RUNS. GPU JOB. Do NOT run without approval. ***
#
# Missing cells (from scripts/validation/audit_dynamic_which_coverage.py):
#   LLaVA-1.5  × TextVQA   (no Dynamic-WHICH outputs)
#   LLaVA-1.5  × DocVQA    (no Dynamic-WHICH outputs)
#   Qwen2.5-VL × GQA       (only an n20 smoke)
#   Qwen2.5-VL × VQAv2     (no Dynamic-WHICH outputs)
#
# Each cell needs its matching dense FINAL + static FINAL (same variant) to exist;
# those are already present for all four cells.
#
# Envs / GPUs (run each model under its own stack):
#   LLaVA : /home/nafees/miniconda3/envs/vlm_env/bin/python  on GPU 0
#   Qwen  : /home/nafees/miniconda3/envs/qwen_env/bin/python on GPU 1
# TextVQA defaults to OCR-on; DocVQA defaults to instruction-on (offline env set).
#
# Run:  bash scripts/dynamic_which/run_dynamic_which_pilots.sh
# -----------------------------------------------------------------------------
set -uo pipefail

REPO="/home/nafees/vlm-thesis"
cd "$REPO"

LLAVA_PY="/home/nafees/miniconda3/envs/vlm_env/bin/python"
QWEN_PY="/home/nafees/miniconda3/envs/qwen_env/bin/python"
LOGDIR="logs/dynamic_which_missing_pilots"
N=200
mkdir -p "$LOGDIR"

# run <python> <gpu> <log-tag> <extra-env> -- <cli args...>
run() {
  local py="$1"; local gpu="$2"; local tag="$3"; local extra_env="$4"; shift 4
  local log="$LOGDIR/${tag}.log"
  echo "===== START ${tag} (GPU ${gpu}) $(date -Is) ====="
  # NOTE: n=200 pilot only (never --full). ${extra_env} is intentionally unquoted (env assignments).
  env ${extra_env} CUDA_VISIBLE_DEVICES="${gpu}" "${py}" \
      -m scripts.dynamic_which.run_dynamic_which --n "${N}" "$@" 2>&1 | tee "${log}"
  local rc=${PIPESTATUS[0]}
  echo "===== DONE ${tag} rc=${rc} $(date -Is) ====="
  echo
}

echo "########## Dynamic-WHICH MISSING n=${N} PILOTS — START $(date -Is) ##########"
echo "Logs -> ${LOGDIR}/"
echo

# ── LLaVA-1.5 (GPU 0, vlm_env) ── TextVQA (OCR-on) + DocVQA (instruction-on, offline) ──
for b in 25 35 50; do
  run "$LLAVA_PY" 0 "llava15_textvqa_textsim_p${b}"          "" \
      --model llava15 --dataset textvqa --budget-pct "$b" --selector textsim
  run "$LLAVA_PY" 0 "llava15_textvqa_textsim_cls_mix_p${b}"  "" \
      --model llava15 --dataset textvqa --budget-pct "$b" --selector textsim_cls_mix
done

for b in 25 35 50; do
  run "$LLAVA_PY" 0 "llava15_docvqa_textsim_p${b}"           "HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1" \
      --model llava15 --dataset docvqa --budget-pct "$b" --selector textsim
  run "$LLAVA_PY" 0 "llava15_docvqa_textsim_cls_mix_p${b}"   "HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1" \
      --model llava15 --dataset docvqa --budget-pct "$b" --selector textsim_cls_mix
done

# ── Qwen2.5-VL (GPU 1, qwen_env) ── GQA + VQAv2 ──
for b in 25 35; do
  run "$QWEN_PY" 1 "qwen25vl7b_gqa_textsim_p${b}"            "" \
      --model qwen25vl7b --dataset gqa --budget-pct "$b" --selector textsim
  run "$QWEN_PY" 1 "qwen25vl7b_gqa_textsim_norm_mix_p${b}"   "" \
      --model qwen25vl7b --dataset gqa --budget-pct "$b" --selector textsim_norm_mix
done

for b in 25 35; do
  run "$QWEN_PY" 1 "qwen25vl7b_vqav2_textsim_p${b}"          "" \
      --model qwen25vl7b --dataset vqav2 --budget-pct "$b" --selector textsim
  run "$QWEN_PY" 1 "qwen25vl7b_vqav2_textsim_norm_mix_p${b}" "" \
      --model qwen25vl7b --dataset vqav2 --budget-pct "$b" --selector textsim_norm_mix
done

echo "########## Dynamic-WHICH MISSING n=${N} PILOTS — ALL DONE $(date -Is) ##########"
echo "Next: python -m scripts.validation.audit_dynamic_which_coverage  (re-audit coverage)"
