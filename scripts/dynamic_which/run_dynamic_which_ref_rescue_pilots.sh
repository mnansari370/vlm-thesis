#!/usr/bin/env bash
# run_dynamic_which_ref_rescue_pilots.sh
# -----------------------------------------------------------------------------
# Launch CLEAN-ROOM Dynamic-WHICH REFERENCE rescue pilots to test whether better selectors
# (saliency_mix_ref, static_guarded_textsim_ref) recover the cells where the original textsim
# failed vs static.
#
#   *** n=200 PILOTS ONLY — NO FULL RUNS. GPU JOB. Do NOT run without approval. ***
#
# Outputs are named dynamic_which_ref_pilot_* → they CANNOT overwrite existing dynamic_which
# results. Same manifests / prompts / scorers / dense+static references as the original path.
#
# Envs / GPUs:
#   LLaVA : /home/nafees/miniconda3/envs/vlm_env/bin/python  on GPU 0
#   Qwen  : /home/nafees/miniconda3/envs/qwen_env/bin/python on GPU 1
# DocVQA runs use HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1.
#
# Run:  bash scripts/dynamic_which/run_dynamic_which_ref_rescue_pilots.sh
# -----------------------------------------------------------------------------
set -uo pipefail

REPO="/home/nafees/vlm-thesis"
cd "$REPO"

LLAVA_PY="/home/nafees/miniconda3/envs/vlm_env/bin/python"
QWEN_PY="/home/nafees/miniconda3/envs/qwen_env/bin/python"
LOGDIR="logs/dynamic_which_ref_rescue"
N=200
SELECTORS=("textsim_ref" "saliency_mix_ref" "static_guarded_textsim_ref")
mkdir -p "$LOGDIR"

# run <python> <gpu> <tag> <extra-env> -- <cli args...>
run() {
  local py="$1"; local gpu="$2"; local tag="$3"; local extra_env="$4"; shift 4
  local log="$LOGDIR/${tag}.log"
  echo "===== START ${tag} (GPU ${gpu}) $(date -Is) ====="
  # n=200 pilot only (the ref CLI has no --full). ${extra_env} intentionally unquoted.
  env ${extra_env} CUDA_VISIBLE_DEVICES="${gpu}" "${py}" \
      -m scripts.dynamic_which.run_dynamic_which_ref --n "${N}" "$@" 2>&1 | tee "${log}"
  local rc=${PIPESTATUS[0]}
  echo "===== DONE ${tag} rc=${rc} $(date -Is) ====="
  echo
}

echo "########## Dynamic-WHICH REF rescue n=${N} pilots — START $(date -Is) ##########"
echo "Logs -> ${LOGDIR}/    Selectors: ${SELECTORS[*]}"
echo

# ── LLaVA-1.5 (GPU 0, vlm_env) ──
for sel in "${SELECTORS[@]}"; do
  for b in 25 35 50; do
    run "$LLAVA_PY" 0 "llava15_textvqa_${sel}_p${b}" "" \
        --model llava15 --dataset textvqa --budget-pct "$b" --selector "$sel"
  done
  for b in 25 35 50; do
    run "$LLAVA_PY" 0 "llava15_docvqa_${sel}_p${b}" "HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1" \
        --model llava15 --dataset docvqa --budget-pct "$b" --selector "$sel"
  done
  for b in 25 35; do
    run "$LLAVA_PY" 0 "llava15_vqav2_${sel}_p${b}" "" \
        --model llava15 --dataset vqav2 --budget-pct "$b" --selector "$sel"
  done
done

# ── Qwen2.5-VL (GPU 1, qwen_env) ──
for sel in "${SELECTORS[@]}"; do
  for b in 25 35; do
    run "$QWEN_PY" 1 "qwen25vl7b_gqa_${sel}_p${b}" "" \
        --model qwen25vl7b --dataset gqa --budget-pct "$b" --selector "$sel"
  done
  for b in 25 35; do
    run "$QWEN_PY" 1 "qwen25vl7b_vqav2_${sel}_p${b}" "" \
        --model qwen25vl7b --dataset vqav2 --budget-pct "$b" --selector "$sel"
  done
  for b in 25 35 50; do
    run "$QWEN_PY" 1 "qwen25vl7b_docvqa_${sel}_p${b}" "HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1" \
        --model qwen25vl7b --dataset docvqa --budget-pct "$b" --selector "$sel"
  done
done

echo "########## Dynamic-WHICH REF rescue n=${N} pilots — ALL DONE $(date -Is) ##########"
echo "Next: python -m scripts.dynamic_which.compare_current_vs_ref"
