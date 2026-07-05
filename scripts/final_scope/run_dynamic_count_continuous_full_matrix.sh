#!/usr/bin/env bash
# run_dynamic_count_continuous_full_matrix.sh
# -----------------------------------------------------------------------------
# Stage 4: DC-C (continuous Dynamic-COUNT, THE main method) for all 8 cells — real second passes
# at each sample's predicted integer K_i.
#
#   *** GPU JOB — do NOT run without approval. Uses BOTH GPUs in parallel:
#       GPU 0 = LLaVA (vlm_env)   GPU 1 = Qwen (qwen_env). ***
#
# Requires: probe outputs + calibration configs (run the probe launcher, then the calibration
# script, then compose DC-D first). Controllers: rule (primary) and ridge (secondary), at the
# config's default λ. Jobs: 16 pure (8 cells × 2 controllers) + 2 SEPARATE COUNT-on-WHICH
# (qwen×textvqa textsim × 2 controllers) = 18. Skip checks are SELECTOR-AWARE (pure vs CoW can
# never suppress each other). Skip-safe; never overwrites; failures summarized at the end.
#
# Run:   bash scripts/final_scope/run_dynamic_count_continuous_full_matrix.sh
# Logs:  logs/dynamic_count_dcc/<tag>.log (+ gpu0.log / gpu1.log)
# -----------------------------------------------------------------------------
set -uo pipefail
REPO="/home/nafees/vlm-thesis"
cd "$REPO"
LLAVA_PY="/home/nafees/miniconda3/envs/vlm_env/bin/python"
QWEN_PY="/home/nafees/miniconda3/envs/qwen_env/bin/python"
LOGDIR="logs/dynamic_count_dcc"
OUTROOT="results/final_scope"
CONTROLLERS=("rule" "ridge")
mkdir -p "$LOGDIR"

# run_job <gpu> <python> <model> <dataset> <controller> <skip-selector> <extra-flags> <extra-env> <failfile>
# skip-selector makes the skip check SELECTOR-AWARE so pure (cls_attn/norm) and COUNT-on-WHICH
# (textsim) outputs can never suppress each other's jobs.
run_job() {
  local gpu="$1" py="$2" model="$3" ds="$4" ctrl="$5" sel="$6" flags="$7" extra_env="$8" failfile="$9"
  local tag="${model}_${ds}_dcc_${ctrl}_${sel}"
  if compgen -G "${OUTROOT}/${model}/${ds}/dynamic_count_dcc_${ctrl}_${sel}_*lam*.json" > /dev/null; then
    echo "===== SKIP ${tag} (exists) $(date -Is) ====="
    return 0
  fi
  local log="${LOGDIR}/${tag}.log"
  echo "===== START ${tag} (GPU ${gpu}) $(date -Is) ====="
  env ${extra_env} CUDA_VISIBLE_DEVICES="${gpu}" "${py}" \
      -m scripts.final_scope.run_dynamic_count_continuous \
      --model "${model}" --dataset "${ds}" --controller "${ctrl}" ${flags} \
      2>&1 | tee "${log}"
  local rc=${PIPESTATUS[0]}
  echo "===== DONE ${tag} rc=${rc} $(date -Is) ====="
  if [[ "${rc}" -ne 0 ]]; then echo "${tag} rc=${rc}" >> "${failfile}"; fi
}

FAIL0="${LOGDIR}/failed_gpu0.txt"; FAIL1="${LOGDIR}/failed_gpu1.txt"
: > "$FAIL0"; : > "$FAIL1"

gpu0_stream() {   # LLaVA pure DC-C (selector cls_attn)
  for ctrl in "${CONTROLLERS[@]}"; do
    for ds in gqa vqav2 textvqa; do
      run_job 0 "$LLAVA_PY" llava15 "$ds" "$ctrl" cls_attn "" "" "$FAIL0"
    done
    run_job 0 "$LLAVA_PY" llava15 docvqa "$ctrl" cls_attn "" "HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1" "$FAIL0"
  done
}
gpu1_stream() {   # Qwen pure DC-C (selector norm) + SEPARATE COUNT-on-WHICH (textsim) at the end
  for ctrl in "${CONTROLLERS[@]}"; do
    for ds in gqa vqav2 textvqa; do
      run_job 1 "$QWEN_PY" qwen25vl7b "$ds" "$ctrl" norm "" "" "$FAIL1"
    done
    run_job 1 "$QWEN_PY" qwen25vl7b docvqa "$ctrl" norm "" "HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1" "$FAIL1"
  done
  for ctrl in "${CONTROLLERS[@]}"; do
    run_job 1 "$QWEN_PY" qwen25vl7b textvqa "$ctrl" textsim "--count-on-which" "" "$FAIL1"
  done
}

echo "########## DC-C full matrix — START $(date -Is) ##########"
gpu0_stream > "${LOGDIR}/gpu0.log" 2>&1 &
P0=$!
gpu1_stream > "${LOGDIR}/gpu1.log" 2>&1 &
P1=$!
wait "$P0"; wait "$P1"

echo "########## DC-C full matrix — ALL DONE $(date -Is) ##########"
NFAIL=$(( $(wc -l < "$FAIL0") + $(wc -l < "$FAIL1") ))
if [[ "$NFAIL" -gt 0 ]]; then
  echo "FAILED JOBS (${NFAIL}):"; cat "$FAIL0" "$FAIL1"; exit 2
fi
echo "All DC-C jobs OK. Next: python -m scripts.final_scope.make_dynamic_count_final_tables"
