#!/usr/bin/env bash
# run_dynamic_count_probe_full_matrix.sh
# -----------------------------------------------------------------------------
# Stage 1: FULL-manifest Dynamic-COUNT PROBE passes with confidence recording, all 8 cells,
# probe budgets p15 AND p25, plus the Qwen×TextVQA textsim probes (COUNT-on-WHICH).
#
#   *** GPU JOB — do NOT run without approval. Uses BOTH GPUs in parallel:
#       GPU 0 = all LLaVA jobs (vlm_env)   GPU 1 = all Qwen jobs (qwen_env). ***
#
# Every probe has a BUILT-IN reproduction gate (predictions must equal the frozen finals);
# a job that fails the gate exits rc=2 and is reported in the failure summary — STOP AND DEBUG.
# Skip-safe: existing probe outputs are skipped, never overwritten (move files to redo).
#
# Run:   bash scripts/dynamic_count/run_probe_full_matrix.sh
# Logs:  logs/dynamic_count_probe/<tag>.log  (+ gpu0.log / gpu1.log stream logs)
# -----------------------------------------------------------------------------
set -uo pipefail
REPO="/home/nafees/vlm-thesis"
cd "$REPO"
LLAVA_PY="/home/nafees/miniconda3/envs/vlm_env/bin/python"
QWEN_PY="/home/nafees/miniconda3/envs/qwen_env/bin/python"
LOGDIR="logs/dynamic_count_probe"
OUTROOT="results/runs"
mkdir -p "$LOGDIR"

probe_base() {  # <dataset> <selector> <pct>
  local ds="$1" sel="$2" pct="$3"
  case "$ds" in
    textvqa) echo "dynamic_count_probe_${sel}_p${pct}_ocr_on" ;;
    docvqa)  echo "dynamic_count_probe_${sel}_p${pct}_instruction_on" ;;
    *)       echo "dynamic_count_probe_${sel}_p${pct}" ;;
  esac
}

# run_job <gpu> <python> <model> <dataset> <selector> <pct> <extra-env> <failfile>
run_job() {
  local gpu="$1" py="$2" model="$3" ds="$4" sel="$5" pct="$6" extra_env="$7" failfile="$8"
  local base tag log
  base="$(probe_base "$ds" "$sel" "$pct")"
  tag="${model}_${ds}_${sel}_p${pct}"
  if [[ -e "${OUTROOT}/${model}/${ds}/${base}.json" && -e "${OUTROOT}/${model}/${ds}/${base}.jsonl" ]]; then
    echo "===== SKIP ${tag} (exists) $(date -Is) ====="
    return 0
  fi
  log="${LOGDIR}/${tag}.log"
  echo "===== START ${tag} (GPU ${gpu}) $(date -Is) ====="
  env ${extra_env} CUDA_VISIBLE_DEVICES="${gpu}" "${py}" \
      -m scripts.dynamic_count.run_probe \
      --model "${model}" --dataset "${ds}" --selector "${sel}" --probe-pct "${pct}" --full \
      2>&1 | tee "${log}"
  local rc=${PIPESTATUS[0]}
  echo "===== DONE ${tag} rc=${rc} $(date -Is) ====="
  if [[ "${rc}" -ne 0 ]]; then echo "${tag} rc=${rc}" >> "${failfile}"; fi
}

FAIL0="${LOGDIR}/failed_gpu0.txt"; FAIL1="${LOGDIR}/failed_gpu1.txt"
: > "$FAIL0"; : > "$FAIL1"

gpu0_stream() {   # LLaVA, GPU 0
  for ds in gqa vqav2 textvqa; do
    for pct in 15 25; do run_job 0 "$LLAVA_PY" llava15 "$ds" cls_attn "$pct" "" "$FAIL0"; done
  done
  for pct in 15 25; do
    run_job 0 "$LLAVA_PY" llava15 docvqa cls_attn "$pct" "HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1" "$FAIL0"
  done
}

gpu1_stream() {   # Qwen, GPU 1
  for ds in gqa vqav2 textvqa; do
    for pct in 15 25; do run_job 1 "$QWEN_PY" qwen25vl7b "$ds" norm "$pct" "" "$FAIL1"; done
  done
  for pct in 15 25; do
    run_job 1 "$QWEN_PY" qwen25vl7b docvqa norm "$pct" "HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1" "$FAIL1"
  done
  # COUNT-on-WHICH probes (textsim, Qwen×TextVQA only)
  for pct in 15 25; do run_job 1 "$QWEN_PY" qwen25vl7b textvqa textsim "$pct" "" "$FAIL1"; done
}

echo "########## Dynamic-COUNT PROBE full matrix — START $(date -Is) ##########"
gpu0_stream > "${LOGDIR}/gpu0.log" 2>&1 &
P0=$!
gpu1_stream > "${LOGDIR}/gpu1.log" 2>&1 &
P1=$!
wait "$P0"; wait "$P1"

echo "########## Dynamic-COUNT PROBE full matrix — ALL DONE $(date -Is) ##########"
NFAIL=$(( $(wc -l < "$FAIL0") + $(wc -l < "$FAIL1") ))
if [[ "$NFAIL" -gt 0 ]]; then
  echo "FAILED JOBS (${NFAIL}) — reproduction-gate failures REQUIRE debugging before continuing:"
  cat "$FAIL0" "$FAIL1"
  exit 2
fi
echo "All probe jobs OK. Next: python -m scripts.dynamic_count.make_controller_calibration"
