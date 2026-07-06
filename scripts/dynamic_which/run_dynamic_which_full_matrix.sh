#!/usr/bin/env bash
# run_dynamic_which_textsim_full_missing.sh
# -----------------------------------------------------------------------------
# Run the MISSING full-final Dynamic-WHICH textsim jobs so the WHICH matrix is complete on the same
# footing as dense/static (2 models × 4 datasets × 5 budgets, selector=textsim). This is a COMPLETE
# evaluation — negative full results are wanted too, not cherry-picked.
#
#   *** FULL manifest runs. GPU JOB. Do NOT run without approval. Run in staged batches. ***
#
# Safety: for each cell, if BOTH the final aggregate .json AND .jsonl already exist → SKIP (never
# overwrite). Qwen×TextVQA is intentionally EXCLUDED (its five full finals are already complete).
#
# Envs / GPUs:
#   LLaVA : CUDA_VISIBLE_DEVICES=0  /home/nafees/miniconda3/envs/vlm_env/bin/python
#   Qwen  : CUDA_VISIBLE_DEVICES=1  /home/nafees/miniconda3/envs/qwen_env/bin/python
# DocVQA runs set HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1.
#
# Logs: per-job tee under logs/dynamic_which_textsim_full_missing/<tag>.log, plus a self-contained
# master log (nohup-compatible). Recommended:
#   nohup bash scripts/dynamic_which/run_dynamic_which_full_matrix.sh \
#         > logs/dynamic_which_textsim_full_missing/nohup.out 2>&1 &
# -----------------------------------------------------------------------------
set -uo pipefail

REPO="/home/nafees/vlm-thesis"
cd "$REPO"

LLAVA_PY="/home/nafees/miniconda3/envs/vlm_env/bin/python"
QWEN_PY="/home/nafees/miniconda3/envs/qwen_env/bin/python"
LOGDIR="logs/dynamic_which_textsim_full_missing"
OUTROOT="results/runs"
BUDGETS=(15 25 35 50 75)
mkdir -p "$LOGDIR"

MASTER_LOG="${LOGDIR}/master.log"
# Duplicate ALL stdout/stderr to a self-contained master log (works with or without nohup redirect).
exec > >(tee -a "${MASTER_LOG}") 2>&1

# variant-aware final basename (matches run_dynamic_which_eval defaults: TextVQA OCR-on, DocVQA instr-on)
final_base() {
  local ds="$1" b="$2"
  case "$ds" in
    textvqa) echo "dynamic_which_final_textsim_p${b}_ocr_on" ;;
    docvqa)  echo "dynamic_which_final_textsim_p${b}_instruction_on" ;;
    *)       echo "dynamic_which_final_textsim_p${b}" ;;
  esac
}

# run_cell <model> <dataset> <gpu> <python> <extra-env>
run_cell() {
  local model="$1" ds="$2" gpu="$3" py="$4" extra_env="$5"
  for b in "${BUDGETS[@]}"; do
    local base outdir jj ll tag log
    base="$(final_base "$ds" "$b")"
    outdir="${OUTROOT}/${model}/${ds}"
    jj="${outdir}/${base}.json"
    ll="${outdir}/${base}.jsonl"
    tag="${model}_${ds}_textsim_p${b}"
    if [[ -e "$jj" && -e "$ll" ]]; then
      echo "===== SKIP ${tag} (final already exists: ${base}) $(date -Is) ====="
      continue
    fi
    log="${LOGDIR}/${tag}.log"
    echo "===== START ${tag} (GPU ${gpu}) $(date -Is) ====="
    # FULL manifest run of the current Dynamic-WHICH runner. ${extra_env} intentionally unquoted.
    env ${extra_env} CUDA_VISIBLE_DEVICES="${gpu}" "${py}" \
        -m scripts.dynamic_which.run_dynamic_which \
        --model "${model}" --dataset "${ds}" --budget-pct "${b}" --selector textsim --full \
        2>&1 | tee "${log}"
    local rc=${PIPESTATUS[0]}
    echo "===== DONE ${tag} rc=${rc} $(date -Is) ====="
    echo
  done
}

echo "########## Dynamic-WHICH textsim FULL missing finals — START $(date -Is) ##########"
echo "Master log -> ${MASTER_LOG} ; per-job logs -> ${LOGDIR}/<tag>.log"
echo

# ── LLaVA-1.5 (GPU 0, vlm_env): gqa, vqav2, textvqa (online) + docvqa (offline) ──
for ds in gqa vqav2 textvqa; do
  run_cell llava15 "$ds" 0 "$LLAVA_PY" ""
done
run_cell llava15 docvqa 0 "$LLAVA_PY" "HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1"

# ── Qwen2.5-VL (GPU 1, qwen_env): gqa, vqav2 (online) + docvqa (offline) — NOT textvqa (done) ──
for ds in gqa vqav2; do
  run_cell qwen25vl7b "$ds" 1 "$QWEN_PY" ""
done
run_cell qwen25vl7b docvqa 1 "$QWEN_PY" "HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1"

echo "########## Dynamic-WHICH textsim FULL missing finals — ALL DONE $(date -Is) ##########"
echo "Next: python -m scripts.validation.audit_dynamic_which"
echo "      python -m scripts.tables.make_dynamic_which_summary"
