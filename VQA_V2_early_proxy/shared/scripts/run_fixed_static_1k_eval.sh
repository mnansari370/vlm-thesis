#!/usr/bin/env bash
# Phase 1 Section 4.3 — Corrected static generation eval (1K val2014)
# Runs all 6 K values sequentially on GPU1 after the ordering fix.
# Skip-classification so no checkpoint is needed.
# Results written to vqa_v2/outputs/static_k{K}_fixed/generation_eval_1k.json
#
# Usage (from repo root):
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_fixed_static_1k_eval.sh

set -euo pipefail

GPU="${CUDA_VISIBLE_DEVICES:-1}"
EVAL_ENV="vlm_env"
MAX_SAMPLES=1000

KS=(64 128 144 192 288 432)

for K in "${KS[@]}"; do
    CONFIG="vqa_v2/VQA_V2_early_proxy/static/llava_static_clsattn_150k_10k_fullvocab_k${K}.yaml"
    OUTPUT_DIR="vqa_v2/outputs/static_k${K}_fixed"
    OUTPUT_PATH="${OUTPUT_DIR}/generation_eval_1k.json"

    mkdir -p "${OUTPUT_DIR}"

    echo ""
    echo "=============================="
    echo "=  Static K=${K} (fixed)     ="
    echo "=============================="

    CUDA_VISIBLE_DEVICES="${GPU}" conda run -n "${EVAL_ENV}" \
        python -m vqa_v2.evaluation.generate_and_score \
            --config "${CONFIG}" \
            --model-type static \
            --output-path "${OUTPUT_PATH}" \
            --max-samples "${MAX_SAMPLES}" \
            --skip-classification \
            --log-every 200

    echo "[Done] K=${K} → ${OUTPUT_PATH}"
done

echo ""
echo "=============================="
echo "=  All K values complete.    ="
echo "=============================="

# Print summary table
conda run -n "${EVAL_ENV}" python -c "
import json

results = []
for k in [64, 128, 144, 192, 288, 432]:
    path = f'vqa_v2/outputs/static_k{k}_fixed/generation_eval_1k.json'
    try:
        d = json.load(open(path))
        acc = d['generation']['vqa_accuracy']
        results.append((k, acc))
        print(f'  K={k:4d}: {acc*100:.2f}%')
    except Exception as e:
        print(f'  K={k:4d}: ERROR — {e}')

if len(results) >= 2:
    # Check monotonicity
    sorted_res = sorted(results, key=lambda x: x[0])
    mono = all(sorted_res[i][1] <= sorted_res[i+1][1] for i in range(len(sorted_res)-1))
    print()
    print(f'Monotonic (higher K → higher acc): {mono}')
    if not mono:
        print('WARNING: non-monotonic curve — re-check fix implementation')
"
