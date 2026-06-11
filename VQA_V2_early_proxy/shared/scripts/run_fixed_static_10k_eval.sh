#!/usr/bin/env bash
# Phase 1 / paper-quality — Corrected static generation eval on full 10K val2014.
# No --max-samples override so config's max_val_samples=10000 applies.
# Results written to VQA_V2/outputs/static_k{K}_fixed/generation_eval_10k.json
#
# Usage (from repo root):
#   CUDA_VISIBLE_DEVICES=1 bash VQA_V2_early_proxy/shared/scripts/run_fixed_static_10k_eval.sh

set -euo pipefail

GPU="${CUDA_VISIBLE_DEVICES:-1}"
EVAL_ENV="vlm_env"

KS=(64 128 144 192 288 432)

for K in "${KS[@]}"; do
    CONFIG="VQA_V2/static/llava_static_clsattn_150k_10k_fullvocab_k${K}.yaml"
    OUTPUT_DIR="VQA_V2/outputs/static_k${K}_fixed"
    OUTPUT_PATH="${OUTPUT_DIR}/generation_eval_10k.json"

    mkdir -p "${OUTPUT_DIR}"

    echo ""
    echo "================================="
    echo "=  Static K=${K} fixed — 10K   ="
    echo "================================="

    CUDA_VISIBLE_DEVICES="${GPU}" conda run -n "${EVAL_ENV}" \
        python -m VQA_V2.shared.evaluation.generate_and_score \
            --config "${CONFIG}" \
            --model-type static \
            --output-path "${OUTPUT_PATH}" \
            --skip-classification \
            --log-every 500

    echo "[Done] K=${K} → ${OUTPUT_PATH}"
done

echo ""
echo "============================="
echo "=  All 10K evals complete.  ="
echo "============================="

# Summary table
conda run -n "${EVAL_ENV}" python -c "
import json

dense_1k  = 77.37
dense_10k = 76.59

print()
print('K    | 1K acc | 10K acc | vs dense-10K')
print('-----|--------|---------|-------------')
for k in [64, 128, 144, 192, 288, 432]:
    p1k  = f'VQA_V2/outputs/static_k{k}_fixed/generation_eval_1k.json'
    p10k = f'VQA_V2/outputs/static_k{k}_fixed/generation_eval_10k.json'
    try:
        a1  = json.load(open(p1k))['generation']['vqa_accuracy'] * 100
        a10 = json.load(open(p10k))['generation']['vqa_accuracy'] * 100
        diff = a10 - dense_10k
        print(f'{k:4d} | {a1:.2f}%  | {a10:.2f}%   | {diff:+.2f}pp')
    except Exception as e:
        print(f'{k:4d} | ERROR: {e}')
print(f' 576 | {dense_1k:.2f}%  | {dense_10k:.2f}%   | (dense baseline)')
"
