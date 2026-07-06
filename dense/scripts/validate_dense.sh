#!/usr/bin/env bash
# This wrapper is a method-facing entry point for the Dense baseline.
# The tested implementation remains in the shared evaluation core under src/ and scripts/;
# this script only calls it. It is safe by default: CPU-only, read-only over results,
# and it never overwrites a saved result.
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "== Dense baseline — CPU validation =="
echo "-- the shared evaluation core still imports and self-checks --"
python -m compileall -q src scripts
python -m src.common.test_evaluation_core

echo
echo "Dense final scores are recorded in:"
echo "  results/tables/final_dense_static_dynamic_comparison.md  (section A)"
echo "  dense/RESULTS.md"
echo
echo "The dense finals are the reference every other method loads; the WHICH and COUNT"
echo "audits below confirm those references resolve correctly:"
echo "  python -m scripts.validation.audit_dynamic_which"
echo "  python -m scripts.validation.audit_dynamic_count"
echo
echo "GPU RERUN (expensive; NOT needed unless reproducing the experiment):"
echo "  bash scripts/dense/run_llava_dense.sh   # LLaVA-1.5, GPU 0, vlm_env"
echo "  bash scripts/dense/run_qwen_dense.sh    # Qwen2.5-VL, GPU 1, qwen_env"
echo "  (the runners skip a cell whose result already exists — no silent overwrite)"
