#!/usr/bin/env bash
# This wrapper is a method-facing entry point for Static pruning.
# The tested implementation remains in the shared evaluation core under src/ and scripts/;
# this script only calls it. It is safe by default: CPU-only, read-only over results,
# and it never overwrites a saved result.
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "== Static pruning — CPU validation =="
echo "-- the shared evaluation core still imports and self-checks --"
python -m compileall -q src scripts
python -m src.final_scope.test_final_scope

echo
echo "Static frontier scores are recorded in:"
echo "  results/final_scope/tables/final_dense_static_dynamic_comparison.md  (section B)"
echo "  results/final_scope/tables/static_final_summary.csv"
echo "  static/RESULTS.md"
echo
echo "The static finals are the floor the dynamic methods must beat; the WHICH audit"
echo "confirms each same-budget static reference resolves:"
echo "  python -m scripts.final_scope.audit_dynamic_which_full_final_matrix"
echo
echo "GPU RERUN (expensive; NOT needed unless reproducing the experiment):"
echo "  bash scripts/final_scope/run_llava_static_final.sh   # LLaVA-1.5 cls_attn, GPU 0"
echo "  bash scripts/final_scope/run_qwen_static_final.sh    # Qwen2.5-VL norm, GPU 1"
echo "  (each static cell requires its matching dense final; runners skip existing results)"
