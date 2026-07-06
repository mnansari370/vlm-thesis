#!/usr/bin/env bash
# This wrapper is a method-facing entry point for Dynamic-WHICH.
# The tested implementation remains in the shared evaluation core under src/ and scripts/;
# this script only calls it. It is safe by default: CPU-only, read-only over results,
# and it never overwrites a saved result.
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "== Dynamic-WHICH — CPU validation =="
python -m compileall -q src scripts
python -m src.common.test_evaluation_core
echo
echo "-- all five Qwen x TextVQA WHICH finals (the headline win) --"
python -m scripts.validation.validate_dynamic_which
echo
echo "-- full 40-cell WHICH matrix completeness --"
python -m scripts.validation.audit_dynamic_which

echo
echo "Result tables:"
echo "  results/tables/dynamic_which_final_report.md"
echo "  results/tables/dynamic_which_textsim_full_final_summary.md"
echo "  results/tables/qwen_textvqa_current_vs_ref_validation.md"
echo "  dynamic_which/RESULTS.md"
echo
echo "GPU RERUN (expensive; NOT needed unless reproducing the experiment):"
echo "  bash scripts/dynamic_which/run_dynamic_which_full_matrix.sh   # 35 full finals"
echo "  (each cell needs its matching dense + same-budget static finals; runners skip"
echo "   any cell whose result already exists — no silent overwrite)"
