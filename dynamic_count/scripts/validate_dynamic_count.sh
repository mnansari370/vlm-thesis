#!/usr/bin/env bash
# This wrapper is a method-facing entry point for Dynamic-COUNT (DC-D and DC-C).
# The tested implementation remains in the shared evaluation core under src/ and scripts/;
# this script only calls it. It is safe by default: CPU-only, read-only over results,
# and it never overwrites a saved result.
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "== Dynamic-COUNT — CPU validation =="
python -m compileall -q src scripts
python -m src.final_scope.test_final_scope
echo
echo "-- every probe / DC-D / DC-C aggregate: gates, n, probe reproduction --"
python -m scripts.final_scope.validate_dynamic_count_final
echo
echo "-- full 8-cell DC matrix completeness --"
python -m scripts.final_scope.audit_dynamic_count_full_matrix

echo
echo "Result tables:"
echo "  results/final_scope/tables/dynamic_count_dc_d_summary.md"
echo "  results/final_scope/tables/dynamic_count_dc_c_summary.md"
echo "  results/final_scope/tables/dynamic_count_win_loss_summary.md"
echo "  results/final_scope/tables/dynamic_count_oracle_summary.md"
echo "  dynamic_count/RESULTS.md"
echo
echo "GPU RERUN (expensive; NOT needed unless reproducing the experiment):"
echo "  # 1. probes -> 2. calibration (CPU) -> 3. DC-D (CPU) -> 4. DC-C (GPU)"
echo "  bash scripts/final_scope/run_dynamic_count_probe_full_matrix.sh"
echo "  python -m scripts.final_scope.make_dynamic_count_controller_calibration"
echo "  python -m scripts.final_scope.compose_dynamic_count_discrete"
echo "  bash scripts/final_scope/run_dynamic_count_continuous_full_matrix.sh"
echo "  (all stages are skip-safe: existing outputs are never overwritten)"
