#!/usr/bin/env bash
# This wrapper is a method-facing entry point for Dynamic-COUNT (DC-D and DC-C).
# The tested implementation remains in the shared evaluation core under src/ and scripts/;
# this script only calls it. It is safe by default: CPU-only, read-only over results,
# and it never overwrites a saved result.
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "== Dynamic-COUNT — CPU validation =="
python -m compileall -q src scripts
python -m src.common.test_evaluation_core
echo
echo "-- every probe / DC-D / DC-C aggregate: gates, n, probe reproduction --"
python -m scripts.validation.validate_dynamic_count
echo
echo "-- full 8-cell DC matrix completeness --"
python -m scripts.validation.audit_dynamic_count

echo
echo "Result tables:"
echo "  results/tables/dynamic_count_dc_d_summary.md"
echo "  results/tables/dynamic_count_dc_c_summary.md"
echo "  results/tables/dynamic_count_win_loss_summary.md"
echo "  results/tables/dynamic_count_oracle_summary.md"
echo "  dynamic_count/RESULTS.md"
echo
echo "GPU RERUN (expensive; NOT needed unless reproducing the experiment):"
echo "  # 1. probes -> 2. calibration (CPU) -> 3. DC-D (CPU) -> 4. DC-C (GPU)"
echo "  bash scripts/dynamic_count/run_probe_full_matrix.sh"
echo "  python -m scripts.dynamic_count.make_controller_calibration"
echo "  python -m scripts.dynamic_count.compose_discrete"
echo "  bash scripts/dynamic_count/run_continuous_full_matrix.sh"
echo "  (all stages are skip-safe: existing outputs are never overwritten)"
