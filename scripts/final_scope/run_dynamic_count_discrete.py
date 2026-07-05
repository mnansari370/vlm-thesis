"""
run_dynamic_count_discrete.py — compose DC-D (discrete cascade baseline) for ONE cell. CPU-only:
both passes are deterministic and already on disk (probe output + frozen target anchor), so the
cascade is composed offline with honest double-pay FLOPs. Uses the calibrated config
(results/final_scope/dynamic_count_configs/{model}_{dataset}.json) — thresholds come from the
calibration split; scoring is on the held-out 80% (default) or the full manifest (--all-samples).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.final_scope import dense_pilot as dp
from src.final_scope import dynamic_count_eval as dce

CFG_DIR = os.path.join("results", "final_scope", "dynamic_count_configs")


def main() -> int:
    ap = argparse.ArgumentParser(description="Compose DC-D for one cell (CPU).")
    ap.add_argument("--model", required=True, choices=list(dp.MODELS))
    ap.add_argument("--dataset", required=True, choices=list(dp.DATASETS))
    ap.add_argument("--count-on-which", action="store_true", dest="cow",
                    help="use the SEPARATE textsim (COUNT-on-WHICH) config — qwen25vl7b×textvqa only")
    ap.add_argument("--all-samples", action="store_true",
                    help="score the FULL manifest (descriptive) instead of the held-out 80%%")
    args = ap.parse_args()
    tag = f"{args.model}_{args.dataset}" + ("_textsim" if args.cow else "")
    cfg_path = os.path.join(CFG_DIR, f"{tag}.json")
    if not os.path.exists(cfg_path):
        print(f"[error] missing calibration config {cfg_path} — run the calibration script first")
        return 2
    cfg = json.load(open(cfg_path))["dcd"]
    js, jl, score, delta, ok = dce.compose_dcd(
        model_name=args.model, dataset=args.dataset, selector=cfg["selector"],
        probe_pct=cfg["probe_pct"], target_pct=cfg["target_pct"],
        risk_signal=cfg["risk_signal"], tau=cfg["tau"],
        eval_split_only=not args.all_samples)
    print(f"\n[done] DC-D {args.model}×{args.dataset}: score={score}% Δcurve={delta:+.2f}pp "
          f"gate={'OK' if ok else 'ISSUES'}\n  {js}\n  {jl}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
