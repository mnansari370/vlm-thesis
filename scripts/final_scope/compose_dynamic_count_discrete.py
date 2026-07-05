"""
compose_dynamic_count_discrete.py — batch driver: compose DC-D for ALL cells with a calibration
config + probe outputs. CPU-only, skip-safe (existing DC-D outputs are skipped, never overwritten).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.final_scope import dense_pilot as dp
from src.final_scope import dynamic_count_eval as dce
from src.pruning import dynamic_count as dc

CFG_DIR = os.path.join("results", "final_scope", "dynamic_count_configs")


def main() -> int:
    argparse.ArgumentParser(description="Compose DC-D for all calibrated cells (CPU).").parse_args()
    done = skipped = failed = 0
    jobs = [(m, d, "") for m in dp.MODELS for d in dp.DATASETS]
    jobs.append(("qwen25vl7b", "textvqa", "_textsim"))     # COUNT-on-WHICH, composed separately
    for m, d, suffix in jobs:
        cfg_path = os.path.join(CFG_DIR, f"{m}_{d}{suffix}.json")
        label = f"{m}×{d}{suffix}"
        if not os.path.exists(cfg_path):
            print(f"[skip] {label}: no calibration config")
            skipped += 1
            continue
        cfg = json.load(open(cfg_path))["dcd"]
        base = dc.dcd_basename(d, cfg["selector"], cfg["probe_pct"], cfg["target_pct"])
        out = os.path.join(dp.OUT_ROOT, m, d, f"{base}.json")
        if os.path.exists(out):
            print(f"[skip] {label}: DC-D already composed ({base})")
            skipped += 1
            continue
        try:
            _, _, score, delta, ok = dce.compose_dcd(
                model_name=m, dataset=d, selector=cfg["selector"], probe_pct=cfg["probe_pct"],
                target_pct=cfg["target_pct"], risk_signal=cfg["risk_signal"], tau=cfg["tau"])
            done += 1
            if not ok:
                failed += 1
        except (FileNotFoundError, ValueError) as e:
            print(f"[fail] {label}: {e}")
            failed += 1
    print(f"\nDC-D compose: done={done} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
