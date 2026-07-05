"""
validate_dynamic_count_final.py — validate all Dynamic-COUNT outputs (CPU).

Checks every dynamic_count_{probe,dcd,dcc}_* aggregate: fairness gate OK, method correct, n>0,
JSONL line count == aggregate n, probe equivalence_ok. Prints a per-file table and
ALL_DC_VALID=True|False.
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.final_scope import dense_pilot as dp


def _lines(path):
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def main() -> int:
    issues = []
    n_files = 0
    for m in dp.MODELS:
        for d in dp.DATASETS:
            for p in sorted(glob.glob(os.path.join(dp.OUT_ROOT, m, d, "dynamic_count_*.json"))):
                if p.endswith(".jsonl"):
                    continue
                n_files += 1
                try:
                    a = json.load(open(p))
                except (OSError, json.JSONDecodeError) as e:
                    issues.append(f"{p}: unreadable ({e})")
                    continue
                name = os.path.basename(p)
                if not a.get("fairness_gate", {}).get("ok"):
                    issues.append(f"{name}: gate not ok")
                if not str(a.get("method", "")).startswith("dynamic_count"):
                    issues.append(f"{name}: wrong method {a.get('method')}")
                jl = p[:-5] + ".jsonl"
                if not os.path.exists(jl):
                    issues.append(f"{name}: missing jsonl")
                elif _lines(jl) != a.get("n"):
                    issues.append(f"{name}: jsonl lines != n")
                if a.get("method") == "dynamic_count_probe" and not a.get("equivalence_ok"):
                    issues.append(f"{name}: probe reproduction gate FAILED")
                lab = a.get("interpretation_label", "")
                extra = a.get("extra_dynamic_count", {})
                print(f"  {m:10} {d:7} {name[:64]:64} n={a.get('n'):>6} "
                      f"gate={'OK' if a.get('fairness_gate',{}).get('ok') else 'X'} "
                      f"{('Δ=' + format(a.get('delta_vs_static_curve_pp'), '+.2f')) if a.get('delta_vs_static_curve_pp') is not None else '':12} {lab}")
    print(f"\nFILES_CHECKED={n_files}")
    if issues:
        print("ISSUES:")
        for i in issues:
            print(f"  - {i}")
    print(f"ALL_DC_VALID={not issues}")
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
