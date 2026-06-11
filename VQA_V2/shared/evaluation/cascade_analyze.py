"""
cascade_analyze.py — Realizable confidence-cascade frontier.

Merges a low-K base pass and a high-K escalation pass (from cascade_pass.py).
For a confidence threshold tau: keep the base answer if base confidence >= tau,
else use the high-K answer. Sweeping tau traces the REALIZABLE accuracy-vs-avg-K
frontier (label-free; uses only the model's own confidence).

Reports the frontier and compares to the uniform static curve at matched avg K.

Usage:
    python -m VQA_V2.shared.evaluation.cascade_analyze \\
        --base VQA_V2/outputs/cascade/base_k64.json \\
        --high VQA_V2/outputs/cascade/high_k334.json \\
        --conf-key mean_conf
"""

import argparse
import json

import numpy as np

# Uniform static curve (VQAv2 10K, generation), for comparison.
UNIFORM = {64: 71.02, 96: 72.92, 128: 74.27, 160: 74.85, 219: 75.48,
           265: 75.71, 276: 75.82, 334: 76.05, 357: 76.17}
UK = np.array(sorted(UNIFORM)); UV = np.array([UNIFORM[k] for k in sorted(UNIFORM)])


def uniform_at(k):
    return float(np.interp(k, UK, UV))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--high", required=True)
    ap.add_argument("--conf-key", default="mean_conf", choices=["mean_conf", "first_token_conf"])
    ap.add_argument("--high-k", type=int, default=None, help="K of high file if generate_and_score format")
    args = ap.parse_args()

    from VQA_V2.shared.datasets.vqav2_answers import normalize_answer
    def _vqa(pred, raw):
        pn = normalize_answer(pred)
        return min(1.0, sum(1 for a in raw if normalize_answer(a) == pn) / 3.0)

    base_raw = json.load(open(args.base))
    base = {r["question_id"]: r for r in base_raw["records"]}
    K_base = base_raw["keep_tokens"]

    hi_raw = json.load(open(args.high))
    if "records" in hi_raw:                      # cascade_pass format
        high = {r["question_id"]: r["score"] for r in hi_raw["records"]}
        K_high = hi_raw["keep_tokens"]
    else:                                        # generate_and_score format
        high = {p["question_id"]: _vqa(p["pred_answer"], p["raw_answers"])
                for p in hi_raw["generation"]["predictions"]}
        K_high = args.high_k
    qids = sorted(set(base) & set(high))
    N = len(qids)

    conf = np.array([base[q][args.conf_key] for q in qids])
    s_base = np.array([base[q]["score"] for q in qids])
    s_high = np.array([high[q] for q in qids])

    print(f"N={N}  base K={K_base} (acc {100*s_base.mean():.2f}%)  "
          f"high K={K_high} (acc {100*s_high.mean():.2f}%)  conf={args.conf_key}")

    # How well does base confidence predict base correctness? (realizability core)
    order = np.argsort(-conf)
    # AUC-ish: fraction of correct among most-confident half vs least-confident half
    half = N // 2
    hi_half = s_base[order[:half]].mean()
    lo_half = s_base[order[half:]].mean()
    print(f"base acc | top-conf half = {100*hi_half:.2f}%   bottom-conf half = {100*lo_half:.2f}%  "
          f"(separation {100*(hi_half-lo_half):+.2f}pp)")

    # Cascade frontier: keep base if conf>=tau else escalate to high.
    print("\nREALIZABLE cascade frontier (keep base if conf>=tau, else high):")
    print(f"  {'tau':>6s}  {'%escal':>7s}  {'avgK':>6s}  {'cascade%':>9s}  {'uniform@avgK':>12s}  {'delta':>7s}")
    best = None
    for tau in np.linspace(0.0, 1.0, 51):
        escalate = conf < tau
        frac = float(escalate.mean())
        avg_k = K_base + frac * (K_high - K_base)
        acc = 100 * np.where(escalate, s_high, s_base).mean()
        u = uniform_at(avg_k)
        d = acc - u
        if (best is None) or (d > best[-1]):
            best = (tau, frac, avg_k, acc, u, d)
        if abs(tau * 10 - round(tau * 10)) < 1e-9:  # print every 0.1
            print(f"  {tau:>6.2f}  {100*frac:>6.1f}%  {avg_k:>6.0f}  {acc:>9.2f}  "
                  f"{u:>12.2f}  {d:>+7.2f}")

    tau, frac, avg_k, acc, u, d = best
    print(f"\nBest cascade point vs uniform: tau={tau:.2f}  avgK={avg_k:.0f}  "
          f"cascade={acc:.2f}%  uniform@{avg_k:.0f}={u:.2f}%  delta={d:+.2f}pp")
    # Efficiency framing: lowest avg K at which cascade matches uniform-K=265 (75.71%).
    target = UNIFORM[265]
    matched_k = None
    for tau in np.linspace(0.0, 1.0, 201):
        escalate = conf < tau
        avg_k = K_base + float(escalate.mean()) * (K_high - K_base)
        acc = 100 * np.where(escalate, s_high, s_base).mean()
        if acc >= target:
            matched_k = (avg_k, acc, tau)
            break
    if matched_k:
        print(f"Efficiency: cascade reaches uniform-K265 accuracy ({target:.2f}%) "
              f"at avg K={matched_k[0]:.0f} (tau={matched_k[2]:.2f}, acc={matched_k[1]:.2f}%)  "
              f"=> {265 - matched_k[0]:.0f} fewer avg tokens ({100*(265-matched_k[0])/265:.0f}%).")
    else:
        print(f"Efficiency: cascade never reaches uniform-K265 accuracy ({target:.2f}%) "
              f"in [{K_base},{K_high}].")


if __name__ == "__main__":
    main()
