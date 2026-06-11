"""
instance_headroom.py — Is there headroom for per-INSTANCE adaptive token allocation?

Uses per-sample VQA scores from static CLS top-K generation evals at several K
values. Computes the ORACLE instance-adaptive frontier: the maximum achievable
average accuracy at each average token budget, where an oracle assigns each
sample its own K using the true label (an unachievable UPPER BOUND on any
learnable per-instance method).

Decision rule:
  oracle(avg_K=B) >> uniform(B)  -> per-instance allocation has headroom (pursue it).
  oracle(avg_K=B) ~= uniform(B)  -> even a perfect allocator can't win (stop).

Method: Lagrangian sweep. For a per-token price lambda, each sample picks
K_i = argmax_K [ score_i(K) - lambda * K ]. Sweeping lambda traces the oracle's
accuracy-vs-average-K Pareto frontier exactly (on the concave hull).

Usage:
    python -m VQA_V2.shared.evaluation.instance_headroom
"""

import json
import os
import sys
from typing import Dict, List

import numpy as np

# Allow running this file directly (repo root = 3 level(s) up); `python -m` does not need this.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from VQA_V2.shared.datasets.vqav2_answers import normalize_answer

# (K, path) for every static per-K generation eval we have.
EVALS = {
    64:  "VQA_V2/outputs/static_k64_pertype/generation_eval_10k.json",
    96:  "VQA_V2/outputs/static_k96_pertype/generation_eval_10k.json",
    128: "VQA_V2/outputs/static_k128_pertype/generation_eval_10k.json",
    160: "VQA_V2/outputs/static_k160_pertype/generation_eval_10k.json",
    219: "VQA_V2/outputs/static_k219_pertype/generation_eval_10k.json",
    265: "VQA_V2/outputs/static_k265_matched/generation_eval_10k.json",
    276: "VQA_V2/outputs/static_k276_pertype/generation_eval_10k.json",
    334: "VQA_V2/outputs/static_k334_pertype/generation_eval_10k.json",
    357: "VQA_V2/outputs/static_k357_pertype/generation_eval_10k.json",
}


def vqa_score(pred: str, raw: List[str]) -> float:
    pn = normalize_answer(pred)
    return min(1.0, sum(1 for a in raw if normalize_answer(a) == pn) / 3.0)


def load_scores() -> (List[int], Dict[int, Dict[int, float]]):
    """Return (sorted K list, {qid: {K: score}}) over qids present in ALL evals."""
    per_k: Dict[int, Dict[int, float]] = {}
    for K, path in EVALS.items():
        with open(path) as f:
            preds = json.load(f)["generation"]["predictions"]
        per_k[K] = {p["question_id"]: vqa_score(p["pred_answer"], p["raw_answers"]) for p in preds}
    Ks = sorted(per_k.keys())
    common = set.intersection(*[set(per_k[K].keys()) for K in Ks])
    scores = {qid: {K: per_k[K][qid] for K in Ks} for qid in common}
    return Ks, scores


def main():
    Ks, scores = load_scores()
    N = len(scores)
    qids = list(scores.keys())
    K_arr = np.array(Ks, dtype=float)
    # S[i, j] = score of sample i at K = Ks[j]
    S = np.array([[scores[q][K] for K in Ks] for q in qids], dtype=float)  # [N, nK]

    print(f"Samples (common across all {len(Ks)} K-evals): {N}")
    print(f"K values: {Ks}\n")

    # Uniform accuracy at each measured K (for reference / comparison).
    uniform = {K: float(S[:, j].mean()) for j, K in enumerate(Ks)}
    print("Uniform static accuracy at each K:")
    for K in Ks:
        print(f"  K={K:3d}: {100*uniform[K]:.2f}%")

    # --- Oracle frontier via Lagrangian sweep ---
    lambdas = np.concatenate([[0.0], np.geomspace(1e-5, 0.05, 400)])
    frontier = []  # (avg_K, avg_score)
    for lam in lambdas:
        util = S - lam * K_arr[None, :]          # [N, nK]
        choice = util.argmax(axis=1)             # best K index per sample
        avg_k = float(K_arr[choice].mean())
        avg_s = float(S[np.arange(N), choice].mean())
        frontier.append((avg_k, avg_s))
    frontier.sort()
    fk = np.array([p[0] for p in frontier])
    fs = np.array([p[1] for p in frontier])

    def oracle_at(budget: float) -> float:
        return float(np.interp(budget, fk, fs))

    print("\nORACLE instance-adaptive frontier (upper bound) vs uniform:")
    print(f"  {'avg K':>6s}  {'uniform%':>9s}  {'oracle%':>8s}  {'headroom_pp':>11s}")
    for B in [96, 128, 160, 200, 219, 265, 334]:
        if B in uniform:
            u = 100 * uniform[B]
        else:
            u = 100 * float(np.interp(B, K_arr, [uniform[K] for K in Ks]))
        o = 100 * oracle_at(B)
        print(f"  {B:>6d}  {u:>9.2f}  {o:>8.2f}  {o-u:>+11.2f}")

    # --- Per-sample token-need characterization ---
    # "min correct K": smallest K with score >= 2/3 (majority-correct); inf if never.
    thresh = 2.0 / 3.0
    min_correct_K = []
    never = 0
    always64 = 0
    for i in range(N):
        ok = np.where(S[i] >= thresh)[0]
        if len(ok) == 0:
            never += 1
            min_correct_K.append(None)
        else:
            mk = Ks[ok[0]]
            min_correct_K.append(mk)
            if mk == Ks[0]:
                always64 += 1
    have = [m for m in min_correct_K if m is not None]
    print("\nPer-sample token-need (min K with VQA score >= 2/3):")
    print(f"  correct already at K={Ks[0]}: {always64} ({100*always64/N:.1f}%)")
    print(f"  never correct (any K):        {never} ({100*never/N:.1f}%)")
    print(f"  K-sensitive (needs K>{Ks[0]}): {N - always64 - never} "
          f"({100*(N-always64-never)/N:.1f}%)")
    if have:
        for K in Ks:
            c = sum(1 for m in have if m == K)
            print(f"    first-correct at K={K:3d}: {c} ({100*c/N:.1f}%)")

    # Ceiling: best-K-per-sample (no budget limit) — absolute oracle.
    best_per_sample = S.max(axis=1).mean()
    print(f"\nAbsolute ceiling (each sample at its best K, no budget cap): "
          f"{100*best_per_sample:.2f}%  (vs dense-ish uniform K=357: {100*uniform[357]:.2f}%)")


if __name__ == "__main__":
    main()
