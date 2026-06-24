"""
Robustness of the 'budget is a mirage' claim (CPU). The dynamic-budget benefit requires
IDENTIFYING the hard tail (samples needing > min budget). We test that directly: binary
classification 'sufficiency-K > 32' from the cheap features, 5-fold CV, report AUC vs the
base rate, plus the best single-feature AUC. If AUC ~ 0.5 the tail is UNPREDICTABLE -> a
per-sample budget cannot beat a fixed budget, independent of predictor design.
"""
import json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
import torch, torch.nn as nn

THRESH = 0.5
FKEYS = ["nvis", "norm_entropy", "eff_support", "top64_mass", "top128_mass", "top256_mass"]


def auc(scores, labels):
    # rank-based (Mann-Whitney); labels in {0,1}
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores); ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(scores)+1)
    # average ranks for ties
    s = np.argsort(scores, kind="mergesort"); sv = scores[s]
    i = 0
    while i < len(sv):
        j = i
        while j+1 < len(sv) and sv[j+1] == sv[i]:
            j += 1
        if j > i:
            ranks[s[i:j+1]] = (i+1 + j+1)/2
        i = j+1
    r_pos = ranks[labels == 1].sum()
    return (r_pos - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="docvqa")
    ap.add_argument("--tag", default=""); a = ap.parse_args()
    tag = a.tag or a.dataset
    print(f"  [tag={tag}]")
    d = json.load(open(f"results/paper_candidates/qwen_budget_data_{tag}.json"))
    ladder = d["ladder"]; rows = d["rows"]; levels = ladder + ["full"]
    X = np.array([[r["feats"][k] for k in FKEYS] for r in rows], dtype=np.float32)
    corr = np.array([[r["soft"][str(K)] >= THRESH for K in levels] for r in rows])

    def suff(i):
        for j in range(len(levels)):
            if all(corr[i, j:]):
                return (0 if levels[j] != "full" else 10**9) if False else (levels[j] if levels[j] != "full" else 10**9)
        return 10**9
    suffK = np.array([suff(i) for i in range(len(rows))])
    hard = (suffK > 32).astype(int)               # needs more than the minimum budget
    n = len(rows); base = hard.mean()
    print("="*64)
    print(f"  Hard-tail predictability [{a.dataset}] (n={n})")
    print(f"  'needs > 32 tokens' base rate: {100*base:.1f}%  ({hard.sum()}/{n})")
    print("-"*64)
    # single-feature AUCs
    print("  single-feature AUC (vs 0.5=chance):")
    for k, name in enumerate(FKEYS):
        a = auc(X[:, k], hard)
        a = max(a, 1-a)                            # direction-agnostic
        print(f"    {name:<14}: {a:.3f}")
    # 5-fold logistic CV out-of-fold AUC
    Xn = (X - X.mean(0))/(X.std(0)+1e-6)
    rng = np.random.RandomState(0); idx = rng.permutation(n); oof = np.zeros(n)
    for f in range(5):
        te = idx[f::5]; tr = np.setdiff1d(idx, te)
        net = nn.Sequential(nn.Linear(X.shape[1], 16), nn.ReLU(), nn.Linear(16, 1))
        opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-2)
        xt = torch.tensor(Xn[tr]); yt = torch.tensor(hard[tr], dtype=torch.float32)
        for _ in range(400):
            opt.zero_grad(); loss = nn.functional.binary_cross_entropy_with_logits(net(xt).squeeze(1), yt); loss.backward(); opt.step()
        with torch.no_grad():
            oof[te] = net(torch.tensor(Xn[te])).squeeze(1).numpy()
    cv_auc = auc(oof, hard)
    print("-"*64)
    print(f"  5-fold CV classifier AUC: {cv_auc:.3f}")
    verdict = "UNPREDICTABLE -> dynamic budget cannot beat fixed (mirage robust)" if cv_auc < 0.6 \
        else "somewhat predictable -> a budget method MIGHT help"
    print(f"  VERDICT: {verdict}")
    print("="*64)
    json.dump({"n": n, "hard_base_rate": round(float(base), 3), "cv_auc": round(float(cv_auc), 3)},
              open("results/thesis_main/highres/qwen_budget_robust.json", "w"), indent=2)


if __name__ == "__main__":
    main()
