"""
Phase 2 analysis (CPU): train a per-sample BUDGET PREDICTOR on the collected features and
test, with 5-fold CV, whether it captures a realistic fraction of the ORACLE dynamic-budget
headroom. Honest confirmation: realistic predictor gain << oracle, and we report both.

predictor target = sufficiency-K (smallest ladder K, robustly correct@0.5). Two predictors:
  (a) simple rule: budget rank ~ attention spread (eff_support), quantile-binned
  (b) small MLP on the 5 distribution features
Eval: dynamic (predicted-K) accuracy & avg budget vs the best FIXED K at the SAME avg budget.
"""
import json, math, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
import torch
import torch.nn as nn

THRESH = 0.5
FKEYS = ["norm_entropy", "eff_support", "top64_mass", "top128_mass", "top256_mass"]


def load(path="results/paper_candidates/qwen_budget_data_docvqa.json"):
    d = json.load(open(path))
    ladder = d["ladder"]; levels = ladder + ["full"]
    rows = d["rows"]
    X = np.array([[r["feats"][k] for k in FKEYS] for r in rows], dtype=np.float32)
    # per-sample correctness at each level; sufficiency-K = smallest level robustly correct
    corr = np.array([[r["soft"][str(K)] >= THRESH for K in levels] for r in rows])  # [n, L]
    Kvals = np.array([(r["feats"]["nvis"] if K == "full" else K) for K in levels for r in [rows[0]]])
    # token cost per level (use K; full -> avg nvis)
    nvis = np.array([r["feats"]["nvis"] for r in rows])
    tok = np.array([(0 if K != "full" else 0) for K in levels], dtype=np.float32)
    tok = np.array([(np.mean(nvis) if K == "full" else K) for K in levels], dtype=np.float32)

    def suff_idx(i):
        for j in range(len(levels)):
            if all(corr[i, j:]):
                return j
        return len(levels) - 1   # never robustly correct -> full
    y = np.array([suff_idx(i) for i in range(len(rows))])
    return X, corr, tok, levels, y


def static_acc_at(corr, tok, budget):
    # binary acc of each fixed level, then interpolate at 'budget' tokens
    accs = corr.mean(0) * 100
    order = np.argsort(tok); xs = tok[order]; ys = accs[order]
    if budget <= xs[0]: return ys[0]
    if budget >= xs[-1]: return ys[-1]
    j = np.searchsorted(xs, budget) - 1
    return ys[j] + (ys[j+1]-ys[j])*(budget-xs[j])/(xs[j+1]-xs[j])


def evaluate(pred_idx, corr, tok):
    n = len(pred_idx)
    acc = 100*np.mean([corr[i, pred_idx[i]] for i in range(n)])
    budget = np.mean([tok[pred_idx[i]] for i in range(n)])
    return acc, budget


def cv_mlp(X, y, L, folds=5, seed=0):
    rng = np.random.RandomState(seed); idx = rng.permutation(len(X))
    Xn = (X - X.mean(0)) / (X.std(0)+1e-6)
    pred = np.zeros(len(X), dtype=int)
    for f in range(folds):
        te = idx[f::folds]; tr = np.setdiff1d(idx, te)
        net = nn.Sequential(nn.Linear(X.shape[1],32), nn.ReLU(), nn.Linear(32,L))
        opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-3)
        xt = torch.tensor(Xn[tr]); yt = torch.tensor(y[tr])
        for _ in range(300):
            opt.zero_grad(); loss = nn.functional.cross_entropy(net(xt), yt); loss.backward(); opt.step()
        with torch.no_grad():
            pred[te] = net(torch.tensor(Xn[te])).argmax(1).numpy()
    return pred


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="docvqa")
    ap.add_argument("--tag", default=""); a = ap.parse_args()
    tag = a.tag or a.dataset
    X, corr, tok, levels, y = load(f"results/paper_candidates/qwen_budget_data_{tag}.json")
    n, L = len(X), len(levels)
    print(f"  [tag={tag}]")
    # references
    oracle_acc, oracle_tok = evaluate(y, corr, tok)
    best_fixed = (corr.mean(0)*100).max()
    print("="*70)
    print(f"  Qwen2.5-VL budget-predictor confirmation [{a.dataset}] (n={n}, QC selection)")
    print(f"  levels(tok): {[round(t) for t in tok]}")
    print(f"  label (sufficiency-K) dist: {np.bincount(y, minlength=L).tolist()}")
    print("-"*70)
    print(f"  best FIXED budget        : {best_fixed:.1f}%")
    s_at_oracle = static_acc_at(corr, tok, oracle_tok)
    print(f"  ORACLE dynamic           : {oracle_acc:.1f}% @ {oracle_tok:.0f} tok  "
          f"(vs static {s_at_oracle:.1f} @ same) -> +{oracle_acc - s_at_oracle:.1f}pp")
    # (a) simple spread rule: bin eff_support into L buckets by quantile -> level
    es = X[:, FKEYS.index("eff_support")]
    ranks = np.argsort(np.argsort(es))            # 0..n-1
    rule_idx = np.clip((ranks / n * L).astype(int), 0, L-1)
    a_acc, a_tok = evaluate(rule_idx, corr, tok)
    print(f"  predictor (spread rule)  : {a_acc:.1f}% @ {a_tok:.0f} tok  "
          f"(vs static {static_acc_at(corr,tok,a_tok):.1f}) -> +{a_acc-static_acc_at(corr,tok,a_tok):.1f}pp")
    # (b) MLP CV
    mlp_idx = cv_mlp(X, y, L)
    b_acc, b_tok = evaluate(mlp_idx, corr, tok)
    print(f"  predictor (MLP, 5-fold)  : {b_acc:.1f}% @ {b_tok:.0f} tok  "
          f"(vs static {static_acc_at(corr,tok,b_tok):.1f}) -> +{b_acc-static_acc_at(corr,tok,b_tok):.1f}pp")
    print("-"*70)
    realistic = b_acc - static_acc_at(corr, tok, b_tok)
    oracle_gain = oracle_acc - s_at_oracle
    frac = 100*realistic/oracle_gain if oracle_gain>0 else 0
    print(f"  => realistic predictor captures {frac:.0f}% of the oracle dynamic-budget gain")
    print("="*70)
    json.dump({"n":n,"oracle_gain":round(oracle_gain,2),"mlp_gain":round(realistic,2),
               "rule_gain":round(a_acc-static_acc_at(corr,tok,a_tok),2),
               "frac_of_oracle_pct":round(frac,1),"best_fixed":round(best_fixed,2)},
              open("results/thesis_main/highres/qwen_budget_eval.json","w"), indent=2)


if __name__ == "__main__":
    main()
