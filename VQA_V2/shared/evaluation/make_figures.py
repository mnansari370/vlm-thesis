"""Generate the key figures for the dynamic-pruning findings write-up."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running this file directly (repo root = 3 level(s) up); `python -m` does not need this.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from VQA_V2.shared.datasets.vqav2_answers import normalize_answer
from VQA_V2.shared.evaluation.flops import fastv_full_flops

OUT = "VQA_V2/outputs/figures"
os.makedirs(OUT, exist_ok=True)

def flops(K):
    # Canonical LM-full prefill TFLOPs (FastV Eq. 5, n = K + 35) — the SAME
    # convention as the GQA track, so all thesis/paper FLOPs are comparable.
    # (The old axis used the attention-only proxy 2·L·S²·H; that remains
    # available as the secondary metric in shared/evaluation/flops.py.)
    return fastv_full_flops(K) / 1e12

# --- static overall curve (locked + computed) ---
STATIC = {64: 71.02, 96: 72.92, 128: 74.27, 144: 74.44, 160: 74.85, 192: 75.31,
          219: 75.48, 255: 75.76, 265: 75.71, 275: 75.71, 276: 75.82, 288: 75.82,
          334: 76.05, 357: 76.17, 432: 76.27, 576: 76.44}
DYN = (264.3, 75.76)  # (avg K, acc)

# --- per-type curves ---
PT = {  # K: (yes/no, attribute, counting, spatial)
    64: (86.60, 64.46, 55.60, 48.48), 96: (88.40, 66.44, 58.64, 48.58),
    128: (89.52, 67.76, 59.67, 52.00), 160: (90.23, 67.94, 61.89, 51.71),
    219: (90.66, 68.72, 61.80, 53.69), 265: (91.03, 68.96, 62.10, 52.92),
    276: (91.26, 69.01, 62.74, 51.95), 334: (91.41, 69.27, 62.99, 52.34),
    357: (91.33, 69.39, 63.38, 53.26)}

def vqa(pred, raw):
    pn = normalize_answer(pred)
    return min(1.0, sum(1 for a in raw if normalize_answer(a) == pn) / 3.0)

# ---------- Fig 1: trade-off (accuracy vs attention FLOPs) ----------
Ks = sorted(STATIC)
fx = [flops(k) for k in Ks]; fy = [STATIC[k] for k in Ks]
plt.figure(figsize=(6, 4.2))
plt.plot(fx, fy, "o-", color="#444", label="Static CLS top-K (uniform)")
plt.plot(flops(DYN[0]), DYN[1], "*", ms=18, color="crimson",
         label=f"Dynamic (type-adaptive, avg K={DYN[0]:.0f})")
plt.xscale("log")
plt.xlabel("LM prefill FLOPs (T, log) — FastV Eq.5, n=K+35")
plt.ylabel("VQAv2 val 10K generation acc (%)")
plt.title("Accuracy vs cost — dynamic point sits ON the static curve")
plt.legend(loc="lower right", fontsize=8); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_tradeoff.png", dpi=130); plt.close()

# ---------- Fig 2: per-type accuracy vs K (saturation) ----------
ptk = sorted(PT)
plt.figure(figsize=(6, 4.2))
for i, name in enumerate(["yes/no", "attribute", "counting", "spatial"]):
    plt.plot(ptk, [PT[k][i] for k in ptk], "o-", label=name)
for tgt, nm in [(219, "y/n"), (276, "attr"), (334, "cnt"), (357, "spat")]:
    plt.axvline(tgt, color="grey", ls=":", alpha=0.4)
plt.xlabel("K (visual tokens kept)"); plt.ylabel("per-type gen acc (%)")
plt.title("Per-type accuracy vs K — all types saturate by K≈160")
plt.legend(fontsize=8); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_pertype.png", dpi=130); plt.close()

# ---------- oracle + cascade frontiers ----------
EVALS = {64: "static_k64_pertype", 96: "static_k96_pertype", 128: "static_k128_pertype",
         160: "static_k160_pertype", 219: "static_k219_pertype", 265: "static_k265_matched",
         276: "static_k276_pertype", 334: "static_k334_pertype", 357: "static_k357_pertype"}
per_k = {}
for K, d in EVALS.items():
    preds = json.load(open(f"VQA_V2/outputs/{d}/generation_eval_10k.json"))["generation"]["predictions"]
    per_k[K] = {p["question_id"]: vqa(p["pred_answer"], p["raw_answers"]) for p in preds}
Kl = sorted(per_k); common = sorted(set.intersection(*[set(per_k[k]) for k in Kl]))
S = np.array([[per_k[k][q] for k in Kl] for q in common]); Karr = np.array(Kl, float)
N = len(common)

# oracle frontier
fr = []
for lam in np.concatenate([[0.0], np.geomspace(1e-5, 0.05, 300)]):
    ch = (S - lam * Karr[None, :]).argmax(1)
    fr.append((Karr[ch].mean(), 100 * S[np.arange(N), ch].mean()))
fr.sort(); ofk = np.array([p[0] for p in fr]); ofy = np.array([p[1] for p in fr])

# cascade frontier (base k64 conf -> escalate to k128)
base = {r["question_id"]: r for r in json.load(open("VQA_V2/outputs/cascade/base_k64.json"))["records"]}
q2 = [q for q in common if q in base]
conf = np.array([base[q]["mean_conf"] for q in q2])
sb = np.array([base[q]["score"] for q in q2]); idx128 = Kl.index(128)
sh = np.array([per_k[128][q] for q in q2])
cfk, cfy = [], []
for tau in np.linspace(0, 1, 101):
    esc = conf < tau
    cfk.append(64 + esc.mean() * (128 - 64)); cfy.append(100 * np.where(esc, sh, sb).mean())

plt.figure(figsize=(6, 4.2))
uk = sorted(STATIC); plt.plot(uk, [STATIC[k] for k in uk], "o-", color="#444", label="uniform static")
plt.plot(ofk, ofy, "-", color="seagreen", lw=2, label="oracle (labelled upper bound)")
plt.plot(cfk, cfy, "s-", color="crimson", ms=3, label="realizable cascade (conf, 64→128)")
plt.xlim(50, 380)
plt.xlabel("average K"); plt.ylabel("VQAv2 gen acc (%)")
plt.title("Headroom: oracle far above; realizable ≈ uniform")
plt.legend(fontsize=8); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_headroom.png", dpi=130); plt.close()

print(f"Figures written to {OUT}/ : fig1_tradeoff.png fig2_pertype.png fig3_headroom.png")
print(f"oracle ceiling={ofy.max():.2f}%  cascade(64->128) max acc={max(cfy):.2f}%")
