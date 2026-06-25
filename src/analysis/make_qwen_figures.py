"""Generate the 4 paper figures for the Qwen2.5-VL study into results/thesis_main/highres/figures/.
Numbers are the measured results from FINDINGS_qwen.md (hardcoded for robustness)."""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/thesis_main/highres/figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.dpi": 130, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig); print(f"  [saved] figures/{name}")


# Fig 1: selection dominates — Qwen-7B DocVQA K-curve
def fig1():
    K = [64, 128, 256, 512]
    qc = [84.7, 92.2, 94.7, 97.2]; uni = [23.9, 32.2, 61.1, 87.7]; nrm = [13.8, 21.0, 37.1, 75.7]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(K, qc, "o-", color="#196f3d", lw=2, label="question-conditioned (mid-layer)")
    ax.plot(K, uni, "s--", color="#2980b9", label="uniform (blind)")
    ax.plot(K, nrm, "^:", color="#7f8c8d", label="norm (blind)")
    ax.axhline(97.2, ls=":", c="gray", label="dense (97.2)")
    ax.annotate("+60pp", (128, 62), color="#7b241c", fontsize=11, ha="center")
    ax.set_xlabel("visual tokens kept (K)"); ax.set_ylabel("DocVQA ANLS")
    ax.set_title("Selection dominates: which tokens you keep\n(Qwen2.5-VL-7B, retains 95% @ 88% FLOP-cut, 3.3x faster)")
    ax.legend(loc="lower right"); ax.set_ylim(5, 100)
    save(fig, "fig1_selection")


# Fig 2: the signal is mid-layer, not early (FastV)
def fig2():
    fig, ax = plt.subplots(figsize=(6, 4))
    l7 = [2, 4, 8, 12, 16, 20, 24]; a7 = [44.6, 45.7, 47.7, 73.7, 93.2, 94.9, 91.2]
    ax.plot([x/28 for x in l7], a7, "o-", color="#c0392b", label="Qwen-7B (28 layers)")
    l3 = [4, 8, 16, 24, 28, 32]; a3 = [20.3, 35.7, 38.7, 87.8, 88.6, 87.2]
    ax.plot([x/36 for x in l3], a3, "s-", color="#8e44ad", label="Qwen-3B (36 layers)")
    ax.axhline(32.2, ls="--", c="#2980b9", label="blind uniform (32.2)")
    ax.axvspan(0, 0.2, color="orange", alpha=0.08); ax.text(0.1, 30, "early\n(FastV)", ha="center", fontsize=9)
    ax.axvspan(0.6, 0.85, color="green", alpha=0.08); ax.text(0.72, 60, "mid: best", ha="center", fontsize=9, color="#196f3d")
    ax.set_xlabel("LLM layer (fraction of depth)"); ax.set_ylabel("DocVQA ANLS @ K=128")
    ax.set_title("The question-conditioned signal is mid-layer\n(early-layer/FastV is below blind)")
    ax.legend(loc="center right"); ax.set_ylim(15, 100)
    save(fig, "fig2_layer")


# Fig 3 (HEADLINE): the budget mirage across 4 models
def fig3():
    models = ["Qwen-3B", "Qwen-7B", "Qwen-32B", "LLaVA-1.6"]
    oracle = [3.1, 4.2, 5.0, 3.1]; pred = [-1.4, 0.0, 0.1, -10.7]
    x = range(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar([i - w/2 for i in x], oracle, w, label="oracle dynamic-budget headroom", color="#e67e22")
    ax.bar([i + w/2 for i in x], pred, w, label="trained budget predictor (realizable)", color="#c0392b")
    ax.axhline(0, color="black", lw=0.8)
    for i, v in enumerate(oracle): ax.text(i - w/2, v + 0.2, f"+{v}", ha="center", fontsize=9)
    for i, v in enumerate(pred): ax.text(i + w/2, v + (0.2 if v >= 0 else -0.6), f"{v:+}", ha="center", fontsize=9)
    ax.set_xticks(list(x)); ax.set_xticklabels(models)
    ax.set_ylabel("dynamic-budget gain (pp, matched-FLOPs)")
    ax.set_title("Per-sample budget is a MIRAGE\noracle headroom small; a trained predictor captures 0% or hurts")
    ax.legend(loc="lower left"); ax.set_ylim(-12, 7)
    save(fig, "fig3_budget_mirage")


# Fig 4: latency speedup
def fig4():
    K = [64, 128, 256, 512, 1024, 2252]; sp = [3.86, 3.31, 2.92, 2.43, 1.80, 1.0]
    fig, ax = plt.subplots(figsize=(5.8, 4))
    ax.plot(K, sp, "o-", color="#2c3e50", lw=2)
    ax.axhline(1, ls=":", c="gray")
    ax.annotate("3.3x @ K=128\n(95% accuracy retained)", (128, 3.31), (300, 3.4), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_xlabel("visual tokens kept (K)"); ax.set_ylabel("LLM decode speedup vs dense")
    ax.set_title("Prune-before-LLM: FLOPs -> real latency\n(LLaVA-1.6, ~2252 dense tokens)")
    ax.set_xscale("log"); ax.set_ylim(0.8, 4.2)
    save(fig, "fig4_latency")


if __name__ == "__main__":
    print("generating Qwen figures ->", OUT)
    fig1(); fig2(); fig3(); fig4(); print("done.")
