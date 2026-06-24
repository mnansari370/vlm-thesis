"""
Generate the 4 paper figures (Path A) from measured data into results/thesis_main/highres/figures/.
All numbers are the n=300 frozen-LLaVA results from FINDINGS.md; effect sizes are large
and stable, so the figures are robust to the n=1000 stability check.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.dpi": 130, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] figures/{name}.png/.pdf")


# ---- Fig 1: resolution x task — retention@K=64 ----
def fig1():
    tasks = ["TextVQA\n(reading)", "GQA\n(reasoning)", "DocVQA\n(document)"]
    lo = [102, 85, None]      # LLaVA-1.5 retention% @K=64 (DocVQA: 336px can't read)
    hi = [57, 66, 19]         # LLaVA-1.6 retention% @K=64
    x = range(len(tasks)); w = 0.38
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - w/2 for i in x], [v if v else 0 for v in lo], w, label="LLaVA-1.5 (576 tok)",
           color="#7fb3d5")
    ax.bar([i + w/2 for i in x], hi, w, label="LLaVA-1.6 (2302 tok)", color="#e59866")
    ax.axhline(100, ls="--", c="gray", lw=1)
    ax.text(2 - w/2, 5, "cannot\nread", ha="center", va="bottom", fontsize=9, color="gray")
    for i, v in enumerate(hi):
        ax.text(i + w/2, v + 1, f"{v}%", ha="center", fontsize=9)
    for i, v in enumerate(lo):
        if v: ax.text(i - w/2, v + 1, f"{v}%", ha="center", fontsize=9)
    ax.set_xticks(list(x)); ax.set_xticklabels(tasks)
    ax.set_ylabel("accuracy retained at K=64\n(% of own dense)")
    ax.set_title("Pruning room = resolution × task token-demand")
    ax.set_ylim(0, 115); ax.legend(loc="upper right")
    save(fig, "fig1_resolution_task")


# ---- Fig 2: selector / layer (TextVQA K=128) ----
def fig2():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    # 2a: layer sweep
    layers = [2, 4, 6, 8, 12, 16, 20]
    acc = [39.13, 40.27, 38.53, 41.47, 51.67, 54.67, 54.0]
    a.plot(layers, acc, "o-", color="#c0392b")
    a.axhline(44.2, ls="--", c="#2980b9", label="blind CLS-attn (44.2)")
    a.fill_between([1, 9], 0, 60, color="orange", alpha=0.06)
    a.fill_between([9, 21], 0, 60, color="green", alpha=0.06)
    a.text(5, 35, "early (FastV)\n< blind", ha="center", fontsize=9, color="#7b241c")
    a.text(16, 57, "mid: > blind", ha="center", fontsize=9, color="#196f3d")
    a.set_xlabel("LLM layer of question→visual attention")
    a.set_ylabel("TextVQA soft-acc @ K=128"); a.set_ylim(33, 60)
    a.set_title("(a) the selection signal is mid-layer"); a.legend(loc="lower right")
    # 2b: selector bars + question-conditioning control
    names = ["FastV\n(L2)", "blind\nCLS", "QC mismatched\nquestion", "QC real\nquestion", "dense"]
    vals = [39.1, 44.2, 43.8, 54.7, 61.4]
    cols = ["#c0392b", "#2980b9", "#95a5a6", "#196f3d", "#34495e"]
    b.bar(names, vals, color=cols)
    for i, v in enumerate(vals):
        b.text(i, v + 0.6, f"{v}", ha="center", fontsize=9)
    b.set_ylabel("TextVQA soft-acc @ K=128"); b.set_ylim(0, 68)
    b.set_title("(b) the gain is question-conditioned\n(mismatched≈blind; 104% from question)")
    save(fig, "fig2_selection")


# ---- Fig 3: accuracy-vs-FLOPs frontier (DocVQA) ----
def fig3():
    from src.analysis.flops import fastv_full_flops
    NT, DENSE = 30, 2302
    Fd = fastv_full_flops(DENSE, NT)
    K = [64, 128, 256, 512, 768, 1152]
    blind = [12.77, 27.76, 41.35, 50.72, 57.86, 63.03]
    qc = [49.23, 55.13, 57.44, 61.44, 63.22, 63.89]
    red = [100 * (1 - fastv_full_flops(k, NT) / Fd) for k in K]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(red, qc, "o-", color="#196f3d", label="QC (mid-layer, question-cond.)")
    ax.plot(red, blind, "s--", color="#2980b9", label="blind CLS-attn")
    ax.axhline(67.19, ls=":", c="gray", label="dense (67.2)")
    for r, q, bl in zip(red, qc, blind):
        ax.annotate(f"{q-bl:+.0f}", (r, (q+bl)/2), fontsize=8, color="#7b241c", ha="center")
    ax.set_xlabel("FLOPs reduction vs dense (%)")
    ax.set_ylabel("DocVQA ANLS"); ax.set_title("Accuracy-vs-FLOPs frontier (matched FLOPs)")
    ax.invert_xaxis(); ax.legend(loc="lower left"); ax.set_ylim(5, 72)
    save(fig, "fig3_flops_frontier")


# ---- Fig 4: oracle-noise decomposition ----
def fig4():
    benches = ["TextVQA", "DocVQA", "ChartQA"]
    naive = [7.0, 9.7, 6.7]
    honest = [2.6, 5.0, 0.5]
    x = range(len(benches)); w = 0.38
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - w/2 for i in x], naive, w, label="naive oracle band (inflated)", color="#e74c3c")
    ax.bar([i + w/2 for i in x], honest, w, label="monotone, noise-free (honest)", color="#27ae60")
    for i, v in enumerate(naive): ax.text(i - w/2, v + 0.1, f"+{v}", ha="center", fontsize=9)
    for i, v in enumerate(honest): ax.text(i + w/2, v + 0.1, f"+{v}", ha="center", fontsize=9)
    ax.set_xticks(list(x)); ax.set_xticklabels(benches)
    ax.set_ylabel("dynamic-budget headroom (pp)")
    ax.set_title("Naive per-sample oracle overstates\ndynamic-budget headroom 2–13×")
    ax.legend(loc="upper right"); ax.set_ylim(0, 11)
    save(fig, "fig4_oracle_decomposition")


if __name__ == "__main__":
    print("generating figures ->", OUT)
    fig1(); fig2(); fig3(); fig4()
    print("done.")
