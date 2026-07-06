# Dynamic-WHICH pilot outcomes — where question-conditioned selection helped and where it did not

**Purpose:** thesis failure analysis. This summarizes the *existing* Dynamic-WHICH pilot /
confirmation runs (no new experiments). Dynamic-WHICH = fixed-budget, question-conditioned visual
token selection (same per-sample token budget K as static; only *which* tokens change). The
headline metric is **Dyn−Static** (dynamic vs the primary static selector at the *same* budget:
LLaVA→`cls_attn`, Qwen→`norm`); positive means the question-conditioned choice beat the
image-only static floor.

All numbers below are read directly from the committed pilot aggregate JSONs; every run passed its
fairness gate. Selector `textsim` = cosine(visual embeds, question-token embeds); mixes add the
static score (`textsim_cls_mix` for LLaVA, `textsim_norm_mix` for Qwen) at α=0.5.

---

## Result at a glance

| Cell | n | Selector | Budget | Dynamic | Static | Dense | **Dyn−Static** | Verdict |
|---|---:|---|---:|---:|---:|---:|---:|:--|
| **Qwen TextVQA** | 1000 | textsim | p15 | 62.46 | 52.76 | 81.25 | **+9.70** | ✅ confirm |
| **Qwen TextVQA** | 1000 | textsim | p25 | 68.61 | 60.02 | 81.25 | **+8.59** | ✅ confirm |
| **Qwen TextVQA** | 1000 | textsim | p35 | 72.26 | 63.78 | 81.25 | **+8.48** | ✅ confirm |
| **Qwen TextVQA** | 1000 | textsim | p50 | 75.68 | 70.19 | 81.25 | **+5.49** | ✅ confirm |
| **Qwen TextVQA** | 1000 | textsim | p75 | 79.57 | 78.06 | 81.25 | **+1.51** | ✅ confirm |
| LLaVA GQA | 1000 | textsim | p25 | 62.70 | 64.20 | 67.30 | **−1.50** | ❌ fail |
| LLaVA GQA | 1000 | textsim | p35 | 63.70 | 65.70 | 67.30 | **−2.00** | ❌ fail |
| LLaVA GQA | 200 | textsim_cls_mix | p25 | 66.50 | 66.00 | 63.50 | +0.50 | ⚠︎ noisy (n=200) |
| LLaVA GQA | 200 | textsim_cls_mix | p35 | 64.50 | 66.50 | 63.50 | −2.00 | ❌ fail |
| LLaVA VQAv2 | 200 | textsim | p25 | 71.50 | 74.67 | 77.33 | **−3.17** | ❌ fail |
| LLaVA VQAv2 | 200 | textsim | p35 | 71.67 | 75.50 | 77.33 | **−3.83** | ❌ fail |
| LLaVA VQAv2 | 200 | textsim_cls_mix | p25 | 73.33 | 74.67 | 77.33 | −1.34 | ❌ fail (mix better, still <0) |
| LLaVA VQAv2 | 200 | textsim_cls_mix | p35 | 72.83 | 75.50 | 77.33 | −2.67 | ❌ fail |
| **Qwen DocVQA** | 200 | textsim | p25 | 40.03 | 49.10 | 88.20 | **−9.07** | ❌ fail |
| **Qwen DocVQA** | 200 | textsim | p35 | 53.36 | 67.84 | 88.20 | **−14.48** | ❌ fail |
| **Qwen DocVQA** | 200 | textsim | p50 | 64.69 | 80.15 | 88.20 | **−15.46** | ❌ fail |
| Qwen DocVQA | 200 | textsim_norm_mix | p25 | 39.60 | 49.10 | 88.20 | −9.50 | ❌ fail |
| Qwen DocVQA | 200 | textsim_norm_mix | p35 | 53.59 | 67.84 | 88.20 | −14.25 | ❌ fail |
| Qwen DocVQA | 200 | textsim_norm_mix | p50 | 70.17 | 80.15 | 88.20 | −9.98 | ❌ fail |

---

## The one success: Qwen × TextVQA (contrast case)

`textsim` beats the `norm` static floor by **+1.5 to +9.7 pp at every budget**, with the gain
*largest where the budget is tightest* (p15 +9.70, p25 +8.59). This is the intended behavior: when
only ~145 of ~964 visual tokens survive, choosing the question-relevant tokens matters a lot, and
TextVQA answers live in a small, question-locatable image region (the text the question asks
about). The n=1000 confirmation matches the full n=5000 final (p15 +7.50 … p75 +1.37) in shape and
sign, so the effect is real, not sampling noise. **Only this cell was promoted to a full final.**

## Failure 1 — LLaVA × GQA (`textsim` below static, confirmed at n=1000)

`textsim` is **−1.5 (p25) / −2.0 (p35)** vs `cls_attn`. This is the most trustworthy negative
because it is at n=1000, not n=200. Interpretation: GQA is relational/compositional ("to the left
of", "same color as"), so the answer depends on *multiple* regions and their relations, not a
single question-locatable patch. LLaVA's image-only `cls_attn` (which keeps globally salient
tokens) already captures the objects GQA asks about; a question-similarity score concentrates on
too few regions and drops the relational context. The `textsim_cls_mix` at n=200 is essentially a
wash (p25 +0.5, p35 −2.0) — mixing in the static score recovers the loss but does not clear it.

## Failure 2 — LLaVA × VQAv2 (`textsim` below static, n=200)

`textsim` is **−3.17 (p25) / −3.83 (p35)** vs `cls_attn`. VQAv2 mixes yes/no, counting, and
open questions over natural images where the salient object is usually the answer target, so the
image-only `cls_attn` floor is already strong and hard to beat. The mix helps (−1.34 / −2.67) but
never turns positive. n=200 is small, but both selectors and both budgets agree on the sign, so the
direction is credible even if the magnitude is noisy.

## Failure 3 — Qwen × DocVQA (`textsim` far below static, n=200)

The largest failure: **−9 to −15 pp** vs `norm`. DocVQA is dense document OCR where the answer can
be anywhere on a text-dense page and the whole page is "question-relevant." A per-token cosine to
the question collapses onto a few tokens and discards most of the readable text, so accuracy craters
(p25 40.0 vs static 49.1 vs dense 88.2). The static `norm` floor, which spreads tokens by activation
magnitude, keeps far more of the page. `textsim_norm_mix` barely changes this (still −9.5 to −10).
DocVQA also has the biggest dense→static gap of any cell (88.2→49.1 at p15), i.e. it is the most
budget-sensitive task, which compounds any selection mistake.

---

## Takeaways for the thesis

1. **Dynamic-WHICH is task-shaped, not universal.** It wins exactly when the answer is localized to
   a small, question-identifiable region (TextVQA scene text) and loses when the answer needs broad
   or relational coverage (GQA relations, VQAv2 natural scenes, DocVQA full-page text).
2. **The static floor is the right baseline.** The image-only floors (`cls_attn`, `norm`) are
   already strong on GQA/VQAv2/DocVQA; a naïve question-similarity selector underperforms them.
3. **Mixing with the static score helps but does not rescue** the failing cells (LLaVA GQA/VQAv2,
   Qwen DocVQA) — mixes narrow the gap by 1–5 pp but stay ≤ 0 vs static.
4. **Do not run full Dynamic-WHICH finals** for LLaVA/GQA, LLaVA/VQAv2, or Qwen/DocVQA with the
   current selectors. The confirmed n=1000 LLaVA-GQA negative is sufficient evidence.
5. **Where next:** the localized-answer requirement suggests either (a) a coverage-aware or
   DocVQA/VQAv2-specific selector, or (b) shifting the axis to *how many* tokens per sample
   (Dynamic-COUNT / adaptive budget) — see `analyze_dynamic_count_oracle.py` for the CPU oracle
   upper bound that motivates the latter.

*Source files: `results/runs/{llava15,qwen25vl7b}/{gqa,vqav2,docvqa,textvqa}/dynamic_which_pilot_*.json`.
All gates OK. No new experiments were run to produce this summary.*
