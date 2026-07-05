# Dynamic-WHICH (textsim) — win/loss summary vs static, full-final matrix

Total rows: **40** (2 models × 4 datasets × 5 budgets; full manifests; all gates OK).
Labels: win = Dyn−Static > 0 · near-tie = −0.50 ≤ Dyn−Static ≤ 0 · loss = Dyn−Static < −0.50.

## Dynamic wins vs static — **6**

| Cell | Dyn−Static | FLOP red.% |
|---|--:|--:|
| qwen25vl7b × textvqa p25 | **+8.36** | 69.64 |
| qwen25vl7b × textvqa p35 | +7.79 | 60.56 |
| qwen25vl7b × textvqa p15 | +7.50 | 78.65 |
| qwen25vl7b × textvqa p50 | +5.94 | 46.84 |
| qwen25vl7b × docvqa p15 | +3.19 | 82.88 |
| qwen25vl7b × textvqa p75 | +1.37 | 23.63 |

All six wins are Qwen; five of six are the TextVQA sweep. The DocVQA p15 win is isolated
(all other DocVQA budgets lose) and reflects the static floor collapsing at 15%, not a DocVQA success.

## Near ties vs static — **2**

| Cell | Dyn−Static | FLOP red.% |
|---|--:|--:|
| qwen25vl7b × vqav2 p50 | −0.39 | 45.55 |
| qwen25vl7b × vqav2 p75 | −0.06 | 22.84 |

## Losses vs static — **32**

| Cell (budgets all losing) | Dyn−Static range |
|---|--:|
| llava15 × gqa p15–p75 (5) | −0.73 … −1.87 |
| llava15 × vqav2 p15–p75 (5) | −1.59 … −5.52 |
| llava15 × textvqa p15–p75 (5) | −3.91 … −8.57 |
| llava15 × docvqa p15–p75 (5) | −3.97 … −6.82 |
| qwen25vl7b × gqa p15–p75 (5) | −0.68 … −2.80 |
| qwen25vl7b × vqav2 p15/p25/p35 (3) | −0.95 … −2.61 |
| qwen25vl7b × docvqa p25/p35/p50/p75 (4) | −2.98 … −13.43 |

All 20 LLaVA cells are losses — the projector's visual embeddings are not natively aligned with the
LLM embedding table, so the cosine textsim score is weak on LLaVA regardless of dataset.

## Headline extremes

- **Best dynamic win:** qwen25vl7b × textvqa **p25 = +8.36 pp** vs static (67.53 vs 59.17;
  69.6% FLOP reduction vs dense).
- **Worst dynamic loss:** qwen25vl7b × docvqa **p50 = −13.43 pp** vs static (71.36 vs 84.79) —
  page-wide dense text is the anti-regime for question-conditioned selection.
- **Strongest efficiency-preserving near-tie:** qwen25vl7b × vqav2 **p50 = −0.39 pp at 45.6% FLOP
  reduction** (the best accuracy-per-FLOP trade among non-wins; p75 is the closest tie at −0.06 pp
  but saves only 22.8%).

*Source: `dynamic_which_dense_static_dynamic_comparison.csv` (40 rows). Read-only summary; no result
files modified.*
