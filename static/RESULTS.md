# Static — results

Source: `results/final_scope/tables/final_dense_static_dynamic_comparison.md` (section B) and
`static_final_summary.csv`. Full manifests, all fairness gates passed. Δdense in parentheses.

| Model | Dataset | Dense | p15 | p25 | p35 | p50 | p75 |
|---|---|--:|--:|--:|--:|--:|--:|
| LLaVA | GQA | 61.42 | 56.14 (−5.28) | 58.15 | 59.41 | 60.53 | **61.53 (+0.11)** |
| LLaVA | VQAv2 | 77.33 | 73.36 (−3.97) | 75.55 | 76.13 | 76.94 | **77.31 (−0.02)** |
| LLaVA | TextVQA | 57.65 | 56.05 (−1.60) | 55.97 | 56.71 | 56.40 | **57.39 (−0.26)** |
| LLaVA | DocVQA | 21.53 | 17.73 (−3.80) | 19.32 | 20.09 | 20.75 | **21.36 (−0.17)** |
| Qwen | GQA | 60.96 | 56.04 (−4.92) | 58.77 | 59.99 | 60.53 | **60.85 (−0.11)** |
| Qwen | VQAv2 | 84.27 | 76.16 (−8.11) | 78.89 | 80.49 | 82.09 | **83.66 (−0.61)** |
| Qwen | TextVQA | 81.06 | 53.06 (−28.00) | 59.17 | 63.57 | 70.14 | **78.43 (−2.63)** |
| Qwen | DocVQA | 94.76 | 29.90 (−64.86) | 49.55 | 68.86 | 84.79 | **93.98 (−0.78)** |

FLOP reduction is ~22–25% at p75 and ~74–83% at p15 on every cell.

Reading:

- **Near-dense at p75 everywhere** (within 0.02–2.63 pp), so roughly a quarter of the FLOPs is
  removed essentially for free, with no question information at all.
- **Graceful on scene tasks**: GQA and VQAv2 lose only ~4–8 pp even at p15 (≈80% of tokens gone).
- **Catastrophic where evidence is dense and spatial**: Qwen TextVQA and Qwen DocVQA collapse at
  p15. An image-only criterion cannot know *which* few tokens the question needs — the opening any
  question-conditioned method must exploit, and the steepness that governs whether adaptive budgets
  can pay off.
