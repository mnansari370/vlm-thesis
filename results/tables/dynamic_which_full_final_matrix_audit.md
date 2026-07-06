# Dynamic-WHICH full-final textsim matrix — completeness audit

Expected cells: **40** (2 models × 4 datasets × 5 budgets, selector=textsim). complete=**40**, missing=**0**, invalid=**0**.

| Model | Dataset | p | Status | json | jsonl | n | gate | Dyn | Static | Dense | Dyn−Static | Dyn−Dense | Tok red.% | FLOP red.% |
|---|---|--:|---|:--:|:--:|--:|:--:|--:|--:|--:|--:|--:|--:|--:|
| llava15 | gqa | 15 | complete | Y | Y | 12578 | True | +55.41 | +56.14 | +61.42 | -0.73 | -6.01 | +80.36 | +80.85 |
| llava15 | gqa | 25 | complete | Y | Y | 12578 | True | +56.89 | +58.15 | +61.42 | -1.26 | -4.53 | +70.85 | +71.48 |
| llava15 | gqa | 35 | complete | Y | Y | 12578 | True | +57.54 | +59.41 | +61.42 | -1.87 | -3.88 | +61.34 | +62.07 |
| llava15 | gqa | 50 | complete | Y | Y | 12578 | True | +58.96 | +60.53 | +61.42 | -1.57 | -2.46 | +47.23 | +48.00 |
| llava15 | gqa | 75 | complete | Y | Y | 12578 | True | +60.80 | +61.53 | +61.42 | -0.73 | -0.62 | +23.62 | +24.17 |
| llava15 | vqav2 | 15 | complete | Y | Y | 25000 | True | +68.33 | +73.36 | +77.33 | -5.03 | -9.00 | +80.72 | +81.19 |
| llava15 | vqav2 | 25 | complete | Y | Y | 25000 | True | +70.03 | +75.55 | +77.33 | -5.52 | -7.30 | +71.16 | +71.79 |
| llava15 | vqav2 | 35 | complete | Y | Y | 25000 | True | +71.32 | +76.13 | +77.33 | -4.81 | -6.01 | +61.61 | +62.33 |
| llava15 | vqav2 | 50 | complete | Y | Y | 25000 | True | +73.14 | +76.94 | +77.33 | -3.80 | -4.19 | +47.44 | +48.21 |
| llava15 | vqav2 | 75 | complete | Y | Y | 25000 | True | +75.72 | +77.31 | +77.33 | -1.59 | -1.61 | +23.72 | +24.28 |
| llava15 | textvqa | 15 | complete | Y | Y | 5000 | True | +47.48 | +56.05 | +57.65 | -8.57 | -10.17 | +73.97 | +74.60 |
| llava15 | textvqa | 25 | complete | Y | Y | 5000 | True | +49.10 | +55.97 | +57.65 | -6.87 | -8.55 | +65.22 | +65.96 |
| llava15 | textvqa | 35 | complete | Y | Y | 5000 | True | +49.76 | +56.71 | +57.65 | -6.95 | -7.89 | +56.46 | +57.27 |
| llava15 | textvqa | 50 | complete | Y | Y | 5000 | True | +50.81 | +56.40 | +57.65 | -5.59 | -6.84 | +43.48 | +44.29 |
| llava15 | textvqa | 75 | complete | Y | Y | 5000 | True | +53.48 | +57.39 | +57.65 | -3.91 | -4.17 | +21.74 | +22.30 |
| llava15 | docvqa | 15 | complete | Y | Y | 5349 | True | +11.41 | +17.73 | +21.53 | -6.32 | -10.12 | +80.18 | +80.67 |
| llava15 | docvqa | 25 | complete | Y | Y | 5349 | True | +12.50 | +19.32 | +21.53 | -6.82 | -9.03 | +70.69 | +71.33 |
| llava15 | docvqa | 35 | complete | Y | Y | 5349 | True | +13.35 | +20.09 | +21.53 | -6.74 | -8.18 | +61.20 | +61.93 |
| llava15 | docvqa | 50 | complete | Y | Y | 5349 | True | +14.47 | +20.75 | +21.53 | -6.28 | -7.06 | +47.13 | +47.90 |
| llava15 | docvqa | 75 | complete | Y | Y | 5349 | True | +17.39 | +21.36 | +21.53 | -3.97 | -4.14 | +23.56 | +24.12 |
| qwen25vl7b | gqa | 15 | complete | Y | Y | 12578 | True | +53.45 | +56.04 | +60.96 | -2.59 | -7.51 | +76.28 | +76.56 |
| qwen25vl7b | gqa | 25 | complete | Y | Y | 12578 | True | +55.97 | +58.77 | +60.96 | -2.80 | -4.99 | +67.37 | +67.71 |
| qwen25vl7b | gqa | 35 | complete | Y | Y | 12578 | True | +57.47 | +59.99 | +60.96 | -2.52 | -3.49 | +58.34 | +58.71 |
| qwen25vl7b | gqa | 50 | complete | Y | Y | 12578 | True | +58.87 | +60.53 | +60.96 | -1.66 | -2.09 | +44.91 | +45.29 |
| qwen25vl7b | gqa | 75 | complete | Y | Y | 12578 | True | +60.17 | +60.85 | +60.96 | -0.68 | -0.79 | +22.44 | +22.71 |
| qwen25vl7b | vqav2 | 15 | complete | Y | Y | 25000 | True | +73.55 | +76.16 | +84.27 | -2.61 | -10.72 | +76.74 | +77.01 |
| qwen25vl7b | vqav2 | 25 | complete | Y | Y | 25000 | True | +77.50 | +78.89 | +84.27 | -1.39 | -6.77 | +67.76 | +68.10 |
| qwen25vl7b | vqav2 | 35 | complete | Y | Y | 25000 | True | +79.54 | +80.49 | +84.27 | -0.95 | -4.73 | +58.68 | +59.06 |
| qwen25vl7b | vqav2 | 50 | complete | Y | Y | 25000 | True | +81.70 | +82.09 | +84.27 | -0.39 | -2.57 | +45.17 | +45.55 |
| qwen25vl7b | vqav2 | 75 | complete | Y | Y | 25000 | True | +83.60 | +83.66 | +84.27 | -0.06 | -0.67 | +22.58 | +22.84 |
| qwen25vl7b | textvqa | 15 | complete | Y | Y | 5000 | True | +60.56 | +53.06 | +81.06 | +7.50 | -20.50 | +77.98 | +78.65 |
| qwen25vl7b | textvqa | 25 | complete | Y | Y | 5000 | True | +67.53 | +59.17 | +81.06 | +8.36 | -13.53 | +68.81 | +69.64 |
| qwen25vl7b | textvqa | 35 | complete | Y | Y | 5000 | True | +71.36 | +63.57 | +81.06 | +7.79 | -9.70 | +59.61 | +60.56 |
| qwen25vl7b | textvqa | 50 | complete | Y | Y | 5000 | True | +76.08 | +70.14 | +81.06 | +5.94 | -4.98 | +45.86 | +46.84 |
| qwen25vl7b | textvqa | 75 | complete | Y | Y | 5000 | True | +79.80 | +78.43 | +81.06 | +1.37 | -1.26 | +22.94 | +23.63 |
| qwen25vl7b | docvqa | 15 | complete | Y | Y | 5349 | True | +33.09 | +29.90 | +94.76 | +3.19 | -61.67 | +82.19 | +82.88 |
| qwen25vl7b | docvqa | 25 | complete | Y | Y | 5349 | True | +46.57 | +49.55 | +94.76 | -2.98 | -48.19 | +72.51 | +73.44 |
| qwen25vl7b | docvqa | 35 | complete | Y | Y | 5349 | True | +57.53 | +68.86 | +94.76 | -11.33 | -37.23 | +62.86 | +63.95 |
| qwen25vl7b | docvqa | 50 | complete | Y | Y | 5349 | True | +71.36 | +84.79 | +94.76 | -13.43 | -23.40 | +48.34 | +49.51 |
| qwen25vl7b | docvqa | 75 | complete | Y | Y | 5349 | True | +88.04 | +93.98 | +94.76 | -5.94 | -6.72 | +24.18 | +25.03 |

## Missing jobs (to launch)

- none — the full-final textsim matrix is complete.

## Framing

This is a **complete** evaluation matrix, not cherry-picking. Qwen×TextVQA textsim is the already-validated full success; the remaining full finals are added so Dynamic-WHICH is compared against dense and static on the SAME footing everywhere — including datasets where the pilot evidence is negative. Negative full results are reported, not hidden.
