# Final thesis comparison — Dense / Static / Dynamic-WHICH / DC-D / DC-C

All numbers read directly from the saved aggregates under `results/runs/`. Dense/static/WHICH rows are FULL-manifest; DC-D/DC-C rows are the held-out 80% eval split (their Δ columns are computed against dense/static over the SAME ids — fair). Pure Dynamic-COUNT (cls_attn/norm) and COUNT-on-WHICH (textsim) are never mixed. Labels vs static curve at matched FLOPs: win > +0.5, near-tie ±0.5, loss < −0.5.

## A. Dense baseline

| Model | Dataset | Method | Selector | Setting | n | Score | Vis/K̄ | TFLOPs | TokRed% | FlopRed% | Δdense | Δstatic-curve | Label | Population |
|---|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|---|---|
| llava15 | gqa | dense | - | all tokens | 12578 | 61.42 | 576.00 | 3.17 | 0.00 | 0.00 | +0.00 | — | reference | full |
| llava15 | vqav2 | dense | - | all tokens | 25000 | 77.33 | 576.00 | 3.15 | 0.00 | 0.00 | +0.00 | — | reference | full |
| llava15 | textvqa | dense | - | all tokens | 5000 | 57.65 | 576.00 | 3.45 | 0.00 | 0.00 | +0.00 | — | reference | full |
| llava15 | docvqa | dense | - | all tokens | 5349 | 21.53 | 576.00 | 3.17 | 0.00 | 0.00 | +0.00 | — | reference | full |
| qwen25vl7b | gqa | dense | - | all tokens | 12578 | 60.96 | 358.60 | 2.12 | 0.00 | 0.00 | +0.00 | — | reference | full |
| qwen25vl7b | vqav2 | dense | - | all tokens | 25000 | 84.27 | 358.60 | 2.11 | 0.00 | 0.00 | +0.00 | — | reference | full |
| qwen25vl7b | textvqa | dense | - | all tokens | 5000 | 81.06 | 963.60 | 5.73 | 0.00 | 0.00 | +0.00 | — | reference | full |
| qwen25vl7b | docvqa | dense | - | all tokens | 5349 | 94.76 | 1229.10 | 6.99 | 0.00 | 0.00 | +0.00 | — | reference | full |

## B. Static pruning (anchors)

| Model | Dataset | Method | Selector | Setting | n | Score | Vis/K̄ | TFLOPs | TokRed% | FlopRed% | Δdense | Δstatic-curve | Label | Population |
|---|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|---|---|
| llava15 | gqa | static | cls_attn | p15 | 12578 | 56.14 | 86.00 | 0.61 | 80.36 | 80.85 | -5.28 | — | baseline | full |
| llava15 | gqa | static | cls_attn | p25 | 12578 | 58.15 | 144.00 | 0.90 | 70.85 | 71.48 | -3.27 | — | baseline | full |
| llava15 | gqa | static | cls_attn | p35 | 12578 | 59.41 | 202.00 | 1.20 | 61.34 | 62.07 | -2.01 | — | baseline | full |
| llava15 | gqa | static | cls_attn | p50 | 12578 | 60.53 | 288.00 | 1.65 | 47.23 | 48.00 | -0.89 | — | baseline | full |
| llava15 | gqa | static | cls_attn | p75 | 12578 | 61.53 | 432.00 | 2.40 | 23.62 | 24.17 | +0.11 | — | baseline | full |
| llava15 | vqav2 | static | cls_attn | p15 | 25000 | 73.36 | 86.00 | 0.59 | 80.72 | 81.19 | -3.97 | — | baseline | full |
| llava15 | vqav2 | static | cls_attn | p25 | 25000 | 75.55 | 144.00 | 0.89 | 71.16 | 71.79 | -1.78 | — | baseline | full |
| llava15 | vqav2 | static | cls_attn | p35 | 25000 | 76.13 | 202.00 | 1.19 | 61.61 | 62.33 | -1.20 | — | baseline | full |
| llava15 | vqav2 | static | cls_attn | p50 | 25000 | 76.94 | 288.00 | 1.63 | 47.44 | 48.21 | -0.39 | — | baseline | full |
| llava15 | vqav2 | static | cls_attn | p75 | 25000 | 77.31 | 432.00 | 2.39 | 23.72 | 24.28 | -0.02 | — | baseline | full |
| llava15 | textvqa | static | cls_attn | p15 | 5000 | 56.05 | 86.00 | 0.88 | 73.97 | 74.60 | -1.60 | — | baseline | full |
| llava15 | textvqa | static | cls_attn | p25 | 5000 | 55.97 | 144.00 | 1.17 | 65.22 | 65.96 | -1.68 | — | baseline | full |
| llava15 | textvqa | static | cls_attn | p35 | 5000 | 56.71 | 202.00 | 1.47 | 56.46 | 57.27 | -0.94 | — | baseline | full |
| llava15 | textvqa | static | cls_attn | p50 | 5000 | 56.40 | 288.00 | 1.92 | 43.48 | 44.29 | -1.25 | — | baseline | full |
| llava15 | textvqa | static | cls_attn | p75 | 5000 | 57.39 | 432.00 | 2.68 | 21.74 | 22.30 | -0.26 | — | baseline | full |
| llava15 | docvqa | static | cls_attn | p15 | 5349 | 17.73 | 86.00 | 0.61 | 80.18 | 80.67 | -3.80 | — | baseline | full |
| llava15 | docvqa | static | cls_attn | p25 | 5349 | 19.32 | 144.00 | 0.91 | 70.69 | 71.33 | -2.21 | — | baseline | full |
| llava15 | docvqa | static | cls_attn | p35 | 5349 | 20.09 | 202.00 | 1.21 | 61.20 | 61.93 | -1.44 | — | baseline | full |
| llava15 | docvqa | static | cls_attn | p50 | 5349 | 20.75 | 288.00 | 1.65 | 47.13 | 47.90 | -0.78 | — | baseline | full |
| llava15 | docvqa | static | cls_attn | p75 | 5349 | 21.36 | 432.00 | 2.41 | 23.56 | 24.12 | -0.17 | — | baseline | full |
| qwen25vl7b | gqa | static | norm | p15 | 12578 | 56.04 | 54.00 | 0.50 | 76.28 | 76.56 | -4.92 | — | baseline | full |
| qwen25vl7b | gqa | static | norm | p25 | 12578 | 58.77 | 89.60 | 0.69 | 67.37 | 67.71 | -2.19 | — | baseline | full |
| qwen25vl7b | gqa | static | norm | p35 | 12578 | 59.99 | 125.70 | 0.88 | 58.34 | 58.71 | -0.97 | — | baseline | full |
| qwen25vl7b | gqa | static | norm | p50 | 12578 | 60.53 | 179.30 | 1.16 | 44.91 | 45.29 | -0.43 | — | baseline | full |
| qwen25vl7b | gqa | static | norm | p75 | 12578 | 60.85 | 269.00 | 1.64 | 22.44 | 22.71 | -0.11 | — | baseline | full |
| qwen25vl7b | vqav2 | static | norm | p15 | 25000 | 76.16 | 54.00 | 0.49 | 76.74 | 77.01 | -8.11 | — | baseline | full |
| qwen25vl7b | vqav2 | static | norm | p25 | 25000 | 78.89 | 89.60 | 0.67 | 67.76 | 68.10 | -5.38 | — | baseline | full |
| qwen25vl7b | vqav2 | static | norm | p35 | 25000 | 80.49 | 125.70 | 0.86 | 58.68 | 59.06 | -3.78 | — | baseline | full |
| qwen25vl7b | vqav2 | static | norm | p50 | 25000 | 82.09 | 179.30 | 1.15 | 45.17 | 45.55 | -2.18 | — | baseline | full |
| qwen25vl7b | vqav2 | static | norm | p75 | 25000 | 83.66 | 269.00 | 1.63 | 22.58 | 22.84 | -0.61 | — | baseline | full |
| qwen25vl7b | textvqa | static | norm | p15 | 5000 | 53.06 | 144.60 | 1.22 | 77.98 | 78.65 | -28.00 | — | baseline | full |
| qwen25vl7b | textvqa | static | norm | p25 | 5000 | 59.17 | 241.00 | 1.74 | 68.81 | 69.64 | -21.89 | — | baseline | full |
| qwen25vl7b | textvqa | static | norm | p35 | 5000 | 63.57 | 337.50 | 2.26 | 59.61 | 60.56 | -17.49 | — | baseline | full |
| qwen25vl7b | textvqa | static | norm | p50 | 5000 | 70.14 | 481.90 | 3.05 | 45.86 | 46.84 | -10.92 | — | baseline | full |
| qwen25vl7b | textvqa | static | norm | p75 | 5000 | 78.43 | 722.70 | 4.38 | 22.94 | 23.63 | -2.63 | — | baseline | full |
| qwen25vl7b | docvqa | static | norm | p15 | 5349 | 29.90 | 184.30 | 1.20 | 82.19 | 82.88 | -64.86 | — | baseline | full |
| qwen25vl7b | docvqa | static | norm | p25 | 5349 | 49.55 | 307.40 | 1.86 | 72.51 | 73.44 | -45.21 | — | baseline | full |
| qwen25vl7b | docvqa | static | norm | p35 | 5349 | 68.86 | 430.10 | 2.52 | 62.86 | 63.95 | -25.90 | — | baseline | full |
| qwen25vl7b | docvqa | static | norm | p50 | 5349 | 84.79 | 614.60 | 3.53 | 48.34 | 49.51 | -9.97 | — | baseline | full |
| qwen25vl7b | docvqa | static | norm | p75 | 5349 | 93.98 | 921.70 | 5.24 | 24.18 | 25.03 | -0.78 | — | baseline | full |

## C. Dynamic-WHICH (textsim, full finals)

| Model | Dataset | Method | Selector | Setting | n | Score | Vis/K̄ | TFLOPs | TokRed% | FlopRed% | Δdense | Δstatic-curve | Label | Population |
|---|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|---|---|
| llava15 | gqa | dynamic_which | textsim | p15 | 12578 | 55.41 | 86.00 | 0.61 | 80.36 | 80.85 | -6.01 | -0.73 | loss | full |
| llava15 | gqa | dynamic_which | textsim | p25 | 12578 | 56.89 | 144.00 | 0.90 | 70.85 | 71.48 | -4.53 | -1.26 | loss | full |
| llava15 | gqa | dynamic_which | textsim | p35 | 12578 | 57.54 | 202.00 | 1.20 | 61.34 | 62.07 | -3.88 | -1.87 | loss | full |
| llava15 | gqa | dynamic_which | textsim | p50 | 12578 | 58.96 | 288.00 | 1.65 | 47.23 | 48.00 | -2.46 | -1.57 | loss | full |
| llava15 | gqa | dynamic_which | textsim | p75 | 12578 | 60.80 | 432.00 | 2.40 | 23.62 | 24.17 | -0.62 | -0.73 | loss | full |
| llava15 | vqav2 | dynamic_which | textsim | p15 | 25000 | 68.33 | 86.00 | 0.59 | 80.72 | 81.19 | -9.00 | -5.03 | loss | full |
| llava15 | vqav2 | dynamic_which | textsim | p25 | 25000 | 70.03 | 144.00 | 0.89 | 71.16 | 71.79 | -7.30 | -5.52 | loss | full |
| llava15 | vqav2 | dynamic_which | textsim | p35 | 25000 | 71.32 | 202.00 | 1.19 | 61.61 | 62.33 | -6.01 | -4.81 | loss | full |
| llava15 | vqav2 | dynamic_which | textsim | p50 | 25000 | 73.14 | 288.00 | 1.63 | 47.44 | 48.21 | -4.19 | -3.80 | loss | full |
| llava15 | vqav2 | dynamic_which | textsim | p75 | 25000 | 75.72 | 432.00 | 2.39 | 23.72 | 24.28 | -1.61 | -1.59 | loss | full |
| llava15 | textvqa | dynamic_which | textsim | p15 | 5000 | 47.48 | 86.00 | 0.88 | 73.97 | 74.60 | -10.17 | -8.57 | loss | full |
| llava15 | textvqa | dynamic_which | textsim | p25 | 5000 | 49.10 | 144.00 | 1.17 | 65.22 | 65.96 | -8.55 | -6.87 | loss | full |
| llava15 | textvqa | dynamic_which | textsim | p35 | 5000 | 49.76 | 202.00 | 1.47 | 56.46 | 57.27 | -7.89 | -6.95 | loss | full |
| llava15 | textvqa | dynamic_which | textsim | p50 | 5000 | 50.81 | 288.00 | 1.92 | 43.48 | 44.29 | -6.84 | -5.59 | loss | full |
| llava15 | textvqa | dynamic_which | textsim | p75 | 5000 | 53.48 | 432.00 | 2.68 | 21.74 | 22.30 | -4.17 | -3.91 | loss | full |
| llava15 | docvqa | dynamic_which | textsim | p15 | 5349 | 11.41 | 86.00 | 0.61 | 80.18 | 80.67 | -10.12 | -6.32 | loss | full |
| llava15 | docvqa | dynamic_which | textsim | p25 | 5349 | 12.50 | 144.00 | 0.91 | 70.69 | 71.33 | -9.03 | -6.82 | loss | full |
| llava15 | docvqa | dynamic_which | textsim | p35 | 5349 | 13.35 | 202.00 | 1.21 | 61.20 | 61.93 | -8.18 | -6.74 | loss | full |
| llava15 | docvqa | dynamic_which | textsim | p50 | 5349 | 14.47 | 288.00 | 1.65 | 47.13 | 47.90 | -7.06 | -6.28 | loss | full |
| llava15 | docvqa | dynamic_which | textsim | p75 | 5349 | 17.39 | 432.00 | 2.41 | 23.56 | 24.12 | -4.14 | -3.97 | loss | full |
| qwen25vl7b | gqa | dynamic_which | textsim | p15 | 12578 | 53.45 | 54.00 | 0.50 | 76.28 | 76.56 | -7.51 | -2.59 | loss | full |
| qwen25vl7b | gqa | dynamic_which | textsim | p25 | 12578 | 55.97 | 89.60 | 0.69 | 67.37 | 67.71 | -4.99 | -2.80 | loss | full |
| qwen25vl7b | gqa | dynamic_which | textsim | p35 | 12578 | 57.47 | 125.70 | 0.88 | 58.34 | 58.71 | -3.49 | -2.52 | loss | full |
| qwen25vl7b | gqa | dynamic_which | textsim | p50 | 12578 | 58.87 | 179.30 | 1.16 | 44.91 | 45.29 | -2.09 | -1.66 | loss | full |
| qwen25vl7b | gqa | dynamic_which | textsim | p75 | 12578 | 60.17 | 269.00 | 1.64 | 22.44 | 22.71 | -0.79 | -0.68 | loss | full |
| qwen25vl7b | vqav2 | dynamic_which | textsim | p15 | 25000 | 73.55 | 54.00 | 0.49 | 76.74 | 77.01 | -10.72 | -2.61 | loss | full |
| qwen25vl7b | vqav2 | dynamic_which | textsim | p25 | 25000 | 77.50 | 89.60 | 0.67 | 67.76 | 68.10 | -6.77 | -1.39 | loss | full |
| qwen25vl7b | vqav2 | dynamic_which | textsim | p35 | 25000 | 79.54 | 125.70 | 0.86 | 58.68 | 59.06 | -4.73 | -0.95 | loss | full |
| qwen25vl7b | vqav2 | dynamic_which | textsim | p50 | 25000 | 81.70 | 179.30 | 1.15 | 45.17 | 45.55 | -2.57 | -0.39 | near-tie | full |
| qwen25vl7b | vqav2 | dynamic_which | textsim | p75 | 25000 | 83.60 | 269.00 | 1.63 | 22.58 | 22.84 | -0.67 | -0.06 | near-tie | full |
| qwen25vl7b | textvqa | dynamic_which | textsim | p15 | 5000 | 60.56 | 144.60 | 1.22 | 77.98 | 78.65 | -20.50 | +7.50 | win | full |
| qwen25vl7b | textvqa | dynamic_which | textsim | p25 | 5000 | 67.53 | 241.00 | 1.74 | 68.81 | 69.64 | -13.53 | +8.36 | win | full |
| qwen25vl7b | textvqa | dynamic_which | textsim | p35 | 5000 | 71.36 | 337.50 | 2.26 | 59.61 | 60.56 | -9.70 | +7.79 | win | full |
| qwen25vl7b | textvqa | dynamic_which | textsim | p50 | 5000 | 76.08 | 481.90 | 3.05 | 45.86 | 46.84 | -4.98 | +5.94 | win | full |
| qwen25vl7b | textvqa | dynamic_which | textsim | p75 | 5000 | 79.80 | 722.70 | 4.38 | 22.94 | 23.63 | -1.26 | +1.37 | win | full |
| qwen25vl7b | docvqa | dynamic_which | textsim | p15 | 5349 | 33.09 | 184.30 | 1.20 | 82.19 | 82.88 | -61.67 | +3.19 | win | full |
| qwen25vl7b | docvqa | dynamic_which | textsim | p25 | 5349 | 46.57 | 307.40 | 1.86 | 72.51 | 73.44 | -48.19 | -2.98 | loss | full |
| qwen25vl7b | docvqa | dynamic_which | textsim | p35 | 5349 | 57.53 | 430.10 | 2.52 | 62.86 | 63.95 | -37.23 | -11.33 | loss | full |
| qwen25vl7b | docvqa | dynamic_which | textsim | p50 | 5349 | 71.36 | 614.60 | 3.53 | 48.34 | 49.51 | -23.40 | -13.43 | loss | full |
| qwen25vl7b | docvqa | dynamic_which | textsim | p75 | 5349 | 88.04 | 921.70 | 5.24 | 24.18 | 25.03 | -6.72 | -5.94 | loss | full |

## D. Dynamic-COUNT DC-D (discrete baseline; eval-80%)

| Model | Dataset | Method | Selector | Setting | n | Score | Vis/K̄ | TFLOPs | TokRed% | FlopRed% | Δdense | Δstatic-curve | Label | Population |
|---|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|---|---|
| llava15 | gqa | dc_d | cls_attn | probe p25→p50 (esc 1.91%) | 10062 | 58.03 | 149.50 | 0.94 | 74.05 | 70.49 | -3.11 | -0.10 | near-tie | eval-80% |
| llava15 | vqav2 | dc_d | cls_attn | probe p25→p50 (esc 9.6%) | 20000 | 75.89 | 171.60 | 1.05 | 70.20 | 66.81 | -1.31 | +0.06 | near-tie | eval-80% |
| llava15 | textvqa | dc_d | cls_attn | probe p15→p75 (esc 50.88%) | 4000 | 57.12 | 305.80 | 2.24 | 46.91 | 35.12 | -0.26 | +0.75 | win | eval-80% |
| llava15 | docvqa | dc_d | cls_attn | probe p25→p75 (esc 74.29%) | 4279 | 20.83 | 464.90 | 2.70 | 19.28 | 14.85 | -1.08 | -0.89 | loss | eval-80% |
| qwen25vl7b | gqa | dc_d | norm | probe p25→p75 (esc 40.1%) | 10062 | 60.62 | 196.30 | 1.34 | 45.01 | 36.78 | -0.46 | -0.03 | near-tie | eval-80% |
| qwen25vl7b | vqav2 | dc_d | norm | probe p25→p75 (esc 4.89%) | 20000 | 79.80 | 102.60 | 0.75 | 71.39 | 64.36 | -4.56 | +0.21 | near-tie | eval-80% |
| qwen25vl7b | textvqa | dc_d | norm | probe p25→p75 (esc 13.2%) | 4000 | 64.06 | 337.00 | 2.32 | 65.02 | 59.46 | -16.96 | +0.02 | near-tie | eval-80% |
| qwen25vl7b | textvqa | dc_d_count_on_which | textsim | probe p15→p50 (esc 5.67%) | 4000 | 62.07 | 172.10 | 1.40 | 82.13 | 75.64 | -18.95 | +6.99 | win | eval-80% |
| qwen25vl7b | docvqa | dc_d | norm | probe p25→p50 (esc 9.46%) | 4279 | 53.09 | 364.90 | 2.19 | 70.25 | 68.64 | -41.57 | -4.61 | loss | eval-80% |

## E. Dynamic-COUNT DC-C (continuous integer K_i — MAIN; eval-80%)

| Model | Dataset | Method | Selector | Setting | n | Score | Vis/K̄ | TFLOPs | TokRed% | FlopRed% | Δdense | Δstatic-curve | Label | Population |
|---|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|---|---|
| llava15 | gqa | dc_c | cls_attn | ridge λ0.6 (esc 89.92%, K̄=233, uniqK=216) | 10062 | 59.77 | 366.70 | 2.19 | 36.33 | 30.68 | -1.37 | -1.26 | loss | eval-80% |
| llava15 | gqa | dc_c | cls_attn | rule λ0.6 (esc 100.0%, K̄=190, uniqK=2) | 10062 | 58.58 | 334.10 | 2.04 | 42.01 | 35.46 | -2.56 | -2.24 | loss | eval-80% |
| llava15 | vqav2 | dc_c | cls_attn | ridge λ0.6 (esc 54.13%, K̄=161, uniqK=219) | 20000 | 76.34 | 269.20 | 1.61 | 53.27 | 48.78 | -0.86 | -0.51 | loss | eval-80% |
| llava15 | vqav2 | dc_c | cls_attn | rule λ0.6 (esc 40.55%, K̄=148, uniqK=230) | 20000 | 76.29 | 239.40 | 1.44 | 58.45 | 54.29 | -0.91 | -0.25 | near-tie | eval-80% |
| llava15 | textvqa | dc_c | cls_attn | ridge λ0.8 (esc 62.3%, K̄=222, uniqK=349) | 4000 | 56.75 | 281.40 | 2.17 | 51.14 | 37.17 | -0.62 | +0.48 | near-tie | eval-80% |
| llava15 | textvqa | dc_c | cls_attn | rule λ1.2 (esc 100.0%, K̄=202, uniqK=349) | 4000 | 56.91 | 287.70 | 2.35 | 50.05 | 31.77 | -0.46 | +0.37 | near-tie | eval-80% |
| llava15 | docvqa | dc_c | cls_attn | ridge λ1.2 (esc 100.0%, K̄=339, uniqK=301) | 4279 | 21.23 | 483.50 | 2.84 | 16.06 | 10.64 | -0.68 | -0.54 | loss | eval-80% |
| llava15 | docvqa | dc_c | cls_attn | rule λ1.0 (esc 100.0%, K̄=305, uniqK=149) | 4279 | 20.86 | 448.70 | 2.65 | 22.10 | 16.40 | -1.05 | -0.84 | loss | eval-80% |
| qwen25vl7b | gqa | dc_c | norm | ridge λ1.0 (esc 43.09%, K̄=108, uniqK=166) | 10062 | 59.89 | 149.20 | 1.09 | 58.22 | 48.41 | -1.19 | -0.45 | near-tie | eval-80% |
| qwen25vl7b | gqa | dc_c | norm | rule λ1.0 (esc 56.63%, K̄=135, uniqK=234) | 10062 | 60.48 | 195.00 | 1.36 | 45.40 | 35.53 | -0.60 | -0.19 | near-tie | eval-80% |
| qwen25vl7b | vqav2 | dc_c | norm | ridge λ0.6 (esc 54.16%, K̄=100, uniqK=234) | 20000 | 81.41 | 167.30 | 1.20 | 53.35 | 43.46 | -2.95 | -0.84 | loss | eval-80% |
| qwen25vl7b | vqav2 | dc_c | norm | rule λ0.6 (esc 69.5%, K̄=106, uniqK=221) | 20000 | 81.21 | 180.10 | 1.29 | 49.78 | 38.81 | -3.15 | -1.36 | loss | eval-80% |
| qwen25vl7b | textvqa | dc_c | norm | ridge λ0.6 (esc 100.0%, K̄=488, uniqK=350) | 4000 | 71.45 | 728.80 | 4.82 | 24.35 | 15.87 | -9.57 | -7.89 | loss | eval-80% |
| qwen25vl7b | textvqa | dc_c_count_on_which | textsim | ridge λ0.6 (esc 94.72%, K̄=479, uniqK=493) | 4000 | 77.09 | 617.90 | 4.20 | 35.86 | 26.61 | -3.93 | -0.35 | near-tie | eval-80% |
| qwen25vl7b | textvqa | dc_c | norm | rule λ0.6 (esc 86.35%, K̄=472, uniqK=529) | 4000 | 71.59 | 691.90 | 4.56 | 28.18 | 20.40 | -9.43 | -7.27 | loss | eval-80% |
| qwen25vl7b | textvqa | dc_c_count_on_which | textsim | rule λ0.6 (esc 100.0%, K̄=520, uniqK=329) | 4000 | 77.92 | 664.70 | 4.48 | 31.00 | 21.80 | -3.10 | -0.79 | loss | eval-80% |
| qwen25vl7b | docvqa | dc_c | norm | ridge λ0.6 (esc 100.0%, K̄=551, uniqK=287) | 4279 | 81.55 | 858.10 | 5.04 | 30.06 | 27.80 | -13.11 | -11.07 | loss | eval-80% |
| qwen25vl7b | docvqa | dc_c | norm | rule λ0.6 (esc 100.0%, K̄=532, uniqK=216) | 4279 | 79.28 | 838.90 | 4.93 | 31.62 | 29.31 | -15.38 | -12.73 | loss | eval-80% |

## COUNT-on-WHICH attribution (qwen25vl7b × textvqa, eval-80%)

| Variant | Score | TFLOPs | Δ vs STATIC curve | Δ vs fixed TEXTSIM curve (COUNT's own increment) |
|---|--:|--:|--:|--:|
| DC-C ridge | 77.09 | 4.204 | -0.35 | -2.30 |
| DC-C rule | 77.92 | 4.480 | -0.79 | -2.03 |
| DC-D | 62.07 | 1.396 | +6.99 | -0.41 |
