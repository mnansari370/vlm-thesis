# Dynamic-WHICH coverage matrix

Tiers: `final` > `n1000_confirm` > `n200_pilot` > `n20_smoke` > `missing`. **Meaningful coverage requires at least an n200 pilot** (an n20 smoke alone does not).

## Per-cell coverage summary (the decision view)

| Model | Dataset | Best tier | # outputs | Meaningful? | **Needs n=200 pilot?** |
|---|---|---|---:|:--:|:--:|
| llava15 | gqa | n1000_confirm | 7 | yes | no |
| llava15 | vqav2 | n200_pilot | 4 | yes | no |
| llava15 | textvqa | n200_pilot | 6 | yes | no |
| llava15 | docvqa | n200_pilot | 6 | yes | no |
| qwen25vl7b | gqa | n200_pilot | 5 | yes | no |
| qwen25vl7b | vqav2 | n200_pilot | 4 | yes | no |
| qwen25vl7b | textvqa | final | 16 | yes | no |
| qwen25vl7b | docvqa | n200_pilot | 6 | yes | no |

### MISSING / INCOMPLETE Dynamic-WHICH coverage — needs n=200 pilots


Meaningfully covered already: llava15×gqa (n1000_confirm), llava15×vqav2 (n200_pilot), llava15×textvqa (n200_pilot), llava15×docvqa (n200_pilot), qwen25vl7b×gqa (n200_pilot), qwen25vl7b×vqav2 (n200_pilot), qwen25vl7b×textvqa (final), qwen25vl7b×docvqa (n200_pilot).

**Only Qwen2.5-VL × TextVQA has a completed Dynamic-WHICH FINAL.** Every other cell has dense+static finals plus at most partial Dynamic-WHICH pilots.

## All found Dynamic-WHICH outputs

| Model | Dataset | Selector | p | n | Status | Dyn | Static | Dense | Dyn−Static | Dyn−Dense | FLOP red.% | Gate |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|:--:|
| llava15 | docvqa | textsim | 25 | 200 | n200_pilot | 7.15 | 14.65 | 16.47 | -7.50 | -9.32 | 71.32 | OK |
| llava15 | docvqa | textsim | 35 | 200 | n200_pilot | 8.01 | 16.20 | 16.47 | -8.19 | -8.46 | 61.92 | OK |
| llava15 | docvqa | textsim | 50 | 200 | n200_pilot | 9.55 | 17.01 | 16.47 | -7.46 | -6.92 | 47.89 | OK |
| llava15 | docvqa | textsim_cls_mix | 25 | 200 | n200_pilot | 11.66 | 14.65 | 16.47 | -2.99 | -4.81 | 71.32 | OK |
| llava15 | docvqa | textsim_cls_mix | 35 | 200 | n200_pilot | 12.37 | 16.20 | 16.47 | -3.83 | -4.10 | 61.92 | OK |
| llava15 | docvqa | textsim_cls_mix | 50 | 200 | n200_pilot | 13.89 | 17.01 | 16.47 | -3.12 | -2.58 | 47.89 | OK |
| llava15 | gqa | textsim | 25 | 1000 | n1000_confirm | 62.70 | 64.20 | 67.30 | -1.50 | -4.60 | 71.50 | OK |
| llava15 | gqa | textsim | 25 | 200 | n200_pilot | 67.00 | 66.00 | 63.50 | 1.00 | 3.50 | 71.49 | OK |
| llava15 | gqa | textsim | 35 | 1000 | n1000_confirm | 63.70 | 65.70 | 67.30 | -2.00 | -3.60 | 62.08 | OK |
| llava15 | gqa | textsim | 35 | 200 | n200_pilot | 66.50 | 66.50 | 63.50 | 0.00 | 3.00 | 62.07 | OK |
| llava15 | gqa | textsim | 75 | 20 | n20_smoke | 65.00 | 55.00 | 60.00 | 10.00 | 5.00 | 24.14 | OK |
| llava15 | gqa | textsim_cls_mix | 25 | 200 | n200_pilot | 66.50 | 66.00 | 63.50 | 0.50 | 3.00 | 71.49 | OK |
| llava15 | gqa | textsim_cls_mix | 35 | 200 | n200_pilot | 64.50 | 66.50 | 63.50 | -2.00 | 1.00 | 62.07 | OK |
| llava15 | textvqa | textsim | 25 | 200 | n200_pilot | 47.65 | 55.70 | 56.75 | -8.05 | -9.10 | 64.82 | OK |
| llava15 | textvqa | textsim | 35 | 200 | n200_pilot | 50.80 | 59.05 | 56.75 | -8.25 | -5.95 | 56.28 | OK |
| llava15 | textvqa | textsim | 50 | 200 | n200_pilot | 50.35 | 57.90 | 56.75 | -7.55 | -6.40 | 43.52 | OK |
| llava15 | textvqa | textsim_cls_mix | 25 | 200 | n200_pilot | 50.20 | 55.70 | 56.75 | -5.50 | -6.55 | 64.82 | OK |
| llava15 | textvqa | textsim_cls_mix | 35 | 200 | n200_pilot | 51.00 | 59.05 | 56.75 | -8.05 | -5.75 | 56.28 | OK |
| llava15 | textvqa | textsim_cls_mix | 50 | 200 | n200_pilot | 53.85 | 57.90 | 56.75 | -4.05 | -2.90 | 43.52 | OK |
| llava15 | vqav2 | textsim | 25 | 200 | n200_pilot | 71.50 | 74.67 | 77.33 | -3.17 | -5.83 | 71.79 | OK |
| llava15 | vqav2 | textsim | 35 | 200 | n200_pilot | 71.67 | 75.50 | 77.33 | -3.83 | -5.66 | 62.33 | OK |
| llava15 | vqav2 | textsim_cls_mix | 25 | 200 | n200_pilot | 73.33 | 74.67 | 77.33 | -1.34 | -4.00 | 71.79 | OK |
| llava15 | vqav2 | textsim_cls_mix | 35 | 200 | n200_pilot | 72.83 | 75.50 | 77.33 | -2.67 | -4.50 | 62.33 | OK |
| qwen25vl7b | docvqa | textsim | 25 | 200 | n200_pilot | 40.03 | 49.10 | 88.20 | -9.07 | -48.17 | 73.47 | OK |
| qwen25vl7b | docvqa | textsim | 35 | 200 | n200_pilot | 53.36 | 67.84 | 88.20 | -14.48 | -34.84 | 63.97 | OK |
| qwen25vl7b | docvqa | textsim | 50 | 200 | n200_pilot | 64.69 | 80.15 | 88.20 | -15.46 | -23.51 | 49.53 | OK |
| qwen25vl7b | docvqa | textsim_norm_mix | 25 | 200 | n200_pilot | 39.60 | 49.10 | 88.20 | -9.50 | -48.60 | 73.47 | OK |
| qwen25vl7b | docvqa | textsim_norm_mix | 35 | 200 | n200_pilot | 53.59 | 67.84 | 88.20 | -14.25 | -34.61 | 63.97 | OK |
| qwen25vl7b | docvqa | textsim_norm_mix | 50 | 200 | n200_pilot | 70.17 | 80.15 | 88.20 | -9.98 | -18.03 | 49.53 | OK |
| qwen25vl7b | gqa | textsim | 25 | 200 | n200_pilot | 60.00 | 67.00 | 68.00 | -7.00 | -8.00 | 67.55 | OK |
| qwen25vl7b | gqa | textsim | 35 | 200 | n200_pilot | 61.00 | 68.00 | 68.00 | -7.00 | -7.00 | 58.59 | OK |
| qwen25vl7b | gqa | textsim | 75 | 20 | n20_smoke | 60.00 | 55.00 | 60.00 | 5.00 | 0.00 | 22.95 | OK |
| qwen25vl7b | gqa | textsim_norm_mix | 25 | 200 | n200_pilot | 67.00 | 67.00 | 68.00 | 0.00 | -1.00 | 67.55 | OK |
| qwen25vl7b | gqa | textsim_norm_mix | 35 | 200 | n200_pilot | 67.50 | 68.00 | 68.00 | -0.50 | -0.50 | 58.59 | OK |
| qwen25vl7b | textvqa | textsim | 15 | 5000 | final | 60.56 | 53.06 | 81.06 | 7.50 | -20.50 | 78.65 | OK |
| qwen25vl7b | textvqa | textsim | 15 | 1000 | n1000_confirm | 62.46 | 52.76 | 81.25 | 9.70 | -18.79 | 78.69 | OK |
| qwen25vl7b | textvqa | textsim | 25 | 5000 | final | 67.53 | 59.17 | 81.06 | 8.36 | -13.53 | 69.64 | OK |
| qwen25vl7b | textvqa | textsim | 25 | 1000 | n1000_confirm | 68.61 | 60.02 | 81.25 | 8.59 | -12.64 | 69.68 | OK |
| qwen25vl7b | textvqa | textsim | 25 | 200 | n200_pilot | 66.25 | 59.35 | 80.60 | 6.90 | -14.35 | 69.03 | OK |
| qwen25vl7b | textvqa | textsim | 35 | 5000 | final | 71.36 | 63.57 | 81.06 | 7.79 | -9.70 | 60.56 | OK |
| qwen25vl7b | textvqa | textsim | 35 | 1000 | n1000_confirm | 72.26 | 63.78 | 81.25 | 8.48 | -8.99 | 60.59 | OK |
| qwen25vl7b | textvqa | textsim | 35 | 200 | n200_pilot | 70.45 | 61.65 | 80.60 | 8.80 | -10.15 | 60.02 | OK |
| qwen25vl7b | textvqa | textsim | 50 | 5000 | final | 76.08 | 70.14 | 81.06 | 5.94 | -4.98 | 46.84 | OK |
| qwen25vl7b | textvqa | textsim | 50 | 1000 | n1000_confirm | 75.68 | 70.19 | 81.25 | 5.49 | -5.57 | 46.86 | OK |
| qwen25vl7b | textvqa | textsim | 50 | 200 | n200_pilot | 73.25 | 72.75 | 80.60 | 0.50 | -7.35 | 46.42 | OK |
| qwen25vl7b | textvqa | textsim | 75 | 5000 | final | 79.80 | 78.43 | 81.06 | 1.37 | -1.26 | 23.63 | OK |
| qwen25vl7b | textvqa | textsim | 75 | 1000 | n1000_confirm | 79.57 | 78.06 | 81.25 | 1.51 | -1.68 | 23.65 | OK |
| qwen25vl7b | textvqa | textsim_norm_mix | 25 | 200 | n200_pilot | 60.60 | 59.35 | 80.60 | 1.25 | -20.00 | 69.03 | OK |
| qwen25vl7b | textvqa | textsim_norm_mix | 35 | 200 | n200_pilot | 66.50 | 61.65 | 80.60 | 4.85 | -14.10 | 60.02 | OK |
| qwen25vl7b | textvqa | textsim_norm_mix | 50 | 200 | n200_pilot | 66.60 | 72.75 | 80.60 | -6.15 | -14.00 | 46.42 | OK |
| qwen25vl7b | vqav2 | textsim | 25 | 200 | n200_pilot | 76.50 | 79.83 | 83.00 | -3.33 | -6.50 | 68.01 | OK |
| qwen25vl7b | vqav2 | textsim | 35 | 200 | n200_pilot | 78.50 | 80.50 | 83.00 | -2.00 | -4.50 | 58.97 | OK |
| qwen25vl7b | vqav2 | textsim_norm_mix | 25 | 200 | n200_pilot | 81.33 | 79.83 | 83.00 | 1.50 | -1.67 | 68.01 | OK |
| qwen25vl7b | vqav2 | textsim_norm_mix | 35 | 200 | n200_pilot | 80.50 | 80.50 | 83.00 | 0.00 | -2.50 | 58.97 | OK |

*Read-only audit of `results/final_scope/*/*/dynamic_which_*.json`. No result files were modified. Dyn−Static is the headline (question-conditioned vs static floor at same budget); Dyn−Dense (=accuracy_delta_pp) is signed vs dense.*
