# Dynamic-WHICH coverage matrix

Tiers: `final` > `n1000_confirm` > `n200_pilot` > `n20_smoke` > `missing`. **Meaningful coverage requires at least an n200 pilot** (an n20 smoke alone does not).

## Per-cell coverage summary (the decision view)

| Model | Dataset | Best tier | # outputs | Meaningful? | **Needs n=200 pilot?** |
|---|---|---|---:|:--:|:--:|
| llava15 | gqa | final | 12 | yes | no |
| llava15 | vqav2 | final | 9 | yes | no |
| llava15 | textvqa | final | 11 | yes | no |
| llava15 | docvqa | final | 11 | yes | no |
| qwen25vl7b | gqa | final | 10 | yes | no |
| qwen25vl7b | vqav2 | final | 9 | yes | no |
| qwen25vl7b | textvqa | final | 16 | yes | no |
| qwen25vl7b | docvqa | final | 11 | yes | no |

### MISSING / INCOMPLETE Dynamic-WHICH coverage — needs n=200 pilots


Meaningfully covered already: llava15×gqa (final), llava15×vqav2 (final), llava15×textvqa (final), llava15×docvqa (final), qwen25vl7b×gqa (final), qwen25vl7b×vqav2 (final), qwen25vl7b×textvqa (final), qwen25vl7b×docvqa (final).

**Only Qwen2.5-VL × TextVQA has a completed Dynamic-WHICH FINAL.** Every other cell has dense+static finals plus at most partial Dynamic-WHICH pilots.

## All found Dynamic-WHICH outputs

| Model | Dataset | Selector | p | n | Status | Dyn | Static | Dense | Dyn−Static | Dyn−Dense | FLOP red.% | Gate |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|:--:|
| llava15 | docvqa | saliency_mix_ref | 25 | 200 | unknown | 11.66 | 14.65 | 16.47 | -2.99 | -4.81 | 71.32 | OK |
| llava15 | docvqa | saliency_mix_ref | 35 | 200 | unknown | 12.37 | 16.20 | 16.47 | -3.83 | -4.10 | 61.92 | OK |
| llava15 | docvqa | saliency_mix_ref | 50 | 200 | unknown | 13.89 | 17.01 | 16.47 | -3.12 | -2.58 | 47.89 | OK |
| llava15 | docvqa | static_guarded_textsim_ref | 25 | 200 | unknown | 13.32 | 14.65 | 16.47 | -1.33 | -3.15 | 71.32 | OK |
| llava15 | docvqa | static_guarded_textsim_ref | 35 | 200 | unknown | 14.66 | 16.20 | 16.47 | -1.54 | -1.81 | 61.92 | OK |
| llava15 | docvqa | static_guarded_textsim_ref | 50 | 200 | unknown | 15.73 | 17.01 | 16.47 | -1.28 | -0.74 | 47.89 | OK |
| llava15 | docvqa | textsim | 15 | 5349 | final | 11.41 | 17.73 | 21.53 | -6.32 | -10.12 | 80.67 | OK |
| llava15 | docvqa | textsim | 25 | 5349 | final | 12.50 | 19.32 | 21.53 | -6.82 | -9.03 | 71.33 | OK |
| llava15 | docvqa | textsim | 25 | 200 | n200_pilot | 7.15 | 14.65 | 16.47 | -7.50 | -9.32 | 71.32 | OK |
| llava15 | docvqa | textsim | 35 | 5349 | final | 13.35 | 20.09 | 21.53 | -6.74 | -8.18 | 61.93 | OK |
| llava15 | docvqa | textsim | 35 | 200 | n200_pilot | 8.01 | 16.20 | 16.47 | -8.19 | -8.46 | 61.92 | OK |
| llava15 | docvqa | textsim | 50 | 5349 | final | 14.47 | 20.75 | 21.53 | -6.28 | -7.06 | 47.90 | OK |
| llava15 | docvqa | textsim | 50 | 200 | n200_pilot | 9.55 | 17.01 | 16.47 | -7.46 | -6.92 | 47.89 | OK |
| llava15 | docvqa | textsim | 75 | 5349 | final | 17.39 | 21.36 | 21.53 | -3.97 | -4.14 | 24.12 | OK |
| llava15 | docvqa | textsim_cls_mix | 25 | 200 | n200_pilot | 11.66 | 14.65 | 16.47 | -2.99 | -4.81 | 71.32 | OK |
| llava15 | docvqa | textsim_cls_mix | 35 | 200 | n200_pilot | 12.37 | 16.20 | 16.47 | -3.83 | -4.10 | 61.92 | OK |
| llava15 | docvqa | textsim_cls_mix | 50 | 200 | n200_pilot | 13.89 | 17.01 | 16.47 | -3.12 | -2.58 | 47.89 | OK |
| llava15 | docvqa | textsim_ref | 25 | 200 | unknown | 7.15 | 14.65 | 16.47 | -7.50 | -9.32 | 71.32 | OK |
| llava15 | docvqa | textsim_ref | 35 | 200 | unknown | 8.01 | 16.20 | 16.47 | -8.19 | -8.46 | 61.92 | OK |
| llava15 | docvqa | textsim_ref | 50 | 200 | unknown | 9.55 | 17.01 | 16.47 | -7.46 | -6.92 | 47.89 | OK |
| llava15 | gqa | textsim | 15 | 12578 | final | 55.41 | 56.14 | 61.42 | -0.73 | -6.01 | 80.85 | OK |
| llava15 | gqa | textsim | 25 | 12578 | final | 56.89 | 58.15 | 61.42 | -1.26 | -4.53 | 71.48 | OK |
| llava15 | gqa | textsim | 25 | 1000 | n1000_confirm | 62.70 | 64.20 | 67.30 | -1.50 | -4.60 | 71.50 | OK |
| llava15 | gqa | textsim | 25 | 200 | n200_pilot | 67.00 | 66.00 | 63.50 | 1.00 | 3.50 | 71.49 | OK |
| llava15 | gqa | textsim | 35 | 12578 | final | 57.54 | 59.41 | 61.42 | -1.87 | -3.88 | 62.07 | OK |
| llava15 | gqa | textsim | 35 | 1000 | n1000_confirm | 63.70 | 65.70 | 67.30 | -2.00 | -3.60 | 62.08 | OK |
| llava15 | gqa | textsim | 35 | 200 | n200_pilot | 66.50 | 66.50 | 63.50 | 0.00 | 3.00 | 62.07 | OK |
| llava15 | gqa | textsim | 50 | 12578 | final | 58.96 | 60.53 | 61.42 | -1.57 | -2.46 | 48.00 | OK |
| llava15 | gqa | textsim | 75 | 12578 | final | 60.80 | 61.53 | 61.42 | -0.73 | -0.62 | 24.17 | OK |
| llava15 | gqa | textsim | 75 | 20 | n20_smoke | 65.00 | 55.00 | 60.00 | 10.00 | 5.00 | 24.14 | OK |
| llava15 | gqa | textsim_cls_mix | 25 | 200 | n200_pilot | 66.50 | 66.00 | 63.50 | 0.50 | 3.00 | 71.49 | OK |
| llava15 | gqa | textsim_cls_mix | 35 | 200 | n200_pilot | 64.50 | 66.50 | 63.50 | -2.00 | 1.00 | 62.07 | OK |
| llava15 | textvqa | saliency_mix_ref | 25 | 200 | unknown | 50.20 | 55.70 | 56.75 | -5.50 | -6.55 | 64.82 | OK |
| llava15 | textvqa | saliency_mix_ref | 35 | 200 | unknown | 51.00 | 59.05 | 56.75 | -8.05 | -5.75 | 56.28 | OK |
| llava15 | textvqa | saliency_mix_ref | 50 | 200 | unknown | 53.85 | 57.90 | 56.75 | -4.05 | -2.90 | 43.52 | OK |
| llava15 | textvqa | static_guarded_textsim_ref | 25 | 200 | unknown | 53.25 | 55.70 | 56.75 | -2.45 | -3.50 | 64.82 | OK |
| llava15 | textvqa | static_guarded_textsim_ref | 35 | 200 | unknown | 56.10 | 59.05 | 56.75 | -2.95 | -0.65 | 56.28 | OK |
| llava15 | textvqa | static_guarded_textsim_ref | 50 | 200 | unknown | 56.80 | 57.90 | 56.75 | -1.10 | 0.05 | 43.52 | OK |
| llava15 | textvqa | textsim | 15 | 5000 | final | 47.48 | 56.05 | 57.65 | -8.57 | -10.17 | 74.60 | OK |
| llava15 | textvqa | textsim | 25 | 5000 | final | 49.10 | 55.97 | 57.65 | -6.87 | -8.55 | 65.96 | OK |
| llava15 | textvqa | textsim | 25 | 200 | n200_pilot | 47.65 | 55.70 | 56.75 | -8.05 | -9.10 | 64.82 | OK |
| llava15 | textvqa | textsim | 35 | 5000 | final | 49.76 | 56.71 | 57.65 | -6.95 | -7.89 | 57.27 | OK |
| llava15 | textvqa | textsim | 35 | 200 | n200_pilot | 50.80 | 59.05 | 56.75 | -8.25 | -5.95 | 56.28 | OK |
| llava15 | textvqa | textsim | 50 | 5000 | final | 50.81 | 56.40 | 57.65 | -5.59 | -6.84 | 44.29 | OK |
| llava15 | textvqa | textsim | 50 | 200 | n200_pilot | 50.35 | 57.90 | 56.75 | -7.55 | -6.40 | 43.52 | OK |
| llava15 | textvqa | textsim | 75 | 5000 | final | 53.48 | 57.39 | 57.65 | -3.91 | -4.17 | 22.30 | OK |
| llava15 | textvqa | textsim_cls_mix | 25 | 200 | n200_pilot | 50.20 | 55.70 | 56.75 | -5.50 | -6.55 | 64.82 | OK |
| llava15 | textvqa | textsim_cls_mix | 35 | 200 | n200_pilot | 51.00 | 59.05 | 56.75 | -8.05 | -5.75 | 56.28 | OK |
| llava15 | textvqa | textsim_cls_mix | 50 | 200 | n200_pilot | 53.85 | 57.90 | 56.75 | -4.05 | -2.90 | 43.52 | OK |
| llava15 | textvqa | textsim_ref | 25 | 200 | unknown | 47.65 | 55.70 | 56.75 | -8.05 | -9.10 | 64.82 | OK |
| llava15 | textvqa | textsim_ref | 25 | 20 | unknown | 31.50 | 36.00 | 41.50 | -4.50 | -10.00 | 64.19 | OK |
| llava15 | textvqa | textsim_ref | 35 | 200 | unknown | 50.80 | 59.05 | 56.75 | -8.25 | -5.95 | 56.28 | OK |
| llava15 | textvqa | textsim_ref | 50 | 200 | unknown | 50.35 | 57.90 | 56.75 | -7.55 | -6.40 | 43.52 | OK |
| llava15 | vqav2 | saliency_mix_ref | 25 | 200 | unknown | 73.33 | 74.67 | 77.33 | -1.34 | -4.00 | 71.79 | OK |
| llava15 | vqav2 | saliency_mix_ref | 35 | 200 | unknown | 72.83 | 75.50 | 77.33 | -2.67 | -4.50 | 62.33 | OK |
| llava15 | vqav2 | static_guarded_textsim_ref | 25 | 200 | unknown | 73.50 | 74.67 | 77.33 | -1.17 | -3.83 | 71.79 | OK |
| llava15 | vqav2 | static_guarded_textsim_ref | 35 | 200 | unknown | 76.50 | 75.50 | 77.33 | 1.00 | -0.83 | 62.33 | OK |
| llava15 | vqav2 | textsim | 15 | 25000 | final | 68.33 | 73.36 | 77.33 | -5.03 | -9.00 | 81.19 | OK |
| llava15 | vqav2 | textsim | 25 | 25000 | final | 70.03 | 75.55 | 77.33 | -5.52 | -7.30 | 71.79 | OK |
| llava15 | vqav2 | textsim | 25 | 200 | n200_pilot | 71.50 | 74.67 | 77.33 | -3.17 | -5.83 | 71.79 | OK |
| llava15 | vqav2 | textsim | 35 | 25000 | final | 71.32 | 76.13 | 77.33 | -4.81 | -6.01 | 62.33 | OK |
| llava15 | vqav2 | textsim | 35 | 200 | n200_pilot | 71.67 | 75.50 | 77.33 | -3.83 | -5.66 | 62.33 | OK |
| llava15 | vqav2 | textsim | 50 | 25000 | final | 73.14 | 76.94 | 77.33 | -3.80 | -4.19 | 48.21 | OK |
| llava15 | vqav2 | textsim | 75 | 25000 | final | 75.72 | 77.31 | 77.33 | -1.59 | -1.61 | 24.28 | OK |
| llava15 | vqav2 | textsim_cls_mix | 25 | 200 | n200_pilot | 73.33 | 74.67 | 77.33 | -1.34 | -4.00 | 71.79 | OK |
| llava15 | vqav2 | textsim_cls_mix | 35 | 200 | n200_pilot | 72.83 | 75.50 | 77.33 | -2.67 | -4.50 | 62.33 | OK |
| llava15 | vqav2 | textsim_ref | 25 | 200 | unknown | 71.50 | 74.67 | 77.33 | -3.17 | -5.83 | 71.79 | OK |
| llava15 | vqav2 | textsim_ref | 35 | 200 | unknown | 71.67 | 75.50 | 77.33 | -3.83 | -5.66 | 62.33 | OK |
| qwen25vl7b | docvqa | saliency_mix_ref | 25 | 200 | unknown | 39.60 | 49.10 | 88.20 | -9.50 | -48.60 | 73.47 | OK |
| qwen25vl7b | docvqa | saliency_mix_ref | 35 | 200 | unknown | 53.59 | 67.84 | 88.20 | -14.25 | -34.61 | 63.97 | OK |
| qwen25vl7b | docvqa | saliency_mix_ref | 50 | 200 | unknown | 70.17 | 80.15 | 88.20 | -9.98 | -18.03 | 49.53 | OK |
| qwen25vl7b | docvqa | static_guarded_textsim_ref | 25 | 200 | unknown | 41.33 | 49.10 | 88.20 | -7.77 | -46.87 | 73.47 | OK |
| qwen25vl7b | docvqa | static_guarded_textsim_ref | 35 | 200 | unknown | 56.46 | 67.84 | 88.20 | -11.38 | -31.74 | 63.97 | OK |
| qwen25vl7b | docvqa | static_guarded_textsim_ref | 50 | 200 | unknown | 71.60 | 80.15 | 88.20 | -8.55 | -16.60 | 49.53 | OK |
| qwen25vl7b | docvqa | textsim | 15 | 5349 | final | 33.09 | 29.90 | 94.76 | 3.19 | -61.67 | 82.88 | OK |
| qwen25vl7b | docvqa | textsim | 25 | 5349 | final | 46.57 | 49.55 | 94.76 | -2.98 | -48.19 | 73.44 | OK |
| qwen25vl7b | docvqa | textsim | 25 | 200 | n200_pilot | 40.03 | 49.10 | 88.20 | -9.07 | -48.17 | 73.47 | OK |
| qwen25vl7b | docvqa | textsim | 35 | 5349 | final | 57.53 | 68.86 | 94.76 | -11.33 | -37.23 | 63.95 | OK |
| qwen25vl7b | docvqa | textsim | 35 | 200 | n200_pilot | 53.36 | 67.84 | 88.20 | -14.48 | -34.84 | 63.97 | OK |
| qwen25vl7b | docvqa | textsim | 50 | 5349 | final | 71.36 | 84.79 | 94.76 | -13.43 | -23.40 | 49.51 | OK |
| qwen25vl7b | docvqa | textsim | 50 | 200 | n200_pilot | 64.69 | 80.15 | 88.20 | -15.46 | -23.51 | 49.53 | OK |
| qwen25vl7b | docvqa | textsim | 75 | 5349 | final | 88.04 | 93.98 | 94.76 | -5.94 | -6.72 | 25.03 | OK |
| qwen25vl7b | docvqa | textsim_norm_mix | 25 | 200 | n200_pilot | 39.60 | 49.10 | 88.20 | -9.50 | -48.60 | 73.47 | OK |
| qwen25vl7b | docvqa | textsim_norm_mix | 35 | 200 | n200_pilot | 53.59 | 67.84 | 88.20 | -14.25 | -34.61 | 63.97 | OK |
| qwen25vl7b | docvqa | textsim_norm_mix | 50 | 200 | n200_pilot | 70.17 | 80.15 | 88.20 | -9.98 | -18.03 | 49.53 | OK |
| qwen25vl7b | docvqa | textsim_ref | 25 | 200 | unknown | 40.03 | 49.10 | 88.20 | -9.07 | -48.17 | 73.47 | OK |
| qwen25vl7b | docvqa | textsim_ref | 35 | 200 | unknown | 53.36 | 67.84 | 88.20 | -14.48 | -34.84 | 63.97 | OK |
| qwen25vl7b | docvqa | textsim_ref | 50 | 200 | unknown | 64.69 | 80.15 | 88.20 | -15.46 | -23.51 | 49.53 | OK |
| qwen25vl7b | gqa | saliency_mix_ref | 25 | 200 | unknown | 67.00 | 67.00 | 68.00 | 0.00 | -1.00 | 67.55 | OK |
| qwen25vl7b | gqa | saliency_mix_ref | 35 | 200 | unknown | 67.50 | 68.00 | 68.00 | -0.50 | -0.50 | 58.59 | OK |
| qwen25vl7b | gqa | static_guarded_textsim_ref | 25 | 200 | unknown | 65.50 | 67.00 | 68.00 | -1.50 | -2.50 | 67.55 | OK |
| qwen25vl7b | gqa | static_guarded_textsim_ref | 35 | 200 | unknown | 67.50 | 68.00 | 68.00 | -0.50 | -0.50 | 58.59 | OK |
| qwen25vl7b | gqa | textsim | 15 | 12578 | final | 53.45 | 56.04 | 60.96 | -2.59 | -7.51 | 76.56 | OK |
| qwen25vl7b | gqa | textsim | 25 | 12578 | final | 55.97 | 58.77 | 60.96 | -2.80 | -4.99 | 67.71 | OK |
| qwen25vl7b | gqa | textsim | 25 | 200 | n200_pilot | 60.00 | 67.00 | 68.00 | -7.00 | -8.00 | 67.55 | OK |
| qwen25vl7b | gqa | textsim | 35 | 12578 | final | 57.47 | 59.99 | 60.96 | -2.52 | -3.49 | 58.71 | OK |
| qwen25vl7b | gqa | textsim | 35 | 200 | n200_pilot | 61.00 | 68.00 | 68.00 | -7.00 | -7.00 | 58.59 | OK |
| qwen25vl7b | gqa | textsim | 50 | 12578 | final | 58.87 | 60.53 | 60.96 | -1.66 | -2.09 | 45.29 | OK |
| qwen25vl7b | gqa | textsim | 75 | 12578 | final | 60.17 | 60.85 | 60.96 | -0.68 | -0.79 | 22.71 | OK |
| qwen25vl7b | gqa | textsim | 75 | 20 | n20_smoke | 60.00 | 55.00 | 60.00 | 5.00 | 0.00 | 22.95 | OK |
| qwen25vl7b | gqa | textsim_norm_mix | 25 | 200 | n200_pilot | 67.00 | 67.00 | 68.00 | 0.00 | -1.00 | 67.55 | OK |
| qwen25vl7b | gqa | textsim_norm_mix | 35 | 200 | n200_pilot | 67.50 | 68.00 | 68.00 | -0.50 | -0.50 | 58.59 | OK |
| qwen25vl7b | gqa | textsim_ref | 25 | 200 | unknown | 60.00 | 67.00 | 68.00 | -7.00 | -8.00 | 67.55 | OK |
| qwen25vl7b | gqa | textsim_ref | 25 | 20 | unknown | 50.00 | 50.00 | 60.00 | 0.00 | -10.00 | 68.11 | OK |
| qwen25vl7b | gqa | textsim_ref | 35 | 200 | unknown | 61.00 | 68.00 | 68.00 | -7.00 | -7.00 | 58.59 | OK |
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
| qwen25vl7b | textvqa | textsim_ref | 15 | 200 | unknown | 58.95 | 47.10 | 80.60 | 11.85 | -21.65 | 77.95 | OK |
| qwen25vl7b | textvqa | textsim_ref | 25 | 200 | unknown | 66.25 | 59.35 | 80.60 | 6.90 | -14.35 | 69.03 | OK |
| qwen25vl7b | textvqa | textsim_ref | 35 | 200 | unknown | 70.45 | 61.65 | 80.60 | 8.80 | -10.15 | 60.02 | OK |
| qwen25vl7b | textvqa | textsim_ref | 50 | 200 | unknown | 73.25 | 72.75 | 80.60 | 0.50 | -7.35 | 46.42 | OK |
| qwen25vl7b | textvqa | textsim_ref | 75 | 200 | unknown | 77.75 | 76.10 | 80.60 | 1.65 | -2.85 | 23.43 | OK |
| qwen25vl7b | vqav2 | saliency_mix_ref | 25 | 200 | unknown | 81.33 | 79.83 | 83.00 | 1.50 | -1.67 | 68.01 | OK |
| qwen25vl7b | vqav2 | saliency_mix_ref | 35 | 200 | unknown | 80.50 | 80.50 | 83.00 | 0.00 | -2.50 | 58.97 | OK |
| qwen25vl7b | vqav2 | static_guarded_textsim_ref | 25 | 200 | unknown | 81.33 | 79.83 | 83.00 | 1.50 | -1.67 | 68.01 | OK |
| qwen25vl7b | vqav2 | static_guarded_textsim_ref | 35 | 200 | unknown | 80.33 | 80.50 | 83.00 | -0.17 | -2.67 | 58.97 | OK |
| qwen25vl7b | vqav2 | textsim | 15 | 25000 | final | 73.55 | 76.16 | 84.27 | -2.61 | -10.72 | 77.01 | OK |
| qwen25vl7b | vqav2 | textsim | 25 | 25000 | final | 77.50 | 78.89 | 84.27 | -1.39 | -6.77 | 68.10 | OK |
| qwen25vl7b | vqav2 | textsim | 25 | 200 | n200_pilot | 76.50 | 79.83 | 83.00 | -3.33 | -6.50 | 68.01 | OK |
| qwen25vl7b | vqav2 | textsim | 35 | 25000 | final | 79.54 | 80.49 | 84.27 | -0.95 | -4.73 | 59.06 | OK |
| qwen25vl7b | vqav2 | textsim | 35 | 200 | n200_pilot | 78.50 | 80.50 | 83.00 | -2.00 | -4.50 | 58.97 | OK |
| qwen25vl7b | vqav2 | textsim | 50 | 25000 | final | 81.70 | 82.09 | 84.27 | -0.39 | -2.57 | 45.55 | OK |
| qwen25vl7b | vqav2 | textsim | 75 | 25000 | final | 83.60 | 83.66 | 84.27 | -0.06 | -0.67 | 22.84 | OK |
| qwen25vl7b | vqav2 | textsim_norm_mix | 25 | 200 | n200_pilot | 81.33 | 79.83 | 83.00 | 1.50 | -1.67 | 68.01 | OK |
| qwen25vl7b | vqav2 | textsim_norm_mix | 35 | 200 | n200_pilot | 80.50 | 80.50 | 83.00 | 0.00 | -2.50 | 58.97 | OK |
| qwen25vl7b | vqav2 | textsim_ref | 25 | 200 | unknown | 76.50 | 79.83 | 83.00 | -3.33 | -6.50 | 68.01 | OK |
| qwen25vl7b | vqav2 | textsim_ref | 35 | 200 | unknown | 78.50 | 80.50 | 83.00 | -2.00 | -4.50 | 58.97 | OK |

*Read-only audit of `results/runs/*/*/dynamic_which_*.json`. No result files were modified. Dyn−Static is the headline (question-conditioned vs static floor at same budget); Dyn−Dense (=accuracy_delta_pp) is signed vs dense.*
