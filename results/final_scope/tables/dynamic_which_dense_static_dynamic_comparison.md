# Dense / Static / Dynamic-WHICH (textsim) — full-final comparison (40 cells)

One row per model × dataset × budget; all runs on the FULL locked manifest with gates OK. Dynamic = current Dynamic-WHICH textsim; Static = the primary static floor at the same budget (LLaVA cls_attn, Qwen norm); Dense = the keep-all reference.

Labels: dynamic_win=6, near_tie=2, dynamic_loss=32  (win: Dyn−Static>0; near_tie: −0.50≤Dyn−Static≤0; loss: Dyn−Static<−0.50).

| Model | Dataset | p | Dense | Static | Dynamic | Dyn−Static | Dyn−Dense | Tok red.% | FLOP red.% | Gate | n | Label |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|:--:|--:|---|
| llava15 | gqa | 15 | 61.42 | 56.14 | 55.41 | -0.73 | -6.01 | 80.36 | 80.85 | OK | 12578 | dynamic_loss |
| llava15 | gqa | 25 | 61.42 | 58.15 | 56.89 | -1.26 | -4.53 | 70.85 | 71.48 | OK | 12578 | dynamic_loss |
| llava15 | gqa | 35 | 61.42 | 59.41 | 57.54 | -1.87 | -3.88 | 61.34 | 62.07 | OK | 12578 | dynamic_loss |
| llava15 | gqa | 50 | 61.42 | 60.53 | 58.96 | -1.57 | -2.46 | 47.23 | 48.00 | OK | 12578 | dynamic_loss |
| llava15 | gqa | 75 | 61.42 | 61.53 | 60.80 | -0.73 | -0.62 | 23.62 | 24.17 | OK | 12578 | dynamic_loss |
| llava15 | vqav2 | 15 | 77.33 | 73.36 | 68.33 | -5.03 | -9.00 | 80.72 | 81.19 | OK | 25000 | dynamic_loss |
| llava15 | vqav2 | 25 | 77.33 | 75.55 | 70.03 | -5.52 | -7.30 | 71.16 | 71.79 | OK | 25000 | dynamic_loss |
| llava15 | vqav2 | 35 | 77.33 | 76.13 | 71.32 | -4.81 | -6.01 | 61.61 | 62.33 | OK | 25000 | dynamic_loss |
| llava15 | vqav2 | 50 | 77.33 | 76.94 | 73.14 | -3.80 | -4.19 | 47.44 | 48.21 | OK | 25000 | dynamic_loss |
| llava15 | vqav2 | 75 | 77.33 | 77.31 | 75.72 | -1.59 | -1.61 | 23.72 | 24.28 | OK | 25000 | dynamic_loss |
| llava15 | textvqa | 15 | 57.65 | 56.05 | 47.48 | -8.57 | -10.17 | 73.97 | 74.60 | OK | 5000 | dynamic_loss |
| llava15 | textvqa | 25 | 57.65 | 55.97 | 49.10 | -6.87 | -8.55 | 65.22 | 65.96 | OK | 5000 | dynamic_loss |
| llava15 | textvqa | 35 | 57.65 | 56.71 | 49.76 | -6.95 | -7.89 | 56.46 | 57.27 | OK | 5000 | dynamic_loss |
| llava15 | textvqa | 50 | 57.65 | 56.40 | 50.81 | -5.59 | -6.84 | 43.48 | 44.29 | OK | 5000 | dynamic_loss |
| llava15 | textvqa | 75 | 57.65 | 57.39 | 53.48 | -3.91 | -4.17 | 21.74 | 22.30 | OK | 5000 | dynamic_loss |
| llava15 | docvqa | 15 | 21.53 | 17.73 | 11.41 | -6.32 | -10.12 | 80.18 | 80.67 | OK | 5349 | dynamic_loss |
| llava15 | docvqa | 25 | 21.53 | 19.32 | 12.50 | -6.82 | -9.03 | 70.69 | 71.33 | OK | 5349 | dynamic_loss |
| llava15 | docvqa | 35 | 21.53 | 20.09 | 13.35 | -6.74 | -8.18 | 61.20 | 61.93 | OK | 5349 | dynamic_loss |
| llava15 | docvqa | 50 | 21.53 | 20.75 | 14.47 | -6.28 | -7.06 | 47.13 | 47.90 | OK | 5349 | dynamic_loss |
| llava15 | docvqa | 75 | 21.53 | 21.36 | 17.39 | -3.97 | -4.14 | 23.56 | 24.12 | OK | 5349 | dynamic_loss |
| qwen25vl7b | gqa | 15 | 60.96 | 56.04 | 53.45 | -2.59 | -7.51 | 76.28 | 76.56 | OK | 12578 | dynamic_loss |
| qwen25vl7b | gqa | 25 | 60.96 | 58.77 | 55.97 | -2.80 | -4.99 | 67.37 | 67.71 | OK | 12578 | dynamic_loss |
| qwen25vl7b | gqa | 35 | 60.96 | 59.99 | 57.47 | -2.52 | -3.49 | 58.34 | 58.71 | OK | 12578 | dynamic_loss |
| qwen25vl7b | gqa | 50 | 60.96 | 60.53 | 58.87 | -1.66 | -2.09 | 44.91 | 45.29 | OK | 12578 | dynamic_loss |
| qwen25vl7b | gqa | 75 | 60.96 | 60.85 | 60.17 | -0.68 | -0.79 | 22.44 | 22.71 | OK | 12578 | dynamic_loss |
| qwen25vl7b | vqav2 | 15 | 84.27 | 76.16 | 73.55 | -2.61 | -10.72 | 76.74 | 77.01 | OK | 25000 | dynamic_loss |
| qwen25vl7b | vqav2 | 25 | 84.27 | 78.89 | 77.50 | -1.39 | -6.77 | 67.76 | 68.10 | OK | 25000 | dynamic_loss |
| qwen25vl7b | vqav2 | 35 | 84.27 | 80.49 | 79.54 | -0.95 | -4.73 | 58.68 | 59.06 | OK | 25000 | dynamic_loss |
| qwen25vl7b | vqav2 | 50 | 84.27 | 82.09 | 81.70 | -0.39 | -2.57 | 45.17 | 45.55 | OK | 25000 | near_tie |
| qwen25vl7b | vqav2 | 75 | 84.27 | 83.66 | 83.60 | -0.06 | -0.67 | 22.58 | 22.84 | OK | 25000 | near_tie |
| qwen25vl7b | textvqa | 15 | 81.06 | 53.06 | 60.56 | +7.50 | -20.50 | 77.98 | 78.65 | OK | 5000 | dynamic_win |
| qwen25vl7b | textvqa | 25 | 81.06 | 59.17 | 67.53 | +8.36 | -13.53 | 68.81 | 69.64 | OK | 5000 | dynamic_win |
| qwen25vl7b | textvqa | 35 | 81.06 | 63.57 | 71.36 | +7.79 | -9.70 | 59.61 | 60.56 | OK | 5000 | dynamic_win |
| qwen25vl7b | textvqa | 50 | 81.06 | 70.14 | 76.08 | +5.94 | -4.98 | 45.86 | 46.84 | OK | 5000 | dynamic_win |
| qwen25vl7b | textvqa | 75 | 81.06 | 78.43 | 79.80 | +1.37 | -1.26 | 22.94 | 23.63 | OK | 5000 | dynamic_win |
| qwen25vl7b | docvqa | 15 | 94.76 | 29.90 | 33.09 | +3.19 | -61.67 | 82.19 | 82.88 | OK | 5349 | dynamic_win |
| qwen25vl7b | docvqa | 25 | 94.76 | 49.55 | 46.57 | -2.98 | -48.19 | 72.51 | 73.44 | OK | 5349 | dynamic_loss |
| qwen25vl7b | docvqa | 35 | 94.76 | 68.86 | 57.53 | -11.33 | -37.23 | 62.86 | 63.95 | OK | 5349 | dynamic_loss |
| qwen25vl7b | docvqa | 50 | 94.76 | 84.79 | 71.36 | -13.43 | -23.40 | 48.34 | 49.51 | OK | 5349 | dynamic_loss |
| qwen25vl7b | docvqa | 75 | 94.76 | 93.98 | 88.04 | -5.94 | -6.72 | 24.18 | 25.03 | OK | 5349 | dynamic_loss |
