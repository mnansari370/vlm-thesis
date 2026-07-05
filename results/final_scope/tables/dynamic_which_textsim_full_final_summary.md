# Dynamic-WHICH textsim — full-final summary (complete matrix)

Completed full textsim finals: **40** / 40 expected. Missing: **0**.

| Model | Dataset | p | Dynamic | Static | Dense | Dyn−Static | Dyn−Dense | Tok red.% | FLOP red.% | Gate | n |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|:--:|--:|
| llava15 | docvqa | 15 | 11.41 | 17.73 | 21.53 | -6.32 | -10.12 | 80.18 | 80.67 | OK | 5349 |
| llava15 | docvqa | 25 | 12.50 | 19.32 | 21.53 | -6.82 | -9.03 | 70.69 | 71.33 | OK | 5349 |
| llava15 | docvqa | 35 | 13.35 | 20.09 | 21.53 | -6.74 | -8.18 | 61.20 | 61.93 | OK | 5349 |
| llava15 | docvqa | 50 | 14.47 | 20.75 | 21.53 | -6.28 | -7.06 | 47.13 | 47.90 | OK | 5349 |
| llava15 | docvqa | 75 | 17.39 | 21.36 | 21.53 | -3.97 | -4.14 | 23.56 | 24.12 | OK | 5349 |
| llava15 | gqa | 15 | 55.41 | 56.14 | 61.42 | -0.73 | -6.01 | 80.36 | 80.85 | OK | 12578 |
| llava15 | gqa | 25 | 56.89 | 58.15 | 61.42 | -1.26 | -4.53 | 70.85 | 71.48 | OK | 12578 |
| llava15 | gqa | 35 | 57.54 | 59.41 | 61.42 | -1.87 | -3.88 | 61.34 | 62.07 | OK | 12578 |
| llava15 | gqa | 50 | 58.96 | 60.53 | 61.42 | -1.57 | -2.46 | 47.23 | 48.00 | OK | 12578 |
| llava15 | gqa | 75 | 60.80 | 61.53 | 61.42 | -0.73 | -0.62 | 23.62 | 24.17 | OK | 12578 |
| llava15 | textvqa | 15 | 47.48 | 56.05 | 57.65 | -8.57 | -10.17 | 73.97 | 74.60 | OK | 5000 |
| llava15 | textvqa | 25 | 49.10 | 55.97 | 57.65 | -6.87 | -8.55 | 65.22 | 65.96 | OK | 5000 |
| llava15 | textvqa | 35 | 49.76 | 56.71 | 57.65 | -6.95 | -7.89 | 56.46 | 57.27 | OK | 5000 |
| llava15 | textvqa | 50 | 50.81 | 56.40 | 57.65 | -5.59 | -6.84 | 43.48 | 44.29 | OK | 5000 |
| llava15 | textvqa | 75 | 53.48 | 57.39 | 57.65 | -3.91 | -4.17 | 21.74 | 22.30 | OK | 5000 |
| llava15 | vqav2 | 15 | 68.33 | 73.36 | 77.33 | -5.03 | -9.00 | 80.72 | 81.19 | OK | 25000 |
| llava15 | vqav2 | 25 | 70.03 | 75.55 | 77.33 | -5.52 | -7.30 | 71.16 | 71.79 | OK | 25000 |
| llava15 | vqav2 | 35 | 71.32 | 76.13 | 77.33 | -4.81 | -6.01 | 61.61 | 62.33 | OK | 25000 |
| llava15 | vqav2 | 50 | 73.14 | 76.94 | 77.33 | -3.80 | -4.19 | 47.44 | 48.21 | OK | 25000 |
| llava15 | vqav2 | 75 | 75.72 | 77.31 | 77.33 | -1.59 | -1.61 | 23.72 | 24.28 | OK | 25000 |
| qwen25vl7b | docvqa | 15 | 33.09 | 29.90 | 94.76 | +3.19 | -61.67 | 82.19 | 82.88 | OK | 5349 |
| qwen25vl7b | docvqa | 25 | 46.57 | 49.55 | 94.76 | -2.98 | -48.19 | 72.51 | 73.44 | OK | 5349 |
| qwen25vl7b | docvqa | 35 | 57.53 | 68.86 | 94.76 | -11.33 | -37.23 | 62.86 | 63.95 | OK | 5349 |
| qwen25vl7b | docvqa | 50 | 71.36 | 84.79 | 94.76 | -13.43 | -23.40 | 48.34 | 49.51 | OK | 5349 |
| qwen25vl7b | docvqa | 75 | 88.04 | 93.98 | 94.76 | -5.94 | -6.72 | 24.18 | 25.03 | OK | 5349 |
| qwen25vl7b | gqa | 15 | 53.45 | 56.04 | 60.96 | -2.59 | -7.51 | 76.28 | 76.56 | OK | 12578 |
| qwen25vl7b | gqa | 25 | 55.97 | 58.77 | 60.96 | -2.80 | -4.99 | 67.37 | 67.71 | OK | 12578 |
| qwen25vl7b | gqa | 35 | 57.47 | 59.99 | 60.96 | -2.52 | -3.49 | 58.34 | 58.71 | OK | 12578 |
| qwen25vl7b | gqa | 50 | 58.87 | 60.53 | 60.96 | -1.66 | -2.09 | 44.91 | 45.29 | OK | 12578 |
| qwen25vl7b | gqa | 75 | 60.17 | 60.85 | 60.96 | -0.68 | -0.79 | 22.44 | 22.71 | OK | 12578 |
| qwen25vl7b | textvqa | 15 | 60.56 | 53.06 | 81.06 | +7.50 | -20.50 | 77.98 | 78.65 | OK | 5000 |
| qwen25vl7b | textvqa | 25 | 67.53 | 59.17 | 81.06 | +8.36 | -13.53 | 68.81 | 69.64 | OK | 5000 |
| qwen25vl7b | textvqa | 35 | 71.36 | 63.57 | 81.06 | +7.79 | -9.70 | 59.61 | 60.56 | OK | 5000 |
| qwen25vl7b | textvqa | 50 | 76.08 | 70.14 | 81.06 | +5.94 | -4.98 | 45.86 | 46.84 | OK | 5000 |
| qwen25vl7b | textvqa | 75 | 79.80 | 78.43 | 81.06 | +1.37 | -1.26 | 22.94 | 23.63 | OK | 5000 |
| qwen25vl7b | vqav2 | 15 | 73.55 | 76.16 | 84.27 | -2.61 | -10.72 | 76.74 | 77.01 | OK | 25000 |
| qwen25vl7b | vqav2 | 25 | 77.50 | 78.89 | 84.27 | -1.39 | -6.77 | 67.76 | 68.10 | OK | 25000 |
| qwen25vl7b | vqav2 | 35 | 79.54 | 80.49 | 84.27 | -0.95 | -4.73 | 58.68 | 59.06 | OK | 25000 |
| qwen25vl7b | vqav2 | 50 | 81.70 | 82.09 | 84.27 | -0.39 | -2.57 | 45.17 | 45.55 | OK | 25000 |
| qwen25vl7b | vqav2 | 75 | 83.60 | 83.66 | 84.27 | -0.06 | -0.67 | 22.58 | 22.84 | OK | 25000 |

## Interpretation

- **Qwen2.5-VL × TextVQA is the only already-validated full success** (Dyn−Static positive at every budget; independently reproduced by the clean-room `textsim_ref` at n=200).
- The other full runs are added for a **fair, complete comparison** against dense and static on the same footing — including datasets where the n=200 pilot evidence was negative. This is a complete evaluation matrix, **not cherry-picking**; negative full results are reported as-is.
- Headline reading: **Dynamic-WHICH textsim is strong for Qwen×TextVQA but is NOT universal** — it under-performs or loses to the static floor on most other datasets. The single primary method (textsim) is held fixed everywhere so the comparison is clean.
- Cells (of the completed 40) where textsim beats static: **6** (qwen25vl7b/docvqa, qwen25vl7b/textvqa).
