# Best method per model × dataset (F)

Best static = highest-accuracy anchor. Best WHICH/DC = best Δ vs the static curve at matched compute. DC rows are eval-80%; verdict thresholds ±0.50pp. COUNT-on-WHICH is excluded here (separate attribution table in the main comparison).

| Model | Dataset | Dense | Best static | Best WHICH | DC-D | Best DC-C | Verdict |
|---|---|--:|---|---|---|---|---|
| llava15 | gqa | 61.42 | 61.53 (p75) | 55.41 (p15, Δ-0.73, loss) | 58.03 (Δ-0.10, near-tie) | 59.77 (cls_attn/ridge, Δ-1.26, loss) | static floor stands |
| llava15 | vqav2 | 77.33 | 77.31 (p75) | 75.72 (p75, Δ-1.59, loss) | 75.89 (Δ+0.06, near-tie) | 76.29 (cls_attn/rule, Δ-0.25, near-tie) | static floor stands |
| llava15 | textvqa | 57.65 | 57.39 (p75) | 53.48 (p75, Δ-3.91, loss) | 57.12 (Δ+0.75, win) | 56.75 (cls_attn/ridge, Δ+0.48, near-tie) | DC-D beats static (+0.75) |
| llava15 | docvqa | 21.53 | 21.36 (p75) | 17.39 (p75, Δ-3.97, loss) | 20.83 (Δ-0.89, loss) | 21.23 (cls_attn/ridge, Δ-0.54, loss) | static floor stands |
| qwen25vl7b | gqa | 60.96 | 60.85 (p75) | 60.17 (p75, Δ-0.68, loss) | 60.62 (Δ-0.03, near-tie) | 60.48 (norm/rule, Δ-0.19, near-tie) | static floor stands |
| qwen25vl7b | vqav2 | 84.27 | 83.66 (p75) | 83.60 (p75, Δ-0.06, near-tie) | 79.80 (Δ+0.21, near-tie) | 81.41 (norm/ridge, Δ-0.84, loss) | static floor stands |
| qwen25vl7b | textvqa | 81.06 | 78.43 (p75) | 67.53 (p25, Δ+8.36, win) | 64.06 (Δ+0.02, near-tie) | 71.59 (norm/rule, Δ-7.27, loss) | WHICH beats static (+8.36) |
| qwen25vl7b | docvqa | 94.76 | 93.98 (p75) | 33.09 (p15, Δ+3.19, win) | 53.09 (Δ-4.61, loss) | 81.55 (norm/ridge, Δ-11.07, loss) | WHICH beats static (+3.19) |
