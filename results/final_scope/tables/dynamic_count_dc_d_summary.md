# DC-D (discrete cascade baseline)

PURE Dynamic-COUNT rows use the locked static selector; rows with selector `textsim` are COUNT-on-WHICH (qwen×textvqa) and are reported separately — never mixed.

| Model | Dataset | Sel | Variant tag | n | Score | avgTFLOPs | static@same | Δcurve | Label | Esc% |
|---|---|---|---|--:|--:|--:|--:|--:|---|--:|
| llava15 | gqa | cls_attn | dc_discrete | 10062 | 58.03 | 0.935 | 58.13 | -0.10 | near_tie | 1.91 |
| llava15 | textvqa | cls_attn | dc_discrete | 4000 | 57.12 | 2.239 | 56.37 | +0.75 | dynamic_win | 50.88 |
| llava15 | docvqa | cls_attn | dc_discrete | 4279 | 20.83 | 2.704 | 21.72 | -0.89 | dynamic_loss | 74.29 |
| llava15 | vqav2 | cls_attn | dc_discrete | 20000 | 75.89 | 1.046 | 75.83 | +0.06 | near_tie | 9.6 |
| qwen25vl7b | gqa | norm | dc_discrete | 10062 | 60.62 | 1.338 | 60.65 | -0.03 | near_tie | 40.1 |
| qwen25vl7b | textvqa | norm | dc_discrete | 4000 | 64.06 | 2.323 | 64.04 | +0.02 | near_tie | 13.2 |
| qwen25vl7b | textvqa | textsim | dc_discrete | 4000 | 62.07 | 1.396 | 55.08 | +6.99 | dynamic_win | 5.67 |
| qwen25vl7b | docvqa | norm | dc_discrete | 4279 | 53.09 | 2.188 | 57.7 | -4.61 | dynamic_loss | 9.46 |
| qwen25vl7b | vqav2 | norm | dc_discrete | 20000 | 79.8 | 0.753 | 79.59 | +0.21 | near_tie | 4.89 |
