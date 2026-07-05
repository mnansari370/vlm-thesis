# Dense / Static / WHICH / DC-D / DC-C — thesis comparison

Static reference = frozen anchors (interpolated curve at each method's average FLOPs, same evaluated ids). WHICH = frozen textsim finals (full manifest).

| Model | Dataset | Dense | Static p25/p75 | WHICH best (p) | DC-D Δcurve | DC-C Δcurve | DC-C label |
|---|---|--:|--:|--:|--:|--:|---|
| llava15 | gqa | 61.42 | 58.15/61.53 | 55.41 (p15) | -0.10 | -2.24 | dynamic_loss |
| llava15 | textvqa | 57.65 | 55.97/57.39 | 53.48 (p75) | +0.75 | +0.37 | near_tie |
| llava15 | docvqa | 21.53 | 19.32/21.36 | 17.39 (p75) | -0.89 | -0.84 | dynamic_loss |
| llava15 | vqav2 | 77.33 | 75.55/77.31 | 75.72 (p75) | +0.06 | -0.25 | near_tie |
| qwen25vl7b | gqa | 60.96 | 58.77/60.85 | 60.17 (p75) | -0.03 | -0.19 | near_tie |
| qwen25vl7b | textvqa | 81.06 | 59.17/78.43 | 67.53 (p25) | +0.02 | -7.27 | dynamic_loss |
| qwen25vl7b | docvqa | 94.76 | 49.55/93.98 | 33.09 (p15) | -4.61 | -12.73 | dynamic_loss |
| qwen25vl7b | vqav2 | 84.27 | 78.89/83.66 | 83.6 (p75) | +0.21 | -1.36 | dynamic_loss |
