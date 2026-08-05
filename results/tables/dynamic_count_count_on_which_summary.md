# Dynamic-COUNT after question-conditioned selection (qwen25vl7b x textvqa)

Held-out 80% evaluation after calibration. Both references are fixed-budget curves
reconstructed on the same held-out ids and interpolated at the matched mean analytical
language-model input-processing computation. Deltas are controller score minus reference.

| Model | Dataset | Sel | Variant | n | Score | avgTFLOPs | WHICH-curve@same | dWHICH | static@same | dstatic |
|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|
| qwen25vl7b | textvqa | textsim | DC-C ridge | 4000 | 77.09 | 4.204 | 79.39 | -2.30 | 77.44 | -0.35 |
| qwen25vl7b | textvqa | textsim | DC-C rule | 4000 | 77.92 | 4.480 | 79.95 | -2.03 | 78.71 | -0.79 |
| qwen25vl7b | textvqa | textsim | DC-D | 4000 | 62.07 | 1.396 | 62.48 | -0.41 | 55.08 | +6.99 |
