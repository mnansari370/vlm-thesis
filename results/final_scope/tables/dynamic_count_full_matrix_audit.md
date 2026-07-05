# Dynamic-COUNT full-matrix audit

Status reflects the PURE Dynamic-COUNT path (locked static selector); COUNT-on-WHICH (textsim, qwen×textvqa) is tracked in its own columns.

| Model | Dataset | p15 probe | equiv | p25 probe | equiv | textsim | cfg | DC-D | DC-C rule | DC-C ridge | CoW DC-D | CoW DC-C | Status |
|---|---|:--:|:--:|:--:|:--:|--:|:--:|:--:|:--:|:--:|--:|--:|---|
| llava15 | gqa | Y | Y | Y | Y | 0 | Y | Y | Y | Y | 0 | 0 | dcc_complete |
| llava15 | textvqa | Y | Y | Y | Y | 0 | Y | Y | Y | Y | 0 | 0 | dcc_complete |
| llava15 | docvqa | Y | Y | Y | Y | 0 | Y | Y | Y | Y | 0 | 0 | dcc_complete |
| llava15 | vqav2 | Y | Y | Y | Y | 0 | Y | Y | Y | Y | 0 | 0 | dcc_complete |
| qwen25vl7b | gqa | Y | Y | Y | Y | 0 | Y | Y | Y | Y | 0 | 0 | dcc_complete |
| qwen25vl7b | textvqa | Y | Y | Y | Y | 2 | Y | Y | Y | Y | 1 | 2 | dcc_complete |
| qwen25vl7b | docvqa | Y | Y | Y | Y | 0 | Y | Y | Y | Y | 0 | 0 | dcc_complete |
| qwen25vl7b | vqav2 | Y | Y | Y | Y | 0 | Y | Y | Y | Y | 0 | 0 | dcc_complete |
