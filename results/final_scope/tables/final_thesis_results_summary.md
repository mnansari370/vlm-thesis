# G. Final thesis results summary

**The complete pruning-method arc on frozen backbones (2 models × 4 datasets, locked manifests, identical prompts/scorers/decoding, honest multi-pass FLOPs):**

| Model | Dataset | Dense | Best static | Best WHICH Δstatic | Best DC-D Δcurve | Best DC-C Δcurve | Who wins? |
|---|---|--:|---|--:|--:|--:|---|
| llava15 | gqa | 61.42 | 61.53 (p75) | -0.73 | -0.10 | -1.26 | static floor stands |
| llava15 | vqav2 | 77.33 | 77.31 (p75) | -1.59 | +0.06 | -0.25 | static floor stands |
| llava15 | textvqa | 57.65 | 57.39 (p75) | -3.91 | +0.75 | +0.48 | DC-D beats static (+0.75) |
| llava15 | docvqa | 21.53 | 21.36 (p75) | -3.97 | -0.89 | -0.54 | static floor stands |
| qwen25vl7b | gqa | 60.96 | 60.85 (p75) | -0.68 | -0.03 | -0.19 | static floor stands |
| qwen25vl7b | vqav2 | 84.27 | 83.66 (p75) | -0.06 | +0.21 | -0.84 | static floor stands |
| qwen25vl7b | textvqa | 81.06 | 78.43 (p75) | +8.36 | +0.02 | -7.27 | WHICH beats static (+8.36) |
| qwen25vl7b | docvqa | 94.76 | 93.98 (p75) | +3.19 | -4.61 | -11.07 | WHICH beats static (+3.19) |

**COUNT-on-WHICH (qwen × textvqa, separate):** DC-C ridge: Δstatic -0.35, Δtextsim-curve -2.30; DC-C rule: Δstatic -0.79, Δtextsim-curve -2.03; DC-D: Δstatic +6.99, Δtextsim-curve -0.41.

**Reading:** Dynamic-WHICH (textsim) beats the static floor only on Qwen×TextVQA (localized, text-addressable evidence + language-aligned visual features) — validated by a clean-room re-implementation. Dynamic-COUNT — discrete (DC-D) and continuous integer-K_i (DC-C) — achieves at best near-ties on flat-curve cells and loses on steep-curve cells, despite real per-sample budget variation (hundreds of unique K_i) and informative confidence signals (47–73% wrong-capture): the oracle headroom is not harvestable under honest double-pay accounting. COUNT adds no increment on top of WHICH. The static accuracy-vs-compute curve is a far stronger baseline than the adaptive-pruning literature's typical comparisons acknowledge.
