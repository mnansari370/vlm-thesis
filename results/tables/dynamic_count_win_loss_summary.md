# Dynamic-COUNT win/loss summary (labels at ±0.50pp vs static curve)

PURE Dynamic-COUNT (static selector) is the headline; COUNT-on-WHICH (textsim) is listed separately so WHICH-vs-COUNT attribution stays clean.

## DC-D (baseline) — pure: dynamic_loss=2, dynamic_win=1, near_tie=5
- llava15 × textvqa: Δ=+0.75pp [dynamic_win] score=57.12
- qwen25vl7b × vqav2: Δ=+0.21pp [near_tie] score=79.8
- llava15 × vqav2: Δ=+0.06pp [near_tie] score=75.89
- qwen25vl7b × textvqa: Δ=+0.02pp [near_tie] score=64.06
- qwen25vl7b × gqa: Δ=-0.03pp [near_tie] score=60.62
- llava15 × gqa: Δ=-0.10pp [near_tie] score=58.03
- llava15 × docvqa: Δ=-0.89pp [dynamic_loss] score=20.83
- qwen25vl7b × docvqa: Δ=-4.61pp [dynamic_loss] score=53.09

## DC-D (baseline) — COUNT-on-WHICH: dynamic_win=1
- qwen25vl7b × textvqa: Δ=+6.99pp [dynamic_win] score=62.07

## DC-C (MAIN) — pure: dynamic_loss=11, near_tie=5
- llava15 × textvqa: Δ=+0.48pp [near_tie] score=56.75
- llava15 × textvqa: Δ=+0.37pp [near_tie] score=56.91
- qwen25vl7b × gqa: Δ=-0.19pp [near_tie] score=60.48
- llava15 × vqav2: Δ=-0.25pp [near_tie] score=76.29
- qwen25vl7b × gqa: Δ=-0.45pp [near_tie] score=59.89
- llava15 × vqav2: Δ=-0.51pp [dynamic_loss] score=76.34
- llava15 × docvqa: Δ=-0.54pp [dynamic_loss] score=21.23
- llava15 × docvqa: Δ=-0.84pp [dynamic_loss] score=20.86
- qwen25vl7b × vqav2: Δ=-0.84pp [dynamic_loss] score=81.41
- llava15 × gqa: Δ=-1.26pp [dynamic_loss] score=59.77
- qwen25vl7b × vqav2: Δ=-1.36pp [dynamic_loss] score=81.21
- llava15 × gqa: Δ=-2.24pp [dynamic_loss] score=58.58
- qwen25vl7b × textvqa: Δ=-7.27pp [dynamic_loss] score=71.59
- qwen25vl7b × textvqa: Δ=-7.89pp [dynamic_loss] score=71.45
- qwen25vl7b × docvqa: Δ=-11.07pp [dynamic_loss] score=81.55
- qwen25vl7b × docvqa: Δ=-12.73pp [dynamic_loss] score=79.28

## DC-C (MAIN) — COUNT-on-WHICH: dynamic_loss=1, near_tie=1
- qwen25vl7b × textvqa: Δ=-0.35pp [near_tie] score=77.09
- qwen25vl7b × textvqa: Δ=-0.79pp [dynamic_loss] score=77.92

