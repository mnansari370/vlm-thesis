# Dynamic-COUNT calibration report (first-20% split; CPU-only)

Pure Dynamic-COUNT uses the locked static selector; COUNT-on-WHICH (textsim) is calibrated and reported SEPARATELY (own config, textsim anchors).

## llava15 × gqa

- chosen probe: **cls_attn p25**, signal **conf_answer_len**, esc-rate 0.05, target p50 → calib Δcurve **-0.02pp**
- DC-C λ_default (compute-matched on calibration): rule=0.6, ridge=0.6
- signal capture@15%false-esc (calibration): conf_first_max_prob=50.7%, conf_first_margin=37.7%, conf_first_entropy=50.3%, conf_mean_logprob=40.7%, conf_mean_token_prob=34.2%, conf_min_token_prob=50.9%, conf_answer_len=8.0%
- config: `results/configs/dynamic_count/llava15_gqa.json`

## llava15 × textvqa

- chosen probe: **cls_attn p15**, signal **conf_mean_token_prob**, esc-rate 0.5, target p75 → calib Δcurve **+0.93pp**
- DC-C λ_default (compute-matched on calibration): rule=1.2, ridge=0.8
- signal capture@15%false-esc (calibration): conf_first_max_prob=57.3%, conf_first_margin=52.9%, conf_first_entropy=57.8%, conf_mean_logprob=65.5%, conf_mean_token_prob=61.1%, conf_min_token_prob=70.6%, conf_answer_len=12.6%
- config: `results/configs/dynamic_count/llava15_textvqa.json`

## llava15 × docvqa

- chosen probe: **cls_attn p25**, signal **conf_min_token_prob**, esc-rate 0.75, target p75 → calib Δcurve **-0.01pp**
- DC-C λ_default (compute-matched on calibration): rule=1.0, ridge=1.2
- signal capture@15%false-esc (calibration): conf_first_max_prob=37.2%, conf_first_margin=30.5%, conf_first_entropy=39.2%, conf_mean_logprob=48.9%, conf_mean_token_prob=45.8%, conf_min_token_prob=39.6%, conf_answer_len=12.7%
- config: `results/configs/dynamic_count/llava15_docvqa.json`

## llava15 × vqav2

- chosen probe: **cls_attn p25**, signal **conf_mean_logprob**, esc-rate 0.1, target p50 → calib Δcurve **+0.25pp**
- DC-C λ_default (compute-matched on calibration): rule=0.6, ridge=0.6
- signal capture@15%false-esc (calibration): conf_first_max_prob=52.9%, conf_first_margin=49.5%, conf_first_entropy=51.9%, conf_mean_logprob=58.0%, conf_mean_token_prob=51.0%, conf_min_token_prob=67.5%, conf_answer_len=22.7%
- config: `results/configs/dynamic_count/llava15_vqav2.json`

## qwen25vl7b × gqa

- chosen probe: **norm p25**, signal **conf_first_margin**, esc-rate 0.4, target p75 → calib Δcurve **+0.22pp**
- DC-C λ_default (compute-matched on calibration): rule=1.0, ridge=1.0
- signal capture@15%false-esc (calibration): conf_first_max_prob=50.8%, conf_first_margin=49.0%, conf_first_entropy=55.2%, conf_mean_logprob=47.7%, conf_mean_token_prob=46.3%, conf_min_token_prob=51.3%, conf_answer_len=28.9%
- config: `results/configs/dynamic_count/qwen25vl7b_gqa.json`

## qwen25vl7b × textvqa

- chosen probe: **norm p25**, signal **conf_min_token_prob**, esc-rate 0.15, target p75 → calib Δcurve **+1.36pp**
- DC-C λ_default (compute-matched on calibration): rule=0.6, ridge=0.6
- signal capture@15%false-esc (calibration): conf_first_max_prob=46.6%, conf_first_margin=39.4%, conf_first_entropy=52.1%, conf_mean_logprob=44.1%, conf_mean_token_prob=39.4%, conf_min_token_prob=53.4%, conf_answer_len=22.9%
- config: `results/configs/dynamic_count/qwen25vl7b_textvqa.json`

## qwen25vl7b × textvqa [COUNT-on-WHICH]

- chosen probe: **textsim p15**, signal **conf_mean_logprob**, esc-rate 0.05, target p50 → calib Δcurve **+9.47pp**
- DC-C λ_default (compute-matched on calibration): rule=0.6, ridge=0.6
- signal capture@15%false-esc (calibration): conf_first_max_prob=38.3%, conf_first_margin=32.1%, conf_first_entropy=45.1%, conf_mean_logprob=39.4%, conf_mean_token_prob=39.4%, conf_min_token_prob=49.5%, conf_answer_len=32.1%
- config: `results/configs/dynamic_count/qwen25vl7b_textvqa_textsim.json`

## qwen25vl7b × docvqa

- chosen probe: **norm p25**, signal **conf_first_entropy**, esc-rate 0.05, target p50 → calib Δcurve **-1.67pp**
- DC-C λ_default (compute-matched on calibration): rule=0.6, ridge=0.6
- signal capture@15%false-esc (calibration): conf_first_max_prob=38.5%, conf_first_margin=36.7%, conf_first_entropy=48.7%, conf_mean_logprob=37.5%, conf_mean_token_prob=39.3%, conf_min_token_prob=37.8%, conf_answer_len=9.6%
- config: `results/configs/dynamic_count/qwen25vl7b_docvqa.json`

## qwen25vl7b × vqav2

- chosen probe: **norm p25**, signal **conf_first_entropy**, esc-rate 0.05, target p75 → calib Δcurve **+0.62pp**
- DC-C λ_default (compute-matched on calibration): rule=0.6, ridge=0.6
- signal capture@15%false-esc (calibration): conf_first_max_prob=68.9%, conf_first_margin=63.2%, conf_first_entropy=73.3%, conf_mean_logprob=67.9%, conf_mean_token_prob=64.5%, conf_min_token_prob=71.3%, conf_answer_len=13.8%
- config: `results/configs/dynamic_count/qwen25vl7b_vqav2.json`

