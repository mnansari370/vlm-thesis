#!/bin/bash
# Section 3 smoke test: budget-variance gate on 200 train / 200 val samples.
# Runs on GPU1. Expected time: ~15-25 minutes (5 epochs × 200 steps × ~2s/step).
# STOP after this: do not launch full gate without human authorization.
set -e

cd /home/nafees/vlm-thesis
LOG=/tmp/gate_smoke.log
OUT=results/gate_smoke_v1

echo "[gate_smoke] Started $(date)" | tee $LOG

CUDA_VISIBLE_DEVICES=1 conda run -n vlm_env python \
    scripts/data/budget_variance_gate.py \
    --config   configs/dynamic_budget/llava_dynamic_gate_smoke.yaml \
    --output-dir "$OUT" \
    --max-train 200 \
    --max-val   200 \
    --epochs    5 \
    --log-every 25 \
    >> $LOG 2>&1

echo "[gate_smoke] Completed $(date)" | tee -a $LOG
echo "[gate_smoke] Results: $OUT/gate_results.json" | tee -a $LOG
echo "[gate_smoke] STOP — awaiting human authorization before full gate run." | tee -a $LOG
