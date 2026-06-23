#!/bin/bash
# Follow-ups: (1) question-conditioning control on the 10-epoch student,
#             (2) retrain 25 epochs reusing the built feature cache,
#             (3) gate the 25-epoch student.
set -o pipefail
cd /home/nafees/vlm-thesis || exit 1
PY=~/miniconda3/envs/qwen_env/bin/python
export CUDA_VISIBLE_DEVICES=1
LOG=v2/outputs/distill/logs
FEATDIR=v2/outputs/distill/featcache_12k
TCACHE=v2/outputs/distill/teacher_docvqa_train12k.pt

echo "[fu] $(date) ===== STAGE 1: question-conditioning control (10-epoch student) ====="
$PY -m v2.distill.eval_control --student v2/outputs/distill/student_12k.pt \
    --gate-split validation --gate-start 0 --n 200 --K 128 --log-every 50 \
    --out v2/outputs/distill/control_12k.json > "$LOG/eval_control_12k.log" 2>&1 || echo "[fu] control failed (non-fatal)"

echo "[fu] $(date) ===== STAGE 2: retrain 25 epochs (reuse feature cache) ====="
$PY -m v2.distill.train_student --teacher "$TCACHE" --out v2/outputs/distill/student_12k_25ep.pt \
    --feat-cache-dir "$FEATDIR" --d 768 --layers 3 --epochs 25 --lr 3e-4 --log-every 2000 \
    > "$LOG/train_student_12k_25ep.log" 2>&1 || { echo "[fu] train FAILED"; exit 1; }
echo "[fu] $(date) train25 done"

echo "[fu] $(date) ===== STAGE 3: gate 25-epoch student on validation[0:400] ====="
$PY -m v2.distill.eval_gate --student v2/outputs/distill/student_12k_25ep.pt \
    --gate-split validation --gate-start 0 --n 400 --K 128 --log-every 100 \
    --out v2/outputs/distill/gate_12k_25ep.json > "$LOG/eval_gate_12k_25ep.log" 2>&1 || { echo "[fu] gate FAILED"; exit 1; }

echo "[fu] $(date) ===== ALL DONE ====="
echo "--- control ---"; cat v2/outputs/distill/control_12k.json 2>/dev/null
echo "--- gate 25ep ---"; cat v2/outputs/distill/gate_12k_25ep.json 2>/dev/null
