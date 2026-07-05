#!/bin/bash
# Option A orchestrator: cache (resume) -> train -> gate, sequential, one tracked job.
# Each stage guarded; aborts if a stage fails or the cache is incomplete.
set -o pipefail
cd /home/nafees/vlm-thesis || exit 1
PY=~/miniconda3/envs/qwen_env/bin/python
export CUDA_VISIBLE_DEVICES=1
LOG=results/thesis_main/highres/distill/logs
TCACHE=results/thesis_main/highres/distill/teacher_docvqa_train12k.pt
STUDENT=results/thesis_main/highres/distill/student_12k.pt
FEATDIR=results/thesis_main/highres/distill/featcache_12k
GATEOUT=results/thesis_main/highres/distill/gate_12k.json

echo "[orch] $(date) ===== STAGE 1: teacher cache (resume) ====="
$PY -m v2.distill.cache_teacher --split train --n-shards 4 --start 0 --end 12000 \
    --out "$TCACHE" --log-every 500 --save-every 500 >> "$LOG/cache_teacher_12k.log" 2>&1 || { echo "[orch] cache FAILED"; exit 1; }

ROWS=$($PY -c "import torch;print(len(torch.load('$TCACHE',map_location='cpu',weights_only=False)['rows']))")
echo "[orch] $(date) cache rows = $ROWS"
if [ "$ROWS" -lt 11900 ]; then echo "[orch] ABORT: cache incomplete ($ROWS < 11900)"; exit 2; fi

echo "[orch] $(date) ===== STAGE 2: train scaled student (d=768, layers=3, 10 epochs) ====="
$PY -m v2.distill.train_student --teacher "$TCACHE" --out "$STUDENT" \
    --feat-cache-dir "$FEATDIR" --d 768 --layers 3 --epochs 10 --lr 3e-4 --log-every 1000 \
    > "$LOG/train_student_12k.log" 2>&1 || { echo "[orch] train FAILED"; exit 1; }
echo "[orch] $(date) train done"

echo "[orch] $(date) ===== STAGE 3: gate on validation[0:400] @ K=128 ====="
$PY -m v2.distill.eval_gate --student "$STUDENT" \
    --gate-split validation --gate-start 0 --n 400 --K 128 --log-every 100 \
    --out "$GATEOUT" > "$LOG/eval_gate_12k.log" 2>&1 || { echo "[orch] gate FAILED"; exit 1; }

echo "[orch] $(date) ===== ALL DONE ====="
cat "$GATEOUT"
