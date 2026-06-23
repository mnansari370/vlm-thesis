#!/usr/bin/env bash
# Detached launcher for the full Stage-1 elastic run.
# Uses setsid so it survives SSH / laptop disconnect (runs independently on vonasah).
#   bash v2/training/launch_stage1_full.sh
# Watch:    tail -f v2/outputs/stage1_full.log
# Stop:     pkill -9 -f "train_stage[1]"
#
# Memory: expandable_segments tames fragmentation from the varying K/answer shapes
#         (bs=16 + gradient checkpointing → ~30 GB reserved, safe on a 48 GB card).
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=/home/nafees/miniconda3/envs/vlm_env/bin/python
LOG="$REPO/v2/outputs/stage1_full.log"
mkdir -p "$REPO/v2/outputs"

setsid bash -c "
  cd '$REPO'
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export CUDA_VISIBLE_DEVICES=0
  exec '$PY' -u -m v2.training.train_stage1 \
    --config v2/configs/stage1_elastic.yaml \
    --output-dir v2/outputs/stage1_full --log-every 20 >> '$LOG' 2>&1
" < /dev/null &

echo "launched (detached, setsid). log: $LOG"
