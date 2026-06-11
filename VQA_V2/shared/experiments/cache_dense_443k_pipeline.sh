#!/usr/bin/env bash
# Full 443K dense pipeline on GPU 0.
# Runs CONCURRENTLY alongside cache_static_gpu0.sh (memory: 14.6+14.6=29.2GB < 49GB).
#
# Chain: cache 443K train → MLP 30ep → gen eval 1K → gen eval 10K (final headline number)

set -euo pipefail
PYTHON=/home/nafees/miniconda3/envs/vlm_env/bin/python
PROJECT_ROOT="${HOME}/vlm-thesis"
cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

CACHE_DIR="VQA_V2/feature_cache/dense_443k"
CONFIG="VQA_V2/dense/llava_dense_443k_10k_fullvocab.yaml"
OUTPUT_DIR="VQA_V2/outputs/dense_443k_v1"

# ── Step 1: Cache full 443K train set ───────────────────────────────────────
if [ -f "${CACHE_DIR}/train/pooled_features.npy" ]; then
    log "443K train cache already exists — skipping"
else
    log "=== Caching dense 443K train set (GPU 0) ==="
    ${PYTHON} -u VQA_V2/shared/scripts/cache_features.py \
        --model-type dense \
        --split train \
        --config "${CONFIG}" \
        --cache-dir "${CACHE_DIR}" \
        --log-every 2000
    log "=== 443K train cache done ==="
fi

# ── Step 2: MLP training on full 443K features, 30 epochs ───────────────────
log "=== MLP training on 443K (30 epochs) ==="
${PYTHON} -u -c "
import sys, math, json, os, torch, torch.nn as nn
from torch.utils.data import DataLoader
sys.path.insert(0, '.')

from VQA_V2.shared.utils.config import load_config
from VQA_V2.shared.utils.seed import set_seed
from VQA_V2.shared.training.cached.train_cached import (
    CachedFeatureDataset, cached_collate, build_answer_head,
    build_optimizer, build_scheduler, train_one_epoch, validate,
    load_id_to_answer, save_checkpoint, save_json
)

cfg = load_config('${CONFIG}')
cfg['training']['epochs'] = 30
set_seed(42, False)

os.makedirs('${OUTPUT_DIR}', exist_ok=True)
save_json('${OUTPUT_DIR}/config.json', cfg)

device = torch.device('cuda')
train_ds = CachedFeatureDataset('${CACHE_DIR}/train')
val_ds   = CachedFeatureDataset('VQA_V2/feature_cache/dense/val')   # 10K val from 150K run
id_to_answer = load_id_to_answer(cfg)
vocab_size = len(id_to_answer)
print(f'Vocab={vocab_size}  train={len(train_ds)}  val={len(val_ds)}', flush=True)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4,
    collate_fn=cached_collate, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=4,
    collate_fn=cached_collate, pin_memory=True, persistent_workers=True)

answer_head = build_answer_head(cfg, vocab_size).to(device)
optimizer = build_optimizer(cfg, answer_head)
steps_per_epoch = math.ceil(len(train_loader) / 2)
scheduler = build_scheduler(cfg, optimizer, steps_per_epoch * 30)

best_acc, best_epoch, history = None, None, []
for epoch in range(1, 31):
    tr = train_one_epoch(answer_head, id_to_answer, train_loader, optimizer, scheduler,
                         2, device, epoch, log_every=9999)
    vl = validate(answer_head, id_to_answer, val_loader, device)
    val_acc = vl['vqa_accuracy']
    print(f'Epoch {epoch:2d}: train={tr[\"vqa_accuracy\"]*100:.2f}%  val={val_acc*100:.2f}%', flush=True)
    history.append({'epoch': epoch, 'train': tr,
                    'val': {k: v for k, v in vl.items() if k != 'predictions'}})
    if best_acc is None or val_acc > best_acc:
        best_acc, best_epoch = val_acc, epoch
        save_checkpoint('${OUTPUT_DIR}/best_model.pt', answer_head, optimizer, scheduler,
                        epoch, {'train': tr, 'val': {k: v for k, v in vl.items() if k != 'predictions'}})
        save_json('${OUTPUT_DIR}/best_predictions.json', {'predictions': vl['predictions']})
    save_json('${OUTPUT_DIR}/history.json', history)

save_json('${OUTPUT_DIR}/metrics.json', {
    'best_epoch': best_epoch, 'best_val_vqa_accuracy': best_acc, 'history': history
})
print(f'Best: epoch={best_epoch}  val_cls={best_acc*100:.2f}%', flush=True)
" 2>&1 | tee VQA_V2/logs/dense_443k_mlp.log
log "=== MLP training done ==="

# ── Step 3: Generation eval on 1K (quick sanity check) ─────────────────────
log "=== Gen eval 1K (443K model) ==="
${PYTHON} -u -m VQA_V2.shared.evaluation.generate_and_score \
    --config      "${CONFIG}" \
    --checkpoint  "${OUTPUT_DIR}/best_model.pt" \
    --model-type  dense \
    --output-path "${OUTPUT_DIR}/generation_eval_1k.json" \
    --max-samples 1000 \
    --skip-classification \
    --log-every 200 \
    2>&1 | tee VQA_V2/logs/dense_443k_gen_1k.log

${PYTHON} -c "
import json
d = json.load(open('${OUTPUT_DIR}/generation_eval_1k.json'))
print(f'Gen eval 1K: {d[\"generation\"][\"vqa_accuracy\"]*100:.2f}%')
" 2>/dev/null
log "=== Gen eval 1K done ==="

# ── Step 4: Full 10K generation eval — the headline number ─────────────────
log "=== Gen eval FULL 10K (443K model) — headline number ==="
${PYTHON} -u -m VQA_V2.shared.evaluation.generate_and_score \
    --config      "${CONFIG}" \
    --checkpoint  "${OUTPUT_DIR}/best_model.pt" \
    --model-type  dense \
    --output-path "${OUTPUT_DIR}/generation_eval_10k.json" \
    --skip-classification \
    --log-every 200 \
    2>&1 | tee VQA_V2/logs/dense_443k_gen_10k.log

${PYTHON} -c "
import json
d = json.load(open('${OUTPUT_DIR}/generation_eval_10k.json'))
print(f'*** HEADLINE: Dense 443K gen eval 10K = {d[\"generation\"][\"vqa_accuracy\"]*100:.2f}% ***')
" 2>/dev/null
log "=== DONE — dense_443k_pipeline complete ==="
