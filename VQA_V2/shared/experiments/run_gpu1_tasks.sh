#!/usr/bin/env bash
# Sequential small tasks on GPU 1 (all ≤2h total).
# Runs while dense 10K gen eval runs concurrently on GPU 0.
#
# Task 1: Dense MLP retrain — 30 epochs (model not converged at epoch 8)
# Task 2: Dense gen eval 1K on the better 30-epoch checkpoint
# Task 3: Static gen eval 1K (untrained) for all 6 K values
#         → zero-shot static baseline curve for the paper

set -euo pipefail

PYTHON=/home/nafees/miniconda3/envs/vlm_env/bin/python
PROJECT_ROOT="${HOME}/vlm-thesis"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
mkdir -p VQA_V2/logs VQA_V2/outputs

# ─── Task 1: Dense MLP retrain with 30 epochs ──────────────────────────────
log "=== Task 1: Dense MLP retrain (30 epochs) ==="

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

cfg = load_config('VQA_V2/dense/llava_dense_150k_10k_fullvocab.yaml')
cfg['training']['epochs'] = 30
set_seed(42, False)

os.makedirs('VQA_V2/outputs/dense_v1_30ep', exist_ok=True)
save_json('VQA_V2/outputs/dense_v1_30ep/config.json', cfg)

device = torch.device('cuda')
train_ds = CachedFeatureDataset('VQA_V2/feature_cache/dense/train')
val_ds   = CachedFeatureDataset('VQA_V2/feature_cache/dense/val')
id_to_answer = load_id_to_answer(cfg)
vocab_size = len(id_to_answer)
print(f'Vocab={vocab_size}, train={len(train_ds)}, val={len(val_ds)}', flush=True)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4,
                          collate_fn=cached_collate, pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=4,
                          collate_fn=cached_collate, pin_memory=True, persistent_workers=True)

answer_head = build_answer_head(cfg, vocab_size).to(device)
optimizer = build_optimizer(cfg, answer_head)
steps_per_epoch = math.ceil(len(train_loader) / 2)
scheduler = build_scheduler(cfg, optimizer, steps_per_epoch * 30)

best_acc, best_epoch, history = None, None, []
for epoch in range(1, 31):
    tr = train_one_epoch(answer_head, id_to_answer, train_loader, optimizer, scheduler, 2, device, epoch, log_every=9999)
    vl = validate(answer_head, id_to_answer, val_loader, device)
    val_acc = vl['vqa_accuracy']
    print(f'Epoch {epoch:2d}: train={tr[\"vqa_accuracy\"]:.4f} val={val_acc:.4f}', flush=True)
    history.append({'epoch': epoch, 'train': tr, 'val': {k:v for k,v in vl.items() if k!='predictions'}})
    if best_acc is None or val_acc > best_acc:
        best_acc, best_epoch = val_acc, epoch
        save_checkpoint('VQA_V2/outputs/dense_v1_30ep/best_model.pt', answer_head, optimizer, scheduler, epoch,
                        {'train': tr, 'val': {k:v for k,v in vl.items() if k!='predictions'}})
        save_json('VQA_V2/outputs/dense_v1_30ep/best_predictions.json', {'predictions': vl['predictions']})
    save_json('VQA_V2/outputs/dense_v1_30ep/history.json', history)

save_json('VQA_V2/outputs/dense_v1_30ep/metrics.json', {
    'best_epoch': best_epoch, 'best_val_vqa_accuracy': best_acc, 'history': history
})
print(f'Best: epoch={best_epoch} val_vqa={best_acc:.4f}', flush=True)
" 2>&1 | tee VQA_V2/logs/dense_mlp_30ep.log

log "=== Task 1 done ==="

# ─── Task 2: Dense gen eval 1K on the 30-epoch checkpoint ──────────────────
log "=== Task 2: Dense gen eval 1K (30-epoch checkpoint) ==="

${PYTHON} -u -m VQA_V2.shared.evaluation.generate_and_score \
    --config      VQA_V2/dense/llava_dense_150k_10k_fullvocab.yaml \
    --checkpoint  VQA_V2/outputs/dense_v1_30ep/best_model.pt \
    --model-type  dense \
    --output-path VQA_V2/outputs/dense_v1_30ep/generation_eval_1k.json \
    --max-samples 1000 \
    --skip-classification \
    --log-every 200 \
    2>&1 | tee VQA_V2/logs/gen_eval_dense_30ep_1k.log

log "=== Task 2 done ==="

# ─── Task 3: Static gen eval 1K (untrained) for all K ──────────────────────
log "=== Task 3: Static zero-shot gen eval 1K for all K ==="

for K in 64 128 144 192 288 432; do
    log "--- Static K=${K} gen eval 1K (untrained) ---"
    mkdir -p "VQA_V2/outputs/static_k${K}_zeroshot"

    ${PYTHON} -u -m VQA_V2.shared.evaluation.generate_and_score \
        --config      "VQA_V2/static/llava_static_clsattn_150k_10k_fullvocab_k${K}.yaml" \
        --model-type  static \
        --output-path "VQA_V2/outputs/static_k${K}_zeroshot/generation_eval_1k.json" \
        --max-samples 1000 \
        --skip-classification \
        --log-every 200 \
        2>&1 | tee "VQA_V2/logs/gen_eval_static_k${K}_zeroshot.log"

    # Print the accuracy
    ${PYTHON} -c "
import json
with open('VQA_V2/outputs/static_k${K}_zeroshot/generation_eval_1k.json') as f:
    d = json.load(f)
acc = d['generation']['vqa_accuracy']
print(f'K=${K}: generation accuracy = {acc:.4f} ({acc*100:.2f}%)')
" 2>/dev/null

    log "--- K=${K} done ---"
done

log "=== All tasks complete ==="
log "Summary:"
for K in 64 128 144 192 288 432; do
    /home/nafees/miniconda3/envs/vlm_env/bin/python -c "
import json, os
p = 'VQA_V2/outputs/static_k${K}_zeroshot/generation_eval_1k.json'
if os.path.exists(p):
    d = json.load(open(p))
    print(f'  Static K=${K}: {d[\"generation\"][\"vqa_accuracy\"]*100:.2f}%')
" 2>/dev/null
done