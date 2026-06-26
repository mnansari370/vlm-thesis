# VQA_V2 Experiments

> Paths refreshed to the post-2026-06-11 layout (`VQA_V2/shared/...`); the workflow itself is the
> historical caching/training pipeline and the GPU phase is closed.

## Strategy
- **vonasah (dev GPU 1)**: short jobs ≤ 1h — val caches, MLP training (~30min), generation eval on 1K subsets
- **HPC IRIS (gpu/l40s partition)**: long jobs >1h — train caches (~15h each)

## HPC submission order

### Step 1: Dense train cache (submit to HPC)
```bash
cd ~/vlm-thesis
sbatch --job-name=vqa2_cache_dense_train \
    VQA_V2/shared/experiments/hpc/cache_features.sh dense train
```

### Step 2: All 6 static train caches (submit to HPC, can be parallel)
```bash
for K in 64 128 144 192 288 432; do
    sbatch --job-name=vqa2_cache_static_k${K}_train \
        VQA_V2/shared/experiments/hpc/cache_features.sh static train ${K}
done
```

### Step 3: Val caches (run on vonasah, ~1h each)
```bash
# Dense val already running on vonasah
# For static val caches:
for K in 64 128 144 192 288 432; do
    CUDA_VISIBLE_DEVICES=1 python VQA_V2/shared/scripts/cache_features.py \
        --model-type static --keep-tokens ${K} --split val \
        --config VQA_V2/static/llava_static_clsattn_150k_10k_fullvocab_k${K}.yaml \
        --cache-dir VQA_V2/feature_cache --log-every 100
done
```

### Step 4: MLP training (run on vonasah as each cache pair completes, ~30min each)
```bash
python -m VQA_V2.shared.training.cached.train_cached \
    --train-cache VQA_V2/feature_cache/dense/train \
    --val-cache   VQA_V2/feature_cache/dense/val \
    --config      VQA_V2/dense/llava_dense_150k_10k_fullvocab.yaml \
    --output-dir  VQA_V2/outputs/dense_v1 \
    --batch-size 128 --eval-batch-size 256
```

### Step 5: Generation eval on full 10K (submit to HPC)
```bash
sbatch --job-name=vqa2_gen_eval_dense \
    VQA_V2/shared/experiments/hpc/generation_eval.sh \
    dense VQA_V2/outputs/dense_v1/best_model.pt \
    VQA_V2/outputs/dense_v1/generation_eval_full.json
```

## IRIS partition note
- `--partition=gpu` → 98.96% busy, may wait in queue
- `--partition=l40s` → 25% busy, faster start (change in script if needed)
