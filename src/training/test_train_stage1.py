"""
Offline smoke test for train_stage1.py — drives the REAL training loop on ~4 synthetic
samples (tempdir, fake images, no download) at fixed K=64 for ~40 steps, and asserts:

  1. the loop runs: losses are finite throughout, no steps skipped;
  2. it actually LEARNS: loss drops clearly (overfitting a tiny set) → the train path
     (forward → loss → backward → optimizer step on LoRA + projector) is wired correctly;
  3. the checkpoint saves ONLY the trainable params (LoRA adapter + projector), non-empty
     and finite, and reloads as well-formed tensors.

Run (one GPU):
    CUDA_VISIBLE_DEVICES=0 /home/nafees/miniconda3/envs/vlm_env/bin/python v2/training/test_train_stage1.py
"""

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch
from PIL import Image

from src.training.train_stage1 import run_training


def _ok(cond, msg):
    if not cond:
        raise AssertionError("FAIL: " + msg)
    print("  ok:", msg)


def _make_synth(tmp):
    img_root = os.path.join(tmp, "images")
    os.makedirs(os.path.join(img_root, "coco"), exist_ok=True)
    os.makedirs(os.path.join(img_root, "gqa"), exist_ok=True)
    rng = np.random.RandomState(0)
    Image.fromarray(rng.randint(0, 255, (32, 40, 3), np.uint8), "RGB").save(
        os.path.join(img_root, "coco", "1.jpg"))
    Image.fromarray(rng.randint(0, 255, (40, 32, 3), np.uint8), "RGB").save(
        os.path.join(img_root, "gqa", "2.jpg"))
    mix = [
        {"id": "a", "image": "coco/1.jpg", "conversations": [
            {"from": "human", "value": "<image>\nWhat color is the box?"},
            {"from": "gpt", "value": "Red."}]},
        {"id": "b", "image": "gqa/2.jpg", "conversations": [
            {"from": "human", "value": "<image>\nHow many circles?"},
            {"from": "gpt", "value": "Two."}]},
        {"id": "c", "image": "coco/1.jpg", "conversations": [
            {"from": "human", "value": "<image>\nIs there a cat?"},
            {"from": "gpt", "value": "No."}]},
        {"id": "d", "image": "gqa/2.jpg", "conversations": [
            {"from": "human", "value": "<image>\nWhat shape is it?"},
            {"from": "gpt", "value": "Square."}]},
    ]
    jpath = os.path.join(tmp, "mini_mix.json")
    with open(jpath, "w") as f:
        json.dump(mix, f)
    return jpath, img_root


def main():
    tmp = tempfile.mkdtemp(prefix="v2_traintest_")
    try:
        jpath, img_root = _make_synth(tmp)
        out_dir = os.path.join(tmp, "run")
        cfg = {
            "seed": 42,
            "data": {"json_path": jpath, "image_root": img_root, "turn_mode": "first",
                     "max_samples": None, "verify_images": True, "drop_missing": True,
                     "num_workers": 0},
            "model": {"model_name": "llava-hf/llava-1.5-7b-hf", "dtype": "bfloat16",
                      "image_pad": True, "lora_rank": 8, "lora_alpha": 16,
                      "lora_dropout": 0.0, "k_ladder": [64]},
            "training": {"epochs": 20, "batch_size": 2, "grad_accum": 1,
                         "learning_rate": 3e-4, "weight_decay": 0.0, "warmup_ratio": 0.05,
                         "max_grad_norm": 1.0, "log_every": 10, "save_every": 0},
        }

        print("[run] driving real training loop on 4 synthetic samples, fixed K=64 ...")
        res = run_training(cfg, out_dir, fixed_k=64)
        losses = res["losses"]

        print("[1] loop ran, losses finite, none skipped")
        _ok(len(losses) >= 30, f"ran enough steps ({len(losses)})")
        _ok(all(np.isfinite(l) for l in losses), "all losses finite")
        _ok(res["skipped"] == 0, "no non-finite steps skipped")

        print("[2] loss decreases (overfit signal)")
        first = sum(losses[:2]) / 2
        last = sum(losses[-4:]) / 4
        print(f"     first≈{first:.3f}  last≈{last:.3f}  Δ={first-last:.3f}")
        _ok(last < first - 0.2, "loss dropped clearly → train path wired correctly")

        print("[3] checkpoint = ONLY trainable params, well-formed")
        ck = torch.load(res["ckpt_path"], map_location="cpu")
        _ok("lora_state" in ck and len(ck["lora_state"]) > 0, "checkpoint has non-empty LoRA adapter")
        _ok("projector_state" in ck and len(ck["projector_state"]) > 0, "checkpoint has projector")
        all_tensors = list(ck["lora_state"].values()) + list(ck["projector_state"].values())
        _ok(all(torch.is_tensor(t) and torch.isfinite(t).all() for t in all_tensors),
            "all saved tensors are finite")
        # size-based proof that ONLY trainable params were saved (no frozen 7B / no 131M embedding)
        total_numel = sum(t.numel() for t in all_tensors)
        max_numel = max(t.numel() for t in all_tensors)
        print(f"     checkpoint params: total={total_numel:,}  largest_tensor={max_numel:,}")
        _ok(total_numel < 100_000_000,
            f"checkpoint small (LoRA+projector, {total_numel:,}) — no frozen 7B leaked")
        _ok(max_numel < 60_000_000,
            "no huge frozen tensor (e.g. 131M embedding) leaked into the checkpoint")

        print("\nALL TRAINER SMOKE-TEST CHECKS PASSED ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
