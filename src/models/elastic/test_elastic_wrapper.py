"""
GPU smoke test for elastic_wrapper.py — ONE fake batch, no training data, no download
(uses the locally-cached LLaVA-1.5-7B). Verifies, before any real run:

  1. ONLY LoRA + projector are trainable (vision tower + base LLM frozen).
  2. Teacher-forced forward gives a finite, grad-enabled loss.
  3. backward() populates grads on LoRA AND projector (so the train path is wired correctly).
  4. Elastic K works: forward at several K values; n_visual == K; bigger K → longer sequence.
  5. A frozen param (vision tower) stays frozen (no grad).

Run (one GPU):
    CUDA_VISIBLE_DEVICES=0 /home/nafees/miniconda3/envs/vlm_env/bin/python src/models/elastic/test_elastic_wrapper.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch
from PIL import Image

from src.models.elastic.elastic_wrapper import ElasticPrunedLlava


def _ok(cond, msg):
    if not cond:
        raise AssertionError("FAIL: " + msg)
    print("  ok:", msg)


def _fake_image(w, h, seed):
    rng = np.random.RandomState(seed)
    return Image.fromarray(rng.randint(0, 255, (h, w, 3), dtype=np.uint8), "RGB")


def main():
    print("[load] building ElasticPrunedLlava (rank=16) ...")
    m = ElasticPrunedLlava(lora_rank=16, dtype="bfloat16", image_pad=True)

    # ── 1) trainable params are ONLY lora + projector ────────────────────────
    print("[1] trainable-parameter audit")
    s = m.param_summary()
    _ok(s["lora"] > 0, f"LoRA params are trainable ({s['lora']:,})")
    _ok(s["projector"] > 0, f"projector params are trainable ({s['projector']:,})")
    _ok(s["other_trainable"] == 0, "NOTHING else is trainable (no stray unfrozen weights)")
    vt_train = sum(p.numel() for n, p in m.model.named_parameters()
                   if p.requires_grad and "vision_tower" in n)
    _ok(vt_train == 0, "vision tower is fully frozen")
    print(f"     trainable {s['trainable']:,} / {s['total']:,} "
          f"({100*s['trainable']/s['total']:.3f}%)")

    # ── 2+3) forward + backward at K=64 ──────────────────────────────────────
    print("[2] teacher-forced forward (K=64)")
    batch = {
        "images":    [_fake_image(640, 480, 1), _fake_image(300, 400, 2)],  # different sizes
        "questions": ["What color is the box?", "How many circles are there?"],
        "answers":   ["Red.", "Two."],
    }
    out = m.forward(batch, K=64)
    loss = out["loss"]
    _ok(torch.isfinite(loss).item(), f"loss is finite ({loss.item():.4f})")
    _ok(loss.requires_grad, "loss carries grad")
    _ok(out["info"]["n_visual"] == 64, "info reports n_visual == K (64)")

    print("[3] backward populates grads on LoRA + projector only")
    loss.backward()
    lora_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                    for n, p in m.model.named_parameters() if "lora_" in n and p.requires_grad)
    proj_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                    for n, p in m.model.named_parameters() if "multi_modal_projector" in n)
    _ok(lora_grad, "at least one LoRA param received a non-zero grad")
    _ok(proj_grad, "projector received a non-zero grad")
    vt_has_grad = any(p.grad is not None
                      for n, p in m.model.named_parameters() if "vision_tower" in n)
    _ok(not vt_has_grad, "vision tower received NO grad (stays frozen)")

    # ── 4) elastic K: several budgets, sequence grows with K ─────────────────
    print("[4] elastic K (32, 256, 576)")
    seq_lens = {}
    for K in (32, 256, 576):
        with torch.no_grad():
            o = m.forward(batch, K=K)
        _ok(torch.isfinite(o["loss"]).item(), f"K={K}: finite loss ({o['loss'].item():.4f})")
        _ok(o["info"]["n_visual"] == K, f"K={K}: n_visual == K")
        seq_lens[K] = o["info"]["seq_len"]
    _ok(seq_lens[32] < seq_lens[256] < seq_lens[576],
        f"sequence length grows with K  {seq_lens}")

    # ── 5) sample_k stays on the ladder ──────────────────────────────────────
    print("[5] sample_k() draws only from the ladder")
    draws = {m.sample_k() for _ in range(50)}
    _ok(draws.issubset(set(m.k_ladder)), f"sampled K ⊆ ladder {m.k_ladder} (saw {sorted(draws)})")

    print("\nALL SMOKE-TEST CHECKS PASSED ✓")


if __name__ == "__main__":
    main()
