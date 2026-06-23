"""
Verify gradient checkpointing (GC) in the elastic wrapper — TWO things:

  (A) CORRECTNESS: with LoRA dropout=0 (deterministic), GC and non-GC must produce
      the SAME loss and the SAME trainable-grad fingerprint. GC is mathematically
      exact; this catches the dropout-RNG / wiring bugs that would silently corrupt
      gradients. Uses a long answer (stresses the real worst case).

  (B) MEMORY/CAPACITY: with GC on, probe bs ∈ {4, 8, 16} (long answers, K=576) and
      report peak GPU memory for each → pick the batch size for the full run.
      (Recall: bs=4 WITHOUT GC hit 47/49 GB — near OOM.)

Run (one GPU):
    CUDA_VISIBLE_DEVICES=0 /home/nafees/miniconda3/envs/vlm_env/bin/python v2/model/test_grad_checkpointing.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch
from PIL import Image

from v2.model.elastic_wrapper import ElasticPrunedLlava

LONG_ANSWER = (" ".join(
    ("The image shows a detailed outdoor scene with several objects arranged across "
     "the frame, including vehicles, people walking on the sidewalk, storefront signs, "
     "trees lining the street, and a clear sky above, all rendered in natural daylight "
     "colors with visible textures and shadows that indicate the time of day.").split()
) + " ") * 3  # ~250 tokens


def _ok(c, m):
    if not c:
        raise AssertionError("FAIL: " + m)
    print("  ok:", m)


def _img(w, h, s):
    return Image.fromarray(np.random.RandomState(s).randint(0, 255, (h, w, 3), np.uint8), "RGB")


def grad_fingerprint(m):
    tot = 0.0
    for _, p in m.model.named_parameters():
        if p.requires_grad and p.grad is not None:
            tot += float(p.grad.float().norm().item())
    return tot


def main():
    torch.manual_seed(0)
    print("[load] wrapper, lora_dropout=0 (deterministic), GC OFF initially ...")
    m = ElasticPrunedLlava(lora_rank=8, lora_dropout=0.0, dtype="bfloat16",
                           image_pad=True, gradient_checkpointing=False)
    m.train(); m._vt.eval()

    batch1 = {"images": [_img(480, 360, 1)], "questions": ["Describe the scene."],
              "answers": [LONG_ANSWER]}

    # ── (A) correctness: non-GC reference ─────────────────────────────────────
    print("[A] correctness — non-GC reference (bs=1, K=256, long answer)")
    m.zero_grad(set_to_none=True)
    torch.manual_seed(123)
    out = m.forward(batch1, K=256)
    loss_nogc = float(out["loss"].item())
    out["loss"].backward()
    fp_nogc = grad_fingerprint(m)
    print(f"     loss_noGC={loss_nogc:.6f}  grad_fp_noGC={fp_nogc:.4f}")

    # enable GC at runtime, redo SAME batch
    print("[A] enabling GC and re-running the identical batch ...")
    m._lm.config.use_cache = False
    m._lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.grad_checkpointing = True
    m.zero_grad(set_to_none=True)
    torch.manual_seed(123)
    out = m.forward(batch1, K=256)
    loss_gc = float(out["loss"].item())
    out["loss"].backward()
    fp_gc = grad_fingerprint(m)
    print(f"     loss_GC  ={loss_gc:.6f}  grad_fp_GC  ={fp_gc:.4f}")

    _ok(abs(loss_gc - loss_nogc) < 1e-3, f"GC loss matches non-GC (|Δ|={abs(loss_gc-loss_nogc):.2e})")
    rel = abs(fp_gc - fp_nogc) / max(fp_nogc, 1e-8)
    _ok(rel < 5e-3, f"GC grad fingerprint matches non-GC (rel Δ={rel:.2e}) → grads not corrupted")

    # ── (B) memory/capacity probe with GC on ──────────────────────────────────
    print("[B] memory probe WITH GC (long answers, K=576):")
    fits = []
    for bs in (4, 8, 16):
        m.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            batch = {"images": [_img(480, 360, i) for i in range(bs)],
                     "questions": ["Describe the scene."] * bs,
                     "answers": [LONG_ANSWER] * bs}
            o = m.forward(batch, K=576)
            o["loss"].backward()
            peak = torch.cuda.max_memory_allocated() / 1e9
            finite = torch.isfinite(o["loss"]).item()
            print(f"     bs={bs:<2d}  peak={peak:5.1f} GB  loss_finite={finite}")
            if finite and peak < 46:
                fits.append(bs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"     bs={bs:<2d}  OOM")
                torch.cuda.empty_cache()
            else:
                raise

    _ok(len(fits) > 0, f"at least one batch size fits with GC (fits: {fits})")
    print(f"\n  → recommended batch size for the full run: {max(fits)} (GC on)")
    print("\nGC VERIFICATION PASSED ✓")


if __name__ == "__main__":
    main()
