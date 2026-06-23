"""
Offline check of the eval path: load the trained Stage-1 elastic model and generate
answers on a few REAL GQA testdev samples. Verifies:
  1. from_checkpoint loads the trained LoRA adapter (a LoRA-B matrix becomes non-zero;
     it inits to 0, so non-zero proves the trained weights were loaded);
  2. generate_answers produces sensible short answers;
  3. it works at both high K (576) and low K (64).

Run (one GPU):
    CUDA_VISIBLE_DEVICES=0 /home/nafees/miniconda3/envs/vlm_env/bin/python v2/eval/test_generate.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from PIL import Image

from v2.model.elastic_wrapper import ElasticPrunedLlava

CKPT = "v2/outputs/stage1_full/final.pt"
GQA_Q = "data/gqa/testdev_balanced_questions.json"
GQA_IMG = "data/gqa/images/images"


def main():
    print(f"[load] {CKPT}")
    m = ElasticPrunedLlava.from_checkpoint(CKPT)

    # 1) confirm the trained adapter actually loaded (a LoRA-B norm must be > 0)
    b_norm = 0.0
    for n, p in m.model.named_parameters():
        if "lora_B" in n:
            b_norm = float(p.float().norm().item())
            if b_norm > 0:
                break
    print(f"[1] LoRA-B norm after load = {b_norm:.4f}  "
          f"({'OK — trained adapter loaded' if b_norm > 0 else 'FAIL — adapter is still zero!'})")
    assert b_norm > 0, "adapter did not load (LoRA-B is zero)"

    # 2) generate on a few real GQA samples
    g = json.load(open(GQA_Q))
    qids = list(g.keys())[:4]
    print("\n[2] real GQA testdev samples — pred vs gold:")
    print(f"  {'gold':<10} {'K=576':<14} {'K=64':<14} question")
    for qid in qids:
        rec = g[qid]
        img = Image.open(os.path.join(GQA_IMG, f"{rec['imageId']}.jpg")).convert("RGB")
        a576 = m.generate_answers([img], [rec["question"]], K=576, max_new_tokens=20)[0]
        a64 = m.generate_answers([img], [rec["question"]], K=64, max_new_tokens=20)[0]
        print(f"  {rec['answer']:<10} {a576[:13]:<14} {a64[:13]:<14} {rec['question']}")

    print("\nEVAL PATH OK ✓ (loads trained model + generates)")


if __name__ == "__main__":
    main()
