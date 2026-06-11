"""
Task 1 — Qualitative check of CLIP-Qcond vs CLS-Attn top-K patch selection.

Renders, for ~15 TextVQA no-OCR images (mix of aspect ratios), the padded image with
the top-K (K=64) selected patches highlighted, side by side for CLS-Attn and CLIP-Qcond.
Decides whether the "CLIP-Qcond below random" result is REAL (CLIP picks background/
border while CLS picks text/object) or an artifact (CLIP picks reasonable patches).

Output: figs/clip_vs_cls_qualitative.png (panel) + per-image overlap stats.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from GQA.static.clip_select import CLIPSpaceLlava

JSONL = "data/textvqa/llava_textvqa_val_v051_ocr.jsonl"
IMG_DIR = "data/textvqa/train_images"
K = 64


def overlay(ax, padded_img, keep_idx, title, color):
    ax.imshow(padded_img)
    grid = padded_img.size[0] // 24  # 14 px per patch on 336; padded square resized to 336
    # work on the 24x24 patch grid over the 336x336 model input
    img336 = padded_img.resize((336, 24 * 14))
    ax.imshow(img336)
    keep = set(int(i) for i in keep_idx)
    for p in range(576):
        r, c = p // 24, p % 24
        if p in keep:
            ax.add_patch(plt.Rectangle((c * 14, r * 14), 14, 14, fill=True,
                                       color=color, alpha=0.45, linewidth=0))
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def main():
    recs = [json.loads(l) for l in open(JSONL)]
    # pick a mix of aspect ratios: wide, tall, square
    picks, seen = [], {"wide": 0, "tall": 0, "square": 0}
    for r in recs:
        im = Image.open(os.path.join(IMG_DIR, r["image"]))
        w, h = im.size
        kind = "square" if 0.9 <= w / h <= 1.1 else ("wide" if w > h else "tall")
        if seen[kind] < 5:
            picks.append(r); seen[kind] += 1
        if len(picks) >= 15:
            break

    model = CLIPSpaceLlava(image_pad=True, honest=True, append_suffix=False)
    rows = []
    overlaps = []
    for r in picks:
        q = r["text"].split("\n")[0]
        img = Image.open(os.path.join(IMG_DIR, r["image"])).convert("RGB")
        sc = model.score_sample(img, q, q)
        cls_keep = sc["cls_scores"].topk(K).indices.tolist()
        clip_keep = sc["clip_scores"].topk(K).indices.tolist()
        ov = len(set(cls_keep) & set(clip_keep))
        overlaps.append(ov)
        rows.append((model._pad_to_square(img), q, cls_keep, clip_keep, ov))

    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    for i, (pimg, q, cls_keep, clip_keep, ov) in enumerate(rows):
        overlay(axes[i, 0], pimg, cls_keep, f"CLS-Attn | {q[:40]}", "lime")
        overlay(axes[i, 1], pimg, clip_keep, f"CLIP-Qcond (overlap {ov}/64)", "red")
    fig.tight_layout()
    os.makedirs("figs", exist_ok=True)
    fig.savefig("figs/clip_vs_cls_qualitative.png", dpi=110)
    print(f"[saved] figs/clip_vs_cls_qualitative.png ({n} images)")
    print(f"mean CLS/CLIP top-64 overlap: {np.mean(overlaps):.1f}/64")
    # also save a compact 4-image panel for the paper
    fig2, axes2 = plt.subplots(4, 2, figsize=(6, 12))
    for i in range(min(4, n)):
        pimg, q, cls_keep, clip_keep, ov = rows[i]
        overlay(axes2[i, 0], pimg, cls_keep, f"CLS-Attn", "lime")
        overlay(axes2[i, 1], pimg, clip_keep, f"CLIP-Qcond", "red")
    fig2.suptitle("Top-64 patch selection: CLS-Attn (saliency) vs CLIP-Qcond (question-CLIP)", fontsize=9)
    fig2.tight_layout()
    fig2.savefig("figs/clip_vs_cls_qualitative_panel4.png", dpi=130)
    print("[saved] figs/clip_vs_cls_qualitative_panel4.png")


if __name__ == "__main__":
    main()
