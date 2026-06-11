"""
Visualize GQA region-supervision targets.

For a few sample questions, overlays:
  - the ground-truth object bounding box(es) from the scene graph
  - the 24x24 patch grid
  - the patches marked as "relevant" (the supervision target)

This is the correctness gate BEFORE building the training loss.
If the highlighted patches sit on the answer object, the target builder is correct.

Outputs PNGs to: outputs/grounding_viz/
"""
import json
import os
import random

import numpy as np
from PIL import Image, ImageDraw

# ----------------------------------------------------------------------------
# CONFIG — matches your model's vision setup
# ----------------------------------------------------------------------------
IMAGE_DIR   = "data/gqa/images/images"   # nested folder confirmed earlier
GRID        = 24                          # 24x24 = 576 patches (CLIP ViT-L/14 @ 336)
OUT_DIR     = "outputs/grounding_viz"
N_SAMPLES   = 8
SEED        = 7

# ----------------------------------------------------------------------------
# CORE: box -> 24x24 patch target. This exact logic will go into the dataset.
# ----------------------------------------------------------------------------
def build_patch_target(objects, img_w, img_h, grid=GRID, soft=False):
    """
    objects: list of dicts each with x, y, w, h in ORIGINAL image pixels.
    img_w, img_h: original image size from the scene graph.

    Returns a [grid*grid] float array in {0,1} (hard) marking patches that
    overlap any object box. The model uses 24x24 patches regardless of the
    original aspect ratio because LLaVA resizes to a square 336x336, so we
    map boxes in NORMALISED coordinates (fraction of width/height) onto the
    square grid — this is consistent with how the patches are produced.
    """
    target = np.zeros((grid, grid), dtype=np.float32)
    if not img_w or not img_h:
        return target.reshape(-1)

    for o in objects:
        # normalised coords [0,1]
        nx0 = o["x"] / img_w
        ny0 = o["y"] / img_h
        nx1 = (o["x"] + o["w"]) / img_w
        ny1 = (o["y"] + o["h"]) / img_h

        # to grid indices (inclusive), clamped
        gx0 = max(0, min(grid - 1, int(np.floor(nx0 * grid))))
        gy0 = max(0, min(grid - 1, int(np.floor(ny0 * grid))))
        gx1 = max(0, min(grid - 1, int(np.ceil (nx1 * grid)) - 1))
        gy1 = max(0, min(grid - 1, int(np.ceil (ny1 * grid)) - 1))
        gx1 = max(gx1, gx0)
        gy1 = max(gy1, gy0)

        target[gy0:gy1 + 1, gx0:gx1 + 1] = 1.0

    return target.reshape(-1)


def referenced_obj_ids(sample):
    ann = sample.get("annotations", {}) or {}
    ids = set()
    for field in ("question", "answer", "fullAnswer"):
        for v in (ann.get(field, {}) or {}).values():
            ids.add(str(v))
    return ids


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading val questions + scene graphs...", flush=True)
    val_q  = json.load(open("data/gqa/val_balanced_questions.json"))
    val_sg = json.load(open("data/gqa/val_sceneGraphs.json"))

    qids = list(val_q.keys())
    random.Random(SEED).shuffle(qids)

    made = 0
    for qid in qids:
        if made >= N_SAMPLES:
            break
        s = val_q[qid]
        img_id = str(s["imageId"])
        g = val_sg.get(img_id)
        if g is None:
            continue
        refs = referenced_obj_ids(s)
        objs = g.get("objects", {})
        resolved = [objs[o] for o in refs if o in objs]
        if not resolved:
            continue

        img_path = os.path.join(IMAGE_DIR, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            continue

        W, H = g["width"], g["height"]
        target = build_patch_target(resolved, W, H).reshape(GRID, GRID)
        n_patches = int(target.sum())

        # ---- render on a 480-wide canvas ----
        disp_w = 480
        scale  = disp_w / W
        disp_h = int(H * scale)
        img = Image.open(img_path).convert("RGB").resize((disp_w, disp_h))
        draw = ImageDraw.Draw(img, "RGBA")

        # shade relevant patches (green)
        cell_w = disp_w / GRID
        cell_h = disp_h / GRID
        for gy in range(GRID):
            for gx in range(GRID):
                if target[gy, gx] > 0:
                    x0 = gx * cell_w; y0 = gy * cell_h
                    draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h],
                                   fill=(0, 255, 0, 70))

        # grid lines (faint)
        for i in range(GRID + 1):
            draw.line([(i * cell_w, 0), (i * cell_w, disp_h)], fill=(255, 255, 255, 40))
            draw.line([(0, i * cell_h), (disp_w, i * cell_h)], fill=(255, 255, 255, 40))

        # ground-truth boxes (red outline)
        for o in resolved:
            x0 = o["x"] * scale; y0 = o["y"] * scale
            x1 = (o["x"] + o["w"]) * scale; y1 = (o["y"] + o["h"]) * scale
            draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0, 255), width=3)

        names = ", ".join(o["name"] for o in resolved)
        fname = f"{made:02d}_{img_id}.png"
        img.save(os.path.join(OUT_DIR, fname))
        print(f"[{made}] {fname}", flush=True)
        print(f"     Q: {s['question']}", flush=True)
        print(f"     A: {s['answer']}   obj(s): {names}", flush=True)
        print(f"     target patches: {n_patches}/576  ({n_patches/576*100:.1f}%)", flush=True)
        made += 1

    print(f"\nWrote {made} images to {OUT_DIR}/", flush=True)
    print("Open them and check: do the GREEN patches sit on the RED box / answer object?", flush=True)


if __name__ == "__main__":
    main()
