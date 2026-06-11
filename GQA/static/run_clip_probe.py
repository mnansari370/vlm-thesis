"""
CLIP-space question-conditioned selection probe (frozen, training-free).

Per sample: ONE scoring pass (CLIP CLS-attn + CLIP-space question->patch), then
generate for every (selector, K) cell. Selectors: random (per-sample seed), cls
(CLS-Attn), clip (CLIP-space qcond), fusion@w (norm(cls)+w*norm(clip), w∈{.25,.5,.75}).

Datasets: textvqa_noocr, textvqa_ocr (VQA soft-acc), gqa (official acc).
CLIP text scoring uses the QUESTION text (not the OCR-augmented prompt).

Usage:
  python -m GQA.static.run_clip_probe --dataset textvqa_noocr --keep_ks 64,96
  python -m GQA.static.run_clip_probe --dataset gqa --keep_ks 64 --max_samples 4000
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from torch.utils.data import DataLoader

from GQA.static.clip_select import CLIPSpaceLlava
from GQA.shared.utils.logger import make_output_dir

FUSION_MIXES = [0.25, 0.5, 0.75]


def build_dataset(name, max_samples):
    if name.startswith("textvqa"):
        from GQA.eval_runners.run_textvqa import TextVQADataset, JSONL, IMAGE_DIR, collate
        use_ocr = name.endswith("_ocr")
        ds = TextVQADataset(JSONL, IMAGE_DIR, use_ocr=use_ocr, max_samples=max_samples)
        return ds, collate, False
    elif name == "gqa":
        from GQA.dense.run_dense_testdev import GQATestdevDataset, collate, IMAGE_DIR
        ds = GQATestdevDataset("data/gqa/testdev_balanced_questions.json", IMAGE_DIR,
                               max_samples=max_samples)
        return ds, collate, True
    raise ValueError(name)


def sample_fields(name, b):
    if name.startswith("textvqa"):
        return (b["image"][0], b["prompt"][0], b["question_id"][0], b["question"][0], None)
    return (b["images"][0], b["questions"][0], b["question_ids"][0], b["questions"][0], b["answers"][0])


def score_cell(name, recs):
    if name.startswith("textvqa"):
        from GQA.shared.textvqa_score import score_textvqa
        return score_textvqa(recs)["accuracy_pct"]
    from GQA.shared.official_score import is_correct
    return round(sum(1 for x in recs if is_correct(x["pred_answer"], x["gold"])) / len(recs) * 100, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["textvqa_noocr", "textvqa_ocr", "gqa"])
    ap.add_argument("--keep_ks", default="64,96")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--output_name", default=None)
    ap.add_argument("--log_every", type=int, default=500)
    args = ap.parse_args()

    keep_ks = [int(x) for x in args.keep_ks.split(",")]
    out_dir = make_output_dir("outputs", args.output_name or f"clip_probe_{args.dataset}")
    print(f"[Output] {out_dir}\n[Config] dataset={args.dataset} K={keep_ks}", flush=True)

    ds, collate, append_suffix = build_dataset(args.dataset, args.max_samples)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate)
    model = CLIPSpaceLlava(image_pad=True, honest=True, append_suffix=append_suffix)

    cells = defaultdict(list)

    def cell_keys():
        keys = []
        for K in keep_ks:
            keys += [("random", None, K), ("cls", None, K), ("clip", None, K)]
            keys += [("fusion", w, K) for w in FUSION_MIXES]
        return keys

    t0 = time.time()
    for step, b in enumerate(loader):
        image, prompt, qid, qtext, gold = sample_fields(args.dataset, b)
        scored = model.score_sample(image, prompt, qtext)
        for (sel, w, K) in cell_keys():
            keep = model.keep_indices(scored, sel, K, sample_offset=step, mix=(w or 0.5))
            ans = model.generate_from_keep(scored, keep)
            rec = {"question_id": qid, "question": qtext, "pred_answer": ans}
            if gold is not None:
                rec["gold"] = gold
            cells[(sel, w, K)].append(rec)
        if (step + 1) % args.log_every == 0:
            sps = (step + 1) / (time.time() - t0)
            msg = []
            for K in keep_ks:
                cl = score_cell(args.dataset, cells[("cls", None, K)])
                cp = score_cell(args.dataset, cells[("clip", None, K)])
                msg.append(f"K{K}: cls={cl:.1f} clip={cp:.1f}")
            print(f"  {step+1}/{len(ds)} {sps:.1f} samp/s  " + "  ".join(msg)
                  + f"  ETA={(len(ds)-step-1)/max(sps,1e-6)/60:.1f}min", flush=True)

    results = {}
    for (sel, w, K), recs in cells.items():
        key = f"{sel}" + (f"_w{w}" if w is not None else "") + f"_K{K}"
        results[key] = {"selector": sel, "mix": w, "keep_k": K,
                        "acc": score_cell(args.dataset, recs), "n": len(recs)}

    print(f"\n{'='*60}\n  CLIP-PROBE RESULTS — {args.dataset}\n{'='*60}")
    for K in keep_ks:
        cls = results.get(f"cls_K{K}", {}).get("acc")
        rnd = results.get(f"random_K{K}", {}).get("acc")
        print(f"\n  K={K}:")
        for key in sorted(results):
            if results[key]["keep_k"] != K:
                continue
            r = results[key]
            extra = f"  (vs cls {r['acc']-cls:+.2f})" if cls is not None and r["selector"] in ("clip", "fusion") else ""
            print(f"    {key:<16} {r['acc']:.2f}%{extra}")
        if cls is not None and rnd is not None:
            print(f"    [cls-random = {cls-rnd:+.2f}]")

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Done] {out_dir} elapsed={(time.time()-t0)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
