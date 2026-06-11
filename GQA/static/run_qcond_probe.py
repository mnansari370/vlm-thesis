"""
Question-conditioned selection probe (frozen, training-free).

For each sample: ONE scoring pass (text->visual attention at layers L), then generate
for every (selector, K) cell from that pass. Selectors: random, cls (CLS-Attn / VisionZip
dominant), qcond@L (question->visual), fusion@L (norm(cls)+norm(qcond)).

Datasets: textvqa_noocr, textvqa_ocr (VQA soft-acc), gqa (official acc).

Usage:
    python -m GQA.static.run_qcond_probe --dataset textvqa_noocr --keep_ks 64,96 --layers 2,5,8
    python -m GQA.static.run_qcond_probe --dataset gqa --keep_ks 64 --layers 2,5,8 --max_samples 12578
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from torch.utils.data import DataLoader

from GQA.static.question_cond import QuestionCondLlava
from GQA.shared.utils.logger import make_output_dir


def build_dataset(name, max_samples):
    if name.startswith("textvqa"):
        from GQA.eval_runners.run_textvqa import TextVQADataset, JSONL, IMAGE_DIR, collate
        use_ocr = name.endswith("ocr") and not name.endswith("noocr")
        ds = TextVQADataset(JSONL, IMAGE_DIR, use_ocr=use_ocr, max_samples=max_samples)
        return ds, collate, False  # append_suffix=False (prompt has instruction)
    elif name == "gqa":
        from GQA.dense.run_dense_testdev import GQATestdevDataset, collate, IMAGE_DIR
        ds = GQATestdevDataset("data/gqa/testdev_balanced_questions.json", IMAGE_DIR,
                               max_samples=max_samples)
        return ds, collate, True  # append_suffix=True (add GQA instruction)
    raise ValueError(name)


def sample_fields(name, b):
    """Extract (image, prompt_or_question, question_id, question_text, gold) for one bs=1 batch."""
    if name.startswith("textvqa"):
        return (b["image"][0], b["prompt"][0], b["question_id"][0], b["question"][0], None)
    else:  # gqa
        return (b["images"][0], b["questions"][0], b["question_ids"][0], b["questions"][0],
                b["answers"][0])


def score_cell(name, recs):
    if name.startswith("textvqa"):
        from GQA.shared.textvqa_score import score_textvqa
        r = score_textvqa(recs)
        return r["accuracy_pct"]
    else:
        from GQA.shared.official_score import is_correct
        c = sum(1 for x in recs if is_correct(x["pred_answer"], x["gold"]))
        return round(c / len(recs) * 100, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["textvqa_noocr", "textvqa_ocr", "gqa"])
    ap.add_argument("--keep_ks", default="64,96")
    ap.add_argument("--layers", default="2,5,8")
    ap.add_argument("--selectors", default="random,cls,qcond,fusion")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--output_name", default=None)
    ap.add_argument("--log_every", type=int, default=500)
    args = ap.parse_args()

    keep_ks = [int(x) for x in args.keep_ks.split(",")]
    layers = [int(x) for x in args.layers.split(",")]
    selectors = args.selectors.split(",")
    name = args.output_name or f"qcond_probe_{args.dataset}"
    out_dir = make_output_dir("outputs", name)
    print(f"[Output] {out_dir}", flush=True)
    print(f"[Config] dataset={args.dataset} K={keep_ks} layers={layers} selectors={selectors}", flush=True)

    ds, collate, append_suffix = build_dataset(args.dataset, args.max_samples)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate)
    model = QuestionCondLlava(layers=layers, image_pad=True, honest=True, append_suffix=append_suffix)

    # cell key -> list of records
    cells = defaultdict(list)

    def cell_keys():
        keys = []
        for K in keep_ks:
            for sel in selectors:
                if sel in ("qcond", "fusion"):
                    for L in layers:
                        keys.append((sel, L, K))
                else:
                    keys.append((sel, None, K))
        return keys

    t0 = time.time()
    for step, b in enumerate(loader):
        image, prompt, qid, qtext, gold = sample_fields(args.dataset, b)
        scored = model.score_sample(image, prompt)
        for (sel, L, K) in cell_keys():
            keep = model.keep_indices(scored, sel, K, layer=L)
            ans = model.generate_from_keep(scored, keep)
            rec = {"question_id": qid, "question": qtext, "pred_answer": ans}
            if gold is not None:
                rec["gold"] = gold
            cells[(sel, L, K)].append(rec)

        if (step + 1) % args.log_every == 0:
            sps = (step + 1) / (time.time() - t0)
            # quick read of the decisive cell
            msg = []
            for K in keep_ks:
                cl = score_cell(args.dataset, cells[("cls", None, K)])
                qc = max(score_cell(args.dataset, cells[("qcond", L, K)]) for L in layers) if "qcond" in selectors else 0
                msg.append(f"K{K}: cls={cl:.1f} qcond={qc:.1f}")
            print(f"  {step+1}/{len(ds)}  {sps:.1f} samp/s  " + "  ".join(msg)
                  + f"  ETA={(len(ds)-step-1)/max(sps,1e-6)/60:.1f}min", flush=True)

    # ── score all cells ───────────────────────────────────────────────────────
    results = {}
    for (sel, L, K), recs in cells.items():
        key = f"{sel}" + (f"_L{L}" if L is not None else "") + f"_K{K}"
        results[key] = {"selector": sel, "layer": L, "keep_k": K,
                        "acc": score_cell(args.dataset, recs), "n": len(recs)}

    print(f"\n{'='*60}\n  PROBE RESULTS — {args.dataset}\n{'='*60}")
    for K in keep_ks:
        print(f"\n  K={K}:")
        rnd = results.get(f"random_K{K}", {}).get("acc")
        cls = results.get(f"cls_K{K}", {}).get("acc")
        for key in sorted(results):
            if results[key]["keep_k"] != K:
                continue
            r = results[key]
            extra = ""
            if cls is not None and r["selector"] in ("qcond", "fusion"):
                extra = f"  (vs cls {r['acc']-cls:+.2f})"
            print(f"    {key:<18} acc={r['acc']:.2f}%{extra}")
        if cls is not None and rnd is not None:
            print(f"    [gaps] cls-random={cls-rnd:+.2f}")

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out_dir, "cells.json"), "w") as f:
        json.dump({f"{s}_{L}_{K}": recs for (s, L, K), recs in cells.items()}, f)
    print(f"\n[Done] {out_dir}  elapsed={(time.time()-t0)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
