"""
Rigor check before declaring the question-conditioned SELECTION axis dead:
does ANY LLM layer's question->visual attention beat blind CLS-attn at K=128?

textattn(layer=3) was WORSE than blind CLS-attn (42.0 vs 44.2 at K=128). But a single
scoring forward contains EVERY layer's attention, so we can sweep many layers for the
price of one forward per sample (+ a cheap K=128 generate per layer). If no layer beats
44.2%, the training-free question-conditioned signal is genuinely weak (frozen features
don't localize — the v1 lesson); if some layer wins, the selection axis is still alive.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.textvqa.textattn_layer_sweep --max-samples 300
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from PIL import Image

from src.evaluation.textvqa.evaluate_textvqa_highres_kcurve import HighResPruner, TEXTVQA_ANN, TEXTVQA_IMG, TEXTVQA_INSTR
from src.metrics.textvqa_score import score_textvqa

BLIND_REF = {64: 34.80, 128: 44.20, 256: 54.67}   # CLS-attn baseline to beat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=300)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--layers", default="2,4,6,8,12,16,20")
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--output", default="results/thesis_main/highres/eval_textattn_layer_sweep.json")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    K = args.K
    p = HighResPruner()
    data = json.load(open(TEXTVQA_ANN))["data"][: args.max_samples]

    preds = {L: [] for L in layers}
    t0 = time.time()
    for idx, rec in enumerate(data):
        ip = os.path.join(TEXTVQA_IMG, f"{rec['image_id']}.jpg")
        if not os.path.exists(ip):
            continue
        img = Image.open(ip).convert("RGB")
        with torch.no_grad():
            prefix, suffix, feats, _, N = p._encode(img, rec["question"])
            full = torch.cat([prefix, feats.to(prefix.dtype), suffix], 0).unsqueeze(0)
            o = p.model.language_model.model(inputs_embeds=full, output_attentions=True, use_cache=False)
            v0 = prefix.shape[0]; v1 = v0 + N
            for L in layers:
                qscore = o.attentions[L][0].float().mean(0)[v1:, v0:v1].mean(0)   # [N]
                keep = qscore.topk(min(K, N)).indices.sort().values
                vis = feats[keep].to(prefix.dtype)
                seq = torch.cat([prefix, vis, suffix], 0).unsqueeze(0)
                mask = torch.ones(1, seq.shape[1], dtype=torch.long, device=p.dev)
                out = p.model.language_model.generate(
                    inputs_embeds=seq, attention_mask=mask, max_new_tokens=args.max_new_tokens,
                    do_sample=False, num_beams=1, pad_token_id=p.eos, use_cache=True)
                pred = p.proc.tokenizer.decode(out[0], skip_special_tokens=True).strip()
                preds[L].append({"question_id": rec["image_id"], "question": rec["question"].strip(),
                                 "pred_answer": pred})
            del o
        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(data)}  ({round((time.time()-t0)/60,1)} min)", flush=True)

    res = {L: score_textvqa(preds[L])["accuracy_pct"] for L in layers}
    blind = BLIND_REF.get(K, None)
    print("\n" + "=" * 56)
    print(f"  question->visual attn by LAYER, K={K}  (blind CLS-attn={blind}%)")
    print("-" * 56)
    best_L, best = None, -1
    for L in layers:
        win = "" if blind is None else ("  <-- beats blind" if res[L] > blind else "")
        print(f"   layer {L:>2}:  {res[L]:6.2f}%{win}")
        if res[L] > best:
            best, best_L = res[L], L
    print("-" * 56)
    if blind is not None:
        verdict = (f"layer {best_L} BEATS blind ({best:.2f} > {blind}) -> selection axis ALIVE"
                   if best > blind else
                   f"NO layer beats blind (best {best:.2f} @ L{best_L} < {blind}) -> "
                   f"training-free QC selection is weak")
        print(f"  {verdict}")
    print("=" * 56)
    json.dump({"K": K, "blind_ref": blind, "by_layer": res, "best_layer": best_L},
              open(args.output, "w"), indent=2)
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
