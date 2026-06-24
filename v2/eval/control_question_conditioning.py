"""
The capstone control: is mid-layer attention selection genuinely QUESTION-CONDITIONED,
or just a better question-BLIND saliency than CLS?

Mid-layer (L16) question->visual selection beats blind CLS-attn by +10.5pp at K=128.
But CLS-attn is question-blind, so maybe L16 is simply a better *blind* saliency. To
isolate question-conditioning: pick the K tokens using a MISMATCHED question's L16
attention, then answer the REAL question with those tokens.
  - real-Q selection  : tokens chosen by the real question  (= the QC result, ~54.7)
  - mismatched-Q sel. : tokens chosen by ANOTHER sample's question, answer real Q (control)
  - blind CLS-attn    : 44.2
If mismatched << real (toward blind), selection genuinely depends on the question.
If mismatched ~ real, the win is just better blind saliency (NOT question-conditioned).

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m v2.eval.control_question_conditioning --max-samples 300
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from PIL import Image

from v2.eval.evaluate_textvqa_highres_kcurve import HighResPruner, TEXTVQA_ANN, TEXTVQA_IMG
from src.metrics.textvqa_score import score_textvqa

LAYER = 16
BLIND = {64: 34.80, 128: 44.20, 256: 54.67}
QC = {64: 52.73, 128: 54.67, 256: 55.30}   # real-Q L16 from the sweep, for reference


def select_keep(p, prefix, feats, suffix, N, K):
    """top-K visual indices by L16 question->visual attention, for the given suffix (question)."""
    full = torch.cat([prefix, feats.to(prefix.dtype), suffix], 0).unsqueeze(0)
    o = p.model.language_model.model(inputs_embeds=full, output_attentions=True, use_cache=False)
    v0 = prefix.shape[0]; v1 = v0 + N
    qscore = o.attentions[LAYER][0].float().mean(0)[v1:, v0:v1].mean(0)
    del o
    return qscore.topk(min(K, N)).indices.sort().values


def gen(p, prefix, feats, keep, suffix, max_new_tokens=20):
    seq = torch.cat([prefix, feats[keep].to(prefix.dtype), suffix], 0).unsqueeze(0)
    mask = torch.ones(1, seq.shape[1], dtype=torch.long, device=p.dev)
    out = p.model.language_model.generate(inputs_embeds=seq, attention_mask=mask,
        max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, pad_token_id=p.eos, use_cache=True)
    return p.proc.tokenizer.decode(out[0], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=300)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--output", default="v2/outputs/eval_control_qcond.json")
    args = ap.parse_args()
    K = args.K
    p = HighResPruner()
    data = json.load(open(TEXTVQA_ANN))["data"][: args.max_samples]

    preds_real, preds_ctrl = [], []
    with torch.no_grad():
        for i, rec in enumerate(data):
            ip = os.path.join(TEXTVQA_IMG, f"{rec['image_id']}.jpg")
            if not os.path.exists(ip):
                continue
            img = Image.open(ip).convert("RGB")
            real_q = rec["question"]
            ctrl_q = data[(i + 1) % len(data)]["question"]      # a DIFFERENT sample's question
            # encode real (for feats + real suffix) and ctrl (for ctrl suffix); feats/prefix identical
            prefix, suffix_real, feats, _, N = p._encode(img, real_q)
            _, suffix_ctrl, _, _, _ = p._encode(img, ctrl_q)

            keep_real = select_keep(p, prefix, feats, suffix_real, N, K)
            keep_ctrl = select_keep(p, prefix, feats, suffix_ctrl, N, K)   # tokens for WRONG question

            a_real = gen(p, prefix, feats, keep_real, suffix_real)
            a_ctrl = gen(p, prefix, feats, keep_ctrl, suffix_real)         # answer REAL q, wrong-q tokens
            preds_real.append({"question_id": rec["image_id"], "question": real_q.strip(), "pred_answer": a_real})
            preds_ctrl.append({"question_id": rec["image_id"], "question": real_q.strip(), "pred_answer": a_ctrl})
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(data)}", flush=True)

    r_real = score_textvqa(preds_real)["accuracy_pct"]
    r_ctrl = score_textvqa(preds_ctrl)["accuracy_pct"]
    blind = BLIND[K]
    # fraction of the QC-over-blind gain that is attributable to the QUESTION
    gain_real = r_real - blind
    gain_ctrl = r_ctrl - blind
    q_fraction = round(100 * (gain_real - gain_ctrl) / gain_real, 1) if gain_real > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"  QUESTION-CONDITIONING CONTROL  (L{LAYER}, K={K}, n={len(preds_real)})")
    print("-" * 60)
    print(f"  blind CLS-attn (question-blind)     : {blind:.2f}%")
    print(f"  L{LAYER} sel by MISMATCHED question    : {r_ctrl:.2f}%   (control)")
    print(f"  L{LAYER} sel by REAL question          : {r_real:.2f}%   (QC)")
    print("-" * 60)
    print(f"  gain over blind: real {gain_real:+.2f} | mismatched {gain_ctrl:+.2f}")
    print(f"  => {q_fraction:.0f}% of the selection gain is DUE TO THE QUESTION")
    verdict = ("genuinely QUESTION-CONDITIONED" if r_real - r_ctrl >= 3
               else "mostly better BLIND saliency (NOT question-conditioned)")
    print(f"  VERDICT: {verdict}")
    print("=" * 60)
    json.dump({"K": K, "layer": LAYER, "blind": blind, "real_q": r_real, "mismatched_q": r_ctrl,
               "question_fraction_pct": q_fraction}, open(args.output, "w"), indent=2)
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
