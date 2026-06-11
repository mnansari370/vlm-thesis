"""
cascade_pass.py — One static-K generation pass that ALSO records the LLM's
label-free answer confidence (for the confidence-cascade realizability test).

For each val sample at a fixed keep_tokens K: CLS top-K selection -> lm.generate()
with output_scores, then record the predicted answer, its VQA-scored correctness,
and two confidence proxies:
  - first_token_conf: top-1 softmax prob of the first generated token
  - mean_conf:        mean top-1 softmax prob over generated tokens (pre-EOS)

Run twice (a low base K and a high escalation K), then cascade_analyze.py sweeps a
confidence threshold to trace the realizable accuracy-vs-avg-K frontier.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m VQA_V2.shared.evaluation.cascade_pass \\
        --config VQA_V2/static/llava_static_clsattn_150k_10k_fullvocab_k64.yaml \\
        --keep-tokens 64 --output-path VQA_V2/outputs/cascade/base_k64.json
"""

import argparse
import json
import os
import sys
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Allow running this file directly (repo root = 3 level(s) up); `python -m` does not need this.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from VQA_V2.shared.datasets import VQACollator, build_vqav2_dataset
from VQA_V2.shared.datasets.vqav2_answers import normalize_answer
from VQA_V2.shared.evaluation.generate_and_score import load_model
from VQA_V2.shared.utils.config import load_config
from VQA_V2.shared.utils.seed import set_seed


def vqa_score(pred: str, raw: List[str]) -> float:
    pn = normalize_answer(pred)
    return min(1.0, sum(1 for a in raw if normalize_answer(a) == pn) / 3.0)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--keep-tokens", type=int, required=True)
    ap.add_argument("--output-path", required=True)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=10)
    ap.add_argument("--log-every", type=int, default=1000)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)), False)
    if args.max_samples is not None:
        cfg["dataset"]["max_val_samples"] = args.max_samples

    ds = build_vqav2_dataset(cfg, "val")
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4,
                        collate_fn=VQACollator(), pin_memory=True)

    model = load_model("static", cfg, None)
    model.eval()
    model.token_selector.keep_tokens = int(args.keep_tokens)  # override K
    lm = model._get_language_model()

    records = []
    for step, batch in enumerate(loader):
        model_inputs = model._prepare_inputs(images=batch["images"], questions=batch["questions"])
        lm_embeds, lm_mask, _ = model._build_pruned_multimodal_inputs(model_inputs)
        out = lm.generate(
            inputs_embeds=lm_embeds, attention_mask=lm_mask,
            max_new_tokens=args.max_new_tokens, do_sample=False,
            return_dict_in_generate=True, output_scores=True,
        )
        seq = out.sequences  # [1, gen_len] (inputs_embeds mode -> only new tokens)
        scores = out.scores  # tuple(len=gen_len) of [1, vocab]
        pred = model.processor.batch_decode(seq, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=True)[0].strip()

        eos = model.processor.tokenizer.eos_token_id
        first_conf = float(F.softmax(scores[0][0], dim=-1).max().item()) if len(scores) else 0.0
        tok_confs = []
        for t, sc in enumerate(scores):
            p = F.softmax(sc[0], dim=-1)
            top1 = int(p.argmax().item())
            tok_confs.append(float(p[top1].item()))
            if eos is not None and top1 == eos:
                break
        mean_conf = float(sum(tok_confs) / max(1, len(tok_confs)))

        records.append({
            "question_id": batch["question_ids"][0],
            "question": batch["questions"][0],
            "pred_answer": pred,
            "raw_answers": batch["raw_answers"][0],
            "score": vqa_score(pred, batch["raw_answers"][0]),
            "first_token_conf": first_conf,
            "mean_conf": mean_conf,
        })
        if (step + 1) % args.log_every == 0:
            acc = sum(r["score"] for r in records) / len(records)
            print(f"[cascade K={args.keep_tokens}] {step+1}/{len(loader)} acc={acc:.4f}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump({"keep_tokens": args.keep_tokens, "num": len(records), "records": records},
                  f, default=str)
    acc = sum(r["score"] for r in records) / len(records)
    print(f"[cascade K={args.keep_tokens}] DONE acc={acc:.4f} -> {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
