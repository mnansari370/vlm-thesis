"""
Step 0b of Option A — the high-res K-curve ("is there room?").

Step 0a proved high-res READS better: LLaVA-1.6 full (~2302 tok) = 61.4% vs LLaVA-1.5
(576) = 42.2% on TextVQA-no-OCR. Now: as we PRUNE the AnyRes visual tokens, does that
61% collapse back toward the ~42% "can't-read" floor? A steep collapse = the high-res
tokens are load-bearing and not redundant = REAL ROOM a question-conditioned selector
could exploit. A flat curve = the tokens are redundant = no room (strong negative).

Selectors (prune BEFORE the LLM, so all 32 layers see K tokens — our v1/v2 methodology):
  - full     : keep all ~2302 tokens (the ceiling; also the correctness check vs stock)
  - base     : keep ONLY the first 576 = the global 336px view (≈ what LLaVA-1.5 sees).
               Needs NO scoring — bulletproof. If this ≈ 42%, the high-res grid tokens
               carry the entire +19pp, i.e. they are demonstrably load-bearing.
  - attn     : keep top-K by CLIP CLS->patch attention (VisionZip-style static selector,
               the realistic question-BLIND baseline; the v2 selector must beat THIS).
  - uniform  : keep every (N/K)-th token (content-blind floor).

AnyRes pruning is done on the FINAL packed sequence. Scores are aligned to that sequence
by passing per-patch CLS-attention THROUGH HF's own pack_image_features (each scalar
broadcast over the channel dim), so we never reimplement the unpad/permute geometry.
newline tokens are scored +inf (always kept; they are few and structural).

CORRECTNESS: --self-test runs selector="full" through the manual splice and asserts the
generated answer matches the stock model.generate() answer. If keep-all reproduces stock,
the splice is correct and pruning is just "keep top-K instead of all".

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m v2.eval.evaluate_textvqa_highres_kcurve --self-test
    CUDA_VISIBLE_DEVICES=0 python -m v2.eval.evaluate_textvqa_highres_kcurve \
        --selectors base,attn --k-values 1152,576,256,128,64 --max-samples 300
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from src.metrics.textvqa_score import score_textvqa

MODEL = "llava-hf/llava-v1.6-vicuna-7b-hf"
TEXTVQA_ANN = "data/textvqa/TextVQA_0.5.1_val.json"
TEXTVQA_IMG = "data/textvqa/train_images"
TEXTVQA_INSTR = "Answer the question using a single word or phrase."
LLAVA15_NOOCR = 42.23   # LLaVA-1.5 full-576 no-OCR (the "can't-read" floor)
HIRES_FULL = 61.40      # LLaVA-1.6 full no-OCR (measured in 0a; the ceiling)
NEWLINE_KEEP = 1e4      # score for newline rows -> always kept (few, structural)


class HighResPruner:
    def __init__(self):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[hires-prune] loading {MODEL} ...", flush=True)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="sdpa",
        ).to(self.dev).eval()
        self.proc = LlavaNextProcessor.from_pretrained(MODEL)
        self.eos = self.proc.tokenizer.eos_token_id
        self.img_tok = int(self.model.config.image_token_index)
        self.vfl = int(self.model.config.vision_feature_layer)        # -2
        self.embed = self.model.get_input_embeddings()
        self.textattn_layer = 3   # early LLM layer for question->visual attention (FastV uses ~2-3)

    @torch.no_grad()
    def _encode(self, image, question):
        """Returns prefix_emb, suffix_emb (text around the image), packed visual feats
        [N,D], aligned packed scores [N], and a 'is-real-token' count."""
        q = question.strip() + " " + TEXTVQA_INSTR
        conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        prompt = self.proc.apply_chat_template(conv, add_generation_prompt=True)
        inp = self.proc(images=image, text=prompt, return_tensors="pt").to(self.dev)
        ids = inp["input_ids"][0]
        img_pos = (ids == self.img_tok).nonzero(as_tuple=True)[0]
        i0, i1 = int(img_pos[0]), int(img_pos[-1]) + 1
        prefix_emb = self.embed(ids[:i0])
        suffix_emb = self.embed(ids[i1:])

        # 1) packed visual features (exactly what stock would scatter in)
        feats_list = self.model.get_image_features(
            inp["pixel_values"], inp["image_sizes"], self.vfl, "default")   # [ [num_tiles,576,D] ]
        packed_feats, _ = self.model.pack_image_features(
            feats_list, inp["image_sizes"], "default", self.model.image_newline)   # [N,D]
        N, D = packed_feats.shape

        # 2) CLS->patch attention per tile, aligned to the packed sequence via pack_image_features
        pv = inp["pixel_values"][0]                                      # [num_tiles,3,336,336]
        vout = self.model.vision_tower(pv, output_attentions=True)
        cls_attn = vout.attentions[self.vfl][:, :, 0, 1:].mean(1).float()   # [num_tiles,576]
        score_feats = cls_attn.unsqueeze(-1).expand(-1, -1, D).contiguous()  # broadcast scalar->D
        newline_const = torch.full((D,), NEWLINE_KEEP, device=self.dev, dtype=score_feats.dtype)
        packed_scores_feats, _ = self.model.pack_image_features(
            [score_feats], inp["image_sizes"], "default", newline_const)   # [N,D]
        packed_scores = packed_scores_feats[:, 0]                          # [N], newline rows == NEWLINE_KEEP
        assert packed_scores.shape[0] == N, f"score/feat misalign {packed_scores.shape[0]} vs {N}"
        return prefix_emb, suffix_emb, packed_feats, packed_scores, N

    @torch.no_grad()
    def generate(self, image, question, selector="full", K=None, max_new_tokens=20):
        prefix_emb, suffix_emb, feats, scores, N = self._encode(image, question)

        if selector == "full":
            keep = torch.arange(N, device=self.dev)
        elif selector == "base":
            keep = torch.arange(min(576, N), device=self.dev)             # global 336px view only
        elif selector == "uniform":
            k = min(K, N)
            keep = torch.linspace(0, N - 1, k, device=self.dev).round().long().unique()
        elif selector == "attn":
            k = min(K, N)
            keep = scores.topk(k).indices.sort().values                   # keep spatial order
        elif selector == "textattn":
            # QUESTION-CONDITIONED (FastV-style): one scoring forward with ALL visual tokens,
            # read how much the QUESTION tokens attend to each visual token at an early LLM
            # layer, keep the top-K. The strong question-aware signal v1's frozen-CLIP selector
            # lacked. (Scoring uses a full forward here; in the real method it'd be a cheap head.)
            full = torch.cat([prefix_emb, feats.to(prefix_emb.dtype), suffix_emb], 0).unsqueeze(0)
            lm = self.model.language_model.model      # LlamaModel (sdpa falls back to eager for attn)
            o = lm(inputs_embeds=full, output_attentions=True, use_cache=False)
            A = o.attentions[self.textattn_layer][0].float().mean(0)       # [S,S] mean over heads
            v0 = prefix_emb.shape[0]; v1 = v0 + N
            qscore = A[v1:, v0:v1].mean(0)                                 # [N] question->visual attn
            del o, A
            k = min(K, N)
            keep = qscore.topk(k).indices.sort().values
        else:
            raise ValueError(selector)

        vis = feats[keep].to(prefix_emb.dtype)
        seq = torch.cat([prefix_emb, vis, suffix_emb], dim=0).unsqueeze(0)
        mask = torch.ones(1, seq.shape[1], dtype=torch.long, device=self.dev)
        out = self.model.language_model.generate(
            inputs_embeds=seq, attention_mask=mask, max_new_tokens=max_new_tokens,
            do_sample=False, num_beams=1, pad_token_id=self.eos, use_cache=True)
        return self.proc.tokenizer.decode(out[0], skip_special_tokens=True).strip(), int(keep.numel())

    @torch.no_grad()
    def generate_stock(self, image, question, max_new_tokens=20):
        """Unmodified HF path — the ground truth for the self-test."""
        q = question.strip() + " " + TEXTVQA_INSTR
        conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        prompt = self.proc.apply_chat_template(conv, add_generation_prompt=True)
        inp = self.proc(images=image, text=prompt, return_tensors="pt").to(self.dev)
        out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                                  num_beams=1, pad_token_id=self.eos, use_cache=True)
        return self.proc.tokenizer.decode(out[0, inp["input_ids"].shape[1]:],
                                          skip_special_tokens=True).strip()


def self_test(pruner, data):
    print("\n=== SELF-TEST: selector='full' (keep-all splice) must match stock generate ===")
    ok = 0
    for rec in data[:6]:
        ip = os.path.join(TEXTVQA_IMG, f"{rec['image_id']}.jpg")
        img = Image.open(ip).convert("RGB")
        a_stock = pruner.generate_stock(img, rec["question"])
        a_full, nkeep = pruner.generate(img, rec["question"], selector="full")
        match = a_stock == a_full
        ok += int(match)
        print(f"  {'OK ' if match else 'MISMATCH'} keep={nkeep:4d} | stock='{a_stock}' | full='{a_full}'")
    print(f"  -> {ok}/6 match" + ("  ✓ splice correct\n" if ok == 6 else "  ✗ SPLICE BUG\n"))
    return ok == 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selectors", default="base,attn")
    ap.add_argument("--k-values", default="1152,576,256,128,64")
    ap.add_argument("--max-samples", type=int, default=300)
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--output", default="v2/outputs/eval_textvqa_highres_kcurve.json")
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()

    pruner = HighResPruner()
    data = json.load(open(TEXTVQA_ANN))["data"][: args.max_samples]

    if args.self_test:
        assert self_test(pruner, data), "self-test failed — fix splice before trusting the curve"
        if args.selectors == "base,attn" and args.max_samples == 300:
            return  # self-test-only invocation

    selectors = args.selectors.split(",")
    Ks = [int(x) for x in args.k_values.split(",")]
    results = {"model": MODEL, "ref": {"llava15_576": LLAVA15_NOOCR, "hires_full": HIRES_FULL}, "runs": {}}

    def run(tag, selector, K):
        t0 = time.time()
        preds, keeps = [], []
        for i, rec in enumerate(data):
            ip = os.path.join(TEXTVQA_IMG, f"{rec['image_id']}.jpg")
            if not os.path.exists(ip):
                continue
            img = Image.open(ip).convert("RGB")
            pred, nkeep = pruner.generate(img, rec["question"], selector=selector, K=K,
                                          max_new_tokens=args.max_new_tokens)
            preds.append({"question_id": rec["image_id"], "question": rec["question"].strip(),
                          "pred_answer": pred})
            keeps.append(nkeep)
            if (i + 1) % args.log_every == 0:
                r = score_textvqa(preds)
                print(f"    {tag} {i+1}/{len(data)} soft-acc={r['accuracy_pct']:.1f}% "
                      f"(~{sum(keeps)//len(keeps)} tok)", flush=True)
        r = score_textvqa(preds)
        avg_keep = sum(keeps) // max(len(keeps), 1)
        results["runs"][tag] = {"selector": selector, "K": K, "avg_tokens_kept": avg_keep,
                                "soft_acc": r["accuracy_pct"], "n": r["n_evaluated"],
                                "minutes": round((time.time() - t0) / 60, 1)}
        print(f"  [{tag}] soft-acc={r['accuracy_pct']:.2f}%  (~{avg_keep} tok, n={r['n_evaluated']})", flush=True)

    # full ceiling (also a sanity re-confirm of 0a)
    run("full", "full", None)
    # base-only (the bulletproof load-bearing test)
    if "base" in selectors:
        run("base_576", "base", None)
    # selector ladders
    for sel in [s for s in selectors if s in ("attn", "uniform", "textattn")]:
        for K in Ks:
            run(f"{sel}_K{K}", sel, K)

    # summary table
    print("\n" + "=" * 64)
    print(f"  TextVQA no-OCR — LLaVA-1.6 high-res — K-curve")
    print(f"  refs: LLaVA-1.5(576)={LLAVA15_NOOCR}%  |  hi-res full={HIRES_FULL}%")
    print("-" * 64)
    for tag, r in results["runs"].items():
        print(f"  {tag:14s} tok~{r['avg_tokens_kept']:>5d}  soft-acc={r['soft_acc']:6.2f}%")
    print("=" * 64)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
