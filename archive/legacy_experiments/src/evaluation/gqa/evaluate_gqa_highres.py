"""
LLaVA-1.6 (high-res) GQA pruning curve — the high-res half of the GQA resolution contrast.
Mirrors src/evaluation/gqa/evaluate_gqa.py (LLaVA-1.5) EXACTLY for comparability:
  - same GQA testdev_balanced, same first-N qids (deterministic)
  - same STRICT exact-match scorer (is_correct) + short-answer extract
  - same instruction "Answer the question using a single word or phrase."
CRITICAL: HighResPruner._encode appends the instruction internally, so we pass the RAW
question (NOT pre-appended) to avoid doubling it — verified by printing prompt[0].

GQA is reasoning (not reading), so it tests whether the high-res "budget binds" effect is
reading-specific or general. Blind CLS-attn selector + full, to match the 1.5 base curve.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.gqa.evaluate_gqa_highres --k-values 64,128,256,576 --max-samples 300
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from PIL import Image
from src.evaluation.textvqa.evaluate_textvqa_highres_kcurve import HighResPruner
from src.metrics.official_score import is_correct
from src.metrics.metrics import extract_short_answer

GQA_Q = "data/gqa/testdev_balanced_questions.json"
GQA_IMG = "data/gqa/images/images"
V15_BASE = {576: 58.67, 128: 54.33, 64: 50.0, 32: 47.0}   # our LLaVA-1.5 frozen base (strict), for contrast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-values", default="64,128,256,576")
    ap.add_argument("--max-samples", type=int, default=300)
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--selector", default="attn", choices=["attn", "textattn", "full"])
    ap.add_argument("--output", default="results/thesis_main/highres/eval_gqa_highres.json")
    args = ap.parse_args()

    p = HighResPruner()
    p.textattn_layer = 16
    g = json.load(open(GQA_Q))
    qids = list(g.keys())[: args.max_samples]
    Ks = [int(x) for x in args.k_values.split(",")]
    levels = Ks + ["full"]

    out = {"model": "llava-1.6", "selector": args.selector, "per_k": {}, "v15_base_strict": V15_BASE}
    printed = False
    for K in levels:
        t0 = time.time()
        ns = ne = n = 0
        for qid in qids:
            rec = g[qid]
            ip = os.path.join(GQA_IMG, f"{rec['imageId']}.jpg")
            if not os.path.exists(ip):
                continue
            img = Image.open(ip).convert("RGB")
            raw_q = rec["question"].strip()          # RAW — pruner appends the instruction once
            sel = "full" if K == "full" else args.selector
            pred, _ = p.generate(img, raw_q, selector=sel, K=(None if K == "full" else K),
                                 max_new_tokens=args.max_new_tokens)
            gold = rec.get("answer", "")
            ext = extract_short_answer(pred, rec["question"])
            n += 1; ns += int(is_correct(pred, gold)); ne += int(is_correct(ext, gold))
        strict = round(100 * ns / max(n, 1), 2)
        extr = round(100 * ne / max(n, 1), 2)
        out["per_k"][str(K)] = {"n": n, "strict_acc": strict, "extract_acc": extr,
                                "minutes": round((time.time() - t0) / 60, 1)}
        ref = "" if K == "full" else f"  [1.5 base K={K}: {V15_BASE.get(K, '-')}]"
        print(f"  [GQA-1.6 K={str(K):>5}] strict={strict}%  extract={extr}%  (n={n}){ref}", flush=True)

    json.dump(out, open(args.output, "w"), indent=2)
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
