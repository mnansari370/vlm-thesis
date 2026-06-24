"""
Verify the dense (full-token) ceiling isn't artificially low. Our DocVQA dense=67.2
(at max_new_tokens=20) sits below published LLaVA-NeXT (~74). Hypothesis: 20-token cap
truncates longer answers. Re-measure dense at a longer cap and report truncation stats.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m v2.analysis.verify_dense --dataset docvqa --max-new-tokens 50
"""
import argparse, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from datasets import load_dataset
from v2.eval.evaluate_textvqa_highres_kcurve import HighResPruner
from src.metrics.docvqa_score import score_docvqa
from src.metrics.chartqa_score import score_chartqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["docvqa", "chartqa", "textvqa"])
    ap.add_argument("--max-samples", type=int, default=300)
    ap.add_argument("--max-new-tokens", type=int, default=50)
    args = ap.parse_args()

    p = HighResPruner()
    if args.dataset == "docvqa":
        ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
        get = lambda ex: (ex["image"], ex["question"], {"golds": [str(a) for a in ex["answers"]]})
        scorer = lambda preds: score_docvqa(preds)["anls_pct"]
    elif args.dataset == "chartqa":
        ds = load_dataset("lmms-lab/ChartQA", split="test")
        get = lambda ex: (ex["image"], ex["question"], {"gold": str(ex["answer"])})
        scorer = lambda preds: score_chartqa(preds)["accuracy_pct"]
    else:
        import json
        data = json.load(open("data/textvqa/TextVQA_0.5.1_val.json"))["data"]
        ds = data
        get = lambda ex: (None, ex["question"], {"image_id": ex["image_id"]})  # special-cased below

    n = min(args.max_samples, len(ds))
    preds, lens, truncs = [], [], 0
    for i in range(n):
        ex = ds[i]
        if args.dataset == "textvqa":
            from PIL import Image
            img = Image.open(f"data/textvqa/train_images/{ex['image_id']}.jpg").convert("RGB")
            q, gold = ex["question"], None
        else:
            img, q, gold = get(ex)
            img = img.convert("RGB")
        pred, _ = p.generate(img, q, selector="full", K=None, max_new_tokens=args.max_new_tokens)
        ntok = len(p.proc.tokenizer(pred, add_special_tokens=False).input_ids)
        lens.append(ntok); truncs += int(ntok >= args.max_new_tokens)
        if args.dataset == "textvqa":
            from src.metrics.textvqa_score import score_textvqa  # noqa
            preds.append({"question_id": ex["image_id"], "question": q.strip(), "pred_answer": pred})
        else:
            preds.append({"pred_answer": pred, **gold})

    if args.dataset == "textvqa":
        from src.metrics.textvqa_score import score_textvqa
        acc = score_textvqa(preds)["accuracy_pct"]; metric = "soft"
    elif args.dataset == "docvqa":
        acc = scorer(preds); metric = "ANLS"
    else:
        acc = scorer(preds); metric = "relaxed"

    print("\n" + "=" * 56)
    print(f"  {args.dataset} dense (full tokens) @ max_new_tokens={args.max_new_tokens}, n={n}")
    print(f"  {metric} = {acc:.2f}%")
    print(f"  answer length: mean={sum(lens)/len(lens):.1f} tok, max={max(lens)}, "
          f"hit-cap={truncs} ({100*truncs/n:.1f}%)")
    print("=" * 56)


if __name__ == "__main__":
    main()
