"""
Phase 2 data collection: for each DocVQA sample, extract cheap QC-attention-distribution
FEATURES (predict how concentrated the answer region is -> how many tokens needed) and the
per-K correctness (for sufficiency-K labels). Feeds the budget-predictor confirmation.

Run in qwen_env.
"""
import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch
from datasets import load_dataset
from v2.qwen.qwen_pruner import QwenPruner
from src.metrics.docvqa_score import anls

LADDER = [32, 64, 128, 256, 512, 768]


def feats(qs):
    p = (qs / qs.sum()).clamp_min(1e-9)
    nvis = p.numel()
    ent = float(-(p * p.log()).sum())
    return {
        "nvis": nvis,
        "norm_entropy": ent / math.log(nvis),            # 0=peaked,1=flat
        "eff_support": float(1.0 / (p * p).sum()),       # participation ratio
        "top64_mass": float(p.topk(min(64, nvis)).values.sum()),
        "top128_mass": float(p.topk(min(128, nvis)).values.sum()),
        "top256_mass": float(p.topk(min(256, nvis)).values.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="docvqa", choices=["docvqa", "infovqa"])
    ap.add_argument("--max-samples", type=int, default=400)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--textattn-layer", type=int, default=16)
    ap.add_argument("--tag", default="")   # output suffix; default = dataset
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    args = ap.parse_args()
    p = QwenPruner(model_name=args.model, textattn_layer=args.textattn_layer,
                   load_in_4bit=args.load_in_4bit, max_pixels=args.max_pixels)
    cfg = "DocVQA" if args.dataset == "docvqa" else "InfographicVQA"
    ds = load_dataset("lmms-lab/DocVQA", cfg, split="validation")
    rows = []
    t0 = time.time()
    for i in range(min(args.max_samples, len(ds))):
        img, q, gold = ds[i]["image"], ds[i]["question"], [str(a) for a in ds[i]["answers"]]
        emb, pos, attn, v0, nvis, qs = p.encode_qc(img, q)
        f = feats(qs)
        soft = {}
        for K in LADDER + ["full"]:
            kl = torch.arange(nvis, device=p.dev) if K == "full" else qs.topk(min(K, nvis)).indices.sort().values
            pred, _ = p.generate_keep(emb, pos, attn, v0, nvis, kl, args.max_new_tokens)
            soft[str(K)] = anls(pred, gold)
        rows.append({"feats": f, "soft": soft})
        if (i + 1) % 50 == 0:
            print(f"  {i+1} ({round((time.time()-t0)/60,1)}m)", flush=True)
    tag = args.tag or args.dataset
    out = f"v2/outputs/qwen_budget_data_{tag}.json"
    json.dump({"ladder": LADDER, "rows": rows}, open(out, "w"))
    print(f"[saved] {out}  n={len(rows)}  dense(full)={100*sum(r['soft']['full'] for r in rows)/len(rows):.2f}")


if __name__ == "__main__":
    main()
