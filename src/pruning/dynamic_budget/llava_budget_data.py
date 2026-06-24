"""
Cross-FAMILY generality: the SAME budget-mirage data collection as Qwen, on LLaVA-1.6
(LlavaNext, a different architecture). Per DocVQA sample: cheap QC-attention features +
per-K correctness (sufficiency-K labels). Feeds qwen_budget_eval/robust (--tag docvqa_llava).
Runs in vlm_env (transformers 4.46.3). QC layer 16 (LLaVA-1.6 best, from Part-A sweep).
"""
import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch
from datasets import load_dataset
from src.evaluation.textvqa.evaluate_textvqa_highres_kcurve import HighResPruner
from src.metrics.docvqa_score import anls

LADDER = [32, 64, 128, 256, 512, 768]


def feats(qs):
    p = (qs / qs.sum()).clamp_min(1e-9)
    nvis = p.numel()
    ent = float(-(p * p.log()).sum())
    return {"nvis": nvis, "norm_entropy": ent / math.log(nvis),
            "eff_support": float(1.0 / (p * p).sum()),
            "top64_mass": float(p.topk(min(64, nvis)).values.sum()),
            "top128_mass": float(p.topk(min(128, nvis)).values.sum()),
            "top256_mass": float(p.topk(min(256, nvis)).values.sum())}


@torch.no_grad()
def encode_qc(p, image, question, layer=16):
    """LLaVA: encode + mid-layer question->visual attention scores (one forward)."""
    prefix, suffix, fts, _, N = p._encode(image, question)
    full = torch.cat([prefix, fts.to(prefix.dtype), suffix], 0).unsqueeze(0)
    o = p.model.language_model.model(inputs_embeds=full, output_attentions=True, use_cache=False)
    A = o.attentions[layer][0].float().mean(0)
    v0 = prefix.shape[0]; v1 = v0 + N
    qs = A[v1:, v0:v1].mean(0)
    del o, A
    return prefix, suffix, fts, N, qs


@torch.no_grad()
def gen_keep(p, prefix, suffix, fts, keep, max_new_tokens=32):
    vis = fts[keep].to(prefix.dtype)
    seq = torch.cat([prefix, vis, suffix], 0).unsqueeze(0)
    mask = torch.ones(1, seq.shape[1], dtype=torch.long, device=p.dev)
    out = p.model.language_model.generate(inputs_embeds=seq, attention_mask=mask,
        max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, pad_token_id=p.eos, use_cache=True)
    return p.proc.tokenizer.decode(out[0], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=400)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()
    p = HighResPruner()
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
    rows = []; t0 = time.time()
    for i in range(min(args.max_samples, len(ds))):
        img = ds[i]["image"].convert("RGB"); q = ds[i]["question"]
        gold = [str(a) for a in ds[i]["answers"]]
        prefix, suffix, fts, N, qs = encode_qc(p, img, q, args.layer)
        f = feats(qs); soft = {}
        for K in LADDER + ["full"]:
            keep = torch.arange(N, device=p.dev) if K == "full" else qs.topk(min(K, N)).indices.sort().values
            pred = gen_keep(p, prefix, suffix, fts, keep, args.max_new_tokens)
            soft[str(K)] = anls(pred, gold)
        rows.append({"feats": f, "soft": soft})
        if (i + 1) % 50 == 0:
            print(f"  {i+1} ({round((time.time()-t0)/60,1)}m)", flush=True)
    json.dump({"ladder": LADDER, "rows": rows}, open("results/paper_candidates/qwen_budget_data_docvqa_llava.json", "w"))
    print(f"[saved] docvqa_llava  n={len(rows)}  dense={100*sum(r['soft']['full'] for r in rows)/len(rows):.2f}")


if __name__ == "__main__":
    main()
