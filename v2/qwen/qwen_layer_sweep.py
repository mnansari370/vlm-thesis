"""
Selection-axis baseline: which LLM layer's question->visual attention is the best selector?
FastV uses early layers (2-3); FEATHER showed mid layers (8/16) are better. We confirm on
Qwen2.5-VL: one scoring forward per sample, evaluate top-K selection at several layers.
Reference: blind uniform @K128 DocVQA = 32.2. Run in qwen_env.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch
from datasets import load_dataset
from v2.qwen.qwen_pruner import QwenPruner
from v2.shared.docvqa_score import anls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=150)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--layers", default="2,4,8,12,16,20,24")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--tag", default="7b")
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()
    p = QwenPruner(model_name=args.model, load_in_4bit=args.load_in_4bit)
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
    layers = [int(x) for x in args.layers.split(",")]
    n = args.max_samples
    sc = {L: [] for L in layers}
    t0 = time.time()
    for i in range(n):
        img, q, gold = ds[i]["image"], ds[i]["question"], [str(a) for a in ds[i]["answers"]]
        emb, pos, attn, v0, nvis, _ = p._encode(img, q)
        out = p.model.model(inputs_embeds=emb, position_ids=pos, attention_mask=attn,
                            use_cache=False, output_attentions=True)
        # extract small per-layer scores, THEN free the all-layer attention before generating (avoid OOM)
        keeps = {}
        for L in layers:
            qs = out.attentions[L][0].float().mean(0)[v0 + nvis:, v0:v0 + nvis].mean(0)
            keeps[L] = qs.topk(min(args.K, nvis)).indices.sort().values.clone()
        del out
        torch.cuda.empty_cache()
        for L in layers:
            pred, _ = p.generate_keep(emb, pos, attn, v0, nvis, keeps[L], args.max_new_tokens)
            sc[L].append(anls(pred, gold))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n} ({round((time.time()-t0)/60,1)}m)", flush=True)
    print("\n" + "=" * 50)
    print(f"  DocVQA Qwen2.5-VL: selector quality by LLM layer (K={args.K}, n={n})")
    print(f"  (blind uniform ref = 32.2)")
    for L in layers:
        a = 100 * sum(sc[L]) / n
        tag = " (FastV-ish)" if L <= 4 else (" <- best-ish" if a == max(100*sum(sc[x])/n for x in layers) else "")
        print(f"    layer {L:>2}: {a:.2f}%{tag}")
    print("=" * 50)
    json.dump({"K": args.K, "n": n, "model": args.model,
               "by_layer": {str(L): round(100*sum(sc[L])/n, 2) for L in layers}},
              open(f"v2/outputs/qwen_layer_sweep_{args.tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
