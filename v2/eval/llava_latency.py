"""
Plan item 4: real LATENCY (FLOPs != latency, per 'Are We Solving the Right Problem?').
Because we prune BEFORE the LLM (all layers see K tokens), the FLOPs cut should translate
to wall-clock speedup. We measure the LLM generate() time (the part pruning affects) at
several K on LLaVA-1.6, with warmup + cuda.synchronize. Selection cost is excluded here
(blind/cheap select); the point is the prune-before-LLM decode speedup.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch
from datasets import load_dataset
from v2.eval.evaluate_textvqa_highres_kcurve import HighResPruner


@torch.no_grad()
def time_generate(p, prefix, suffix, feats, keep, max_new_tokens=32, reps=3):
    vis = feats[keep].to(prefix.dtype)
    seq = torch.cat([prefix, vis, suffix], 0).unsqueeze(0)
    mask = torch.ones(1, seq.shape[1], dtype=torch.long, device=p.dev)
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        t0 = time.time()
        p.model.language_model.generate(inputs_embeds=seq, attention_mask=mask,
            max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
            pad_token_id=p.eos, use_cache=True)
        torch.cuda.synchronize(); ts.append(time.time() - t0)
    return min(ts) * 1000  # ms (min = least-noisy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--k-values", default="64,128,256,512,1024")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()
    p = HighResPruner()
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
    Ks = [int(x) for x in args.k_values.split(",")]
    # warmup
    for i in range(3):
        pr, su, ft, _, N = p._encode(ds[i]["image"].convert("RGB"), ds[i]["question"])
        time_generate(p, pr, su, ft, torch.arange(min(128, N), device=p.dev), args.max_new_tokens, 1)

    lat = {K: [] for K in Ks + ["full"]}
    ntok = []
    for i in range(3, 3 + args.n):
        pr, su, ft, _, N = p._encode(ds[i]["image"].convert("RGB"), ds[i]["question"])
        ntok.append(N)
        for K in Ks + ["full"]:
            keep = torch.arange(N, device=p.dev) if K == "full" else torch.arange(min(K, N), device=p.dev)
            lat[K].append(time_generate(p, pr, su, ft, keep, args.max_new_tokens))

    full_ms = sum(lat["full"]) / len(lat["full"])
    print("\n" + "=" * 56)
    print(f"  LLaVA-1.6 LLM generate latency by K (DocVQA, n={args.n}, ~{sum(ntok)//len(ntok)} dense tok)")
    print(f"  {'K':>6}{'latency(ms)':>14}{'speedup':>10}")
    print("-" * 56)
    res = {}
    for K in Ks + ["full"]:
        ms = sum(lat[K]) / len(lat[K])
        sp = full_ms / ms
        res[str(K)] = {"ms": round(ms, 1), "speedup": round(sp, 2)}
        print(f"  {str(K):>6}{ms:>14.1f}{sp:>9.2f}x")
    print("=" * 56)
    json.dump({"n": args.n, "dense_tok": sum(ntok)//len(ntok), "by_K": res},
              open("v2/outputs/llava_latency.json", "w"), indent=2)


if __name__ == "__main__":
    main()
