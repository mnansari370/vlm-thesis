"""
Phase 1d: question-conditioning control on Qwen2.5-VL. Proves the mid-layer selector
genuinely USES THE QUESTION (not just a better blind saliency): select the K visual
tokens by a MISMATCHED question's mid-layer attention, then answer the REAL question.
If accuracy falls toward the blind baseline, the selection is genuinely question-conditioned.

Run in qwen_env.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch
from datasets import load_dataset
from src.pruning.question_conditioned_selection.qwen_pruner import QwenPruner
from src.metrics.docvqa_score import anls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=100)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()
    p = QwenPruner()
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
    n = args.max_samples
    real_s, mis_s = [], []
    t0 = time.time()
    for i in range(n):
        img, q, gold = ds[i]["image"], ds[i]["question"], [str(a) for a in ds[i]["answers"]]
        mq = ds[(i + 1) % n]["question"]                                  # a DIFFERENT question
        er = p.encode_qc(img, q)                                          # real: emb/pos/attn + qscores
        em, _, _, _, _, qs_mis = p.encode_qc(img, mq)                     # mismatched scores (same image)
        emb, pos, attn, v0, nvis, qs_real = er
        keep_real = qs_real.topk(min(args.K, nvis)).indices.sort().values
        keep_mis = qs_mis.topk(min(args.K, nvis)).indices.sort().values
        a_real, _ = p.generate_keep(emb, pos, attn, v0, nvis, keep_real, args.max_new_tokens)
        a_mis, _ = p.generate_keep(emb, pos, attn, v0, nvis, keep_mis, args.max_new_tokens)
        real_s.append(anls(a_real, gold)); mis_s.append(anls(a_mis, gold))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n} ({round((time.time()-t0)/60,1)}m) real={100*sum(real_s)/(i+1):.1f} mis={100*sum(mis_s)/(i+1):.1f}", flush=True)
    real = 100 * sum(real_s) / n
    mis = 100 * sum(mis_s) / n
    BLIND = 32.19   # uniform @K128 DocVQA (the question-blind reference)
    qfrac = round(100 * (real - mis) / (real - BLIND), 1) if real > BLIND else 0.0
    print("\n" + "=" * 58)
    print(f"  DocVQA Qwen2.5-VL — question-conditioning control (K={args.K}, n={n})")
    print(f"  blind (uniform)        : {BLIND}")
    print(f"  QC by MISMATCHED q     : {mis:.2f}   (control)")
    print(f"  QC by REAL q           : {real:.2f}")
    print(f"  -> {qfrac:.0f}% of (real-blind) gain is DUE TO THE QUESTION")
    print(f"  VERDICT: {'genuinely QUESTION-CONDITIONED' if real-mis>=5 else 'mostly blind saliency'}")
    print("=" * 58)
    json.dump({"K": args.K, "n": n, "real": real, "mismatched": mis, "blind": BLIND,
               "question_fraction_pct": qfrac}, open("results/thesis_main/highres/qwen_control_docvqa.json", "w"), indent=2)


if __name__ == "__main__":
    main()
