"""
Evaluate the Stage-1 elastic model on GQA testdev (generation protocol) at one or
more K. Reports TWO accuracies per K:
  - strict   : exact match after rstrip('.').lower()  (v1's honest scorer)
  - extracted: same, but after pulling the short answer out of a verbose reply
               (diagnostic — tells us if the KNOWLEDGE is there even when the model
                is chatty). The gap between the two measures the model's verbosity.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m v2.eval.evaluate_gqa \
        --checkpoint v2/outputs/stage1_full/final.pt \
        --k-values 576 --max-samples 500
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from PIL import Image

from v2.model.elastic_wrapper import ElasticPrunedLlava
from src.metrics.official_score import is_correct
from src.metrics.metrics import extract_short_answer

GQA_Q = "data/gqa/testdev_balanced_questions.json"
GQA_IMG = "data/gqa/images/images"
V1_DENSE = 61.42  # v1 frozen dense GQA, for reference
# eval appends the benchmark instruction (training uses the mix's per-sample instruction;
# raw GQA testdev questions have none, so we add the GQA-standard one to match the format
# the short-answer mix samples were trained with).
GQA_INSTR = "Answer the question using a single word or phrase."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="v2/outputs/stage1_full/final.pt")
    ap.add_argument("--base", action="store_true",
                    help="eval the UNTRAINED base model (no checkpoint) — the frozen baseline")
    ap.add_argument("--k-values", default="576")
    ap.add_argument("--max-samples", type=int, default=500)
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--output", default=None)
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()

    Ks = [int(x) for x in args.k_values.split(",")]
    if args.base:
        print("[eval] UNTRAINED BASE model (no checkpoint)")
        m = ElasticPrunedLlava(lora_rank=16, dtype="bfloat16", image_pad=True,
                               gradient_checkpointing=False)
        m.eval()
    else:
        m = ElasticPrunedLlava.from_checkpoint(args.checkpoint)

    g = json.load(open(GQA_Q))
    qids = list(g.keys())[: args.max_samples]   # deterministic first-N (held-out testdev)

    out = {"checkpoint": args.checkpoint, "n": len(qids), "v1_dense_ref": V1_DENSE, "per_k": {}}
    for K in Ks:
        t0 = time.time()
        n_strict = n_extract = n = 0
        samples = []
        for qid in qids:
            rec = g[qid]
            ipath = os.path.join(GQA_IMG, f"{rec['imageId']}.jpg")
            if not os.path.exists(ipath):
                continue
            img = Image.open(ipath).convert("RGB")
            q_fmt = rec["question"].strip() + " " + GQA_INSTR
            pred = m.generate_answers([img], [q_fmt], K=K,
                                      max_new_tokens=args.max_new_tokens)[0]
            gold = rec.get("answer", "")
            ext = extract_short_answer(pred, rec["question"])
            s_ok = is_correct(pred, gold)
            e_ok = is_correct(ext, gold)
            n += 1; n_strict += int(s_ok); n_extract += int(e_ok)
            samples.append({"qid": qid, "q": rec["question"], "gold": gold,
                            "pred": pred, "extracted": ext, "strict": s_ok, "extract": e_ok})
            if n % args.log_every == 0:
                print(f"  K={K} {n}/{len(qids)}  strict={100*n_strict/n:.1f}%  "
                      f"extracted={100*n_extract/n:.1f}%", flush=True)
        strict = round(100 * n_strict / max(n, 1), 2)
        extract = round(100 * n_extract / max(n, 1), 2)
        out["per_k"][K] = {"n": n, "strict_acc": strict, "extract_acc": extract,
                           "minutes": round((time.time() - t0) / 60, 1)}
        print(f"\n[GQA K={K}]  strict={strict}%   extracted={extract}%   "
              f"(n={n}, v1 frozen dense={V1_DENSE}%)", flush=True)

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"summary": out, "samples": samples}, f, indent=2)
        print(f"[saved] {args.output}")
    print("\n" + json.dumps(out["per_k"], indent=2))


if __name__ == "__main__":
    main()
