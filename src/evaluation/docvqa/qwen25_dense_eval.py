"""
Phase 0b: reproduce Qwen2.5-VL-7B DENSE (no pruning) numbers on DocVQA / ChartQA to
confirm harness fidelity (published: DocVQA 95.7 ANLS, ChartQA 87.3 relaxed-acc).
Runs in the ISOLATED qwen_env (transformers 4.51.3); scorers are stdlib-only copies.

Standard Qwen2.5-VL inference: chat template + qwen_vl_utils.process_vision_info.
Reports avg visual tokens (image_grid_thw // merge^2) so we know the pruning budget range.

Usage (qwen_env python!):
  CUDA_VISIBLE_DEVICES=0 /home/nafees/miniconda3/envs/qwen_env/bin/python \
      -m src.evaluation.docvqa.qwen25_dense_eval --dataset docvqa --max-samples 200
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from src.metrics.docvqa_score import anls
from src.metrics.chartqa_score import relaxed_correct

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
INSTR = "Answer the question using a single word or phrase."


def load_bench(name):
    if name == "docvqa":
        ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
        get = lambda ex: (ex["image"], ex["question"], [str(a) for a in ex["answers"]])
        score = lambda pred, gold: anls(pred, gold)
        ref = 95.7
    else:
        ds = load_dataset("lmms-lab/ChartQA", split="test")
        get = lambda ex: (ex["image"], ex["question"], str(ex["answer"]))
        score = lambda pred, gold: 1.0 if relaxed_correct(pred, gold) else 0.0
        ref = 87.3
    return ds, get, score, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["docvqa", "chartqa"])
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)  # Qwen default-ish
    ap.add_argument("--instr", action="store_true", help="append the single-word instruction")
    args = ap.parse_args()

    print(f"[qwen] loading {MODEL} ...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda:0").eval()
    processor = AutoProcessor.from_pretrained(MODEL, min_pixels=args.min_pixels, max_pixels=args.max_pixels)

    ds, get, score, ref = load_bench(args.dataset)
    n = min(args.max_samples, len(ds))
    scores, toks = [], []
    t0 = time.time()
    for i in range(n):
        img, q, gold = get(ds[i])
        img = img.convert("RGB")
        qtext = q.strip() + (" " + INSTR if args.instr else "")
        messages = [{"role": "user", "content": [{"type": "image", "image": img},
                                                 {"type": "text", "text": qtext}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to("cuda:0")
        # visual tokens = sum over grid_thw of t*h*w // merge^2
        thw = inputs["image_grid_thw"]
        merge = processor.image_processor.merge_size
        nvis = int((thw[:, 0] * thw[:, 1] * thw[:, 2]).sum() // (merge * merge))
        toks.append(nvis)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        pred = processor.decode(gen, skip_special_tokens=True).strip()
        scores.append(score(pred, gold))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n}  acc={100*sum(scores)/len(scores):.2f}%  (~{sum(toks)//len(toks)} vis-tok)", flush=True)

    acc = 100 * sum(scores) / n
    metric = "ANLS" if args.dataset == "docvqa" else "relaxed-acc"
    print("\n" + "=" * 56)
    print(f"  Qwen2.5-VL-7B DENSE {args.dataset}  (n={n}, instr={args.instr})")
    print(f"  {metric} = {acc:.2f}%   (published ref {ref})")
    print(f"  avg visual tokens = {sum(toks)//len(toks)}  (max {max(toks)})")
    print(f"  {round((time.time()-t0)/60,1)} min")
    print("=" * 56)
    json.dump({"dataset": args.dataset, "n": n, "acc": acc, "ref": ref,
               "avg_vis_tokens": sum(toks)//len(toks), "max_vis_tokens": max(toks),
               "instr": args.instr, "max_pixels": args.max_pixels},
              open(f"results/thesis_main/highres/qwen25_dense_{args.dataset}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
