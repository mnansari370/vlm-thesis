"""
GQA testdev_balanced evaluation — honest locked protocol.

Produces predictions in the format GQA's official eval.py expects:
    [{"questionId": "...", "prediction": "..."}]  →  testdev_balanced_predictions.json

Prompt: LLaVA-1.5 vicuna_v1 + GQA suffix (from GQA.sh + convert_gqa_for_eval.py):
    USER: <image>\\n{question}\\nAnswer the question using a single word or phrase.
    ASSISTANT:

Post-processing (convert_gqa_for_eval.py, line 14):
    text.rstrip('.').lower()   ← ONLY this. No article removal. No plural stripping.

Decoding (honest — matches LLaVA's model_vqa_loader.py):
    greedy (do_sample=False, temperature=0), max_new_tokens=64,
    NO repetition_penalty, NO min_new_tokens, natural EOS stop.

Scoring: GQA/shared/official_score.py (canonical, used everywhere in this project).

Published reference: LLaVA-1.5-7B → 62.0% on testdev_balanced (Table 1, LLaVA-1.5 paper)

Usage
-----
    python -m GQA.dense.run_dense_testdev --output_name testdev_dense_honest
    python -m GQA.dense.run_dense_testdev --output_name testdev_dense_pad --image_pad
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration

from GQA.shared.official_score import is_correct, score_testdev_format, print_result
from GQA.shared.utils.logger import make_output_dir


MODEL_NAME   = "llava-hf/llava-1.5-7b-hf"
QUESTIONS    = "data/gqa/testdev_balanced_questions.json"
IMAGE_DIR    = "data/gqa/images/images"

# Exact suffix from LLaVA's GQA evaluation (vicuna_v1 conv mode, GQA.sh)
PROMPT_SUFFIX = "\nAnswer the question using a single word or phrase."


# ── dataset ───────────────────────────────────────────────────────────────────

class GQATestdevDataset(Dataset):
    def __init__(self, questions_path: str, image_dir: str,
                 max_samples: int | None = None):
        with open(questions_path) as f:
            raw = json.load(f)
        self.image_dir = image_dir
        self.records = []
        missing = 0
        for qid, rec in raw.items():
            img_path = os.path.join(image_dir, f"{rec['imageId']}.jpg")
            if not os.path.exists(img_path):
                missing += 1
                continue
            self.records.append({
                "question_id": qid,
                "question":    rec["question"],
                "answer":      rec.get("answer", ""),
                "image_path":  img_path,
                "semantic_type": rec.get("types", {}).get("semantic", "unknown"),
            })
        if max_samples:
            self.records = self.records[:max_samples]
        if missing:
            print(f"[Data] Skipped {missing} missing images.", flush=True)
        print(f"[Data] Loaded {len(self.records):,} testdev_balanced samples.", flush=True)

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return {
            "question_id":   r["question_id"],
            "question":      r["question"],
            "answer":        r["answer"],
            "semantic_type": r["semantic_type"],
            "image":         Image.open(r["image_path"]).convert("RGB"),
        }


def collate(batch):
    return {
        "question_ids":   [b["question_id"]   for b in batch],
        "questions":      [b["question"]       for b in batch],
        "answers":        [b["answer"]         for b in batch],
        "semantic_types": [b["semantic_type"]  for b in batch],
        "images":         [b["image"]          for b in batch],
    }


# ── model (inline — matches LLaVA protocol exactly) ──────────────────────────

class LlavaTestdevEval:
    """
    Frozen LLaVA-1.5-7B using vicuna_v1 conv format for GQA evaluation.
    Honest protocol — matches LLaVA's GQA.sh + convert_gqa_for_eval.py exactly.

    image_pad=True: pads images to square before encoding (LLaVA-1.5 default).
    image_pad=False: uses processor default (center-crop for non-square images).
    """
    def __init__(self, image_pad: bool = False):
        print(f"[Model] Loading {MODEL_NAME} (image_pad={image_pad}) ...", flush=True)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to(self.dev)
        self.model.eval()
        self.model.config.use_cache = True   # enable KV cache for generation

        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
        self.processor.tokenizer.padding_side = "left"
        vc = getattr(self.model.config, "vision_config", None)
        self.processor.patch_size = getattr(vc, "patch_size", 14)
        self.processor.vision_feature_select_strategy = "default"
        self.processor.num_additional_image_tokens = 0
        self.image_pad = image_pad

        # Print the tokenized prompt so we can verify template parity
        _sample_q = "What color is the car?"
        _sample_conv = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": _sample_q + PROMPT_SUFFIX},
        ]}]
        _prompt = self.processor.apply_chat_template(
            _sample_conv, add_generation_prompt=True, tokenize=False
        )
        print(f"[Model] Prompt template (sample):\n{repr(_prompt)}", flush=True)
        print(f"[Model] Ready on {self.dev}.", flush=True)

    def _pad_to_square(self, img: Image.Image) -> Image.Image:
        """Pad non-square image to square with black border (LLaVA-1.5 default)."""
        w, h = img.size
        if w == h:
            return img
        side = max(w, h)
        padded = Image.new("RGB", (side, side), (0, 0, 0))
        padded.paste(img, ((side - w) // 2, (side - h) // 2))
        return padded

    @torch.no_grad()
    def generate_raw(self, images: list, questions: list[str]) -> list[str]:
        """Return the raw decoded generation (`.strip()` only, no rstrip/lower).

        Used by the B0 isolation test so the SAME generation can be scored
        both honestly (rstrip('.').lower()) and via extract_short_answer.
        """
        if self.image_pad:
            images = [self._pad_to_square(img) for img in images]

        # Build prompts with LLaVA's vicuna_v1 template + GQA instruction suffix
        convs = [
            [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": q.strip() + PROMPT_SUFFIX},
            ]}]
            for q in questions
        ]
        prompts = [
            self.processor.apply_chat_template(c, add_generation_prompt=True, tokenize=False)
            for c in convs
        ]
        inp = self.processor(
            text=prompts, images=images, return_tensors="pt", padding=True
        )
        inp = {k: (v.to(self.dev) if hasattr(v, "to") else v) for k, v in inp.items()}

        # Honest decoding — matches LLaVA's model_vqa_loader.py:
        #   do_sample=False (temperature=0), max_new_tokens=64, no repetition_penalty
        out = self.model.generate(
            **inp,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        # Decode only generated tokens
        in_len = inp["input_ids"].shape[1]
        raw_texts = self.processor.tokenizer.batch_decode(
            out[:, in_len:], skip_special_tokens=True
        )
        return [t.strip() for t in raw_texts]

    def generate(self, images: list, questions: list[str]) -> list[str]:
        # LLaVA's convert_gqa_for_eval.py post-processing (line 14):
        #   text.rstrip('.').lower()   ← the only transform
        return [t.rstrip(".").lower() for t in self.generate_raw(images, questions)]


# Normalization imported from canonical scorer — do not duplicate here.
# normalize(text) = text.strip().rstrip('.').lower()
# is_correct(pred, gold) uses that normalization for both sides.


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_name",  default="testdev_dense_honest")
    ap.add_argument("--questions",    default=QUESTIONS)
    ap.add_argument("--image_dir",    default=IMAGE_DIR)
    ap.add_argument("--batch_size",   type=int, default=4)
    ap.add_argument("--num_workers",  type=int, default=4)
    ap.add_argument("--log_every",    type=int, default=200)
    ap.add_argument("--max_samples",  type=int, default=None,
                    help="Cap to N samples for quick probes.")
    ap.add_argument("--image_pad",    action="store_true",
                    help="Pad non-square images to square before encoding "
                         "(LLaVA-1.5 original uses image_aspect_ratio='pad').")
    args = ap.parse_args()

    out_dir = make_output_dir("outputs", args.output_name)
    print(f"[Output] {out_dir}", flush=True)
    print(f"[Config] image_pad={args.image_pad}  max_samples={args.max_samples}", flush=True)

    dataset = GQATestdevDataset(args.questions, args.image_dir,
                                max_samples=args.max_samples)
    loader  = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0),
    )

    model = LlavaTestdevEval(image_pad=args.image_pad)

    # predictions → GQA official eval.py format
    # results     → full records for official_score.py (val format)
    predictions: list[dict] = []
    results:     list[dict] = []
    n_correct = 0
    t0 = time.time()

    for step, batch in enumerate(loader):
        preds = model.generate(batch["images"], batch["questions"])

        for qid, pred, gold, q, stype in zip(
            batch["question_ids"], preds, batch["answers"],
            batch["questions"], batch["semantic_types"]
        ):
            ok = is_correct(pred, gold)
            n_correct += int(ok)

            predictions.append({"questionId": qid, "prediction": pred})
            results.append({
                "question_id":   qid,
                "question":      q,
                "pred_answer":   pred,
                "answer":        gold,
                "semantic_type": stype,
                "correct":       ok,
            })

        if (step + 1) % args.log_every == 0:
            n_done  = len(results)
            acc     = n_correct / n_done
            elapsed = time.time() - t0
            sps     = (step + 1) / elapsed
            eta_min = (len(loader) - step - 1) / max(sps, 1e-6) / 60
            print(f"  {n_done:>6}/{len(dataset):,}  acc={acc*100:.2f}%  "
                  f"speed={sps:.2f} batch/s  ETA={eta_min:.1f}min", flush=True)

    elapsed_h = (time.time() - t0) / 3600

    # ── Score with canonical scorer ───────────────────────────────────────────
    from GQA.shared.official_score import score_val_format
    scored = score_val_format(results, {})
    print_result(scored,
                 label=f"testdev_balanced  image_pad={args.image_pad}",
                 reference=62.0)
    print(f"  Elapsed  : {elapsed_h:.2f}h", flush=True)

    # Save GQA official eval.py format
    pred_path = os.path.join(out_dir, "testdev_balanced_predictions.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f)

    # Save full results for re-scoring
    with open(os.path.join(out_dir, "results.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    acc   = scored["accuracy_pct"]
    diff  = acc - 62.0
    metrics = {
        "scorer":          "official_score.py (rstrip('.').lower(), strict equality)",
        "n_evaluated":     scored["n_total"],
        "n_correct":       scored["n_correct"],
        "accuracy":        scored["accuracy"],
        "accuracy_pct":    acc,
        "n_empty":         scored["n_empty"],
        "per_type":        scored["per_type"],
        "reference_llava_7b": 62.0,
        "diff_from_reference_pp": round(diff, 2),
        "within_0_5pp":    abs(diff) <= 0.5,
        "prompt_suffix":   PROMPT_SUFFIX,
        "decoding":        "greedy max_new_tokens=64 no repetition_penalty no min_new_tokens",
        "image_pad":       args.image_pad,
        "elapsed_hours":   round(elapsed_h, 3),
        "split":           "testdev_balanced",
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[Files] {out_dir}/")
    print(f"  testdev_balanced_predictions.json  (GQA official format)")
    print(f"  results.jsonl                      (full records)")
    print(f"  metrics.json                       (scored with official_score.py)")


if __name__ == "__main__":
    main()
