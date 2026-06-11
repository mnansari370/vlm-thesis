"""
Phase 1.5 / Phase 2 — Evaluate our fixed static model on the EXACT FasterVLM question subset.

Loads the 10K question IDs used by FasterVLM (from scripts/diag1_data/val2014_questions.jsonl),
runs our static model on those questions, and scores with the same VQA-consensus normalizer
used in generate_and_score.py.

Preprocessing is read from the config's image_aspect_ratio field ("pad" or "center_crop").
The --override-aspect-ratio flag allows overriding the config value for ablations.

Usage (from repo root):
  # Uses preprocessing from config (now "pad" by default):
  CUDA_VISIBLE_DEVICES=0 conda run -n vlm_env python VQA_V2_early_proxy/shared/scripts/eval_static_on_fvlm_subset.py \\
      --config vqa_v2/VQA_V2_early_proxy/static/llava_static_clsattn_150k_10k_fullvocab_k128.yaml \\
      --output-path vqa_v2/outputs/static_k128_pad/generation_eval_fvlm_subset_10k.json

  # Force center-crop for ablation:
  CUDA_VISIBLE_DEVICES=0 conda run -n vlm_env python VQA_V2_early_proxy/shared/scripts/eval_static_on_fvlm_subset.py \\
      --config ... --override-aspect-ratio center_crop --output-path /tmp/cc_test.json
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Allow running this file directly (repo root = 3 levels up); `python -m` does not need this.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from VQA_V2.shared.datasets.collate import VQACollator
from VQA_V2.shared.datasets.image_transforms import build_image_transform
from VQA_V2.shared.datasets.vqav2_answers import normalize_answer
from VQA_V2.shared.evaluation.generate_and_score import (
    load_model,
    run_generation_eval,
    vqa_consensus_score,
)
from VQA_V2.shared.utils.config import load_config
from VQA_V2.shared.utils.seed import set_seed


class FVLMSubsetDataset(Dataset):
    """
    Loads exactly the question IDs from scripts/diag1_data/val2014_questions.jsonl,
    matched against the full VQAv2 val2014 questions + annotations JSONs.
    Uses the same build_image_transform as the main pipeline.
    """

    def __init__(
        self,
        fvlm_jsonl: str,
        questions_json: str,
        annotations_json: str,
        image_dir: str,
        image_aspect_ratio: str = "center_crop",
        max_samples: Optional[int] = None,
    ):
        self.image_dir = image_dir
        self.transform = build_image_transform(
            image_size=336,
            is_train=False,
            image_aspect_ratio=image_aspect_ratio,
        )

        fvlm_qids: List[int] = []
        with open(fvlm_jsonl) as f:
            for line in f:
                fvlm_qids.append(json.loads(line)["question_id"])
        fvlm_qid_set = set(fvlm_qids)

        with open(annotations_json) as f:
            anns_data = json.load(f)
        anns_by_qid = {a["question_id"]: a for a in anns_data["annotations"]}

        with open(questions_json) as f:
            qs_data = json.load(f)
        by_qid = {q["question_id"]: q for q in qs_data["questions"] if q["question_id"] in fvlm_qid_set}

        samples: List[Dict[str, Any]] = []
        for qid in fvlm_qids:
            q = by_qid.get(qid)
            if q is None:
                continue
            ann = anns_by_qid.get(qid, {})
            raw_answers = [a["answer"] for a in ann.get("answers", [])]
            img_id = q["image_id"]
            img_path = os.path.join(image_dir, f"COCO_val2014_{img_id:012d}.jpg")
            if not os.path.exists(img_path):
                continue
            samples.append({
                "question_id": qid,
                "image_id": img_id,
                "question": q["question"],
                "raw_answers": raw_answers,
                "image_path": img_path,
            })

        if max_samples is not None:
            samples = samples[:max_samples]

        self.samples = samples
        print(f"[FVLMSubset] {len(self.samples)} samples loaded (image_aspect_ratio={image_aspect_ratio})", flush=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        img = Image.open(s["image_path"]).convert("RGB")
        img = self.transform(img)
        return {
            "image": img,
            "question": s["question"],
            "answer": s["raw_answers"][0] if s["raw_answers"] else "",
            "answer_label": -1,
            "raw_answers": s["raw_answers"],
            "normalized_answers": [],
            "question_id": s["question_id"],
            "image_id": s["image_id"],
            "image_path": s["image_path"],
            "active_split": "val",
        }


def main():
    parser = argparse.ArgumentParser(description="Eval model on FasterVLM question subset")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model-type", default="static")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--override-aspect-ratio", default=None,
                        help="Override config's image_aspect_ratio (pad or center_crop)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    image_aspect_ratio = args.override_aspect_ratio or cfg["dataset"].get("image_aspect_ratio", "center_crop")
    batch_size = int(cfg["training"].get("eval_batch_size", 1))

    fvlm_jsonl = "scripts/diag1_data/val2014_questions.jsonl"
    questions_json = cfg["dataset"]["questions_val"]
    annotations_json = cfg["dataset"]["annotations_val"]
    image_dir = cfg["dataset"]["image_dir_val"]

    print(f"[Info] image_aspect_ratio: {image_aspect_ratio}", flush=True)
    print(f"[Info] model_type: {args.model_type}", flush=True)

    dataset = FVLMSubsetDataset(
        fvlm_jsonl=fvlm_jsonl,
        questions_json=questions_json,
        annotations_json=annotations_json,
        image_dir=image_dir,
        image_aspect_ratio=image_aspect_ratio,
        max_samples=args.max_samples,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=VQACollator(),
        num_workers=4,
        persistent_workers=True,
    )

    print(f"[Info] Loading model...", flush=True)
    model = load_model(args.model_type, cfg, args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"[Info] Running generation eval on {len(dataset)} samples...", flush=True)
    gen_cfg = cfg.get("generation", cfg.get("model", {}))
    predictions = run_generation_eval(
        model=model,
        model_type=args.model_type,
        loader=loader,
        max_new_tokens=int(cfg.get("generation", {}).get("max_new_tokens", 10)),
        do_sample=bool(cfg.get("generation", {}).get("do_sample", False)),
        log_every=args.log_every,
    )

    scores = [vqa_consensus_score(p["pred_answer"], p["raw_answers"]) for p in predictions]
    vqa_acc = float(sum(scores) / len(scores)) if scores else 0.0

    result = {
        "description": f"{args.model_type} eval on FasterVLM 10K question subset — preprocessing={image_aspect_ratio}",
        "n_predictions": len(predictions),
        "vqa_accuracy": vqa_acc,
        "vqa_accuracy_pct": round(vqa_acc * 100, 2),
        "fastervlm_K128_pct": 74.40,
        "delta_vs_fastervlm_pp": round(vqa_acc * 100 - 74.40, 2),
        "predictions": predictions,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"Our {args.model_type} on FasterVLM subset: {vqa_acc*100:.2f}%  (preprocessing={image_aspect_ratio})", flush=True)
    print(f"FasterVLM K=128 (reference):           74.40%", flush=True)
    print(f"Delta:                                 {vqa_acc*100 - 74.40:+.2f}pp", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"[Done] → {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
