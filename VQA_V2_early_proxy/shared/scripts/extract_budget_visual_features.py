import argparse
import json
import math
from pathlib import Path
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModel

from VQA_V2_early_proxy.shared.utils.config import load_config
from VQA_V2_early_proxy.shared.datasets import build_vqav2_dataset


def infer_qtype(question: str) -> str:
    q = question.lower().strip()
    words = q.split()
    first = words[0] if words else ""

    if first in {"is", "are", "was", "were", "do", "does", "did", "can", "could", "has", "have"}:
        return "yes_no"
    if first in {"what", "which"}:
        if "color" in q or "colour" in q:
            return "color"
        if "number" in q or "many" in q or "count" in q:
            return "count"
        return "what_which"
    if first == "how":
        if "many" in q or "much" in q:
            return "count"
        return "how"
    if first in {"where", "who", "why", "when"}:
        return first
    return "other"


def safe_float(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def score_stats(scores: torch.Tensor, prefix: str):
    """
    scores: [N]
    """
    scores = scores.float()
    probs = torch.softmax(scores, dim=0)

    top_vals, _ = torch.topk(scores, k=min(25, scores.numel()))
    top2, _ = torch.topk(scores, k=2)

    entropy = -(probs * torch.log(probs + 1e-12)).sum()
    norm_entropy = entropy / math.log(scores.numel())
    effective_tokens = torch.exp(entropy)

    out = {
        f"{prefix}_mean": safe_float(scores.mean()),
        f"{prefix}_std": safe_float(scores.std(unbiased=False)),
        f"{prefix}_min": safe_float(scores.min()),
        f"{prefix}_max": safe_float(scores.max()),
        f"{prefix}_top1": safe_float(top_vals[0]),
        f"{prefix}_top2": safe_float(top2[1]),
        f"{prefix}_top1_top2_margin": safe_float(top2[0] - top2[1]),
        f"{prefix}_top5_mean": safe_float(top_vals[:5].mean()),
        f"{prefix}_top10_mean": safe_float(top_vals[:10].mean()),
        f"{prefix}_top25_mean": safe_float(top_vals[:25].mean()),
        f"{prefix}_entropy": safe_float(entropy),
        f"{prefix}_norm_entropy": safe_float(norm_entropy),
        f"{prefix}_effective_tokens": safe_float(effective_tokens),
    }
    return out


def load_budget_labels(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    by_qid = {}
    for r in data["records"]:
        by_qid[int(r["question_id"])] = r
    return by_qid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="VQA_V2_early_proxy/static/llava_static_clsattn_60k_10k_top3500_k288.yaml")
    parser.add_argument("--labels", default="data/budget_oracle/val_budget_labels_binary.json")
    parser.add_argument("--output", default="data/budget_oracle/val_budget_visual_features_binary.jsonl")
    parser.add_argument("--vision_model", default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--split", default="val")
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Force validation sample count for this feature extraction.
    cfg["dataset"]["max_val_samples"] = args.max_samples

    labels_by_qid = load_budget_labels(args.labels)

    dataset = build_vqav2_dataset(cfg, args.split)
    print(f"[Info] Dataset samples: {len(dataset)}", flush=True)
    print(f"[Info] Budget label records: {len(labels_by_qid)}", flush=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    dtype = torch.float16 if args.dtype == "float16" and device.type == "cuda" else torch.float32

    print(f"[Info] Loading CLIP vision model: {args.vision_model}", flush=True)
    image_processor = CLIPImageProcessor.from_pretrained(args.vision_model)
    vision_model = CLIPVisionModel.from_pretrained(args.vision_model)
    vision_model.to(device=device, dtype=dtype)
    vision_model.eval()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_no_label = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for idx in tqdm(range(len(dataset)), desc="Extracting visual budget features"):
            sample = dataset[idx]
            qid = int(sample["question_id"])

            if qid not in labels_by_qid:
                skipped_no_label += 1
                continue

            label = labels_by_qid[qid]
            image = sample["image"]
            question = sample["question"]

            inputs = image_processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)

            with torch.no_grad():
                out = vision_model(
                    pixel_values=pixel_values,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=True,
                )

            # attentions[-1]: [1, heads, 577, 577]
            # CLS attends to patch tokens: position 0 -> positions 1:
            cls_attn = out.attentions[-1][0, :, 0, 1:].mean(dim=0).float()  # [576]

            # hidden_states[-2]: [1, 577, D], remove CLS
            patch_features = out.hidden_states[-2][0, 1:, :].float()  # [576, D]
            patch_norms = patch_features.norm(dim=-1)  # [576]

            feats = {}
            feats.update(score_stats(cls_attn, "cls_attn"))
            feats.update(score_stats(patch_norms, "patch_norm"))

            q_words = question.strip().split()
            qtype = infer_qtype(question)

            record = {
                "question_id": qid,
                "image_id": int(sample["image_id"]),
                "question": question,
                "qtype": qtype,
                "question_length": len(q_words),
                "oracle_budget": int(label["oracle_budget"]),
                "budget_class": int(label["budget_class"]),
                "budget_class_name": label["budget_class_name"],
                "mapped_tokens": int(label.get("mapped_tokens", 576 if int(label["budget_class"]) == 1 else 144)),
                "features": feats,
            }

            fout.write(json.dumps(record) + "\n")
            written += 1

    print("=" * 100)
    print(f"Saved features to: {out_path}")
    print(f"Written records: {written}")
    print(f"Skipped no label: {skipped_no_label}")
    print("=" * 100)


if __name__ == "__main__":
    main()
