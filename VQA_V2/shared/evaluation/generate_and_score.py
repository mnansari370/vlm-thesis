"""
Generation evaluator — THE canonical evaluation protocol for this track.

Loads any trained checkpoint (dense / static / dynamic), bypasses the answer
head, runs model.generate() on the validation set, and scores with VQA
consensus accuracy.  Reports BOTH classification accuracy (from the answer head,
if present) and generation accuracy in a single output JSON so the two metrics
can be compared directly. All thesis/paper numbers come from the GENERATION
section; the classification numbers are the retired early proxy
(see docs/vqav2_findings.md, protocol note).

Usage:
    python -m VQA_V2.shared.evaluation.generate_and_score \
        --config VQA_V2/dense/llava_dense_150k_10k_fullvocab.yaml \
        --checkpoint VQA_V2/outputs/<run>/best_model.pt \
        --model-type dense \
        --output-path VQA_V2/outputs/<run>/generation_eval.json \
        [--max-samples 1000]   # optional, for quick smoke tests

Runs from repo root: cd /home/nafees/vlm-thesis && python -m VQA_V2.shared.evaluation.generate_and_score ...
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Allow running from repo root
# Allow running this file directly (repo root = 3 level(s) up); `python -m` does not need this.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from VQA_V2.shared.datasets import VQACollator, build_vqav2_dataset
from VQA_V2.shared.datasets.vqav2_answers import normalize_answer
from VQA_V2.shared.utils.config import load_config
from VQA_V2.shared.utils.seed import set_seed


# ── VQA scoring ────────────────────────────────────────────────────────────

def vqa_consensus_score(pred: str, raw_answers: List[str]) -> float:
    pred_norm = normalize_answer(pred)
    matches = sum(1 for a in raw_answers if normalize_answer(a) == pred_norm)
    return min(1.0, matches / 3.0)


def compute_mean_accuracy(preds: List[Dict[str, Any]]) -> float:
    if not preds:
        return 0.0
    scores = [vqa_consensus_score(p["pred_answer"], p["raw_answers"]) for p in preds]
    return float(sum(scores) / len(scores))


# ── Model loading ───────────────────────────────────────────────────────────

def load_model(model_type: str, cfg: Dict[str, Any], checkpoint_path: Optional[str]):
    if model_type == "dense":
        from VQA_V2.dense import LlavaDenseVQAModel
        model = LlavaDenseVQAModel(cfg)
    elif model_type == "static":
        from VQA_V2.static import LlavaStaticVQAModel
        model = LlavaStaticVQAModel(cfg)
    elif model_type == "dynamic":
        from VQA_V2.dynamic import LlavaDynamicVQAModel
        model = LlavaDynamicVQAModel(cfg)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[Info] Loading checkpoint: {checkpoint_path}", flush=True)
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        if "answer_head_state_dict" in ckpt and hasattr(model, "answer_head") and model.answer_head is not None:
            model.answer_head.load_state_dict(ckpt["answer_head_state_dict"], strict=True)
            print("[Info] Answer head loaded.", flush=True)

        if "token_selector_state_dict" in ckpt and hasattr(model, "token_selector") and model.token_selector is not None:
            model.token_selector.load_state_dict(ckpt["token_selector_state_dict"], strict=True)
            print("[Info] Token selector loaded.", flush=True)
    elif checkpoint_path:
        print(f"[Warn] Checkpoint not found: {checkpoint_path}. Running with random answer head.", flush=True)

    return model


# ── Classification forward (reuse existing model forward) ──────────────────

@torch.no_grad()
def run_classification_eval(
    model,
    loader: DataLoader,
    log_every: int = 100,
) -> List[Dict[str, Any]]:
    model.eval()
    predictions = []
    for step, batch in enumerate(loader):
        outputs = model(batch)
        pred_answers = outputs["predictions"].get("pred_answers", None)
        if pred_answers is None:
            continue
        for i, pred in enumerate(pred_answers):
            predictions.append({
                "question_id": batch["question_ids"][i],
                "image_id": batch["image_ids"][i],
                "question": batch["questions"][i],
                "pred_answer": pred,
                "raw_answers": batch["raw_answers"][i],
            })
        if (step + 1) % log_every == 0:
            print(f"[Classification eval] {step+1}/{len(loader)}", flush=True)
    return predictions


# ── Generation forward ──────────────────────────────────────────────────────

def _prepare_inputs_for_generation(model, images, questions):
    """Re-use the model's own _prepare_inputs but bypass answer head path."""
    return model._prepare_inputs(images=images, questions=questions)


def _get_token_stats(model, model_type: str, batch, model_inputs) -> Dict[str, Any]:
    """Extract token stats from the model for FLOPs reporting."""
    try:
        if model_type == "dense":
            ts = model._build_dense_token_stats(
                batch_size=len(batch["images"]),
                device=model._get_model_device(),
            )
            return {
                "num_visual_tokens_after": int(ts["num_visual_tokens_after_selection"].float().mean().item()),
            }
        elif model_type == "static":
            pv = model_inputs.get("pixel_values")
            if pv is not None:
                vf, attn = model._run_vision_encoder(pv)
                sel = model.token_selector(visual_features=vf, final_layer_attentions=attn, token_mask=None)
                return {"num_visual_tokens_after": int(sel["num_tokens_after"].float().mean().item())}
    except Exception:
        pass
    return {}


@torch.no_grad()
def run_generation_eval(
    model,
    model_type: str,
    loader: DataLoader,
    max_new_tokens: int = 10,
    do_sample: bool = False,
    log_every: int = 100,
) -> List[Dict[str, Any]]:
    """
    Bypass the answer head completely. Run the frozen LLM's generate() on
    the selected (or full) visual token sequence.

    For dense:  full 576 tokens → generate()
    For static: K selected tokens → LM forward → generate()
    For dynamic: hard-selected tokens → LM forward → generate()
    """
    model.eval()
    predictions = []

    for step, batch in enumerate(loader):
        images = batch["images"]
        questions = batch["questions"]

        model_inputs = _prepare_inputs_for_generation(model, images, questions)
        device = model._get_model_device()

        try:
            if model_type == "dense":
                # Full forward: use model.model.generate() directly
                gen_ids = model.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                )
                prompt_len = model_inputs["input_ids"].shape[1]
                answer_ids = gen_ids[:, prompt_len:]
                decoded = model.processor.batch_decode(
                    answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )

            elif model_type == "static":
                # Build pruned multimodal inputs then use language_model.generate().
                # When inputs_embeds is passed to generate(), the output contains ONLY
                # the newly generated token IDs (not the embedding positions), so no
                # slicing by input length is needed.
                lm_embeds, lm_mask, _ = model._build_pruned_multimodal_inputs(model_inputs)
                lm = model._get_language_model()
                gen_ids = lm.generate(
                    inputs_embeds=lm_embeds,
                    attention_mask=lm_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                )
                decoded = model.processor.batch_decode(
                    gen_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

            elif model_type == "dynamic":
                # Same: inputs_embeds mode → gen_ids contains only new tokens.
                lm_embeds, lm_mask, _, _ = model._build_dynamic_multimodal_inputs(
                    model_inputs=model_inputs,
                    questions=questions,
                )
                lm = model._get_language_model()
                gen_ids = lm.generate(
                    inputs_embeds=lm_embeds,
                    attention_mask=lm_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                )
                decoded = model.processor.batch_decode(
                    gen_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

        except Exception as e:
            print(f"[Warn] Generation failed at step {step}: {e}", flush=True)
            decoded = [""] * len(questions)

        for i, pred in enumerate(decoded):
            predictions.append({
                "question_id": batch["question_ids"][i],
                "image_id": batch["image_ids"][i],
                "question": batch["questions"][i],
                "pred_answer": pred.strip(),
                "raw_answers": batch["raw_answers"][i],
            })

        if (step + 1) % log_every == 0:
            acc_so_far = compute_mean_accuracy(predictions)
            print(f"[Generation eval] {step+1}/{len(loader)} | running_acc={acc_so_far:.4f}", flush=True)

    return predictions


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model-type", required=True, choices=["dense", "static", "dynamic"])
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-samples", type=int, default=None, help="Override val set size for smoke tests")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--skip-classification", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)), False)

    # Override val samples if requested (smoke test)
    if args.max_samples is not None:
        cfg["dataset"]["max_val_samples"] = args.max_samples

    print("[Info] Building dataset...", flush=True)
    dataset = build_vqav2_dataset(cfg, args.split)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["training"].get("eval_batch_size", 1)),
        shuffle=False,
        num_workers=int(cfg["dataset"].get("num_workers", 4)),
        collate_fn=VQACollator(),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(cfg["dataset"].get("num_workers", 4)) > 0,
    )

    print("[Info] Loading model...", flush=True)
    model = load_model(args.model_type, cfg, args.checkpoint)
    model.eval()

    results: Dict[str, Any] = {
        "experiment_name": cfg.get("experiment_name"),
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "num_samples": len(dataset),
    }

    # ── Classification eval ────────────────────────────────────────────────
    if not args.skip_classification and hasattr(model, "answer_head") and model.answer_head is not None:
        print("\n[Info] === Classification evaluation ===", flush=True)
        cls_preds = run_classification_eval(model, loader, log_every=args.log_every)
        cls_acc = compute_mean_accuracy(cls_preds)
        print(f"[Info] Classification VQA accuracy: {cls_acc:.4f} ({cls_acc*100:.2f}%)", flush=True)
        results["classification"] = {
            "vqa_accuracy": cls_acc,
            "num_predictions": len(cls_preds),
            "predictions": cls_preds,
        }

    # ── Generation eval ────────────────────────────────────────────────────
    if not args.skip_generation:
        print("\n[Info] === Generation evaluation ===", flush=True)
        gen_preds = run_generation_eval(
            model=model,
            model_type=args.model_type,
            loader=loader,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            log_every=args.log_every,
        )
        gen_acc = compute_mean_accuracy(gen_preds)
        print(f"[Info] Generation VQA accuracy: {gen_acc:.4f} ({gen_acc*100:.2f}%)", flush=True)
        results["generation"] = {
            "vqa_accuracy": gen_acc,
            "num_predictions": len(gen_preds),
            "max_new_tokens": args.max_new_tokens,
            "predictions": gen_preds,
        }

    # ── Save ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Info] Results saved to: {args.output_path}", flush=True)

    # Print summary
    print("\n=== SUMMARY ===")
    if "classification" in results:
        print(f"  Classification: {results['classification']['vqa_accuracy']*100:.2f}%")
    if "generation" in results:
        print(f"  Generation:     {results['generation']['vqa_accuracy']*100:.2f}%")


if __name__ == "__main__":
    main()
