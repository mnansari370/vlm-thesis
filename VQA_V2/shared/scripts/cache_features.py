"""
Feature caching script for dense and static models.

Because the backbone is fully frozen, the LLM's last-hidden-state for every
sample is deterministic given (image, question, K). This script runs one
forward pass per sample per model variant, saves results to memory-mapped
numpy arrays, and never touches the GPU again during answer-head training.

Outputs (per cache key):
    <cache_dir>/
        pooled_features.npy        float16  [N, 4096]   last-hidden @ answer pos
        per_layer_answer_pos.npy   float16  [N, 32, 4096]  per-layer, answer pos only
        question_type_ids.npy      int32    [N]
        question_ids.npy           int64    [N]
        answer_labels.npy          int32    [N]
        raw_answers.json                    list[list[str]]
        metadata.json

Cache keys:
    dense/train,  dense/val
    static_k64/train,  static_k64/val,  ...  static_k432/val

Usage (run from repo root):
    python VQA_V2/shared/scripts/cache_features.py \
        --model-type dense \
        --split train \
        --config VQA_V2/dense/llava_dense_150k_10k_fullvocab.yaml \
        --cache-dir VQA_V2/feature_cache

    python VQA_V2/shared/scripts/cache_features.py \
        --model-type static \
        --keep-tokens 288 \
        --split train \
        --config VQA_V2/static/llava_static_clsattn_150k_10k_fullvocab_k288.yaml \
        --cache-dir VQA_V2/feature_cache
"""

import argparse
import json
import os
import sys
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Allow running this file directly (repo root = 3 level(s) up); `python -m` does not need this.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from VQA_V2.shared.datasets import VQACollator, build_vqav2_dataset
from VQA_V2.shared.utils.config import load_config
from VQA_V2.shared.utils.seed import set_seed


NUM_LAYERS = 32
HIDDEN_SIZE = 4096


def _get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _gather_last_valid_hidden(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_indices = attention_mask.sum(dim=1) - 1
    last_indices = last_indices.clamp(min=0)
    batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_idx, last_indices, :]   # [B, H]


def _all_layers_answer_pos(all_hidden: tuple, attention_mask: torch.Tensor) -> torch.Tensor:
    """Stack all layer hidden states at the answer position. Returns [B, L, H]."""
    last_indices = attention_mask.sum(dim=1) - 1
    last_indices = last_indices.clamp(min=0)
    batch_idx = torch.arange(attention_mask.size(0), device=attention_mask.device)
    layers = []
    for layer_h in all_hidden:    # each is [B, S, H]
        layers.append(layer_h[batch_idx, last_indices, :])  # [B, H]
    return torch.stack(layers, dim=1)  # [B, L, H]


@torch.no_grad()
def cache_dense(
    cfg,
    split: str,
    cache_dir: str,
    log_every: int = 200,
):
    from VQA_V2.dense import LlavaDenseVQAModel

    key = f"dense/{split}"
    out_dir = os.path.join(cache_dir, key)
    os.makedirs(out_dir, exist_ok=True)

    dataset = build_vqav2_dataset(cfg, split)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg["dataset"].get("num_workers", 4)),
        collate_fn=VQACollator(),
        pin_memory=True,
        persistent_workers=True,
    )
    N = len(dataset)
    print(f"[Cache] dense/{split}: {N} samples", flush=True)

    model = LlavaDenseVQAModel(cfg)
    model.eval()
    device = _get_model_device(model)

    pooled = np.zeros((N, HIDDEN_SIZE), dtype=np.float16)
    per_layer = np.zeros((N, NUM_LAYERS, HIDDEN_SIZE), dtype=np.float16)
    q_type_ids = np.zeros(N, dtype=np.int32)
    q_ids = np.zeros(N, dtype=np.int64)
    ans_labels = np.zeros(N, dtype=np.int32)
    raw_answers_list: List[List[str]] = []

    idx = 0
    for step, batch in enumerate(loader):
        model_inputs = model._prepare_inputs(
            images=batch["images"], questions=batch["questions"]
        )

        outputs = model.model(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        attn_mask = model_inputs["attention_mask"]
        last_h = outputs.hidden_states[-1]   # [B, S, H]
        pooled_h = _gather_last_valid_hidden(last_h, attn_mask)    # [B, H]
        all_layers_h = _all_layers_answer_pos(outputs.hidden_states[1:], attn_mask)  # [B, L, H]

        B = pooled_h.size(0)
        for b in range(B):
            pooled[idx] = pooled_h[b].float().cpu().numpy().astype(np.float16)
            per_layer[idx] = all_layers_h[b].float().cpu().numpy().astype(np.float16)
            ans_labels[idx] = int(batch["answer_labels"][b].item())
            q_ids[idx] = int(batch["question_ids"][b])
            raw_answers_list.append(batch["raw_answers"][b])

            # Question type using the same heuristic as DynamicTokenSelector
            from VQA_V2.shared.datasets.vqav2 import _question_type_id
            q_type_ids[idx] = _question_type_id(batch["questions"][b])
            idx += 1

        if (step + 1) % log_every == 0:
            print(f"[Cache] dense/{split}: {idx}/{N}", flush=True)

    print(f"[Cache] dense/{split}: {idx}/{N} done. Saving...", flush=True)

    np.save(os.path.join(out_dir, "pooled_features.npy"), pooled[:idx])
    np.save(os.path.join(out_dir, "per_layer_answer_pos.npy"), per_layer[:idx])
    np.save(os.path.join(out_dir, "question_type_ids.npy"), q_type_ids[:idx])
    np.save(os.path.join(out_dir, "question_ids.npy"), q_ids[:idx])
    np.save(os.path.join(out_dir, "answer_labels.npy"), ans_labels[:idx])

    with open(os.path.join(out_dir, "raw_answers.json"), "w") as f:
        json.dump(raw_answers_list, f)

    meta = {
        "model_type": "dense",
        "split": split,
        "num_samples": idx,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "pooled_shape": [idx, HIDDEN_SIZE],
        "per_layer_shape": [idx, NUM_LAYERS, HIDDEN_SIZE],
        "dtype": "float16",
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[Cache] Saved to {out_dir}", flush=True)
    return out_dir


@torch.no_grad()
def cache_static(
    cfg,
    keep_tokens: int,
    split: str,
    cache_dir: str,
    log_every: int = 200,
):
    from VQA_V2.static import LlavaStaticVQAModel

    key = f"static_k{keep_tokens}/{split}"
    out_dir = os.path.join(cache_dir, key)
    os.makedirs(out_dir, exist_ok=True)

    dataset = build_vqav2_dataset(cfg, split)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg["dataset"].get("num_workers", 4)),
        collate_fn=VQACollator(),
        pin_memory=True,
        persistent_workers=True,
    )
    N = len(dataset)
    print(f"[Cache] static_k{keep_tokens}/{split}: {N} samples", flush=True)

    model = LlavaStaticVQAModel(cfg)
    model.eval()

    pooled = np.zeros((N, HIDDEN_SIZE), dtype=np.float16)
    per_layer = np.zeros((N, NUM_LAYERS, HIDDEN_SIZE), dtype=np.float16)
    q_type_ids = np.zeros(N, dtype=np.int32)
    q_ids = np.zeros(N, dtype=np.int64)
    ans_labels = np.zeros(N, dtype=np.int32)
    raw_answers_list: List[List[str]] = []

    idx = 0
    for step, batch in enumerate(loader):
        model_inputs = model._prepare_inputs(
            images=batch["images"], questions=batch["questions"]
        )

        # Static: build pruned multimodal inputs, then run LM
        lm_embeds, lm_mask, _ = model._build_pruned_multimodal_inputs(model_inputs)
        lm = model._get_language_model()

        lm_out = lm(
            inputs_embeds=lm_embeds,
            attention_mask=lm_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        last_h = lm_out.hidden_states[-1]
        pooled_h = _gather_last_valid_hidden(last_h, lm_mask)
        all_layers_h = _all_layers_answer_pos(lm_out.hidden_states[1:], lm_mask)

        B = pooled_h.size(0)
        for b in range(B):
            pooled[idx] = pooled_h[b].float().cpu().numpy().astype(np.float16)
            per_layer[idx] = all_layers_h[b].float().cpu().numpy().astype(np.float16)
            ans_labels[idx] = int(batch["answer_labels"][b].item())
            q_ids[idx] = int(batch["question_ids"][b])
            raw_answers_list.append(batch["raw_answers"][b])

            from VQA_V2.shared.datasets.vqav2 import _question_type_id
            q_type_ids[idx] = _question_type_id(batch["questions"][b])
            idx += 1

        if (step + 1) % log_every == 0:
            print(f"[Cache] static_k{keep_tokens}/{split}: {idx}/{N}", flush=True)

    print(f"[Cache] static_k{keep_tokens}/{split}: {idx}/{N} done. Saving...", flush=True)

    np.save(os.path.join(out_dir, "pooled_features.npy"), pooled[:idx])
    np.save(os.path.join(out_dir, "per_layer_answer_pos.npy"), per_layer[:idx])
    np.save(os.path.join(out_dir, "question_type_ids.npy"), q_type_ids[:idx])
    np.save(os.path.join(out_dir, "question_ids.npy"), q_ids[:idx])
    np.save(os.path.join(out_dir, "answer_labels.npy"), ans_labels[:idx])

    with open(os.path.join(out_dir, "raw_answers.json"), "w") as f:
        json.dump(raw_answers_list, f)

    meta = {
        "model_type": "static",
        "keep_tokens": keep_tokens,
        "split": split,
        "num_samples": idx,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "pooled_shape": [idx, HIDDEN_SIZE],
        "per_layer_shape": [idx, NUM_LAYERS, HIDDEN_SIZE],
        "dtype": "float16",
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[Cache] Saved to {out_dir}", flush=True)
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-type", required=True, choices=["dense", "static"])
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--keep-tokens", type=int, default=None,
                        help="Required for --model-type static. Number of tokens to keep.")
    parser.add_argument("--cache-dir", default="VQA_V2/feature_cache")
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Override dataset size (train or val) for smoke tests.")
    args = parser.parse_args()

    if args.model_type == "static" and args.keep_tokens is None:
        parser.error("--keep-tokens is required for --model-type static")

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)), False)

    # Apply --max-samples override before dataset construction
    if args.max_samples is not None:
        if args.split == "train":
            cfg["dataset"]["max_samples"] = args.max_samples
        else:
            cfg["dataset"]["max_val_samples"] = args.max_samples
        print(f"[Cache] --max-samples override: {args.max_samples} ({args.split})", flush=True)

    N = args.max_samples if args.max_samples else (150000 if args.split == "train" else 10000)
    pooled_gb = N * HIDDEN_SIZE * 2 / 1e9
    per_layer_gb = N * NUM_LAYERS * HIDDEN_SIZE * 2 / 1e9
    print(f"[Cache] Estimated disk: pooled={pooled_gb:.3f}GB, per_layer={per_layer_gb:.3f}GB", flush=True)

    if args.model_type == "dense":
        cache_dense(cfg, args.split, args.cache_dir, log_every=args.log_every)
    else:
        cache_static(cfg, args.keep_tokens, args.split, args.cache_dir, log_every=args.log_every)


if __name__ == "__main__":
    main()
