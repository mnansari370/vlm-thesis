"""
Cached answer-head trainer for dense and static models.

Loads pre-computed pooled features from disk (output of cache_features.py)
and trains only the AnswerHeadMLP. No backbone forward passes — one epoch
over 150K samples runs in under 30 minutes.

Usage:
    python -m VQA_V2.shared.training.cached.train_cached \
        --train-cache VQA_V2/feature_cache/dense/train \
        --val-cache   VQA_V2/feature_cache/dense/val \
        --config      VQA_V2/dense/llava_dense_150k_10k_fullvocab.yaml \
        --output-dir  VQA_V2/outputs/dense_cached_v1
"""

import argparse
import json
import math
import os
import sys
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Allow running this file directly (repo root = 4 level(s) up); `python -m` does not need this.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from VQA_V2.dense.answer_head import AnswerHeadMLP
from VQA_V2.shared.datasets.vqav2_answers import normalize_answer
from VQA_V2.shared.utils.config import load_config
from VQA_V2.shared.utils.seed import set_seed


# ── Cached feature dataset ─────────────────────────────────────────────────

class CachedFeatureDataset(Dataset):
    """
    Reads pooled_features.npy + answer_labels.npy + raw_answers.json from
    a cache directory produced by cache_features.py.
    """

    def __init__(self, cache_dir: str):
        pooled_path = os.path.join(cache_dir, "pooled_features.npy")
        labels_path = os.path.join(cache_dir, "answer_labels.npy")
        raw_ans_path = os.path.join(cache_dir, "raw_answers.json")
        meta_path = os.path.join(cache_dir, "metadata.json")

        if not os.path.exists(pooled_path):
            raise FileNotFoundError(f"Cache not found: {pooled_path}\n"
                                    f"Run cache_features.py first.")

        self.features = np.load(pooled_path, mmap_mode="r")   # [N, H], float16
        self.labels = np.load(labels_path, mmap_mode="r")     # [N], int32

        with open(raw_ans_path) as f:
            self.raw_answers = json.load(f)

        with open(meta_path) as f:
            self.meta = json.load(f)

        assert len(self.features) == len(self.labels) == len(self.raw_answers), \
            "Cache arrays have mismatched lengths"

        print(f"[CachedFeatureDataset] Loaded {len(self.features)} samples from {cache_dir}", flush=True)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        feat = torch.from_numpy(self.features[idx].astype(np.float32))  # [H]
        label = int(self.labels[idx])
        return {
            "features": feat,
            "answer_label": label,
            "raw_answers": self.raw_answers[idx],
        }


def cached_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    features = torch.stack([b["features"] for b in batch], dim=0)
    labels = torch.tensor([b["answer_label"] for b in batch], dtype=torch.long)
    raw_answers = [b["raw_answers"] for b in batch]
    return {"features": features, "answer_labels": labels, "raw_answers": raw_answers}


# ── VQA accuracy ───────────────────────────────────────────────────────────

def vqa_consensus_score(pred: str, raw_answers: List[str]) -> float:
    pred_norm = normalize_answer(pred)
    matches = sum(1 for a in raw_answers if normalize_answer(a) == pred_norm)
    return min(1.0, matches / 3.0)


# ── Build answer head ──────────────────────────────────────────────────────

def build_answer_head(cfg: Dict[str, Any], vocab_size: int) -> AnswerHeadMLP:
    model_cfg = cfg["model"]
    return AnswerHeadMLP(
        input_dim=4096,  # Vicuna-7B hidden size
        hidden_dim=int(model_cfg["answer_head_hidden_dim"]),
        output_dim=vocab_size,
        dropout=float(model_cfg.get("answer_head_dropout", 0.1)),
        train_dtype=model_cfg.get("answer_head_train_dtype", "float32"),
    )


def load_id_to_answer(cfg: Dict[str, Any]) -> Dict[int, str]:
    path = cfg["dataset"].get("answer_vocab_path")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"answer_vocab_path not found: {path}")
    with open(path) as f:
        data = json.load(f)
    if "answer_to_id" in data:
        return {int(v): k for k, v in data["answer_to_id"].items()}
    return {int(k): v for k, v in data["id_to_answer"].items()}


# ── Optimizer / scheduler ──────────────────────────────────────────────────

def build_optimizer(cfg: Dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    opt_cfg = cfg["optimizer"]
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        eps=float(opt_cfg.get("eps", 1e-8)),
    )


def build_scheduler(cfg: Dict[str, Any], optimizer, total_steps: int):
    sched_cfg = cfg.get("scheduler", {})
    warmup_ratio = float(sched_cfg.get("warmup_ratio", 0.1))
    min_lr = float(sched_cfg.get("min_lr", 1e-6))
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        base_lr = float(cfg["training"]["learning_rate"])
        return max(min_lr / base_lr, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Training / validation loops ────────────────────────────────────────────

def train_one_epoch(
    answer_head: nn.Module,
    id_to_answer: Dict[int, str],
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    grad_accum_steps: int,
    device: torch.device,
    epoch: int,
    log_every: int = 500,
) -> Dict[str, float]:
    answer_head.train()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    total_loss = 0.0
    num_samples = 0
    acc_scores: List[float] = []
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        features = batch["features"].to(device)
        labels = batch["answer_labels"].to(device)

        logits = answer_head(features)
        loss = loss_fn(logits, labels) / max(1, grad_accum_steps)
        loss.backward()

        if ((step + 1) % grad_accum_steps == 0) or ((step + 1) == len(loader)):
            torch.nn.utils.clip_grad_norm_(answer_head.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        bs = features.size(0)
        total_loss += float(loss.item()) * grad_accum_steps * bs
        num_samples += bs

        pred_ids = logits.argmax(dim=-1)
        for i, (pid, raw_ans) in enumerate(zip(pred_ids.tolist(), batch["raw_answers"])):
            pred_str = id_to_answer.get(int(pid), "")
            acc_scores.append(vqa_consensus_score(pred_str, raw_ans))

        if (step + 1) % log_every == 0:
            avg_loss = total_loss / max(1, num_samples)
            avg_acc = sum(acc_scores) / max(1, len(acc_scores))
            print(
                f"[Train] Epoch {epoch} step {step+1}/{len(loader)} "
                f"loss={avg_loss:.4f} vqa_acc={avg_acc:.4f}",
                flush=True,
            )

    return {
        "loss": total_loss / max(1, num_samples),
        "vqa_accuracy": sum(acc_scores) / max(1, len(acc_scores)),
    }


@torch.no_grad()
def validate(
    answer_head: nn.Module,
    id_to_answer: Dict[int, str],
    loader: DataLoader,
    device: torch.device,
    log_every: int = 200,
) -> Dict[str, float]:
    answer_head.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    total_loss = 0.0
    num_samples = 0
    acc_scores: List[float] = []
    predictions: List[Dict[str, Any]] = []

    for step, batch in enumerate(loader):
        features = batch["features"].to(device)
        labels = batch["answer_labels"].to(device)

        logits = answer_head(features)
        loss = loss_fn(logits, labels)

        bs = features.size(0)
        total_loss += float(loss.item()) * bs
        num_samples += bs

        pred_ids = logits.argmax(dim=-1)
        for i, (pid, raw_ans) in enumerate(zip(pred_ids.tolist(), batch["raw_answers"])):
            pred_str = id_to_answer.get(int(pid), "")
            score = vqa_consensus_score(pred_str, raw_ans)
            acc_scores.append(score)
            predictions.append({"pred_answer": pred_str, "raw_answers": raw_ans})

        if (step + 1) % log_every == 0:
            print(f"[Val] step {step+1}/{len(loader)}", flush=True)

    return {
        "loss": total_loss / max(1, num_samples),
        "vqa_accuracy": sum(acc_scores) / max(1, len(acc_scores)),
        "predictions": predictions,
    }


# ── Save / load ────────────────────────────────────────────────────────────

def save_checkpoint(path: str, answer_head: nn.Module, optimizer, scheduler, epoch: int, metrics: dict):
    torch.save({
        "epoch": epoch,
        "metrics": metrics,
        "answer_head_state_dict": answer_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }, path)


def save_json(path: str, data: Any):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True, help="Path to train cache dir")
    parser.add_argument("--val-cache", required=True, help="Path to val cache dir")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override training.cached_batch_size from config.")
    parser.add_argument("--eval-batch-size", type=int, default=None,
                        help="Override eval batch size.")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override training.epochs. Use ~60 for convergence.")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping: halt after this many epochs with no val improvement.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)), False)
    os.makedirs(args.output_dir, exist_ok=True)
    save_json(os.path.join(args.output_dir, "config.json"), cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}", flush=True)

    train_ds = CachedFeatureDataset(args.train_cache)
    val_ds = CachedFeatureDataset(args.val_cache)
    id_to_answer = load_id_to_answer(cfg)
    vocab_size = len(id_to_answer)
    print(f"[Info] Vocab size: {vocab_size}", flush=True)

    grad_accum = int(cfg["training"].get("grad_accum_steps", 2))
    batch_size = args.batch_size or int(cfg["training"].get("cached_batch_size", 128))
    eval_batch_size = args.eval_batch_size or int(cfg["training"].get("cached_batch_size", 128))
    max_epochs = args.max_epochs or int(cfg["training"]["epochs"])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, collate_fn=cached_collate, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=eval_batch_size, shuffle=False,
        num_workers=4, collate_fn=cached_collate, pin_memory=True, persistent_workers=True,
    )

    answer_head = build_answer_head(cfg, vocab_size).to(device)
    optimizer = build_optimizer(cfg, answer_head)

    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = steps_per_epoch * max_epochs
    scheduler = build_scheduler(cfg, optimizer, total_steps)

    print(f"[Info] Train samples: {len(train_ds)}, val: {len(val_ds)}", flush=True)
    print(f"[Info] batch_size={batch_size}, grad_accum={grad_accum}, max_epochs={max_epochs}, patience={args.patience}", flush=True)

    best_acc = None
    best_epoch = None
    history = []
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        print(f"\n[Info] ===== Epoch {epoch}/{max_epochs} =====", flush=True)

        train_metrics = train_one_epoch(
            answer_head=answer_head,
            id_to_answer=id_to_answer,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_accum_steps=grad_accum,
            device=device,
            epoch=epoch,
            log_every=args.log_every,
        )

        val_metrics = validate(
            answer_head=answer_head,
            id_to_answer=id_to_answer,
            loader=val_loader,
            device=device,
        )

        val_acc = val_metrics["vqa_accuracy"]
        print(
            f"[Info] Epoch {epoch} | "
            f"train_loss={train_metrics['loss']:.4f} train_vqa={train_metrics['vqa_accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_vqa={val_acc:.4f}",
            flush=True,
        )

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": {k: v for k, v in val_metrics.items() if k != "predictions"},
        })

        is_best = best_acc is None or val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            save_checkpoint(
                os.path.join(args.output_dir, "best_model.pt"),
                answer_head, optimizer, scheduler, epoch,
                {"train": train_metrics, "val": {k: v for k, v in val_metrics.items() if k != "predictions"}},
            )
            save_json(os.path.join(args.output_dir, "best_predictions.json"),
                      {"predictions": val_metrics["predictions"]})
            print(f"[Info] New best: epoch={epoch} val_vqa={best_acc:.4f}", flush=True)
        else:
            epochs_no_improve += 1
            print(f"[Info] No improvement for {epochs_no_improve}/{args.patience} epochs.", flush=True)
            if epochs_no_improve >= args.patience:
                print(f"[EarlyStopping] Stopping at epoch {epoch}. Best val_vqa={best_acc:.4f} at epoch {best_epoch}.", flush=True)
                save_json(os.path.join(args.output_dir, "history.json"), history)
                break

        save_json(os.path.join(args.output_dir, "history.json"), history)

    # Final metrics file
    final_metrics = {
        "experiment_name": cfg.get("experiment_name"),
        "best_epoch": best_epoch,
        "best_val_vqa_accuracy": best_acc,
        "history": history,
    }
    save_json(os.path.join(args.output_dir, "metrics.json"), final_metrics)
    print(f"\n[Done] Best val VQA accuracy: {best_acc:.4f} at epoch {best_epoch}", flush=True)


if __name__ == "__main__":
    main()
