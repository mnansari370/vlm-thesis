"""
train_dynamic.py — Full training for the dynamic visual-token-pruning model.

Trains the LlavaDynamicVQAModel's token_selector (budget controller + scorer
projection) and answer_head on top of a FROZEN LLaVA backbone. The contribution
is the adaptive per-question-type token budget K; token RANKING is CLS-attention
(scoring_mode=cls_only), matching the locked static baseline so the comparison is
apples-to-apples and isolates the budget allocation.

Primary metric = GENERATION accuracy (LLM generate(), answer head bypassed). The
answer-head classification accuracy is a diagnostic only.

Saves checkpoints in the format generate_and_score.py expects:
    {"token_selector_state_dict": ..., "answer_head_state_dict": ..., ...}

Each epoch reports:
    - train loss / ce / budget (with NaN-skip count)
    - val K distribution (mean/std/min/max) and per-type mean K
    - optional small-subset generation accuracy (the PRIMARY metric)

Usage (smoke):
    CUDA_VISIBLE_DEVICES=1 python -m VQA_V2.dynamic.train_dynamic \\
        --config VQA_V2/dynamic/llava_dynamic_150k_10k_fullvocab.yaml \\
        --output-dir VQA_V2/outputs/dynamic_smoke \\
        --max-train 300 --max-val 200 --epochs 2 --gen-eval-samples 100

Usage (full run): drop the --max-* overrides, set --epochs and --gen-eval-samples.
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

# Allow running this file directly (repo root = 2 level(s) up); `python -m` does not need this.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from VQA_V2.dynamic.llava_wrapper import LlavaDynamicVQAModel
from VQA_V2.shared.datasets import VQACollator, build_vqav2_dataset
from VQA_V2.shared.evaluation.generate_and_score import run_generation_eval, compute_mean_accuracy
from VQA_V2.shared.utils.config import load_config
from VQA_V2.shared.utils.seed import set_seed


QTYPE_NAMES = {0: "yes/no", 1: "attribute", 2: "counting", 3: "spatial"}


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def train_one_epoch(model, loader, optimizer, scheduler, grad_accum_steps, epoch, log_every):
    model.train()
    total_loss = total_ce = total_budget = 0.0
    num_samples = 0
    num_skipped_nonfinite = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        out = model(batch)
        loss = out["predictions"]["loss"]
        if loss is None:
            continue

        # Never backprop a non-finite loss (would poison every trainable weight).
        # The root cause (all-ignored CE at bs=1) is fixed in the wrapper; this is
        # defense-in-depth for any other source (e.g. fp16 overflow).
        if not torch.isfinite(loss):
            num_skipped_nonfinite += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        (loss / max(1, grad_accum_steps)).backward()

        if ((step + 1) % grad_accum_steps == 0) or ((step + 1) == len(loader)):
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        bs = len(batch["images"])
        total_loss += float(loss.item()) * bs
        total_ce += float(out["predictions"]["ce_loss"].item()) * bs
        total_budget += float(out["dynamic_losses"]["budget_loss"].item()) * bs
        num_samples += bs

        if (step + 1) % log_every == 0:
            n = max(1, num_samples)
            print(f"[Train] Epoch {epoch} step {step+1}/{len(loader)} "
                  f"loss={total_loss/n:.4f} ce={total_ce/n:.4f} budget={total_budget/n:.5f}",
                  flush=True)

    n = max(1, num_samples)
    if num_skipped_nonfinite > 0:
        print(f"[Train] Epoch {epoch}: skipped {num_skipped_nonfinite} non-finite-loss sample(s).",
              flush=True)
    return {
        "loss": total_loss / n,
        "ce_loss": total_ce / n,
        "budget_loss": total_budget / n,
        "skipped_nonfinite": num_skipped_nonfinite,
    }


@torch.no_grad()
def eval_k_distribution(model, loader, max_batches: Optional[int] = None) -> Dict[str, Any]:
    """Hard-selection eval: per-sample K and question type. Forward only (no generate)."""
    model.eval()
    k_values: List[int] = []
    qtype_k: Dict[int, List[int]] = defaultdict(list)
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        out = model(batch)
        k = int(out["token_stats"]["num_visual_tokens_after_selection"].float().mean().item())
        qt = int(out["token_stats"]["question_type_ids"].float().mean().item())
        k_values.append(k)
        qtype_k[qt].append(k)

    k_arr = np.array(k_values, dtype=float)
    res: Dict[str, Any] = {
        "k_mean": float(k_arr.mean()), "k_std": float(k_arr.std()),
        "k_min": float(k_arr.min()), "k_max": float(k_arr.max()),
        "k_median": float(np.median(k_arr)), "num_samples": len(k_values),
        "per_type": {},
    }
    for qt in [0, 1, 2, 3]:
        name = QTYPE_NAMES[qt]
        if qtype_k[qt]:
            arr = np.array(qtype_k[qt], dtype=float)
            res["per_type"][name] = {"mean_K": float(arr.mean()), "std_K": float(arr.std()),
                                     "count": len(arr)}
        else:
            res["per_type"][name] = {"mean_K": None, "std_K": None, "count": 0}
    return res


def small_generation_acc(model, loader, max_new_tokens: int, n_samples: int) -> Optional[float]:
    """Run generation on the first n_samples of the val loader (PRIMARY metric proxy)."""
    if n_samples <= 0:
        return None
    # Build a capped loader view by slicing the underlying dataset deterministically.
    ds = loader.dataset
    n = min(n_samples, len(ds))
    subset = torch.utils.data.Subset(ds, list(range(n)))
    sub_loader = DataLoader(
        subset, batch_size=loader.batch_size, shuffle=False,
        num_workers=0, collate_fn=loader.collate_fn, pin_memory=False,
    )
    preds = run_generation_eval(
        model=model, model_type="dynamic", loader=sub_loader,
        max_new_tokens=max_new_tokens, do_sample=False, log_every=max(1, n // 4),
    )
    return compute_mean_accuracy(preds)


def main():
    ap = argparse.ArgumentParser(description="Full dynamic-model training.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--gen-eval-samples", type=int, default=500,
                    help="Per-epoch generation eval subset size (0 = skip). PRIMARY metric proxy.")
    ap.add_argument("--kdist-max-samples", type=int, default=2000,
                    help="Cap val samples for the (cheap) per-epoch K-distribution eval.")
    ap.add_argument("--num-workers", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.max_train is not None:
        cfg["dataset"]["max_samples"] = args.max_train
    if args.max_val is not None:
        cfg["dataset"]["max_val_samples"] = args.max_val
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.num_workers is not None:
        cfg["dataset"]["num_workers"] = args.num_workers

    set_seed(int(cfg.get("seed", 42)), False)
    os.makedirs(args.output_dir, exist_ok=True)
    save_json(cfg, os.path.join(args.output_dir, "config.json"))

    print("[Train] Loading dynamic model...", flush=True)
    model = LlavaDynamicVQAModel(cfg)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    tcfg, tk = cfg["training"], cfg["token_selection"]
    print(f"[Train] Trainable params: {n_trainable:,}", flush=True)
    print(f"[Train] scoring_mode={tk.get('scoring_mode')} budget_strategy={tk.get('budget_strategy')} "
          f"min_keep={tk.get('min_keep_tokens')} max_keep={tk.get('max_keep_tokens')}", flush=True)
    print(f"[Train] budget_loss_type={tcfg.get('budget_loss_type')} "
          f"budget_loss_weight={tcfg.get('budget_loss_weight')} "
          f"grad_through_lm={tcfg.get('enable_selector_grad_through_lm')}", flush=True)

    print("[Train] Loading datasets...", flush=True)
    train_ds = build_vqav2_dataset(cfg, "train")
    val_ds = build_vqav2_dataset(cfg, "val")
    num_workers = int(cfg["dataset"].get("num_workers", 2))
    collate = VQACollator()
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=num_workers,
                              collate_fn=collate, pin_memory=False,
                              persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers,
                            collate_fn=collate, pin_memory=False,
                            persistent_workers=num_workers > 0)

    epochs = int(cfg["training"]["epochs"])
    lr = float(cfg["training"]["learning_rate"])
    wd = float(cfg["training"]["weight_decay"])
    grad_accum = int(cfg["training"].get("grad_accum_steps", 1))
    max_new_tokens = int(cfg["model"].get("generation_max_new_tokens", 10))

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)
    total_steps = math.ceil(len(train_loader) / grad_accum) * epochs
    warmup = max(1, int(total_steps * float(cfg.get("scheduler", {}).get("warmup_ratio", 0.1))))
    min_lr = float(cfg.get("scheduler", {}).get("min_lr", 1e-6))

    def lr_lambda(step):
        if step < warmup:
            return float(step) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return max(min_lr / lr, 0.5 * (1.0 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = []
    best_metric = -1.0
    best_path = os.path.join(args.output_dir, "best_model.pt")

    for epoch in range(1, epochs + 1):
        print(f"\n[Train] ===== Epoch {epoch}/{epochs} =====", flush=True)
        tr = train_one_epoch(model, train_loader, optimizer, scheduler, grad_accum,
                             epoch, args.log_every)

        kd = eval_k_distribution(model, val_loader, max_batches=args.kdist_max_samples)
        gen_acc = small_generation_acc(model, val_loader, max_new_tokens, args.gen_eval_samples)

        pt = kd["per_type"]
        def _mk(name):
            v = pt[name]["mean_K"]
            return f"{v:.0f}" if v is not None else "NA"
        print(f"[Train] Epoch {epoch} | loss={tr['loss']:.4f} ce={tr['ce_loss']:.4f} "
              f"budget={tr['budget_loss']:.5f}", flush=True)
        print(f"[Val]   K mean={kd['k_mean']:.1f} std={kd['k_std']:.1f} "
              f"[{int(kd['k_min'])},{int(kd['k_max'])}] | per-type K: "
              f"yes/no={_mk('yes/no')} attr={_mk('attribute')} "
              f"count={_mk('counting')} spatial={_mk('spatial')}", flush=True)
        if gen_acc is not None:
            print(f"[Val]   GENERATION acc (n={args.gen_eval_samples}) = "
                  f"{gen_acc:.4f} ({gen_acc*100:.2f}%)  <-- PRIMARY", flush=True)

        rec = {"epoch": epoch, "train": tr, "val_k_dist": kd,
               "val_gen_acc_subset": gen_acc}
        history.append(rec)
        save_json({"history": history}, os.path.join(args.output_dir, "history.json"))

        # Track best by generation acc if available, else by inverse budget loss.
        cur = gen_acc if gen_acc is not None else -tr["budget_loss"]
        if cur > best_metric:
            best_metric = cur
            torch.save({
                "epoch": epoch,
                "token_selector_state_dict": model.token_selector.state_dict(),
                "answer_head_state_dict": model.answer_head.state_dict(),
                "val_gen_acc_subset": gen_acc,
                "val_k_dist": kd,
                "config": cfg,
            }, best_path)
            print(f"[Train] New best (metric={cur:.4f}) -> saved {best_path}", flush=True)

    # Always save a final checkpoint too.
    final_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save({
        "epoch": epochs,
        "token_selector_state_dict": model.token_selector.state_dict(),
        "answer_head_state_dict": model.answer_head.state_dict(),
        "config": cfg,
    }, final_path)
    print(f"\n[Train] Final checkpoint -> {final_path}", flush=True)
    print(f"[Train] Best checkpoint  -> {best_path} (metric={best_metric:.4f})", flush=True)
    print("[Train] DONE.", flush=True)


if __name__ == "__main__":
    main()
