"""
budget_variance_gate.py — Dynamic model K-variance smoke test.

Trains the dynamic model (frozen backbone, trains token_selector + answer_head)
on a small dataset subset and reports the predicted-K distribution per question type.

PASS condition: std(K) > PASS_STD_THRESHOLD tokens across the eval set.
FAIL condition: near-constant K (std < threshold) — increase budget_loss_weight or retrain.

NOTE on diversity loss: with batch_size=1, keep_ratio.std() = 0 (std of a scalar is 0),
so budget_diversity_loss is inactive. K-variance is driven exclusively by the
question_type_target budget_loss (MSE between predicted keep_ratio and per-type targets).
This is expected and sufficient for the gate test. Document in paper.

Usage (smoke test, 200 samples, GPU1):
    CUDA_VISIBLE_DEVICES=1 python VQA_V2/shared/scripts/budget_variance_gate.py \\
        --config VQA_V2/dynamic/llava_dynamic_gate_smoke.yaml \\
        --output-dir VQA_V2/outputs/gate_smoke_v1 \\
        --max-train 200 \\
        --max-val 200 \\
        --epochs 5
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

# Allow running this file directly (repo root = 3 level(s) up); `python -m` does not need this.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from VQA_V2.dynamic.llava_wrapper import LlavaDynamicVQAModel
from VQA_V2.dynamic.budget_controller import BudgetController
from VQA_V2.shared.datasets.vqav2 import build_vqav2_dataset
from VQA_V2.shared.datasets.vqav2_answers import normalize_answer
from VQA_V2.shared.utils.config import load_config
from VQA_V2.shared.utils.seed import set_seed


QTYPE_NAMES = {0: "yes/no", 1: "attribute", 2: "counting", 3: "spatial"}
PASS_STD_THRESHOLD = 20  # tokens — std(K) must exceed this to PASS


def gate_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate for LlavaDynamicVQAModel.forward(): images as list, labels stacked."""
    return {
        "images": [b["image"] for b in batch],
        "questions": [b["question"] for b in batch],
        "answer_labels": torch.tensor([b["answer_label"] for b in batch], dtype=torch.long),
        "raw_answers": [b["raw_answers"] for b in batch],
        "question_ids": [b["question_id"] for b in batch],
        "image_ids": [b["image_id"] for b in batch],
    }


def vqa_consensus_score(pred: str, raw_answers: List[str]) -> float:
    pred_norm = normalize_answer(pred)
    matches = sum(1 for a in raw_answers if normalize_answer(a) == pred_norm)
    return min(1.0, matches / 3.0)


def train_one_epoch(
    model: LlavaDynamicVQAModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    grad_accum_steps: int,
    epoch: int,
    log_every: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_budget = 0.0
    num_samples = 0
    num_skipped_nonfinite = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        out = model(batch)
        loss = out["predictions"]["loss"]
        if loss is None:
            continue

        # Defense-in-depth: never backprop a non-finite loss (it would poison every
        # trainable weight). The root NaN cause (all-ignored CE at batch_size=1) is
        # fixed in the wrapper; this catches any other source (e.g. fp16 overflow).
        if not torch.isfinite(loss):
            num_skipped_nonfinite += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        (loss / max(1, grad_accum_steps)).backward()

        if ((step + 1) % grad_accum_steps == 0) or ((step + 1) == len(loader)):
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
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
            print(
                f"[Train] Epoch {epoch} step {step+1}/{len(loader)} "
                f"loss={total_loss/n:.4f} ce={total_ce/n:.4f} budget={total_budget/n:.4f}",
                flush=True,
            )

    n = max(1, num_samples)
    if num_skipped_nonfinite > 0:
        print(f"[Train] Epoch {epoch}: skipped {num_skipped_nonfinite} non-finite-loss "
              f"sample(s) (no backprop).", flush=True)
    return {
        "loss": total_loss / n,
        "ce_loss": total_ce / n,
        "budget_loss": total_budget / n,
        "skipped_nonfinite": num_skipped_nonfinite,
    }


@torch.no_grad()
def eval_k_distribution(
    model: LlavaDynamicVQAModel,
    loader: DataLoader,
    id_to_answer: Optional[Dict[int, str]],
    log_every: int = 50,
) -> Dict[str, Any]:
    """Hard-selection eval: collect K and question type per sample."""
    model.eval()
    k_values: List[int] = []
    qtype_k: Dict[int, List[int]] = defaultdict(list)
    acc_scores: List[float] = []

    for i, batch in enumerate(loader):
        out = model(batch)
        k = int(out["token_stats"]["num_visual_tokens_after_selection"].item())
        qtype = int(out["token_stats"]["question_type_ids"].item())
        k_values.append(k)
        qtype_k[qtype].append(k)

        if id_to_answer is not None:
            pred_id = int(out["predictions"]["pred_answer_ids"].item())
            pred_str = id_to_answer.get(pred_id, "")
            raw = batch["raw_answers"][0]
            acc_scores.append(vqa_consensus_score(pred_str, raw))

        if (i + 1) % log_every == 0:
            print(f"[Eval] {i+1}/{len(loader)}", flush=True)

    k_arr = np.array(k_values, dtype=float)
    result: Dict[str, Any] = {
        "k_mean": float(k_arr.mean()),
        "k_std": float(k_arr.std()),
        "k_min": float(k_arr.min()),
        "k_max": float(k_arr.max()),
        "k_median": float(np.median(k_arr)),
        "num_samples": len(k_values),
        "per_type": {},
    }

    for qt in [0, 1, 2, 3]:
        name = QTYPE_NAMES[qt]
        if qtype_k[qt]:
            arr = np.array(qtype_k[qt], dtype=float)
            result["per_type"][name] = {
                "mean_K": float(arr.mean()),
                "std_K": float(arr.std()),
                "count": len(arr),
            }
        else:
            result["per_type"][name] = {"mean_K": None, "std_K": None, "count": 0}

    if acc_scores:
        result["classification_vqa_acc"] = float(np.mean(acc_scores))

    return result


def main():
    parser = argparse.ArgumentParser(description="Budget-variance gate smoke test.")
    parser.add_argument("--config", required=True, help="Dynamic model YAML config.")
    parser.add_argument("--output-dir", required=True, help="Where to write gate_results.json.")
    parser.add_argument("--max-train", type=int, default=None,
                        help="Override dataset.max_samples (train subset for smoke test).")
    parser.add_argument("--max-val", type=int, default=None,
                        help="Override dataset.max_val_samples (val subset for eval).")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training.epochs.")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--zero-qproj-in-budget", action="store_true",
                        help="Zero out question_projected before the budget controller. "
                             "Forces the controller to use ONLY score_stats + question_type_ids. "
                             "Use for the gate test to verify the type-conditioning mechanism "
                             "without memorization of question embeddings.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.max_train is not None:
        cfg["dataset"]["max_samples"] = args.max_train
    if args.max_val is not None:
        cfg["dataset"]["max_val_samples"] = args.max_val
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    set_seed(int(cfg.get("seed", 42)), False)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    print("[Gate] Loading dynamic model...", flush=True)
    model = LlavaDynamicVQAModel(cfg)
    id_to_answer = model.id_to_answer

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"[Gate] Trainable params: {n_trainable:,}", flush=True)
    print(f"[Gate] max_keep_tokens={cfg['token_selection']['max_keep_tokens']}, "
          f"min_keep_tokens={cfg['token_selection']['min_keep_tokens']}", flush=True)
    print(f"[Gate] budget_loss_weight={cfg['training']['budget_loss_weight']}, "
          f"budget_diversity_weight={cfg['training']['budget_diversity_weight']}", flush=True)
    print(f"[Gate] NOTE: diversity_loss=0 with batch_size=1 (std of scalar = 0). "
          f"K-variance driven by question_type_target budget_loss.", flush=True)

    if args.zero_qproj_in_budget:
        _orig_bc_forward = model.token_selector.budget_controller.forward
        def _patched_bc_forward(question_projected, score_stats, question_type_ids=None):
            return _orig_bc_forward(
                torch.zeros_like(question_projected),
                torch.zeros_like(score_stats),
                question_type_ids,
            )
        model.token_selector.budget_controller.forward = _patched_bc_forward
        print("[Gate] PATCHED: question_projected AND score_stats zeroed in budget_controller. "
              "Controller uses ONLY question_type_ids. "
              "Tests pure type→K mapping — no memorization possible.", flush=True)

    print("[Gate] Loading datasets...", flush=True)
    train_ds = build_vqav2_dataset(cfg, "train")
    val_ds = build_vqav2_dataset(cfg, "val")

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        num_workers=2, collate_fn=gate_collate, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=2, collate_fn=gate_collate, pin_memory=False,
    )

    epochs = int(cfg["training"]["epochs"])
    lr = float(cfg["training"]["learning_rate"])
    weight_decay = float(cfg["training"]["weight_decay"])
    grad_accum = int(cfg["training"].get("grad_accum_steps", 1))

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    total_steps = math.ceil(len(train_loader) / grad_accum) * epochs
    warmup_ratio = float(cfg.get("scheduler", {}).get("warmup_ratio", 0.05))
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    min_lr = float(cfg.get("scheduler", {}).get("min_lr", 1e-7))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr / lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = []
    for epoch in range(1, epochs + 1):
        print(f"\n[Gate] ===== Train epoch {epoch}/{epochs} =====", flush=True)
        m = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_accum_steps=grad_accum,
            epoch=epoch,
            log_every=args.log_every,
        )
        print(
            f"[Gate] Epoch {epoch} | loss={m['loss']:.4f} "
            f"ce={m['ce_loss']:.4f} budget={m['budget_loss']:.4f}",
            flush=True,
        )
        history.append({"epoch": epoch, **m})

    # Post-training type sweep: verify budget_controller produces different K per type.
    # Calls budget_controller directly with zeroed qproj/stats and each type_id.
    print("\n[Gate] ===== Post-training type-sweep (direct BC call, eval mode) =====",
          flush=True)
    bc = model.token_selector.budget_controller
    bc.eval()
    type_targets = [0.38, 0.48, 0.58, 0.62]
    with torch.no_grad():
        try:
            bc_device = next(bc.parameters()).device
            bc_dtype = bc.net[1].weight.dtype
            q_dim = int(cfg["token_selection"].get("shared_dim", 512))
            qproj_zeros = torch.zeros(1, q_dim, device=bc_device, dtype=bc_dtype)
            stats_zeros = torch.zeros(1, 7, device=bc_device, dtype=bc_dtype)
            sweep_krs = []
            for t in range(4):
                qtype_t = torch.tensor([t], dtype=torch.long, device=bc_device)
                # Call budget_controller directly with zeroed inputs (original unpatched method).
                out_t = BudgetController.forward(bc, qproj_zeros, stats_zeros, qtype_t)
                kr = float(out_t["keep_ratio"].item())
                k_t = round(kr * int(cfg["token_selection"]["num_visual_tokens"]))
                target_k = round(type_targets[t] * int(cfg["token_selection"]["num_visual_tokens"]))
                print(
                    f"  Type {t} ({QTYPE_NAMES[t]:10s}): keep_ratio={kr:.4f}  K={k_t:3d} "
                    f"(target K={target_k}  delta={k_t-target_k:+d})",
                    flush=True,
                )
                sweep_krs.append(kr)
            sweep_std = float(np.std([round(kr * int(cfg["token_selection"]["num_visual_tokens"])) for kr in sweep_krs]))
            print(f"  std(K) across 4 types = {sweep_std:.1f}", flush=True)
        except Exception as e:
            print(f"  [TypeSweep failed: {e}]", flush=True)

    print(f"\n[Gate] ===== K-distribution eval ({len(val_ds)} samples, hard selection) =====",
          flush=True)
    k_stats = eval_k_distribution(model, val_loader, id_to_answer, log_every=args.log_every)

    # ── Report ─────────────────────────────────────────────────────────────
    sep = "=" * 62
    print(f"\n{sep}", flush=True)
    print("  GATE SMOKE TEST — K-DISTRIBUTION REPORT", flush=True)
    print(sep, flush=True)
    print(f"  Samples evaluated : {k_stats['num_samples']}", flush=True)
    print(f"  K  mean={k_stats['k_mean']:.1f}  std={k_stats['k_std']:.1f}  "
          f"min={int(k_stats['k_min'])}  max={int(k_stats['k_max'])}  "
          f"median={k_stats['k_median']:.1f}", flush=True)
    print(f"  max_keep_tokens   : {cfg['token_selection']['max_keep_tokens']}", flush=True)
    if "classification_vqa_acc" in k_stats:
        print(f"  Classification acc: {k_stats['classification_vqa_acc']:.4f} "
              f"(untrained head — low is expected)", flush=True)

    print("\n  Per question type (mean K ± std):", flush=True)
    print(f"  {'Type':12s}  {'mean K':>8s}  {'std K':>7s}  {'count':>6s}", flush=True)
    for qt in [0, 1, 2, 3]:
        name = QTYPE_NAMES[qt]
        stats = k_stats["per_type"][name]
        if stats["count"] > 0:
            print(
                f"  {name:12s}  {stats['mean_K']:>8.1f}  {stats['std_K']:>7.1f}  {stats['count']:>6d}",
                flush=True,
            )
        else:
            print(f"  {name:12s}  {'N/A':>8s}  {'N/A':>7s}  {0:>6d}", flush=True)

    # Loss progression
    if len(history) >= 2:
        print(f"\n  Loss trajectory:", flush=True)
        for h in history:
            print(f"    epoch {h['epoch']:2d}: loss={h['loss']:.4f} "
                  f"ce={h['ce_loss']:.4f} budget={h['budget_loss']:.4f}", flush=True)
        loss_decreased = history[-1]["loss"] < history[0]["loss"]
        print(f"  Loss decreasing: {'YES' if loss_decreased else 'NO — check LR/optimizer'}", flush=True)
    else:
        loss_decreased = True

    # PASS / FAIL
    k_var_pass = k_stats["k_std"] > PASS_STD_THRESHOLD
    print(f"\n  std(K) = {k_stats['k_std']:.1f}  threshold = {PASS_STD_THRESHOLD}", flush=True)
    if k_var_pass:
        print("  RESULT: PASS — non-trivial K-variance observed.", flush=True)
    else:
        print("  RESULT: FAIL — K near-constant.", flush=True)
        print("  → Increase budget_loss_weight (try 1.0) and retrain.", flush=True)
        print("  → Or investigate whether question_type_target ratios are reachable.", flush=True)
    print(sep + "\n", flush=True)

    passed = k_var_pass and (loss_decreased if len(history) >= 2 else True)

    results = {
        "passed": passed,
        "k_stats": k_stats,
        "history": history,
        "gate_config": {
            "max_keep_tokens": cfg["token_selection"]["max_keep_tokens"],
            "min_keep_tokens": cfg["token_selection"]["min_keep_tokens"],
            "budget_loss_weight": cfg["training"]["budget_loss_weight"],
            "budget_diversity_weight": cfg["training"]["budget_diversity_weight"],
            "budget_loss_type": cfg["training"]["budget_loss_type"],
            "question_type_target_ratios": cfg["training"]["question_type_target_ratios"],
            "pass_std_threshold": PASS_STD_THRESHOLD,
            "zero_qproj_in_budget": args.zero_qproj_in_budget,
            "note": "diversity_loss=0 with batch_size=1; K-variance from question_type_target",
        },
    }

    out_path = os.path.join(args.output_dir, "gate_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"[Gate] Results saved → {out_path}", flush=True)
    print("[Gate] DONE. STOP. Await human authorization before full gate run.", flush=True)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
