"""
train_stage1.py — Stage-1 elastic-backbone training.

Feeds the LLaVA-665K mix through ElasticPrunedLlava with a RANDOM K each step,
trains ONLY the LoRA adapters + projector (61M params) on the teacher-forced answer
loss, and checkpoints just those trainable params.

This is the run that unlocks the dense ceiling, the static frontier, AND the pilot
measurement — all from one model.

Usage (after the data download finishes):
    CUDA_VISIBLE_DEVICES=0 python -m v2.training.train_stage1 \
        --config v2/configs/stage1_elastic.yaml \
        --output-dir v2/outputs/stage1_elastic_run

Quick subset run:
    ... --max-train 20000 --epochs 1 --batch-size 4

The training loop is factored into run_training(cfg, output_dir) so the offline
smoke test (test_train_stage1.py) can drive it on tiny synthetic data.
"""

import argparse
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from peft import get_peft_model_state_dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from v2.data_setup.mix_dataset import LlavaMixDataset, mix_collate
from v2.model.elastic_wrapper import ElasticPrunedLlava
from v2.shared.utils.config import load_config
from v2.shared.utils.seed import set_seed


# ── build helpers ─────────────────────────────────────────────────────────────

def build_model(cfg: Dict[str, Any]) -> ElasticPrunedLlava:
    m = cfg["model"]
    return ElasticPrunedLlava(
        model_name=m.get("model_name", "llava-hf/llava-1.5-7b-hf"),
        lora_rank=int(m.get("lora_rank", 16)),
        lora_alpha=int(m.get("lora_alpha", 32)),
        lora_dropout=float(m.get("lora_dropout", 0.05)),
        dtype=str(m.get("dtype", "bfloat16")),
        image_pad=bool(m.get("image_pad", True)),
        k_ladder=tuple(m.get("k_ladder", (32, 64, 128, 256, 576))),
        seed=int(cfg.get("seed", 42)),
        gradient_checkpointing=bool(m.get("gradient_checkpointing", False)),
    )


def build_loader(cfg: Dict[str, Any], batch_size: int) -> DataLoader:
    d = cfg["data"]
    ds = LlavaMixDataset(
        json_path=d["json_path"],
        image_root=d["image_root"],
        turn_mode=str(d.get("turn_mode", "first")),
        max_samples=d.get("max_samples"),
        stratify=True,
        verify_images=bool(d.get("verify_images", False)),
        drop_missing=bool(d.get("drop_missing", True)),
        seed=int(cfg.get("seed", 42)),
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=int(d.get("num_workers", 4)), collate_fn=mix_collate,
        pin_memory=False, persistent_workers=int(d.get("num_workers", 4)) > 0,
        drop_last=False,
    )


def save_checkpoint(model: ElasticPrunedLlava, cfg: Dict[str, Any], step: int, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        # adapter only — save_embedding_layers=False so the frozen 131M token
        # embedding is NOT bundled in (it is identical to the base model).
        "lora_state": get_peft_model_state_dict(model.peft_lm, save_embedding_layers=False),
        "projector_state": model._proj.state_dict(),
        "step": step,
        "config": cfg,
    }, path)


# ── training loop (factored for the smoke test) ───────────────────────────────

def run_training(
    cfg: Dict[str, Any],
    output_dir: str,
    fixed_k: Optional[int] = None,
) -> Dict[str, Any]:
    set_seed(int(cfg.get("seed", 42)))
    os.makedirs(output_dir, exist_ok=True)

    t = cfg["training"]
    batch_size = int(t.get("batch_size", 4))
    grad_accum = int(t.get("grad_accum", 1))
    epochs = int(t.get("epochs", 1))
    lr = float(t.get("learning_rate", 2e-4))
    wd = float(t.get("weight_decay", 0.0))
    warmup_ratio = float(t.get("warmup_ratio", 0.03))
    max_grad_norm = float(t.get("max_grad_norm", 1.0))
    log_every = int(t.get("log_every", 50))
    save_every = int(t.get("save_every", 2000))

    loader = build_loader(cfg, batch_size)
    model = build_model(cfg)

    # train mode: LoRA + projector train (dropout on); vision tower stays eval (frozen)
    model.train()
    model._vt.eval()

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=lr, weight_decay=wd)
    steps_per_epoch = math.ceil(len(loader) / max(1, grad_accum))
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"[Stage1] {len(loader.dataset):,} examples | bs={batch_size} accum={grad_accum} "
          f"epochs={epochs} steps/epoch={steps_per_epoch} total_steps={total_steps} lr={lr}",
          flush=True)

    losses: List[float] = []
    n_skipped = 0
    opt_step = 0
    samples_seen = 0
    total_samples = len(loader.dataset) * epochs
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, epochs + 1):
        for i, batch in enumerate(loader):
            samples_seen += len(batch["images"])
            out = model.forward(batch, K=fixed_k)   # K=None → random per step
            loss = out["loss"]

            if not torch.isfinite(loss):
                n_skipped += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            (loss / grad_accum).backward()
            losses.append(float(loss.item()))

            is_boundary = ((i + 1) % grad_accum == 0) or ((i + 1) == len(loader))
            if is_boundary:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                opt_step += 1

                if opt_step % log_every == 0:
                    recent = sum(losses[-log_every:]) / min(len(losses), log_every)
                    elapsed = time.time() - t0
                    sps = samples_seen / max(elapsed, 1e-6)
                    eta_h = (total_samples - samples_seen) / max(sps, 1e-6) / 3600
                    print(f"[Stage1] epoch {epoch} opt_step {opt_step}/{total_steps} "
                          f"K={out['info']['K']} loss(recent)={recent:.4f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e} "
                          f"{sps:.1f} samp/s ETA={eta_h:.1f}h", flush=True)

                if save_every > 0 and opt_step % save_every == 0:
                    save_checkpoint(model, cfg, opt_step,
                                    os.path.join(output_dir, f"ckpt_step{opt_step}.pt"))

    final_path = os.path.join(output_dir, "final.pt")
    save_checkpoint(model, cfg, opt_step, final_path)
    if n_skipped:
        print(f"[Stage1] skipped {n_skipped} non-finite-loss step(s).", flush=True)
    print(f"[Stage1] done. final checkpoint → {final_path}", flush=True)
    return {"losses": losses, "ckpt_path": final_path, "opt_steps": opt_step, "skipped": n_skipped}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-train", type=int, default=None, help="override data.max_samples")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--fixed-k", type=int, default=None, help="debug: pin K (disable random)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.max_train is not None:  cfg["data"]["max_samples"] = args.max_train
    if args.epochs is not None:     cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None: cfg["training"]["batch_size"] = args.batch_size
    if args.grad_accum is not None: cfg["training"]["grad_accum"] = args.grad_accum
    if args.num_workers is not None: cfg["data"]["num_workers"] = args.num_workers
    if args.log_every is not None:  cfg["training"]["log_every"] = args.log_every
    if args.save_every is not None: cfg["training"]["save_every"] = args.save_every

    run_training(cfg, args.output_dir, fixed_k=args.fixed_k)


if __name__ == "__main__":
    main()
