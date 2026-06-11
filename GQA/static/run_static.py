"""
Static visual token pruning evaluation on GQA val_balanced (full 132,062 questions).

Methods evaluated
-----------------
  none            — No pruning (K=576). Sanity check; must reproduce ~67.73%.
  spatial_uniform — Deterministic stride-based 24×24 grid subsampling.
  random          — Uniform random K-subset, seeded per sample.
  cls_attn        — VisionZip Dominant: CLS-to-patch CLIP attention, top-K.
  l2_norm         — L2 norm of CLIP hidden states, top-K.
  fastv_style     — LLM-layer-2 received attention (eager mode, for ablation).

Smoke-test first (--max_samples 500 ≈ 3 min), then launch full runs.

Usage
-----
# Sanity check (must match ~67.73% dense baseline):
python -m GQA.static.run_static \\
    --method none --keep_k 576 --max_samples 500 \\
    --output_name smoke_none_576

# Smoke tests:
python -m GQA.static.run_static --method cls_attn  --keep_k 288 --max_samples 500
python -m GQA.static.run_static --method random    --keep_k 288 --max_samples 500
python -m GQA.static.run_static --method spatial_uniform --keep_k 288 --max_samples 500
python -m GQA.static.run_static --method l2_norm   --keep_k 288 --max_samples 500

# Full run (example — 3.5 h on one RTX 6000 Ada):
python -m GQA.static.run_static \\
    --method cls_attn --keep_k 288 \\
    --output_name paper_static_cls_attn_k288

# Resume a crashed run:
python -m GQA.static.run_static \\
    --method cls_attn --keep_k 288 \\
    --output_name paper_static_cls_attn_k288 \\
    --resume

Expected smoke-test ranges (500-sample subset, ±noise):
  none       K=576 : 65–71%   (must match zero-shot; any large gap = bug)
  cls_attn   K=288 : 63–67%
  cls_attn   K=144 : 60–64%
  random     K=288 : 56–62%
  spatial_u  K=288 : 60–64%
  l2_norm    K=288 : 58–63%

If cls_attn K=288 returns > 67% on 500 samples, verify the masked-position
count in the output log before assuming it is correct.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from torch.utils.data import DataLoader

from GQA.shared.dataset import GQAValDataset, collate_val
from GQA.shared.static import SPATIAL_INDICES, StaticPrunedLlava
from GQA.shared.flops import flops_row, print_flops_table
from GQA.shared.metrics import (
    compute_accuracy,
    gqa_exact_match,
    print_accuracy_table,
)
from GQA.shared.utils.logger import make_output_dir


# ── checkpoint helpers ────────────────────────────────────────────────────────

def _ckpt_path(out_dir: str) -> str:
    return os.path.join(out_dir, "predictions_partial.json")


def load_checkpoint(out_dir: str) -> tuple[list[dict], set[str]]:
    path = _ckpt_path(out_dir)
    if not os.path.exists(path):
        return [], set()
    with open(path) as f:
        data = json.load(f)
    preds = data.get("predictions", [])
    done  = {str(p["question_id"]) for p in preds}
    print(f"[Resume] Loaded {len(preds):,} predictions from checkpoint.", flush=True)
    return preds, done


def save_checkpoint(out_dir: str, predictions: list[dict]) -> None:
    with open(_ckpt_path(out_dir), "w") as f:
        json.dump({"predictions": predictions}, f)


# ── evaluation loop ───────────────────────────────────────────────────────────

def evaluate(
    model:              StaticPrunedLlava,
    loader:             DataLoader,
    out_dir:            str,
    save_every:         int,
    resume_predictions: list[dict],
    done_qids:          set[str],
    log_every:          int,
    method:             str,
    keep_k:             int,
) -> list[dict]:
    predictions = list(resume_predictions)
    correct_so_far = sum(
        1 for p in predictions
        if gqa_exact_match(p.get("pred_answer"), p.get("answer"))
    )
    total_so_far  = len(predictions)
    n_steps       = len(loader)
    sample_offset = total_so_far   # global index of the next un-processed sample
    t0            = time.time()

    for step, batch in enumerate(loader):
        qids = [str(q) for q in batch["question_ids"]]

        # Skip batches already done (resume mode)
        if done_qids and all(q in done_qids for q in qids):
            sample_offset += len(qids)
            continue

        preds = model.generate_answers(
            images        = batch["images"],
            questions     = batch["questions"],
            sample_offset = sample_offset,
        )

        for i, (pred, gold) in enumerate(zip(preds, batch["answers"])):
            qid = str(batch["question_ids"][i])
            if qid in done_qids:
                continue
            is_ok = gqa_exact_match(pred, gold)
            correct_so_far += int(is_ok)
            total_so_far   += 1
            done_qids.add(qid)
            predictions.append({
                "question_id":   qid,
                "image_id":      batch["image_ids"][i],
                "question":      batch["questions"][i],
                "pred_answer":   pred,
                "answer":        gold,
                "semantic_type": batch["semantic_types"][i],
                "correct":       is_ok,
            })

        sample_offset += len(qids)

        if (step + 1) % save_every == 0:
            save_checkpoint(out_dir, predictions)

        if (step + 1) % log_every == 0:
            acc     = correct_so_far / max(total_so_far, 1)
            elapsed = time.time() - t0
            sps     = (step + 1) / max(elapsed, 1e-6)
            eta_h   = (n_steps - step - 1) / max(sps, 1e-6) / 3600
            print(
                f"[Eval] {step+1:>6}/{n_steps}  "
                f"acc={acc*100:.2f}%  "
                f"method={method}  K={keep_k}  "
                f"n={total_so_far:>8,}  "
                f"speed={sps:.2f} batch/s  "
                f"ETA={eta_h:.1f}h",
                flush=True,
            )

    save_checkpoint(out_dir, predictions)
    return predictions


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Static visual token pruning evaluation on GQA val_balanced."
    )

    # Data
    p.add_argument("--questions_path", default="./data/gqa/val_balanced_questions.json",
                   help="Path to GQA questions JSON (val or train_balanced).")
    p.add_argument("--image_dir",      default="./data/gqa/images/images")
    p.add_argument("--max_samples",    type=int, default=None,
                   help="Cap dataset for smoke tests (e.g. 500).")
    p.add_argument("--question_ids_file", default=None,
                   help="JSON file containing a list of question IDs to evaluate on. "
                        "Used for oracle training runs on train_balanced subsets.")

    # Method
    p.add_argument(
        "--method",
        required=True,
        choices=["none", "random", "spatial_uniform", "cls_attn",
                 "l2_norm", "fastv_style"],
        help="Token selection strategy.",
    )
    p.add_argument(
        "--keep_k",
        required=True,
        type=int,
        choices=[576, 432, 288, 192, 144],
        help="Number of visual tokens to keep.",
    )
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for 'random' method.")

    # Output
    p.add_argument("--output_name", default=None,
                   help="Experiment name (auto-generated if omitted).")
    p.add_argument("--resume",      action="store_true",
                   help="Resume from a partial checkpoint in output_name dir.")

    # Loader
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--save_every",  type=int, default=500,
                   help="Flush checkpoint every N batches.")
    p.add_argument("--log_every",   type=int, default=200,
                   help="Print progress every N batches.")

    args = p.parse_args()

    # Validate none/K combo
    if args.method == "none" and args.keep_k != 576:
        p.error("--method none requires --keep_k 576")

    # Auto-generate output name if not given
    if args.output_name is None:
        args.output_name = (
            f"paper_static_{args.method}_k{args.keep_k}"
            + (f"_seed{args.seed}" if args.method == "random" else "")
            + (f"_n{args.max_samples}" if args.max_samples else "")
        )

    out_dir = make_output_dir(base_dir="outputs", experiment_name=args.output_name)
    print(f"[Output] {out_dir}", flush=True)

    # Persist config
    cfg = vars(args)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # Load optional question_ids filter (for oracle training runs)
    question_ids = None
    if args.question_ids_file:
        with open(args.question_ids_file) as f:
            question_ids = json.load(f)
        print(f"[Data]   Filtering to {len(question_ids):,} question IDs "
              f"from {args.question_ids_file}", flush=True)

    # Dataset
    dataset = GQAValDataset(
        questions_path = args.questions_path,
        image_dir      = args.image_dir,
        max_samples    = args.max_samples,
        seed           = args.seed,
        question_ids   = question_ids,
    )
    loader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        collate_fn  = collate_val,
        pin_memory  = torch.cuda.is_available(),
        persistent_workers = (args.num_workers > 0),
    )
    print(
        f"[Data]   {len(dataset):,} samples, "
        f"{len(loader):,} batches @ bs={args.batch_size}",
        flush=True,
    )

    # Spatial-uniform: log index counts as early validation
    if args.method == "spatial_uniform":
        k = args.keep_k
        n = len(SPATIAL_INDICES[k])
        print(f"[Method] spatial_uniform K={k}: {n} pre-computed indices", flush=True)
        assert n == k, f"BUG: spatial_uniform K={k} has {n} indices (expected {k})"

    # Resume
    resume_preds, done_qids = [], set()
    if args.resume:
        resume_preds, done_qids = load_checkpoint(out_dir)

    # Model
    model = StaticPrunedLlava(
        method = args.method,
        keep_k = args.keep_k,
        seed   = args.seed,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    t_start = time.time()
    print(f"\n[Eval] Starting — method={args.method}, K={args.keep_k} ...", flush=True)

    predictions = evaluate(
        model              = model,
        loader             = loader,
        out_dir            = out_dir,
        save_every         = args.save_every,
        resume_predictions = resume_preds,
        done_qids          = done_qids,
        log_every          = args.log_every,
        method             = args.method,
        keep_k             = args.keep_k,
    )
    elapsed_h = (time.time() - t_start) / 3600

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_accuracy(predictions)
    label   = f"{args.method.upper()} K={args.keep_k}"
    print_accuracy_table(metrics, label=label)

    # ── FLOPs ─────────────────────────────────────────────────────────────────
    print("\n[FLOPs] Paper table:")
    print_flops_table([576, 432, 288, 192, 144])
    flops = flops_row(args.keep_k, label=label)

    # ── Smoke-test advisory ───────────────────────────────────────────────────
    acc = metrics["gqa_accuracy"]
    if args.max_samples:
        # Calibrated for physical-removal pipeline on 500-sample subset.
        # Physical removal gives higher accuracy than attention-masking because
        # the LM processes a clean shorter sequence with no masked-but-present tokens.
        expected_ranges = {
            ("none",            576): (0.65, 0.71),
            ("cls_attn",        576): (0.65, 0.71),
            ("cls_attn",        432): (0.66, 0.70),
            ("cls_attn",        288): (0.63, 0.69),
            ("cls_attn",        192): (0.61, 0.68),
            ("cls_attn",        144): (0.58, 0.66),
            ("random",          432): (0.66, 0.70),
            ("random",          288): (0.63, 0.68),
            ("random",          192): (0.60, 0.67),
            ("random",          144): (0.55, 0.65),
            ("spatial_uniform", 432): (0.66, 0.70),
            ("spatial_uniform", 288): (0.63, 0.69),
            ("spatial_uniform", 192): (0.61, 0.68),
            ("spatial_uniform", 144): (0.57, 0.66),
            ("l2_norm",         432): (0.65, 0.70),
            ("l2_norm",         288): (0.63, 0.68),
            ("l2_norm",         192): (0.60, 0.67),
            ("l2_norm",         144): (0.56, 0.65),
            ("fastv_style",     288): (0.58, 0.68),
            ("fastv_style",     144): (0.52, 0.65),
        }
        lo, hi = expected_ranges.get((args.method, args.keep_k), (0.0, 1.0))
        if lo <= acc <= hi:
            print(f"\n[Smoke] PASS — acc={acc*100:.2f}% in expected range "
                  f"[{lo*100:.0f}%, {hi*100:.0f}%]", flush=True)
        elif acc > hi:
            print(f"\n[Smoke] WARN — acc={acc*100:.2f}% ABOVE expected "
                  f"[{lo*100:.0f}%, {hi*100:.0f}%]. "
                  "Verify masked-position count in log before launching full run.",
                  flush=True)
        else:
            print(f"\n[Smoke] FAIL — acc={acc*100:.2f}% BELOW expected "
                  f"[{lo*100:.0f}%, {hi*100:.0f}%]. "
                  "Debug image-token positions / attention layer before full run.",
                  flush=True)

    # ── Save results ──────────────────────────────────────────────────────────
    result = {
        "experiment_name":  args.output_name,
        "model":            "llava-hf/llava-1.5-7b-hf",
        "method":           args.method,
        "keep_k":           args.keep_k,
        "seed":             args.seed,
        "n_evaluated":      metrics["n_evaluated"],
        "n_correct":        metrics["n_correct"],
        "gqa_accuracy":     metrics["gqa_accuracy"],
        "per_type":         metrics["per_type"],
        "flops":            flops,
        "elapsed_hours":    round(elapsed_h, 2),
        "batch_size":       args.batch_size,
        "prompt_suffix":    " Answer with one word or a short phrase only.",
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(out_dir, "predictions.json"), "w") as f:
        json.dump({"predictions": predictions}, f)

    print(
        f"\n[Done] {label}  "
        f"acc={acc*100:.2f}%  "
        f"({metrics['n_correct']:,}/{metrics['n_evaluated']:,})  "
        f"K={args.keep_k}  "
        f"FastV-full={flops['fastv_full_TFLOPs']:.4f}T ({flops['fastv_full_reduction_pct']:.1f}% reduction)  "
        f"Attn-only={flops['attention_only_GFLOPs']:.1f}G ({flops['attention_only_reduction_pct']:.1f}% reduction)  "
        f"elapsed={elapsed_h:.2f}h"
    )
    print(f"[Done] Output: {out_dir}")


if __name__ == "__main__":
    main()
