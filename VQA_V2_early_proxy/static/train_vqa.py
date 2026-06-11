"""Training entry point for static CLS-attention pruning (60k/top-3500 era). Run: python -m VQA_V2_early_proxy.static.train_vqa --config <yaml>."""

import argparse
import json
import math
import os
from copy import deepcopy
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from VQA_V2_early_proxy.shared.datasets import VQACollator, build_vqav2_dataset
from VQA_V2_early_proxy.static import LlavaStaticVQAModel
from VQA_V2_early_proxy.static import (
    build_optimizer,
    build_scheduler,
    measure_latency,
    train_one_epoch,
    validate_one_epoch,
)
from VQA_V2_early_proxy.shared.utils.config import load_config
from VQA_V2_early_proxy.shared.utils.logger import make_output_dir
from VQA_V2_early_proxy.shared.utils.seed import set_seed


def save_json(path: str, data: Dict[str, Any]) -> None:
    """
    Save a dictionary as JSON.
    Torch tensors are converted to Python values/lists automatically.
    """

    def _convert(obj):
        if torch.is_tensor(obj):
            if obj.numel() == 1:
                return obj.detach().cpu().item()
            return obj.detach().cpu().tolist()
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_convert)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Count only parameters that require gradients."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: torch.nn.Module) -> int:
    """Count all model parameters."""
    return sum(p.numel() for p in model.parameters())


def build_loader(dataset, cfg: Dict[str, Any], split_name: str):
    """
    Build a DataLoader for the requested split.
    """
    collator = VQACollator()

    if split_name == "train":
        batch_size = int(cfg["training"]["batch_size"])
        shuffle = True
    else:
        batch_size = int(cfg["training"].get("eval_batch_size", cfg["training"]["batch_size"]))
        shuffle = False

    num_workers = int(cfg["dataset"]["num_workers"])
    pin_memory = bool(cfg["system"].get("use_cuda", True) and torch.cuda.is_available())

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )


def save_checkpoint(
    path: str,
    model,
    optimizer=None,
    scheduler=None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,
):
    """
    Save a compact checkpoint for the static pruning experiment.
    """
    checkpoint = {
        "epoch": epoch,
        "metrics": metrics,
        "experiment_name": getattr(model, "cfg", {}).get("experiment_name", None),
        "backbone_name_or_path": getattr(model, "model_cfg", {}).get(
            "pretrained_model_name_or_path", None
        ),
    }

    if getattr(model, "answer_head", None) is not None:
        checkpoint["answer_head_state_dict"] = model.answer_head.state_dict()
    else:
        checkpoint["model_state_dict"] = model.state_dict()

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)


def build_grad_scaler(use_amp: bool):
    """
    Build a GradScaler only when AMP is actually enabled.
    """
    amp_enabled = bool(use_amp and torch.cuda.is_available())
    if not amp_enabled:
        return None

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=True)

    return torch.cuda.amp.GradScaler(enabled=True)


def load_answer_head_checkpoint_if_requested(model, cfg: Dict[str, Any]) -> None:
    """
    Load a saved answer-head checkpoint for eval-only runs.

    This is needed when we want to evaluate a previously trained static baseline
    on another split, for example train60k, without retraining.
    """
    checkpoint_path = cfg.get("evaluation", {}).get("checkpoint_path", None)

    if checkpoint_path is None or str(checkpoint_path).strip() == "":
        return

    checkpoint_path = str(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"evaluation.checkpoint_path not found: {checkpoint_path}")

    print(f"[Info] Loading answer head checkpoint from: {checkpoint_path}", flush=True)

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if "answer_head_state_dict" not in ckpt:
        raise KeyError(
            f"Checkpoint does not contain 'answer_head_state_dict': {checkpoint_path}"
        )

    if getattr(model, "answer_head", None) is None:
        raise ValueError("Model has no answer_head, but checkpoint loading was requested.")

    model.answer_head.load_state_dict(ckpt["answer_head_state_dict"], strict=True)

    device = next(model.model.parameters()).device
    model.answer_head.to(device)

    print("[Info] Answer head checkpoint loaded.", flush=True)


def estimate_analytical_attention_flops(
    model: LlavaStaticVQAModel,
    metrics: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Analytical attention-only FLOPs proxy.

    This remains the main thesis-side FLOPs metric, because static pruning
    directly changes multimodal sequence length before the frozen language model.

    Approximation:
        FLOPs ~= 2 * L * S^2 * H
    """
    seq_len = metrics.get("avg_multimodal_sequence_length", None)
    if seq_len is None:
        return None

    model_config = model.model.config
    text_config = getattr(model_config, "text_config", None)

    if text_config is not None:
        hidden_size = int(getattr(text_config, "hidden_size"))
        num_layers = int(getattr(text_config, "num_hidden_layers"))
    else:
        hidden_size = int(getattr(model_config, "hidden_size"))
        num_layers = int(getattr(model_config, "num_hidden_layers"))

    seq_len = float(seq_len)
    flops = 2.0 * num_layers * (seq_len ** 2) * hidden_size

    return {
        "method": "analytical_attention_proxy_v2",
        "avg_multimodal_sequence_length": seq_len,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "attention_flops": flops,
        "attention_flops_giga": flops / 1e9,
    }


class StaticClassificationFlopsWrapper(nn.Module):
    """
    Tensor-only wrapper around the static pruning classification path for fvcore.

    It traces the actual static pipeline:
        vision tower -> CLS-attention top-k selection -> projector -> language model -> answer head
    """

    def __init__(self, static_model: LlavaStaticVQAModel):
        super().__init__()
        self.static_model = static_model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }

        lm_inputs_embeds, lm_attention_mask, _ = self.static_model._build_pruned_multimodal_inputs(
            model_inputs=model_inputs
        )

        language_model = self.static_model._get_language_model()
        lm_outputs = language_model(
            inputs_embeds=lm_inputs_embeds,
            attention_mask=lm_attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        if lm_outputs.hidden_states is None:
            raise ValueError("Language model did not return hidden states during fvcore tracing.")

        last_hidden = lm_outputs.hidden_states[-1]
        pooled_features = self.static_model._gather_last_valid_hidden(
            hidden_states=last_hidden,
            attention_mask=lm_attention_mask,
        )

        if self.static_model.answer_head is None:
            return pooled_features

        logits = self.static_model.answer_head(pooled_features)
        return logits


def _extract_single_sample_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reduce a collated batch to a single-sample batch for cheaper and more stable tracing.
    """
    return {
        "images": [batch["images"][0]],
        "questions": [batch["questions"][0]],
        "answers": [batch["answers"][0]],
        "raw_answers": [batch["raw_answers"][0]],
        "normalized_answers": [batch["normalized_answers"][0]],
        "answer_labels": batch["answer_labels"][:1],
        "question_ids": [batch["question_ids"][0]],
        "image_ids": [batch["image_ids"][0]],
        "image_paths": [batch["image_paths"][0]],
        "active_splits": [batch["active_splits"][0]],
    }


def estimate_fvcore_flops(
    model: LlavaStaticVQAModel,
    loader,
) -> Optional[Dict[str, Any]]:
    """
    Estimate FLOPs with fvcore on one validation sample.

    Notes:
    - This is a secondary best-effort traced metric.
    - Unsupported operators can appear on modern Hugging Face / Transformer models.
    - The main thesis metric should still be the analytical attention FLOPs.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception as e:
        return {
            "available": False,
            "error": f"fvcore import failed: {e}",
        }

    try:
        batch = next(iter(loader))
    except StopIteration:
        return {
            "available": False,
            "error": "Could not get a batch from loader for fvcore FLOPs.",
        }

    single_batch = _extract_single_sample_batch(batch)

    model.eval()
    with torch.no_grad():
        model_inputs = model._prepare_inputs(
            images=single_batch["images"],
            questions=single_batch["questions"],
        )

    required_keys = {"input_ids", "attention_mask", "pixel_values"}
    missing = required_keys.difference(model_inputs.keys())
    if missing:
        return {
            "available": False,
            "error": f"Missing keys for fvcore FLOPs: {sorted(missing)}",
        }

    flops_model = StaticClassificationFlopsWrapper(model)
    flops_model.eval()

    inputs = (
        model_inputs["input_ids"],
        model_inputs["attention_mask"],
        model_inputs["pixel_values"],
    )

    try:
        analysis = FlopCountAnalysis(flops_model, inputs)

        total_flops = analysis.total()
        unsupported = analysis.unsupported_ops()
        uncalled_modules = analysis.uncalled_modules()

        by_operator = {}
        try:
            op_breakdown = analysis.by_operator()
            by_operator = {str(k): float(v) for k, v in op_breakdown.items()}
        except Exception:
            by_operator = {}

        return {
            "available": True,
            "method": "fvcore_traced_static_classification_forward",
            "num_samples_traced": 1,
            "flops": float(total_flops),
            "flops_giga": float(total_flops) / 1e9,
            "unsupported_ops": {str(k): int(v) for k, v in unsupported.items()},
            "uncalled_modules": sorted(list(uncalled_modules)),
            "by_operator": by_operator,
        }

    except Exception as e:
        return {
            "available": False,
            "error": f"fvcore tracing failed: {e}",
        }


def run_eval_only(cfg: Dict[str, Any], output_dir: str):
    """
    Run evaluation-only mode, typically used for static debug sanity checks.
    """
    split_name = cfg["dataset"].get("active_split", "val")
    if split_name not in {"train", "val"}:
        raise ValueError(f"active_split must be 'train' or 'val', got {split_name}")

    dataset = build_vqav2_dataset(cfg, split_name)
    loader = build_loader(dataset, cfg, split_name=split_name)

    model = LlavaStaticVQAModel(cfg)
    load_answer_head_checkpoint_if_requested(model, cfg)

    use_amp = str(cfg["training"].get("mixed_precision", "")).lower() == "fp16"

    val_metrics = validate_one_epoch(
        model=model,
        loader=loader,
        use_amp=use_amp,
        log_every_n_steps=int(cfg["logging"]["log_every_n_steps"]),
        save_predictions=bool(cfg["evaluation"].get("save_predictions", True)),
    )

    latency_metrics = None
    if cfg["evaluation"].get("compute_latency", False):
        latency_cfg = cfg["evaluation"].get("latency", {})
        latency_metrics = measure_latency(
            model=model,
            loader=loader,
            num_warmup_steps=int(latency_cfg.get("num_warmup_steps", 10)),
            num_measure_steps=int(latency_cfg.get("num_measure_steps", 50)),
            synchronize_cuda=bool(latency_cfg.get("synchronize_cuda", True)),
            use_amp=use_amp,
        )

    flops_metrics = None
    if cfg["evaluation"].get("compute_flops", False):
        flops_metrics = estimate_analytical_attention_flops(
            model=model,
            metrics=val_metrics,
        )

    if flops_metrics is not None and cfg["evaluation"].get("compute_fvcore_flops", False):
        flops_metrics["fvcore"] = estimate_fvcore_flops(model=model, loader=loader)

    run_metrics = {
        "experiment_name": cfg["experiment_name"],
        "mode": cfg["training"]["mode"],
        "answer_mode": cfg["dataset"]["answer_mode"],
        "dataset_split": split_name,
        "num_samples": len(dataset),
        "total_parameters": count_total_parameters(model),
        "trainable_parameters": count_trainable_parameters(model),
        "validation": {k: v for k, v in val_metrics.items() if k != "predictions"},
        "flops": flops_metrics,
        "latency": latency_metrics,
    }

    save_json(os.path.join(output_dir, "metrics.json"), run_metrics)

    if cfg["evaluation"].get("save_predictions", True):
        save_json(
            os.path.join(output_dir, "predictions.json"),
            {"predictions": val_metrics["predictions"]},
        )

    return run_metrics


def run_train_answer_head(cfg: Dict[str, Any], output_dir: str):
    """
    Main static-pruning training path.

    This keeps the same successful idea from your existing static setup:
    - CLS-attention-based fixed top-k pruning
    - frozen backbone
    - train only the answer head
    """
    train_dataset = build_vqav2_dataset(cfg, "train")
    val_dataset = build_vqav2_dataset(cfg, "val")

    print(f"[Info] Train samples: {len(train_dataset)}", flush=True)
    print(f"[Info] Val samples: {len(val_dataset)}", flush=True)

    train_loader = build_loader(train_dataset, cfg, split_name="train")
    val_loader = build_loader(val_dataset, cfg, split_name="val")

    model = LlavaStaticVQAModel(cfg)

    optimizer = build_optimizer(cfg, model)

    steps_per_epoch = math.ceil(
        len(train_loader) / max(1, int(cfg["training"]["grad_accum_steps"]))
    )
    total_training_steps = steps_per_epoch * int(cfg["training"]["epochs"])
    scheduler = build_scheduler(cfg, optimizer, total_training_steps)

    use_amp = str(cfg["training"].get("mixed_precision", "")).lower() == "fp16"
    scaler = build_grad_scaler(use_amp)

    history = []
    best_epoch = None
    best_metric = None
    best_val_metrics = None

    monitor = cfg["logging"].get("monitor", "val_accuracy")
    save_best_only = bool(cfg["logging"].get("save_best_only", True))

    print(
        f"[Info] Starting training for {cfg['training']['epochs']} epochs | "
        f"steps/epoch={len(train_loader)} | "
        f"grad_accum={cfg['training']['grad_accum_steps']}",
        flush=True,
    )

    for epoch in range(int(cfg["training"]["epochs"])):
        print(f"[Info] ===== Epoch {epoch + 1}/{cfg['training']['epochs']} =====", flush=True)

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            use_amp=use_amp,
            grad_accum_steps=int(cfg["training"]["grad_accum_steps"]),
            log_every_n_steps=int(cfg["logging"]["log_every_n_steps"]),
            use_wandb=False,
            epoch_index=epoch,
        )

        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            use_amp=use_amp,
            log_every_n_steps=int(cfg["logging"]["log_every_n_steps"]),
            save_predictions=bool(cfg["evaluation"].get("save_predictions", True)),
        )

        epoch_record = {
            "epoch": epoch + 1,
            "train": {k: v for k, v in train_metrics.items()},
            "val": {k: v for k, v in val_metrics.items() if k != "predictions"},
        }
        history.append(epoch_record)

        print(
            f"[Info] Epoch {epoch + 1} summary | "
            f"train_loss={train_metrics.get('loss')} | "
            f"train_vqa={train_metrics.get('vqa_accuracy')} | "
            f"val_loss={val_metrics.get('loss')} | "
            f"val_vqa={val_metrics.get('vqa_accuracy')}",
            flush=True,
        )

        if monitor == "val_accuracy":
            current_metric = val_metrics.get("vqa_accuracy", None)
        elif monitor == "val_loss":
            current_metric = val_metrics.get("loss", None)
        else:
            raise ValueError(f"Unsupported monitor: {monitor}")

        is_better = False
        if current_metric is not None:
            if best_metric is None:
                is_better = True
            elif monitor == "val_loss":
                is_better = current_metric < best_metric
            else:
                is_better = current_metric > best_metric

        if is_better:
            best_metric = current_metric
            best_epoch = epoch + 1
            best_val_metrics = deepcopy(val_metrics)

            print(
                f"[Info] New best model at epoch {best_epoch} | "
                f"{monitor}={best_metric}",
                flush=True,
            )

            save_checkpoint(
                path=os.path.join(output_dir, "best_model.pt"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                metrics={
                    "train": train_metrics,
                    "val": {k: v for k, v in val_metrics.items() if k != "predictions"},
                },
            )

            if cfg["evaluation"].get("save_predictions", True):
                save_json(
                    os.path.join(output_dir, "best_predictions.json"),
                    {"predictions": val_metrics["predictions"]},
                )

        if not save_best_only:
            save_checkpoint(
                path=os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}.pt"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                metrics={
                    "train": train_metrics,
                    "val": {k: v for k, v in val_metrics.items() if k != "predictions"},
                },
            )

    # Always save the last checkpoint for completeness.
    save_checkpoint(
        path=os.path.join(output_dir, "last_model.pt"),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=int(cfg["training"]["epochs"]),
        metrics={"history": history},
    )

    latency_metrics = None
    if cfg["evaluation"].get("compute_latency", False):
        print("[Info] Measuring latency on validation loader...", flush=True)
        latency_cfg = cfg["evaluation"].get("latency", {})
        latency_metrics = measure_latency(
            model=model,
            loader=val_loader,
            num_warmup_steps=int(latency_cfg.get("num_warmup_steps", 10)),
            num_measure_steps=int(latency_cfg.get("num_measure_steps", 50)),
            synchronize_cuda=bool(latency_cfg.get("synchronize_cuda", True)),
            use_amp=use_amp,
        )

    flops_metrics = None
    if cfg["evaluation"].get("compute_flops", False):
        metrics_source = (
            {k: v for k, v in best_val_metrics.items() if k != "predictions"}
            if best_val_metrics is not None
            else None
        )
        if metrics_source is not None:
            flops_metrics = estimate_analytical_attention_flops(
                model=model,
                metrics=metrics_source,
            )

    if flops_metrics is not None and cfg["evaluation"].get("compute_fvcore_flops", False):
        flops_metrics["fvcore"] = estimate_fvcore_flops(model=model, loader=val_loader)

    run_metrics = {
        "experiment_name": cfg["experiment_name"],
        "mode": cfg["training"]["mode"],
        "answer_mode": cfg["dataset"]["answer_mode"],
        "train_num_samples": len(train_dataset),
        "val_num_samples": len(val_dataset),
        "total_parameters": count_total_parameters(model),
        "trainable_parameters": count_trainable_parameters(model),
        "best_epoch": best_epoch,
        "best_monitor_value": best_metric,
        "best_validation": None if best_val_metrics is None else {
            k: v for k, v in best_val_metrics.items() if k != "predictions"
        },
        "flops": flops_metrics,
        "latency": latency_metrics,
        "history": history,
    }

    save_json(os.path.join(output_dir, "metrics.json"), run_metrics)

    return run_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    set_seed(
        int(cfg["seed"]),
        bool(cfg["system"].get("deterministic", False)),
    )

    output_dir = make_output_dir(
        base_dir=cfg["logging"]["save_dir"],
        experiment_name=cfg["experiment_name"],
    )
    print(f"Outputs will be saved to: {output_dir}", flush=True)

    save_json(os.path.join(output_dir, "config_resolved.json"), cfg)

    training_mode = cfg["training"]["mode"]
    if training_mode == "eval_only":
        metrics = run_eval_only(cfg, output_dir)
    elif training_mode == "train_answer_head":
        metrics = run_train_answer_head(cfg, output_dir)
    else:
        raise ValueError(f"Unsupported training.mode: {training_mode}")

    print("Run completed successfully.", flush=True)
    print(
        json.dumps(
            metrics,
            indent=2,
            default=lambda x: x.tolist() if torch.is_tensor(x) else x,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()