"""Train/validate loops + latency measurement for the dense baseline."""

import contextlib
import time
from typing import Any, Dict, List, Optional

import torch

from VQA_V2_early_proxy.shared.datasets.vqav2_answers import normalize_answer
from .losses import extract_model_loss


def _to_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _mean_tensor_list(values: List[torch.Tensor]) -> Optional[float]:
    if len(values) == 0:
        return None
    stacked = torch.cat([v.detach().cpu().reshape(-1) for v in values], dim=0)
    return float(stacked.float().mean().item())


def _mean_float_list(values: List[float]) -> Optional[float]:
    if len(values) == 0:
        return None
    return float(sum(values) / len(values))


def _compute_vqa_consensus_score(pred_answer: str, gt_answers: List[str]) -> float:
    """
    Lightweight VQA-style consensus score:
      min(1, matches / 3)
    """
    pred = normalize_answer(pred_answer)
    normalized_gt = [normalize_answer(a) for a in gt_answers]

    matches = sum(1 for ans in normalized_gt if ans == pred)
    return min(1.0, matches / 3.0)


def _compute_batch_vqa_accuracy(
    pred_answers: Optional[List[str]],
    raw_answers: Optional[List[List[str]]],
) -> Optional[float]:
    if pred_answers is None or raw_answers is None:
        return None
    if len(pred_answers) == 0:
        return None

    scores = []
    for pred, gt_list in zip(pred_answers, raw_answers):
        scores.append(_compute_vqa_consensus_score(pred, gt_list))

    return float(sum(scores) / len(scores)) if len(scores) > 0 else None


def _autocast_context(use_amp: bool):
    if use_amp and torch.cuda.is_available():
        return torch.cuda.amp.autocast()
    return contextlib.nullcontext()


def _set_training_mode_for_answer_head(model) -> None:
    """
    Keep the frozen backbone in eval mode while allowing the answer head to train.
    """
    model.eval()

    if hasattr(model, "answer_head") and model.answer_head is not None:
        model.answer_head.train()


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler=None,
    scaler=None,
    use_amp: bool = False,
    grad_accum_steps: int = 1,
    log_every_n_steps: int = 10,
    use_wandb: bool = False,
    epoch_index: int = 0,
):
    """
    Train for one epoch.

    Used for:
    frozen LLaVA backbone + trainable answer head.
    """
    if optimizer is None:
        raise ValueError("Optimizer is required for training.")

    _set_training_mode_for_answer_head(model)

    total_loss = 0.0
    num_loss_examples = 0

    batch_acc_values: List[float] = []
    token_before_values: List[torch.Tensor] = []
    token_after_values: List[torch.Tensor] = []
    retention_values: List[torch.Tensor] = []
    raw_question_length_values: List[torch.Tensor] = []
    processor_input_length_values: List[torch.Tensor] = []
    multimodal_seq_values: List[torch.Tensor] = []

    non_finite_batch_count = 0

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        with _autocast_context(use_amp):
            outputs = model(batch)
            loss = extract_model_loss(outputs)

        if not torch.isfinite(loss):
            non_finite_batch_count += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        loss_for_backward = loss / max(1, grad_accum_steps)

        if scaler is not None:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        if ((step + 1) % grad_accum_steps == 0) or ((step + 1) == len(loader)):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()

        batch_size = len(batch["questions"])
        total_loss += _to_float(loss) * batch_size
        num_loss_examples += batch_size

        pred_answers = outputs["predictions"].get("pred_answers", None)
        batch_acc = _compute_batch_vqa_accuracy(
            pred_answers=pred_answers,
            raw_answers=batch.get("raw_answers", None),
        )
        if batch_acc is not None:
            batch_acc_values.append(batch_acc)

        token_stats = outputs["token_stats"]
        token_before_values.append(token_stats["num_visual_tokens_before_selection"])
        token_after_values.append(token_stats["num_visual_tokens_after_selection"])
        retention_values.append(token_stats["retention_ratio"])

        analysis = outputs.get("analysis", {})
        if analysis.get("raw_question_lengths", None) is not None:
            raw_question_length_values.append(analysis["raw_question_lengths"])
        if analysis.get("processor_input_lengths", None) is not None:
            processor_input_length_values.append(analysis["processor_input_lengths"])
        if analysis.get("multimodal_sequence_length", None) is not None:
            multimodal_seq_values.append(analysis["multimodal_sequence_length"])

        if use_wandb:
            try:
                import wandb

                log_dict = {
                    "train/step_loss": _to_float(loss),
                    "train/epoch": epoch_index + 1,
                }
                if batch_acc is not None:
                    log_dict["train/step_vqa_accuracy"] = batch_acc
                wandb.log(log_dict)
            except Exception:
                pass

        if (step + 1) % log_every_n_steps == 0:
            msg = (
                f"[Train] Epoch {epoch_index + 1} "
                f"Step {step + 1}/{len(loader)} "
                f"loss={_to_float(loss):.4f}"
            )
            if batch_acc is not None:
                msg += f" vqa_acc={batch_acc:.4f}"
            print(msg, flush=True)

    if non_finite_batch_count > 0:
        print(f"[Train] Skipped {non_finite_batch_count} batches due to non-finite loss.", flush=True)

    avg_loss = None
    if num_loss_examples > 0:
        avg_loss = total_loss / num_loss_examples

    return {
        "loss": avg_loss,
        "vqa_accuracy": _mean_float_list(batch_acc_values),
        "avg_num_visual_tokens_before": _mean_tensor_list(token_before_values),
        "avg_num_visual_tokens_after": _mean_tensor_list(token_after_values),
        "avg_retention_ratio": _mean_tensor_list(retention_values),
        "avg_raw_question_length": _mean_tensor_list(raw_question_length_values),
        "avg_processor_input_length": _mean_tensor_list(processor_input_length_values),
        "avg_multimodal_sequence_length": _mean_tensor_list(multimodal_seq_values),
    }


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    use_amp: bool = False,
    log_every_n_steps: int = 10,
    save_predictions: bool = True,
):
    """
    Validate for one epoch.

    Works for both:
    - generation debug mode
    - classification answer-head mode
    """
    model.eval()

    total_loss = 0.0
    num_loss_examples = 0

    batch_acc_values: List[float] = []
    token_before_values: List[torch.Tensor] = []
    token_after_values: List[torch.Tensor] = []
    retention_values: List[torch.Tensor] = []
    raw_question_length_values: List[torch.Tensor] = []
    processor_input_length_values: List[torch.Tensor] = []
    multimodal_seq_values: List[torch.Tensor] = []

    saved_predictions = []
    non_finite_val_loss_count = 0

    for step, batch in enumerate(loader):
        with _autocast_context(use_amp):
            outputs = model(batch)

        loss = outputs["predictions"].get("loss", None)
        if loss is not None:
            if torch.isfinite(loss):
                batch_size = len(batch["questions"])
                total_loss += _to_float(loss) * batch_size
                num_loss_examples += batch_size
            else:
                non_finite_val_loss_count += 1

        pred_answers = outputs["predictions"].get("pred_answers", None)
        batch_acc = _compute_batch_vqa_accuracy(
            pred_answers=pred_answers,
            raw_answers=batch.get("raw_answers", None),
        )
        if batch_acc is not None:
            batch_acc_values.append(batch_acc)

        token_stats = outputs["token_stats"]
        token_before_values.append(token_stats["num_visual_tokens_before_selection"])
        token_after_values.append(token_stats["num_visual_tokens_after_selection"])
        retention_values.append(token_stats["retention_ratio"])

        analysis = outputs.get("analysis", {})
        if analysis.get("raw_question_lengths", None) is not None:
            raw_question_length_values.append(analysis["raw_question_lengths"])
        if analysis.get("processor_input_lengths", None) is not None:
            processor_input_length_values.append(analysis["processor_input_lengths"])
        if analysis.get("multimodal_sequence_length", None) is not None:
            multimodal_seq_values.append(analysis["multimodal_sequence_length"])

        if save_predictions and pred_answers is not None:
            for idx in range(len(batch["questions"])):
                saved_predictions.append(
                    {
                        "question_id": batch["question_ids"][idx],
                        "image_id": batch["image_ids"][idx],
                        "question": batch["questions"][idx],
                        "pred_answer": pred_answers[idx],
                        "raw_answers": batch["raw_answers"][idx],
                        "normalized_answers": batch["normalized_answers"][idx],
                    }
                )

        if (step + 1) % log_every_n_steps == 0:
            msg = f"[Val] Step {step + 1}/{len(loader)}"
            if batch_acc is not None:
                msg += f" vqa_acc={batch_acc:.4f}"
            print(msg, flush=True)

    if non_finite_val_loss_count > 0:
        print(
            f"[Val] Skipped loss aggregation for {non_finite_val_loss_count} batches due to non-finite loss.",
            flush=True,
        )

    avg_loss = None
    if num_loss_examples > 0:
        avg_loss = total_loss / num_loss_examples

    return {
        "loss": avg_loss,
        "vqa_accuracy": _mean_float_list(batch_acc_values),
        "avg_num_visual_tokens_before": _mean_tensor_list(token_before_values),
        "avg_num_visual_tokens_after": _mean_tensor_list(token_after_values),
        "avg_retention_ratio": _mean_tensor_list(retention_values),
        "avg_raw_question_length": _mean_tensor_list(raw_question_length_values),
        "avg_processor_input_length": _mean_tensor_list(processor_input_length_values),
        "avg_multimodal_sequence_length": _mean_tensor_list(multimodal_seq_values),
        "predictions": saved_predictions,
    }


@torch.no_grad()
def measure_latency(
    model,
    loader,
    num_warmup_steps: int = 10,
    num_measure_steps: int = 50,
    synchronize_cuda: bool = True,
    use_amp: bool = False,
):
    """
    Measure inference latency on a few batches.
    """
    model.eval()

    times = []
    measured_batch_sizes = []

    data_iter = iter(loader)

    for _ in range(num_warmup_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        with _autocast_context(use_amp):
            _ = model(batch)

    for _ in range(num_measure_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        batch_size = len(batch["questions"])
        measured_batch_sizes.append(batch_size)

        if torch.cuda.is_available() and synchronize_cuda and next(model.parameters()).is_cuda:
            torch.cuda.synchronize()

        start = time.perf_counter()

        with _autocast_context(use_amp):
            _ = model(batch)

        if torch.cuda.is_available() and synchronize_cuda and next(model.parameters()).is_cuda:
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append(end - start)

    if len(times) == 0:
        return None

    avg_batch_time = float(sum(times) / len(times))
    avg_batch_size = (
        float(sum(measured_batch_sizes) / len(measured_batch_sizes))
        if len(measured_batch_sizes) > 0
        else 1.0
    )
    avg_sample_time = avg_batch_time / max(avg_batch_size, 1.0)
    throughput = 1.0 / max(avg_sample_time, 1e-12)

    return {
        "avg_batch_time_sec": avg_batch_time,
        "avg_sample_time_sec": avg_sample_time,
        "throughput_samples_per_sec": throughput,
    }