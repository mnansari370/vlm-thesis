"""
GQA datasets for the paper.

GQAValDataset  — evaluation (deterministic ordered subset or full val).
GQATrainDataset — training (shuffled, filters empty answers).
"""

import json
import os
import random
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from GQA.shared.metrics import get_semantic_type


class GQAValDataset(Dataset):
    """
    GQA validation dataset.
    Sorted deterministically so checkpoint/resume gives same order.
    """

    def __init__(
        self,
        questions_path: str,
        image_dir: str,
        max_samples: int | None = None,
        seed: int = 42,
        question_ids: list[str] | None = None,
    ):
        self.image_dir = image_dir

        print(f"[GQA] Loading val questions: {questions_path}", flush=True)
        with open(questions_path) as f:
            raw = json.load(f)

        qids = sorted(raw.keys())   # deterministic order

        if question_ids is not None:
            # Filter to a specific pre-sampled subset (e.g. oracle training set)
            qid_set = set(str(q) for q in question_ids)
            qids = sorted(q for q in qids if q in qid_set)
        elif max_samples is not None and max_samples < len(qids):
            rng = random.Random(seed)
            rng.shuffle(qids)
            qids = sorted(qids[:max_samples])   # re-sort after sampling

        self.records: list[dict[str, Any]] = []
        for qid in qids:
            rec = raw[qid]
            img_path = os.path.join(image_dir, f"{rec['imageId']}.jpg")
            if not os.path.exists(img_path):
                continue
            self.records.append({
                "question_id":   qid,
                "image_id":      rec["imageId"],
                "question":      rec["question"],
                "answer":        rec.get("answer", ""),
                "semantic_type": get_semantic_type(rec.get("types")),
                "image_path":    img_path,
            })

        print(f"[GQA] Val loaded: {len(self.records):,} samples", flush=True)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        return {
            "image":         Image.open(r["image_path"]).convert("RGB"),
            "question":      r["question"],
            "answer":        r["answer"],
            "question_id":   r["question_id"],
            "image_id":      r["image_id"],
            "semantic_type": r["semantic_type"],
        }


class GQATrainDataset(Dataset):
    """
    GQA training dataset.
    Shuffled with fixed seed; skips samples with no answer or missing images.
    """

    def __init__(
        self,
        questions_path: str,
        image_dir: str,
        max_samples: int | None = None,
        seed: int = 42,
    ):
        self.image_dir = image_dir

        print(f"[GQA] Loading train questions: {questions_path}", flush=True)
        with open(questions_path) as f:
            raw = json.load(f)

        qids = list(raw.keys())
        random.Random(seed).shuffle(qids)
        if max_samples:
            qids = qids[:max_samples]

        self.records: list[dict[str, Any]] = []
        missing = 0
        for qid in qids:
            rec = raw[qid]
            if not rec.get("answer"):
                continue
            img_path = os.path.join(image_dir, f"{rec['imageId']}.jpg")
            if not os.path.exists(img_path):
                missing += 1
                continue
            self.records.append({
                "question": rec["question"],
                "answer":   rec["answer"],
                "path":     img_path,
            })

        if missing:
            print(f"[GQA] Skipped {missing} missing images.", flush=True)
        print(f"[GQA] Train loaded: {len(self.records):,} samples", flush=True)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        return {
            "image":    Image.open(r["path"]).convert("RGB"),
            "question": r["question"],
            "answer":   r["answer"],
        }


class GQATrainDatasetWithRegions(Dataset):
    """
    GQA training dataset with region supervision targets.

    Loads region_targets_train.json alongside the questions so the model
    can receive per-sample BCE supervision on the patch scorer.

    Each sample returns:
      image        : PIL image
      question     : str
      answer       : str
      patch_target : float32 [576] binary mask (zeros if no region target)
      has_target   : bool — False means region_weight should be 0
    """

    def __init__(
        self,
        questions_path:      str,
        image_dir:           str,
        region_targets_path: str,
        max_samples:         int | None = None,
        seed:                int = 42,
    ):
        import torch
        self.image_dir = image_dir

        print(f"[GQA] Loading train questions: {questions_path}", flush=True)
        with open(questions_path) as f:
            raw = json.load(f)

        print(f"[GQA] Loading region targets: {region_targets_path}", flush=True)
        with open(region_targets_path) as f:
            region_targets = json.load(f)

        qids = list(raw.keys())
        random.Random(seed).shuffle(qids)
        if max_samples:
            qids = qids[:max_samples]

        self.records: list[dict[str, Any]] = []
        missing = 0
        for qid in qids:
            rec = raw[qid]
            if not rec.get("answer"):
                continue
            img_path = os.path.join(image_dir, f"{rec['imageId']}.jpg")
            if not os.path.exists(img_path):
                missing += 1
                continue
            patch_indices = region_targets.get(str(qid))
            has_target    = patch_indices is not None
            self.records.append({
                "question":      rec["question"],
                "answer":        rec["answer"],
                "path":          img_path,
                "has_target":    has_target,
                "patch_indices": patch_indices if has_target else [],
            })

        if missing:
            print(f"[GQA] Skipped {missing} missing images.", flush=True)
        n_with = sum(1 for r in self.records if r["has_target"])
        print(
            f"[GQA] Train loaded: {len(self.records):,} samples  "
            f"({n_with:,} with region targets, "
            f"{n_with/len(self.records)*100:.1f}%)",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        import torch
        r = self.records[idx]
        target = torch.zeros(576, dtype=torch.float32)
        if r["has_target"] and r["patch_indices"]:
            idx_tensor = torch.tensor(r["patch_indices"], dtype=torch.long)
            idx_tensor = idx_tensor.clamp(0, 575)
            target[idx_tensor] = 1.0
        return {
            "image":        Image.open(r["path"]).convert("RGB"),
            "question":     r["question"],
            "answer":       r["answer"],
            "patch_target": target,
            "has_target":   r["has_target"],
        }


def collate_with_regions(batch: list[dict]) -> dict:
    import torch
    return {
        "images":         [b["image"]        for b in batch],
        "questions":      [b["question"]     for b in batch],
        "answers":        [b["answer"]       for b in batch],
        "region_targets": torch.stack([b["patch_target"] for b in batch]),  # [B, 576]
        "region_weights": torch.tensor(
            [1.0 if b["has_target"] else 0.0 for b in batch],
            dtype=torch.float32,
        ),  # [B]
    }


# ── collators ─────────────────────────────────────────────────────────────────

def collate_val(batch: list[dict]) -> dict:
    return {
        "images":         [b["image"]         for b in batch],
        "questions":      [b["question"]       for b in batch],
        "answers":        [b["answer"]         for b in batch],
        "question_ids":   [b["question_id"]    for b in batch],
        "image_ids":      [b["image_id"]       for b in batch],
        "semantic_types": [b["semantic_type"]  for b in batch],
    }


def collate_train(batch: list[dict]) -> dict:
    return {
        "images":    [b["image"]    for b in batch],
        "questions": [b["question"] for b in batch],
        "answers":   [b["answer"]   for b in batch],
    }
