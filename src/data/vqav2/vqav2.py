"""
VQAv2 dataset for vqa_v2 pipeline.

Key differences from the original datasets/vqav2.py:
- Stratified sampling: max_samples is applied proportionally across 4 question types
  (yes/no, attribute, counting, spatial) using the same heuristic as DynamicTokenSelector.
- filter_unknown_answers defaults to False (we use full vocabulary).
- answer_label = -1 for answers not in vocab → CrossEntropyLoss(ignore_index=-1) handles them.
"""

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset

from .image_transforms import build_image_transform
from .vqav2_answers import (
    answer_to_label,
    build_id_to_answer,
    get_majority_answer,
    load_answer_vocab,
    normalize_answers,
)


# ---------------------------------------------------------------------------
# Question-type heuristic (exact copy of _infer_question_type_ids logic from
# models/dynamic/llava_wrapper.py — using it here for stratified sampling)
# ---------------------------------------------------------------------------

_YES_NO_STARTS = (
    "is ", "are ", "was ", "were ",
    "do ", "does ", "did ",
    "can ", "could ", "will ", "would ",
    "has ", "have ", "had ",
    "is there", "are there",
)

_COUNTING_PATTERNS = (
    "how many",
    "number of",
    "count ",
    "amount of",
)

_SPATIAL_PATTERNS = (
    "where", "left", "right", "behind", "front", "in front",
    "on top", "under", "above", "below", "next to", "near",
    "between", "side", "position", "located",
)


def _question_type_id(question: str) -> int:
    """
    Returns:
        0 = yes/no
        1 = attribute/object
        2 = counting
        3 = spatial/complex
    Counting is checked before yes/no (matches DynamicTokenSelector precedence).
    """
    q = " ".join(question.lower().strip().split())
    if any(q.startswith(p) or p in q for p in _COUNTING_PATTERNS):
        return 2
    if any(q.startswith(p) for p in _YES_NO_STARTS):
        return 0
    if any(p in q for p in _SPATIAL_PATTERNS):
        return 3
    return 1


def _stratified_sample(
    samples: List[Dict[str, Any]],
    max_samples: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Sample proportionally from 4 question-type buckets.
    Preserves the natural type distribution in the resulting subset.
    """
    import random
    rng = random.Random(seed)

    buckets: Dict[int, List[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        qt = _question_type_id(s["question"])
        buckets[qt].append(i)

    total = len(samples)
    selected_indices: List[int] = []

    for qt in [0, 1, 2, 3]:
        indices = buckets[qt]
        rng.shuffle(indices)
        # Proportional quota for this type
        quota = round(max_samples * len(indices) / total)
        quota = min(quota, len(indices))
        selected_indices.extend(indices[:quota])

    # Cap precisely at max_samples (rounding can push slightly over/under)
    rng.shuffle(selected_indices)
    selected_indices = selected_indices[:max_samples]
    selected_indices.sort()  # restore file order for reproducibility

    result = [samples[i] for i in selected_indices]

    # Print type distribution for verification
    counts = defaultdict(int)
    for s in result:
        counts[_question_type_id(s["question"])] += 1
    names = {0: "yes/no", 1: "attribute", 2: "counting", 3: "spatial"}
    print(
        f"[Stratified sample] Total={len(result)} | " +
        " | ".join(f"{names[t]}={counts[t]}" for t in [0, 1, 2, 3]),
        flush=True,
    )
    return result


class VQAv2Dataset(Dataset):
    """
    VQAv2 dataset for the vqa_v2 pipeline.

    Changes vs original:
    - max_samples applied via stratified sampling (preserves qtype distribution).
    - filter_unknown_answers=False by default (full vocab, ignore_index=-1 handles unknowns).
    - Same __getitem__ return dict as original for collator compatibility.
    """

    def __init__(
        self,
        active_split: str,
        image_dir: str,
        questions_path: str,
        annotations_path: Optional[str],
        image_size: int = 336,
        answer_mode: str = "classification",
        answer_vocab_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        is_train: bool = False,
        filter_unknown_answers: bool = False,
        stratify: bool = True,
        seed: int = 42,
        image_aspect_ratio: str = "center_crop",
    ):
        if active_split not in {"train", "val"}:
            raise ValueError(f"active_split must be 'train' or 'val', got: {active_split}")
        if answer_mode not in {"generation", "classification"}:
            raise ValueError(f"answer_mode must be 'generation' or 'classification'")

        self.active_split = active_split
        self.image_dir = image_dir
        self.questions_path = questions_path
        self.annotations_path = annotations_path
        self.image_size = image_size
        self.answer_mode = answer_mode
        self.max_samples = max_samples
        self.filter_unknown_answers = filter_unknown_answers
        self.stratify = stratify
        self.seed = seed
        self.transform = build_image_transform(
            image_size=image_size,
            is_train=is_train,
            image_aspect_ratio=image_aspect_ratio,
        )

        self.answer_to_id: Optional[Dict[str, int]] = None
        self.id_to_answer: Optional[Dict[int, str]] = None
        if answer_vocab_path is not None and os.path.exists(answer_vocab_path):
            self.answer_to_id = load_answer_vocab(answer_vocab_path)
            self.id_to_answer = build_id_to_answer(self.answer_to_id)

        self.samples = self._build_samples()

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _infer_image_filename(self, image_id: int) -> str:
        if "train2014" in self.image_dir:
            return f"COCO_train2014_{image_id:012d}.jpg"
        if "val2014" in self.image_dir:
            return f"COCO_val2014_{image_id:012d}.jpg"
        raise ValueError(f"Cannot infer COCO prefix from image_dir: {self.image_dir}")

    def _build_samples(self) -> List[Dict[str, Any]]:
        questions_data = self._load_json(self.questions_path)
        questions = questions_data["questions"]

        annotations_by_qid: Dict[int, Any] = {}
        if self.annotations_path and os.path.exists(self.annotations_path):
            ann_data = self._load_json(self.annotations_path)
            for ann in ann_data["annotations"]:
                annotations_by_qid[ann["question_id"]] = ann

        samples: List[Dict[str, Any]] = []
        num_skipped = 0

        for q in questions:
            question_id = q["question_id"]
            image_id = q["image_id"]
            question_text = q["question"]
            ann = annotations_by_qid.get(question_id)

            raw_answers: List[str] = []
            normalized_ans: List[str] = []
            majority_answer = ""
            answer_label = -1

            if ann is not None:
                raw_answers = [a["answer"] for a in ann.get("answers", [])]
                normalized_ans = normalize_answers(raw_answers)
                majority_answer = get_majority_answer(raw_answers)

                if self.answer_mode == "classification":
                    answer_label = answer_to_label(
                        majority_answer, self.answer_to_id, unknown_index=-1
                    )
                    if self.filter_unknown_answers and answer_label == -1:
                        num_skipped += 1
                        continue

            samples.append({
                "question_id": question_id,
                "image_id": image_id,
                "question": question_text,
                "raw_answers": raw_answers,
                "normalized_answers": normalized_ans,
                "majority_answer": majority_answer,
                "answer_label": answer_label,
                "image_path": os.path.join(
                    self.image_dir, self._infer_image_filename(image_id)
                ),
                "active_split": self.active_split,
            })

        if num_skipped > 0:
            print(
                f"[Info] {self.active_split}: skipped {num_skipped} unknown-answer samples",
                flush=True,
            )

        # Apply stratified or plain truncation
        if self.max_samples is not None and len(samples) > self.max_samples:
            if self.stratify and self.active_split == "train":
                samples = _stratified_sample(samples, self.max_samples, seed=self.seed)
            else:
                samples = samples[: self.max_samples]

        print(
            f"[Info] {self.active_split}: {len(samples)} samples loaded",
            flush=True,
        )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        s = self.samples[index]

        if not os.path.exists(s["image_path"]):
            raise FileNotFoundError(f"Image not found: {s['image_path']}")

        image = Image.open(s["image_path"]).convert("RGB")
        image = self.transform(image)

        return {
            "image": image,
            "question": s["question"],
            "answer": s["majority_answer"],
            "answer_label": s["answer_label"],
            "raw_answers": s["raw_answers"],
            "normalized_answers": s["normalized_answers"],
            "question_id": s["question_id"],
            "image_id": s["image_id"],
            "image_path": s["image_path"],
            "active_split": s["active_split"],
        }


def build_vqav2_dataset(cfg: Dict[str, Any], split_name: str) -> VQAv2Dataset:
    dataset_cfg = cfg["dataset"]
    answer_mode = dataset_cfg["answer_mode"]
    filter_unknown = (
        answer_mode == "classification"
        and bool(dataset_cfg.get("filter_unknown_answers", False))
    )
    seed = int(cfg.get("seed", 42))
    stratify = bool(dataset_cfg.get("stratify", True))
    image_aspect_ratio = dataset_cfg.get("image_aspect_ratio", "center_crop")

    if split_name == "train":
        return VQAv2Dataset(
            active_split=dataset_cfg["train_split"],
            image_dir=dataset_cfg["image_dir_train"],
            questions_path=dataset_cfg["questions_train"],
            annotations_path=dataset_cfg["annotations_train"],
            image_size=dataset_cfg["image_size"],
            answer_mode=answer_mode,
            answer_vocab_path=dataset_cfg.get("answer_vocab_path"),
            max_samples=dataset_cfg.get("max_samples"),
            is_train=True,
            filter_unknown_answers=filter_unknown,
            stratify=stratify,
            seed=seed,
            image_aspect_ratio=image_aspect_ratio,
        )
    elif split_name == "val":
        return VQAv2Dataset(
            active_split=dataset_cfg["val_split"],
            image_dir=dataset_cfg["image_dir_val"],
            questions_path=dataset_cfg["questions_val"],
            annotations_path=dataset_cfg["annotations_val"],
            image_size=dataset_cfg["image_size"],
            answer_mode=answer_mode,
            answer_vocab_path=dataset_cfg.get("answer_vocab_path"),
            max_samples=dataset_cfg.get("max_val_samples"),
            is_train=False,
            filter_unknown_answers=filter_unknown,
            stratify=False,   # val: plain truncation, preserve order
            seed=seed,
            image_aspect_ratio=image_aspect_ratio,
        )
    else:
        raise ValueError(f"split_name must be 'train' or 'val', got: {split_name}")
