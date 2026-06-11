"""VQAv2 dataset (60k/top-3500 era): questions + annotations + COCO images; majority-answer labels."""

import json
import os
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


class VQAv2Dataset(Dataset):
    """
    VQA v2 dataset aligned with the dense baseline pipeline.

    Supports:
    - train / val split loading
    - generation mode
    - classification mode
    - optional answer vocabulary
    - optional early truncation with max_samples

    Important classification-mode behavior:
    - if answer_mode == "classification" and filter_unknown_answers == True,
      samples whose majority answer is not present in the answer vocabulary
      are skipped during dataset construction.
    """

    def __init__(
        self,
        active_split: str,
        image_dir: str,
        questions_path: str,
        annotations_path: Optional[str],
        image_size: int = 336,
        answer_mode: str = "generation",
        answer_vocab_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        is_train: bool = False,
        filter_unknown_answers: bool = False,
    ):
        if active_split not in {"train", "val"}:
            raise ValueError(
                f"active_split must be 'train' or 'val', got: {active_split}"
            )

        if answer_mode not in {"generation", "classification"}:
            raise ValueError(
                f"answer_mode must be 'generation' or 'classification', got: {answer_mode}"
            )

        self.active_split = active_split
        self.image_dir = image_dir
        self.questions_path = questions_path
        self.annotations_path = annotations_path
        self.image_size = image_size
        self.answer_mode = answer_mode
        self.max_samples = max_samples
        self.filter_unknown_answers = filter_unknown_answers
        self.transform = build_image_transform(image_size=image_size, is_train=is_train)

        self.answer_to_id = None
        self.id_to_answer = None
        if answer_vocab_path is not None and os.path.exists(answer_vocab_path):
            self.answer_to_id = load_answer_vocab(answer_vocab_path)
            self.id_to_answer = build_id_to_answer(self.answer_to_id)

        self.samples = self._build_samples()

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _infer_image_filename(self, image_id: int) -> str:
        """
        COCO naming in VQA v2:
        train: COCO_train2014_000000000000.jpg
        val:   COCO_val2014_000000000000.jpg
        """
        if "train2014" in self.image_dir:
            prefix = "COCO_train2014_"
        elif "val2014" in self.image_dir:
            prefix = "COCO_val2014_"
        else:
            raise ValueError(
                f"Could not infer COCO prefix from image_dir: {self.image_dir}"
            )

        return f"{prefix}{image_id:012d}.jpg"

    def _build_samples(self) -> List[Dict[str, Any]]:
        questions_data = self._load_json(self.questions_path)
        questions = questions_data["questions"]

        annotations_by_qid = {}
        if self.annotations_path is not None and os.path.exists(self.annotations_path):
            annotations_data = self._load_json(self.annotations_path)
            for ann in annotations_data["annotations"]:
                annotations_by_qid[ann["question_id"]] = ann

        samples: List[Dict[str, Any]] = []
        num_skipped_unknown = 0

        for q in questions:
            question_id = q["question_id"]
            image_id = q["image_id"]
            question_text = q["question"]

            ann = annotations_by_qid.get(question_id)

            raw_answers: List[str] = []
            normalized_answers: List[str] = []
            majority_answer = ""
            answer_label = -1

            if ann is not None:
                raw_answers = [a["answer"] for a in ann.get("answers", [])]
                normalized_answers = normalize_answers(raw_answers)
                majority_answer = get_majority_answer(raw_answers)

                if self.answer_mode == "classification":
                    answer_label = answer_to_label(
                        majority_answer,
                        self.answer_to_id,
                        unknown_index=-1,
                    )

                    if self.filter_unknown_answers and answer_label == -1:
                        num_skipped_unknown += 1
                        continue

            image_filename = self._infer_image_filename(image_id)
            image_path = os.path.join(self.image_dir, image_filename)

            samples.append(
                {
                    "question_id": question_id,
                    "image_id": image_id,
                    "question": question_text,
                    "raw_answers": raw_answers,
                    "normalized_answers": normalized_answers,
                    "majority_answer": majority_answer,
                    "answer_label": answer_label,
                    "image_path": image_path,
                    "active_split": self.active_split,
                }
            )

            if self.max_samples is not None and len(samples) >= self.max_samples:
                break

        if self.answer_mode == "classification" and self.filter_unknown_answers:
            print(
                f"[Info] Split={self.active_split} | "
                f"kept {len(samples)} samples after filtering unknown answers | "
                f"skipped {num_skipped_unknown} samples",
                flush=True,
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]

        if not os.path.exists(sample["image_path"]):
            raise FileNotFoundError(f"Image not found: {sample['image_path']}")

        image = Image.open(sample["image_path"]).convert("RGB")
        image = self.transform(image)

        return {
            "image": image,
            "question": sample["question"],
            "answer": sample["majority_answer"],
            "answer_label": sample["answer_label"],
            "raw_answers": sample["raw_answers"],
            "normalized_answers": sample["normalized_answers"],
            "question_id": sample["question_id"],
            "image_id": sample["image_id"],
            "image_path": sample["image_path"],
            "active_split": sample["active_split"],
        }


def build_vqav2_dataset(cfg: Dict[str, Any], split_name: str) -> VQAv2Dataset:
    """
    Build VQAv2Dataset from config.

    split_name:
      - "train"
      - "val"
    """
    dataset_cfg = cfg["dataset"]

    train_split = dataset_cfg["train_split"]
    val_split = dataset_cfg["val_split"]

    filter_unknown_answers = (
        dataset_cfg["answer_mode"] == "classification"
        and bool(dataset_cfg.get("filter_unknown_answers", True))
    )

    if split_name == "train":
        active_split = train_split
        image_dir = dataset_cfg["image_dir_train"]
        questions_path = dataset_cfg["questions_train"]
        annotations_path = dataset_cfg["annotations_train"]
        max_samples = dataset_cfg.get("max_samples")
        is_train = True

    elif split_name == "val":
        active_split = val_split
        image_dir = dataset_cfg["image_dir_val"]
        questions_path = dataset_cfg["questions_val"]
        annotations_path = dataset_cfg["annotations_val"]
        max_samples = dataset_cfg.get("max_val_samples", dataset_cfg.get("max_samples"))
        is_train = False

    else:
        raise ValueError(f"split_name must be 'train' or 'val', got: {split_name}")

    return VQAv2Dataset(
        active_split=active_split,
        image_dir=image_dir,
        questions_path=questions_path,
        annotations_path=annotations_path,
        image_size=dataset_cfg["image_size"],
        answer_mode=dataset_cfg["answer_mode"],
        answer_vocab_path=dataset_cfg.get("answer_vocab_path"),
        max_samples=max_samples,
        is_train=is_train,
        filter_unknown_answers=filter_unknown_answers,
    )