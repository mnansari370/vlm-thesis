"""VQACollator: PIL images + raw strings stay as lists; labels become tensors."""

from typing import Any, Dict, List

import torch


class VQACollator:
    """
    Batch collator for VQA v2.

    Important design choice:
    - images remain as a list of PIL images
    - question strings remain raw
    - tokenization / image preprocessing is deferred to the LLaVA processor
      inside the model wrapper

    This keeps the dataset code model-agnostic and the wrapper model-aware.
    """

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = [item["image"] for item in batch]

        questions = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]
        raw_answers = [item["raw_answers"] for item in batch]
        normalized_answers = [item["normalized_answers"] for item in batch]

        answer_labels = torch.tensor(
            [item["answer_label"] for item in batch],
            dtype=torch.long,
        )

        question_ids = [item["question_id"] for item in batch]
        image_ids = [item["image_id"] for item in batch]
        image_paths = [item["image_path"] for item in batch]
        active_splits = [item["active_split"] for item in batch]

        return {
            "images": images,
            "questions": questions,
            "answers": answers,
            "raw_answers": raw_answers,
            "normalized_answers": normalized_answers,
            "answer_labels": answer_labels,
            "question_ids": question_ids,
            "image_ids": image_ids,
            "image_paths": image_paths,
            "active_splits": active_splits,
        }