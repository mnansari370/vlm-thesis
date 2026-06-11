"""Answer normalization (lowercase, strip punctuation, number words, articles), majority answer, vocab IO."""

import json
import re
from collections import Counter
from typing import Dict, List, Optional


_NUMBER_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

_ARTICLES = {"a", "an", "the"}


def normalize_answer(text: Optional[str]) -> str:
    """
    Lightweight VQA-style answer normalization.

    This is intentionally simpler than the full official VQA normalization,
    but it is stable and suitable for:
    - answer vocabulary building
    - majority-answer supervision
    - simple answer comparison
    """
    if text is None:
        return ""

    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)

    tokens = text.split()
    tokens = [_NUMBER_MAP.get(tok, tok) for tok in tokens]
    tokens = [tok for tok in tokens if tok not in _ARTICLES]

    return " ".join(tokens)


def normalize_answers(answers: List[str]) -> List[str]:
    return [normalize_answer(a) for a in answers]


def get_majority_answer(answers: List[str]) -> str:
    """
    Return the most frequent normalized answer.
    Used for classification-mode supervision.
    """
    normalized = normalize_answers(answers)
    if not normalized:
        return ""

    counter = Counter(normalized)
    return counter.most_common(1)[0][0]


def load_answer_vocab(path: str) -> Dict[str, int]:
    """
    Load answer vocabulary from JSON.

    Supported formats:
    1) {"answer_to_id": {...}, "id_to_answer": {...}}
    2) {"yes": 0, "no": 1, ...}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "answer_to_id" in data:
        return data["answer_to_id"]
    return data


def build_id_to_answer(answer_to_id: Dict[str, int]) -> Dict[int, str]:
    return {idx: ans for ans, idx in answer_to_id.items()}


def answer_to_label(
    answer: str,
    answer_to_id: Optional[Dict[str, int]],
    unknown_index: int = -1,
) -> int:
    if answer_to_id is None:
        return unknown_index
    return answer_to_id.get(answer, unknown_index)