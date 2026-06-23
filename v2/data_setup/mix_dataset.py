"""
mix_dataset.py — loader for the LLaVA-1.5 instruction mix (llava_v1_5_mix665k.json).

The mix JSON is a list of records:
    {"id": ..., "image": "<source>/.../file.jpg",
     "conversations": [{"from": "human"|"gpt", "value": "..."}, ...]}

Design choices (deliberate, see ../README.md):
  - Text-only rows (ShareGPT, no "image" key) are DROPPED — nothing to prune.
  - PURE PARSING (clean_question / source_of / parse_record) has no file I/O, so the
    format edge-cases are unit-testable offline with synthetic records (no download).
  - Examples are GENERATION-style (image, question, answer) — never answer classification.
  - One image per record; multi-turn conversations are handled by `turn_mode`.

turn_mode:
  "first" : one example per record = the first (question, answer) turn (clean single-question;
            best for the question-conditioned selector / budget head).
  "all"   : expand every (question, answer) turn into its own example, all sharing the image
            (more data for the elastic backbone; later turns lack conversation history, so
            use with care — documented, not default).

Returned __getitem__ dict:
    {"image": PIL.Image (RGB), "question": str, "answer": str, "source": str, "id": str}
Compatible with mix_collate() below (keeps PIL images as a list, like the v1 collators).
"""

import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset


IMAGE_TOKEN = "<image>"
VALID_TURN_MODES = frozenset({"first", "all"})


# ── pure parsing (no I/O — unit-testable) ─────────────────────────────────────

def clean_question(text: Optional[str]) -> str:
    """Remove the <image> placeholder and collapse whitespace/newlines."""
    if not text:
        return ""
    text = text.replace(IMAGE_TOKEN, " ")
    return " ".join(text.split()).strip()


def source_of(image_path: Optional[str]) -> str:
    """First path component identifies the source dataset (coco / gqa / ocr_vqa / vg / textvqa)."""
    if not image_path:
        return "unknown"
    return image_path.replace("\\", "/").split("/", 1)[0]


def parse_record(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert one mix entry → {"id", "image", "source", "turns": [(q, a), ...]} or None.

    Returns None for:
      - text-only rows (no "image"), or
      - records with no usable (human → gpt) pair.

    Pairing is robust to ordering: each human turn's text becomes the pending question,
    answered by the next gpt turn. Stray/duplicate turns are ignored, not crashed on.
    """
    image = raw.get("image")
    if not image:
        return None  # text-only (ShareGPT) — nothing to prune

    convs = raw.get("conversations") or []
    turns: List[Tuple[str, str]] = []
    pending_q: Optional[str] = None

    for turn in convs:
        if not isinstance(turn, dict):
            continue
        role = turn.get("from")
        val = turn.get("value", "")
        if role == "human":
            pending_q = clean_question(val)
        elif role == "gpt":
            if pending_q is not None and pending_q != "":
                turns.append((pending_q, (val or "").strip()))
                pending_q = None

    if not turns:
        return None

    return {
        "id": raw.get("id"),
        "image": image,
        "source": source_of(image),
        "turns": turns,
    }


def records_to_examples(records: List[Dict[str, Any]], turn_mode: str) -> List[Dict[str, Any]]:
    """Flatten parsed records into per-example dicts according to turn_mode."""
    if turn_mode not in VALID_TURN_MODES:
        raise ValueError(f"turn_mode must be one of {VALID_TURN_MODES}, got '{turn_mode}'")
    examples: List[Dict[str, Any]] = []
    for rec in records:
        turn_list = rec["turns"][:1] if turn_mode == "first" else rec["turns"]
        for q, a in turn_list:
            examples.append({
                "id": rec["id"],
                "image": rec["image"],
                "source": rec["source"],
                "question": q,
                "answer": a,
            })
    return examples


def stratified_subsample(
    examples: List[Dict[str, Any]],
    max_samples: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Sample proportionally across `source` buckets, preserving the natural source mix."""
    if max_samples >= len(examples):
        return examples
    rng = random.Random(seed)
    buckets: Dict[str, List[int]] = defaultdict(list)
    for i, ex in enumerate(examples):
        buckets[ex["source"]].append(i)

    total = len(examples)
    chosen: List[int] = []
    for src, idxs in buckets.items():
        rng.shuffle(idxs)
        quota = min(round(max_samples * len(idxs) / total), len(idxs))
        chosen.extend(idxs[:quota])

    rng.shuffle(chosen)
    chosen = sorted(chosen[:max_samples])  # deterministic order
    return [examples[i] for i in chosen]


# ── dataset (file I/O) ────────────────────────────────────────────────────────

class LlavaMixDataset(Dataset):
    """
    LLaVA-665K instruction-mix dataset for v2 training.

    Args:
      json_path     : path to llava_v1_5_mix665k.json (or any subset in the same format).
      image_root    : directory the record "image" paths are relative to
                      (e.g. data/llava_mix/images).
      turn_mode     : "first" (default) or "all".
      max_samples   : optional cap; applied via source-stratified subsampling.
      stratify      : if True (default) and max_samples set, subsample proportionally by source.
      verify_images : if True, check each image exists on disk up front.
      drop_missing  : with verify_images, drop records whose image is missing (else raise).
      seed          : subsample RNG seed.
    """

    def __init__(
        self,
        json_path: str,
        image_root: str,
        turn_mode: str = "first",
        max_samples: Optional[int] = None,
        stratify: bool = True,
        verify_images: bool = False,
        drop_missing: bool = True,
        seed: int = 42,
    ):
        if turn_mode not in VALID_TURN_MODES:
            raise ValueError(f"turn_mode must be one of {VALID_TURN_MODES}, got '{turn_mode}'")
        self.image_root = image_root
        self.turn_mode = turn_mode

        with open(json_path, "r", encoding="utf-8") as f:
            raw_list = json.load(f)

        n_raw = len(raw_list)
        records = [r for r in (parse_record(x) for x in raw_list) if r is not None]
        n_text_only = n_raw - len(records)

        examples = records_to_examples(records, turn_mode)

        n_missing = 0
        if verify_images:
            kept = []
            for ex in examples:
                if os.path.exists(os.path.join(image_root, ex["image"])):
                    kept.append(ex)
                else:
                    n_missing += 1
                    if not drop_missing:
                        raise FileNotFoundError(
                            f"Missing image: {os.path.join(image_root, ex['image'])}"
                        )
            examples = kept

        if max_samples is not None and stratify and len(examples) > max_samples:
            examples = stratified_subsample(examples, max_samples, seed=seed)
        elif max_samples is not None and len(examples) > max_samples:
            examples = examples[:max_samples]

        self.examples = examples

        src_counts = defaultdict(int)
        for ex in examples:
            src_counts[ex["source"]] += 1
        print(
            f"[LlavaMix] {n_raw:,} raw → dropped {n_text_only:,} text-only"
            + (f", {n_missing:,} missing-image" if verify_images else "")
            + f" → {len(examples):,} examples ({turn_mode}). "
            + "by source: " + ", ".join(f"{k}={v}" for k, v in sorted(src_counts.items())),
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        path = os.path.join(self.image_root, ex["image"])
        image = Image.open(path).convert("RGB")
        return {
            "image": image,
            "question": ex["question"],
            "answer": ex["answer"],
            "source": ex["source"],
            "id": ex["id"],
        }


def mix_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Keep PIL images and strings as lists (the v1 collator convention)."""
    return {
        "images":    [b["image"]    for b in batch],
        "questions": [b["question"] for b in batch],
        "answers":   [b["answer"]   for b in batch],
        "sources":   [b["source"]   for b in batch],
        "ids":       [b["id"]       for b in batch],
    }
