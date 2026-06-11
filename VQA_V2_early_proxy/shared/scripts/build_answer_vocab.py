import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Tuple

from VQA_V2_early_proxy.shared.datasets.vqav2_answers import get_majority_answer


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_answer_vocab_from_annotations(
    annotations_path: str,
    top_k: int,
) -> Tuple[Dict[str, int], Dict[str, str], Dict[str, int]]:
    """
    Build answer vocabulary from VQA v2 training annotations.

    Strategy:
    - For each question, compute the normalized majority answer
    - Count frequency across training questions
    - Keep the top_k most frequent answers
    """
    data = load_json(annotations_path)
    annotations = data["annotations"]

    answer_counter = Counter()

    for ann in annotations:
        raw_answers = [a["answer"] for a in ann.get("answers", [])]
        majority_answer = get_majority_answer(raw_answers)
        if majority_answer != "":
            answer_counter[majority_answer] += 1

    most_common = answer_counter.most_common(top_k)

    answer_to_id = {answer: idx for idx, (answer, _) in enumerate(most_common)}
    id_to_answer = {str(idx): answer for answer, idx in answer_to_id.items()}

    metadata = {
        "num_total_unique_answers": len(answer_counter),
        "num_selected_answers": len(answer_to_id),
        "top_k_requested": int(top_k),
    }

    return answer_to_id, id_to_answer, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Build top-K VQA answer vocabulary from training annotations."
    )
    parser.add_argument(
        "--annotations-path",
        type=str,
        default="./data/vqav2/v2_mscoco_train2014_annotations.json",
        help="Path to VQA v2 training annotations JSON.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./data/vqav2/answer_vocab_topk.json",
        help="Path to save answer vocabulary JSON.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3000,
        help="Maximum number of answers to keep.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.annotations_path):
        raise FileNotFoundError(f"Annotations file not found: {args.annotations_path}")

    answer_to_id, id_to_answer, metadata = build_answer_vocab_from_annotations(
        annotations_path=args.annotations_path,
        top_k=args.top_k,
    )

    output_data = {
        "answer_to_id": answer_to_id,
        "id_to_answer": id_to_answer,
        "metadata": metadata,
    }

    parent = os.path.dirname(args.output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Saved answer vocabulary to: {args.output_path}")
    print(f"Selected answers: {metadata['num_selected_answers']}")
    print(f"Total unique normalized majority answers: {metadata['num_total_unique_answers']}")


if __name__ == "__main__":
    main()