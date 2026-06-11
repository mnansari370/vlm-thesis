import argparse
from pprint import pprint

from torch.utils.data import DataLoader

from VQA_V2_early_proxy.shared.datasets import VQACollator, build_vqav2_dataset
from VQA_V2_early_proxy.shared.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Inspect one batch from the VQA dataset pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Which split to inspect.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2,
        help="Number of items to inspect in one batch.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = build_vqav2_dataset(cfg, args.split)
    collator = VQACollator()

    loader = DataLoader(
        dataset,
        batch_size=args.num_samples,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    batch = next(iter(loader))

    print("=" * 80)
    print(f"Split: {args.split}")
    print(f"Batch size: {len(batch['questions'])}")
    print("=" * 80)

    print("Keys in batch:")
    pprint(list(batch.keys()))
    print()

    print("Questions:")
    pprint(batch["questions"])
    print()

    print("Answers:")
    pprint(batch["answers"])
    print()

    print("Raw answers:")
    pprint(batch["raw_answers"])
    print()

    print("Answer labels:")
    print(batch["answer_labels"])
    print()

    print("Question IDs:")
    pprint(batch["question_ids"])
    print()

    print("Image IDs:")
    pprint(batch["image_ids"])
    print()

    print("Image paths:")
    pprint(batch["image_paths"])
    print()

    print("Images type:")
    print(type(batch["images"]))
    if len(batch["images"]) > 0:
        print("First image type:", type(batch["images"][0]))
        print("First image size:", getattr(batch["images"][0], "size", None))
        print("First image mode:", getattr(batch["images"][0], "mode", None))

    print("=" * 80)
    print("Batch inspection completed.")


if __name__ == "__main__":
    main()