"""
Offline unit test for mix_dataset.py — builds tiny synthetic fixtures in a temp dir
(no download, no real images). Run from repo root:

    /home/nafees/miniconda3/envs/vlm_env/bin/python v2/data_setup/test_mix_dataset.py
or  python -m v2.data_setup.test_mix_dataset
"""

import json
import os
import shutil
import sys
import tempfile

# allow running as a plain script from anywhere (repo root = 2 levels up)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from PIL import Image

from v2.data_setup.mix_dataset import (
    clean_question,
    source_of,
    parse_record,
    records_to_examples,
    stratified_subsample,
    LlavaMixDataset,
    mix_collate,
)


def _ok(cond, msg):
    if not cond:
        raise AssertionError("FAIL: " + msg)
    print("  ok:", msg)


def test_pure_parsing():
    print("[1] pure parsing (no I/O)")
    # <image> at start, with newline
    _ok(clean_question("<image>\nWhat color is the bus?") == "What color is the bus?",
        "clean_question strips leading <image> + newline")
    # <image> at end
    _ok(clean_question("What does the sign say?\n<image>") == "What does the sign say?",
        "clean_question strips trailing <image>")
    _ok(clean_question("  spaced   out \n text ") == "spaced out text",
        "clean_question collapses whitespace")
    _ok(clean_question(None) == "" and clean_question("") == "",
        "clean_question handles None/empty")

    _ok(source_of("coco/train2017/000000033471.jpg") == "coco", "source_of coco")
    _ok(source_of("gqa/images/2375429.jpg") == "gqa", "source_of gqa")
    _ok(source_of("ocr_vqa/images/12.jpg") == "ocr_vqa", "source_of ocr_vqa")
    _ok(source_of("vg/VG_100K/2317.jpg") == "vg", "source_of vg")
    _ok(source_of(None) == "unknown", "source_of None")

    # text-only (ShareGPT) → None
    _ok(parse_record({"id": "s1", "conversations": [
        {"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}]}) is None,
        "parse_record drops text-only (no image)")
    # image but empty conversations → None
    _ok(parse_record({"id": "e1", "image": "coco/x.jpg", "conversations": []}) is None,
        "parse_record drops empty-conversation record")
    # image but only a human turn (no answer) → None
    _ok(parse_record({"id": "e2", "image": "coco/x.jpg", "conversations": [
        {"from": "human", "value": "<image>\nq?"}]}) is None,
        "parse_record drops record with a question but no answer")

    # multi-turn
    rec = parse_record({"id": "m1", "image": "coco/train2017/1.jpg", "conversations": [
        {"from": "human", "value": "<image>\nWhat color is the bus?"},
        {"from": "gpt", "value": "White and red."},
        {"from": "human", "value": "What is on the back?"},
        {"from": "gpt", "value": "An advertisement."},
    ]})
    _ok(rec is not None and len(rec["turns"]) == 2, "multi-turn parsed into 2 turns")
    _ok(rec["turns"][0] == ("What color is the bus?", "White and red."), "turn 0 (q,a) correct")
    _ok(rec["source"] == "coco", "record source = coco")


def test_examples_and_subsample():
    print("[2] turn_mode expansion + stratified subsample")
    recs = [
        {"id": "m1", "image": "coco/1.jpg", "source": "coco",
         "turns": [("q1", "a1"), ("q2", "a2")]},
        {"id": "g1", "image": "gqa/2.jpg", "source": "gqa",
         "turns": [("q3", "a3")]},
    ]
    first = records_to_examples(recs, "first")
    _ok(len(first) == 2, "turn_mode='first' → 1 example per record (2 total)")
    allx = records_to_examples(recs, "all")
    _ok(len(allx) == 3, "turn_mode='all' → every turn expanded (3 total)")
    _ok(allx[0]["question"] == "q1" and allx[1]["question"] == "q2", "expanded order preserved")

    # stratified subsample keeps proportions roughly and is deterministic
    big = [{"id": str(i), "image": "coco/x.jpg", "source": ("coco" if i % 2 else "gqa"),
            "question": "q", "answer": "a"} for i in range(100)]
    s1 = stratified_subsample(big, 20, seed=42)
    s2 = stratified_subsample(big, 20, seed=42)
    _ok(len(s1) == 20, "subsample returns requested size")
    _ok([e["id"] for e in s1] == [e["id"] for e in s2], "subsample is deterministic (same seed)")
    srcs = {e["source"] for e in s1}
    _ok(srcs == {"coco", "gqa"}, "subsample keeps both sources represented")


def test_dataset_io():
    print("[3] Dataset end-to-end (synthetic fixtures in tempdir)")
    tmp = tempfile.mkdtemp(prefix="v2_mixtest_")
    try:
        img_root = os.path.join(tmp, "images")
        os.makedirs(os.path.join(img_root, "coco"), exist_ok=True)
        os.makedirs(os.path.join(img_root, "gqa"), exist_ok=True)
        # two tiny real images
        Image.new("RGB", (8, 8), (255, 0, 0)).save(os.path.join(img_root, "coco", "1.jpg"))
        Image.new("RGB", (8, 8), (0, 255, 0)).save(os.path.join(img_root, "gqa", "2.jpg"))
        # gqa/3.jpg is intentionally MISSING

        mix = [
            {"id": "m1", "image": "coco/1.jpg", "conversations": [
                {"from": "human", "value": "<image>\nWhat color?"},
                {"from": "gpt", "value": "Red."},
                {"from": "human", "value": "Anything else?"},
                {"from": "gpt", "value": "No."}]},
            {"id": "g1", "image": "gqa/2.jpg", "conversations": [
                {"from": "human", "value": "<image>\nWhich object?"},
                {"from": "gpt", "value": "A box."}]},
            {"id": "sharegpt1", "conversations": [  # text-only, must drop
                {"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}]},
            {"id": "miss1", "image": "gqa/3.jpg", "conversations": [  # missing image
                {"from": "human", "value": "<image>\nq?"}, {"from": "gpt", "value": "a."}]},
        ]
        jpath = os.path.join(tmp, "mini_mix.json")
        with open(jpath, "w") as f:
            json.dump(mix, f)

        # turn_mode="first", verify_images=True → drops text-only AND missing image → 2 examples
        ds = LlavaMixDataset(jpath, img_root, turn_mode="first",
                             verify_images=True, drop_missing=True)
        _ok(len(ds) == 2, "first+verify → 2 examples (text-only + missing dropped)")
        item = ds[0]
        _ok(item["image"].size == (8, 8) and item["image"].mode == "RGB", "getitem returns RGB PIL")
        _ok(item["question"] == "What color?" and item["answer"] == "Red.", "getitem q/a clean")
        _ok(item["source"] == "coco", "getitem source")

        # turn_mode="all" → m1 contributes 2 turns, g1 one → 3 examples
        ds_all = LlavaMixDataset(jpath, img_root, turn_mode="all",
                                 verify_images=True, drop_missing=True)
        _ok(len(ds_all) == 3, "all+verify → 3 examples (multi-turn expanded)")

        # collate
        batch = mix_collate([ds[0], ds[1]])
        _ok(batch["questions"] == ["What color?", "Which object?"], "collate batches questions")
        _ok(len(batch["images"]) == 2, "collate keeps images as list")

        # drop_missing=False should raise on the missing image
        raised = False
        try:
            LlavaMixDataset(jpath, img_root, turn_mode="first",
                            verify_images=True, drop_missing=False)
        except FileNotFoundError:
            raised = True
        _ok(raised, "drop_missing=False raises on missing image")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_pure_parsing()
    test_examples_and_subsample()
    test_dataset_io()
    print("\nALL TESTS PASSED ✓")


if __name__ == "__main__":
    main()
