# `v2/data_setup/` — LLaVA-665K training data setup

The mix JSON is a **list of relative image paths + conversations**. It does **not** download anything —
you place the images on disk once, then the paths resolve to local files. Everything below lives under
the repo's `data/` (git-ignored), and **reuses** the GQA/TextVQA images you already have via symlinks.

## Target layout

```
data/llava_mix/                      ← NEW (added; replaces nothing)
├── llava_v1_5_mix665k.json          ← the one JSON (a few hundred MB of text)
└── images/
    ├── coco/train2017/              ← NEW download  (~18 GB)   COCO train2017, bare filenames 000000000000.jpg
    ├── ocr_vqa/images/              ← NEW download  (~few GB)  via OCR-VQA's own download script
    ├── vg/VG_100K/                  ← NEW download  (~15 GB)   Visual Genome part 1
    ├── vg/VG_100K_2/                ←               Visual Genome part 2
    ├── gqa/images        ─► symlink to ../../../gqa/images/images        (REUSE — already on disk)
    └── textvqa/train_images ─► symlink to ../../../textvqa/train_images  (REUSE — already on disk)
```

## Sources

| Item | Where | Note |
|---|---|---|
| `llava_v1_5_mix665k.json` | HF `liuhaotian/LLaVA-Instruct-150K` → file `llava_v1_5_mix665k.json` | one file |
| COCO **train2017** | `http://images.cocodataset.org/zips/train2017.zip` | **2017, not 2014** — different filenames; can't reuse your val2014 |
| OCR-VQA images | OCR-VQA project download script (`loadDataset.py`) | a few images may be dead URLs — normal |
| Visual Genome | VG_100K.zip + VG_100K_2.zip (Visual Genome site / HF mirror) | two parts |
| GQA images | already at `data/gqa/images/images` | symlink, don't re-download |
| TextVQA train images | already at `data/textvqa/train_images` | symlink, don't re-download |

## Setup steps (run from repo root)

```bash
mkdir -p data/llava_mix/images/{coco,ocr_vqa,vg}

# 1) the JSON (use huggingface-cli or wget the raw file)
#    -> data/llava_mix/llava_v1_5_mix665k.json

# 2) COCO train2017  -> data/llava_mix/images/coco/train2017/
#    wget the zip, unzip into that folder (filenames look like 000000033471.jpg)

# 3) OCR-VQA  -> data/llava_mix/images/ocr_vqa/images/   (run OCR-VQA's loadDataset.py)

# 4) Visual Genome  -> data/llava_mix/images/vg/VG_100K/  and  .../vg/VG_100K_2/

# 5) REUSE existing images via symlinks (no copy, no re-download):
ln -s "$(pwd)/data/gqa/images/images"      data/llava_mix/images/gqa/images   # if mix uses gqa/images/<id>.jpg
ln -s "$(pwd)/data/textvqa/train_images"   data/llava_mix/images/textvqa/train_images
#   (create the parent dirs first if the symlink target path needs them, e.g. mkdir -p data/llava_mix/images/gqa)
```

## Before training

- **Drop ShareGPT** (text-only rows with no `image` field) — nothing to prune there.
- **Subsample to ~300K** for the pilot (stratify across VQA/GQA/OCR/Ref/VG), scale to full mix later.
- **Validate paths:** scan the JSON and assert every referenced image exists on disk; log+skip any
  missing (expect a few from OCR-VQA). A small `verify_images.py` will live here.
- The loader (built next, `mix_dataset.py`) reads the `conversations` format and yields
  `{image, question, answer, ...}` compatible with the existing collator pattern — **generation targets,
  no answer classification**.

## Safety

All paths are under the git-ignored `data/`. Symlinks point at existing folders read-only. Nothing here
modifies or moves any v1 data; it only adds `data/llava_mix/`.
