# Package marker only. The legacy package-level imports
# (`from .vqav2 import VQAv2Dataset, build_vqav2_dataset` and
# `from .collate import VQACollator`, the retired classification-era dataset/collator)
# were removed in cleanup Pass 2 (2026-07-05) so importing the active scorer helper
# (`from src.data.vqav2.vqav2_answers import normalize_answer`) no longer pulls retired
# modules (torch Dataset, image transforms) into the final-scope runtime. The retired
# modules live in archive/legacy_datasets/src/data/vqav2/.
