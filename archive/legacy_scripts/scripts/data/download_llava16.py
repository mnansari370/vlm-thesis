"""
Resumable download of LLaVA-1.6-vicuna-7b (HF format) into the HF cache.
Same CLIP-336 vision tower + Vicuna-7B LLM as our LLaVA-1.5, but with AnyRes
high-res tiling (~2880 visual tokens) — the regime where the visual-token budget
actually binds (Option A diagnostic). safetensors only (no .bin). ~14.1 GB.

snapshot_download resumes partial files automatically, so re-running is safe.
"""
import sys
from huggingface_hub import snapshot_download

REPO = "llava-hf/llava-v1.6-vicuna-7b-hf"

if __name__ == "__main__":
    path = snapshot_download(REPO, ignore_patterns=["*.bin"], max_workers=8)
    print(f"DOWNLOAD_COMPLETE: {path}", flush=True)
