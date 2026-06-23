from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen2.5-VL-3B-Instruct", ignore_patterns=["*.bin"], max_workers=8)
print("DONE:", p, flush=True)
