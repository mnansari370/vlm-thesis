from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct", ignore_patterns=["*.bin"], max_workers=8)
print("DOWNLOAD_COMPLETE:", p, flush=True)
