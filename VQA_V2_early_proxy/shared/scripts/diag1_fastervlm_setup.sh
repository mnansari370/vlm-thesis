#!/usr/bin/env bash
# Diagnostic 1 — FasterVLM on our val2014 subset
#
# Runs FasterVLM's original eval pipeline on our 10K val2014 split.
# Purpose: confirm their K=128 number is consistent with our corrected pipeline.
#
# Usage (from repo root, after running diag1_convert_data.py first):
#   bash scripts/diag1_fastervlm_setup.sh       # step 1: env + install
#   bash scripts/diag1_fastervlm_run.sh 128     # step 2: eval at K=128
#
# Requirements:
#   - FasterVLM repo already cloned at /home/nafees/FasterVLM
#   - This script creates a separate conda env 'fastervlm_eval'
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
REPO="/home/nafees/FasterVLM"

echo "=== Step 1: Create conda env fastervlm_eval ==="
conda create -y -n fastervlm_eval python=3.10 2>&1 | tail -5

echo ""
echo "=== Step 2: Install FasterVLM dependencies ==="
# FasterVLM requires torch==2.1.2, transformers==4.37.2 — install in their env
conda run -n fastervlm_eval pip install \
    torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3

conda run -n fastervlm_eval pip install \
    transformers==4.37.2 tokenizers==0.15.1 sentencepiece shortuuid \
    accelerate==0.21.0 peft bitsandbytes einops==0.6.1 timm==0.6.13 \
    protobuf scikit-learn==1.2.2 2>&1 | tail -5

echo ""
echo "=== Step 3: Install FasterVLM package ==="
conda run -n fastervlm_eval pip install -e "$REPO" 2>&1 | tail -3

echo ""
echo "=== Step 4: Convert val2014 data to FasterVLM JSONL format ==="
conda run -n fastervlm_eval python VQA_V2_early_proxy/shared/scripts/diag1_convert_data.py

echo ""
echo "Done. Now run:  bash scripts/diag1_fastervlm_run.sh 128"
