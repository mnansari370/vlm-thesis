#!/usr/bin/env bash
# ============================================================================
# download_mix.sh — overnight, resumable download of the LLaVA-665K training data.
#
# Fire-and-forget. Designed to run unattended (e.g. before sleeping):
#     nohup bash scripts/data/download_mix.sh > /tmp/v2_download.log 2>&1 &
#     tail -f /tmp/v2_download.log        # to watch
#
# Safety / design:
#   - ADDS only under data/llava_mix/ ; never touches data/gqa, data/textvqa, or any v1 data.
#   - Resumable: wget -c continues partial files; extraction is skipped if already done.
#   - Does NOT abort on one failure — it logs, continues, and prints a summary at the end,
#     so you wake up to maximum progress and a clear list of what (if anything) needs a retry.
#   - REUSES your existing GQA + TextVQA images via symlinks (no re-download).
#
# NOTE: URLs below are the standard public sources but can change / rate-limit. If a step
# fails, the summary tells you which one; re-running the script resumes only the missing parts.
# OCR-VQA is the known-finicky one (book-cover URLs via its own script) — handled separately,
# flagged, and NON-fatal here.
# ============================================================================

set -uo pipefail   # NOT -e: we want to continue past a failed step and report at the end

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ROOT="data/llava_mix"
IMG="$ROOT/images"
mkdir -p "$IMG/coco" "$IMG/vg" "$IMG/ocr_vqa" "$IMG/gqa" "$IMG/textvqa"

declare -A STATUS   # step -> OK / FAIL / SKIP

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
done_marker() { echo "$IMG/.done_$1"; }   # presence => that step already completed

# disk space guard (warn if < 100 GB free on the data partition)
free_gb=$(df -BG --output=avail "$ROOT" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
log "Free space on data partition: ${free_gb} GB"
if [ "${free_gb:-0}" -lt 100 ]; then
  log "WARNING: < 100 GB free. COCO(~18G)+VG(~15G)+zips need transient room. Continuing anyway."
fi

# ── 1) the mix JSON ──────────────────────────────────────────────────────────
fetch_json() {
  local url="https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json"
  local out="$ROOT/llava_v1_5_mix665k.json"
  if [ -s "$out" ]; then STATUS[json]=SKIP; log "JSON already present, skip."; return; fi
  log "Downloading mix JSON ..."
  if wget -c -q --show-progress -O "$out" "$url"; then STATUS[json]=OK; else STATUS[json]=FAIL; fi
}

# ── 2) COCO train2017 (~18 GB) ───────────────────────────────────────────────
fetch_coco() {
  local m; m="$(done_marker coco)"
  if [ -f "$m" ]; then STATUS[coco]=SKIP; log "COCO already extracted, skip."; return; fi
  local zip="$IMG/coco/train2017.zip"
  log "Downloading COCO train2017 (~18 GB, resumable) ..."
  if ! wget -c -O "$zip" "http://images.cocodataset.org/zips/train2017.zip"; then
    STATUS[coco]=FAIL; log "COCO download failed."; return; fi
  log "Extracting COCO train2017 ..."
  if unzip -q -n "$zip" -d "$IMG/coco/"; then
    rm -f "$zip"; touch "$m"; STATUS[coco]=OK
  else STATUS[coco]=FAIL; fi
}

# ── 3) Visual Genome (two parts, ~15 GB) ─────────────────────────────────────
fetch_vg() {
  local m; m="$(done_marker vg)"
  if [ -f "$m" ]; then STATUS[vg]=SKIP; log "VG already extracted, skip."; return; fi
  local ok=1
  log "Downloading Visual Genome part 1 ..."
  if wget -c -O "$IMG/vg/vg1.zip" "https://cs.stanford.edu/people/rak248/VG_100K/images.zip"; then
    unzip -q -n "$IMG/vg/vg1.zip" -d "$IMG/vg/" && rm -f "$IMG/vg/vg1.zip" || ok=0
  else ok=0; fi
  log "Downloading Visual Genome part 2 ..."
  if wget -c -O "$IMG/vg/vg2.zip" "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip"; then
    unzip -q -n "$IMG/vg/vg2.zip" -d "$IMG/vg/" && rm -f "$IMG/vg/vg2.zip" || ok=0
  else ok=0; fi
  # VG zips may extract to VG_100K/ and VG_100K_2/ already — verify
  if [ "$ok" = 1 ] && { [ -d "$IMG/vg/VG_100K" ] || [ -d "$IMG/vg/VG_100K_2" ]; }; then
    touch "$m"; STATUS[vg]=OK
  else
    STATUS[vg]=FAIL
    log "VG may have failed or extracted to an unexpected folder — check $IMG/vg/ layout."
  fi
}

# ── 4) symlinks to REUSE existing GQA + TextVQA images (no re-download) ───────
link_existing() {
  local ok=1
  # GQA jpgs live at data/gqa/images/images (v1 layout) -> mix path prefix "gqa/images/"
  if [ -d "$REPO_ROOT/data/gqa/images/images" ]; then
    ln -sfn "$REPO_ROOT/data/gqa/images/images" "$IMG/gqa/images"
  else log "WARN: data/gqa/images/images not found — GQA symlink skipped."; ok=0; fi
  # TextVQA train images -> mix path prefix "textvqa/train_images/"
  if [ -d "$REPO_ROOT/data/textvqa/train_images" ]; then
    ln -sfn "$REPO_ROOT/data/textvqa/train_images" "$IMG/textvqa/train_images"
  else log "WARN: data/textvqa/train_images not found — TextVQA symlink skipped."; ok=0; fi
  [ "$ok" = 1 ] && STATUS[symlinks]=OK || STATUS[symlinks]="PARTIAL"
}

# ── 5) OCR-VQA (known-finicky; flagged, non-fatal) ───────────────────────────
note_ocrvqa() {
  STATUS[ocr_vqa]="MANUAL"
  log "OCR-VQA images are NOT auto-downloaded here (book-cover URLs need OCR-VQA's own"
  log "loadDataset.py). Do this one together later, or follow the LLaVA data-prep notes,"
  log "placing images under $IMG/ocr_vqa/images/. The rest of training can start without it."
}

# ── run ──────────────────────────────────────────────────────────────────────
log "=== LLaVA-665K download starting (root: $ROOT) ==="
fetch_json
fetch_coco
fetch_vg
link_existing
note_ocrvqa

echo ""
log "================ SUMMARY ================"
for k in json coco vg symlinks ocr_vqa; do
  printf '  %-10s : %s\n' "$k" "${STATUS[$k]:-?}"
done
log "Re-running this script resumes only the missing parts."
log "Next: validate paths with  python -m v2.data_setup.verify_images  (built next)."
log "=== done ==="
