#!/usr/bin/env bash
# ============================================================================
# download_fast.sh — aria2c (16-connection) version of the big-image fetch.
# Resumes the COCO partial left by download_mix.sh, fetches Visual Genome,
# extracts, and symlinks the already-present GQA + TextVQA images.
#
# cocodataset.org throttles per-connection (~0.3 MB/s single, ~5 MB/s x16),
# so this finishes the remaining COCO+VG in ~1h instead of ~18h.
#
# Fire detached:
#   setsid nohup bash scripts/data/download_fast.sh > results/thesis_main/highres/download_fast.log 2>&1 < /dev/null &
#
# Resumable + idempotent (.done_coco / .done_vg markers; aria2c --continue).
# ============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ARIA="/home/nafees/miniconda3/bin/aria2c"
A_OPTS=(-x16 -s16 -k1M --file-allocation=none --continue=true
        --max-tries=8 --retry-wait=5 --console-log-level=warn --summary-interval=30)
IMG="data/llava_mix/images"
mkdir -p "$IMG/coco" "$IMG/vg" "$IMG/gqa" "$IMG/textvqa"

declare -A STATUS
log(){ echo "[$(date '+%H:%M:%S')] $*"; }

# ── COCO train2017 (resume the existing wget partial) ────────────────────────
if [ -f "$IMG/.done_coco" ]; then
  STATUS[coco]=SKIP; log "COCO already done, skip."
else
  log "aria2c resuming COCO train2017 (x16) ..."
  if "$ARIA" "${A_OPTS[@]}" -d "$IMG/coco" -o train2017.zip \
        "http://images.cocodataset.org/zips/train2017.zip"; then
    log "unzip COCO ..."
    if unzip -q -n "$IMG/coco/train2017.zip" -d "$IMG/coco/"; then
      rm -f "$IMG/coco/train2017.zip"; touch "$IMG/.done_coco"; STATUS[coco]=OK
    else
      STATUS[coco]=UNZIP_FAIL
      log "COCO unzip failed (zip may be corrupt) — delete train2017.zip and re-run to refetch."
    fi
  else
    STATUS[coco]=FAIL
  fi
fi

# ── Visual Genome (two parts) ────────────────────────────────────────────────
if [ -f "$IMG/.done_vg" ]; then
  STATUS[vg]=SKIP; log "VG already done, skip."
else
  ok=1
  log "aria2c VG part 1 (x16) ..."
  "$ARIA" "${A_OPTS[@]}" -d "$IMG/vg" -o vg1.zip \
      "https://cs.stanford.edu/people/rak248/VG_100K/images.zip" \
    && unzip -q -n "$IMG/vg/vg1.zip" -d "$IMG/vg/" && rm -f "$IMG/vg/vg1.zip" || ok=0
  log "aria2c VG part 2 (x16) ..."
  "$ARIA" "${A_OPTS[@]}" -d "$IMG/vg" -o vg2.zip \
      "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip" \
    && unzip -q -n "$IMG/vg/vg2.zip" -d "$IMG/vg/" && rm -f "$IMG/vg/vg2.zip" || ok=0
  # images.zip can extract FLAT into vg/ instead of vg/VG_100K/ — normalize so the
  # mix's "vg/VG_100K/<id>.jpg" paths resolve (images2.zip already makes VG_100K_2/).
  if compgen -G "$IMG/vg/*.jpg" >/dev/null 2>&1; then
    log "normalizing flat VG extraction into VG_100K/ ..."
    mkdir -p "$IMG/vg/VG_100K"
    find "$IMG/vg" -maxdepth 1 -name '*.jpg' -exec mv -t "$IMG/vg/VG_100K/" {} +
  fi
  if [ "$ok" = 1 ] && [ -d "$IMG/vg/VG_100K" ] && [ -d "$IMG/vg/VG_100K_2" ]; then
    touch "$IMG/.done_vg"; STATUS[vg]=OK
  else
    STATUS[vg]=FAIL
    log "VG may have failed or extracted to an unexpected folder — check $IMG/vg/."
  fi
fi

# ── symlinks: reuse existing GQA + TextVQA images (no re-download) ────────────
[ -d "$REPO_ROOT/data/gqa/images/images" ] \
  && ln -sfn "$REPO_ROOT/data/gqa/images/images" "$IMG/gqa/images"
[ -d "$REPO_ROOT/data/textvqa/train_images" ] \
  && ln -sfn "$REPO_ROOT/data/textvqa/train_images" "$IMG/textvqa/train_images"
STATUS[symlinks]=OK
STATUS[ocr_vqa]=MANUAL

# ── summary ──────────────────────────────────────────────────────────────────
echo ""
log "================ SUMMARY ================"
for k in coco vg symlinks ocr_vqa; do printf '  %-10s : %s\n' "$k" "${STATUS[$k]:-?}"; done
log "Re-running resumes only the missing parts."
log "=== done ==="
