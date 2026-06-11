#!/usr/bin/env bash
# Pull a completed train cache from HPC IRIS scratch to vonasah.
# Run this on vonasah after an HPC cache job finishes.
#
# Usage:
#   bash VQA_V2/shared/experiments/rsync_from_hpc.sh dense
#   bash VQA_V2/shared/experiments/rsync_from_hpc.sh static 288
#
# Or pull all at once (slower, skips already-synced):
#   bash VQA_V2/shared/experiments/rsync_from_hpc.sh all

set -euo pipefail

HPC_USER="nmo"
HPC_HOST="access-iris.uni.lu"
HPC_PORT="8022"
HPC_SCRATCH="/scratch/users/nmo/vlm-thesis/vqa_v2_cache"
LOCAL_CACHE="/home/nafees/vlm-thesis/VQA_V2/feature_cache"

MODEL_TYPE="${1:?Usage: $0 <dense|static|all> [keep_tokens]}"
KEEP_TOKENS="${2:-}"

rsync_key() {
    local key="$1"
    local remote="${HPC_SCRATCH}/${key}/train"
    local local_dir="${LOCAL_CACHE}/${key}/train"

    echo "[rsync] Pulling ${key}/train from HPC..."
    mkdir -p "${local_dir}"
    rsync -av --progress \
        -e "ssh -p ${HPC_PORT}" \
        "${HPC_USER}@${HPC_HOST}:${remote}/" \
        "${local_dir}/"

    if [ -f "${local_dir}/pooled_features.npy" ]; then
        SIZE=$(du -sh "${local_dir}" | cut -f1)
        echo "[rsync] ${key}/train: done (${SIZE})"
    else
        echo "[rsync] ERROR: ${key}/train incomplete — pooled_features.npy missing"
        exit 1
    fi
}

if [ "${MODEL_TYPE}" = "all" ]; then
    rsync_key "dense"
    for K in 64 128 144 192 288 432; do
        rsync_key "static_k${K}"
    done
elif [ "${MODEL_TYPE}" = "dense" ]; then
    rsync_key "dense"
elif [ "${MODEL_TYPE}" = "static" ]; then
    [ -n "${KEEP_TOKENS}" ] || { echo "Usage: $0 static <K>"; exit 1; }
    rsync_key "static_k${KEEP_TOKENS}"
else
    echo "Unknown model_type: ${MODEL_TYPE}"; exit 1
fi

echo ""
echo "Done. The vonasah pipeline will auto-detect the cache and start MLP training."
echo "Check progress: tail -f VQA_V2/logs/vonasah_pipeline.log"
