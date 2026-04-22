#!/bin/bash
# Flare-style CTDG launcher:
#   1. Single-process prepare with the target global partition count
#   2. Multi-node/multi-GPU torchrun train
#   3. Multi-node/multi-GPU torchrun predict
#
# Usage:
#   bash run_tgn_ctdg_flare.sh [prepare|train|predict|all]
#
# Key environment variables:
#   CONFIG=configs/tgn_wiki.yaml
#   ARTIFACT_ROOT=/shared/path/artifacts/WIKI
#   NNODES=2
#   NODE_RANK=0
#   NPROC_PER_NODE=4
#   MASTER_ADDR=node0.example.com
#   MASTER_PORT=29500
#
# Notes:
# - `prepare` is single-process, but it still needs the target global worker count
#   so artifact partitioning matches the later distributed run.
# - After `prepare`, copy `${ARTIFACT_ROOT}` to every training node if it is not on
#   shared storage.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODE="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/tgn_wiki.yaml}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${SCRIPT_DIR}/artifacts/$(basename "${CONFIG}" .yaml)}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
WORLD_SIZE="$((NNODES * NPROC_PER_NODE))"

log() {
    printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

run_prepare() {
    log "prepare | config=${CONFIG} | artifact_root=${ARTIFACT_ROOT} | world_size=${WORLD_SIZE}"
    WORLD_SIZE="${WORLD_SIZE}" \
    "${PYTHON_BIN}" -m starry_unigraph \
        --config "${CONFIG}" \
        --artifact-root "${ARTIFACT_ROOT}" \
        --phase prepare
    log "prepare complete"
    log "copy ${ARTIFACT_ROOT} to every node before train/predict if storage is not shared"
}

run_train() {
    log "train | nnodes=${NNODES} | node_rank=${NODE_RANK} | nproc_per_node=${NPROC_PER_NODE}"
    torchrun \
        --nnodes="${NNODES}" \
        --node_rank="${NODE_RANK}" \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        -m starry_unigraph \
        --config "${CONFIG}" \
        --artifact-root "${ARTIFACT_ROOT}" \
        --phase train
    log "train complete"
}

run_predict() {
    log "predict | nnodes=${NNODES} | node_rank=${NODE_RANK} | nproc_per_node=${NPROC_PER_NODE}"
    torchrun \
        --nnodes="${NNODES}" \
        --node_rank="${NODE_RANK}" \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="$((MASTER_PORT + 1))" \
        -m starry_unigraph \
        --config "${CONFIG}" \
        --artifact-root "${ARTIFACT_ROOT}" \
        --phase predict
    log "predict complete"
}

case "${MODE}" in
    prepare) run_prepare ;;
    train) run_train ;;
    predict) run_predict ;;
    all)
        run_prepare
        run_train
        run_predict
        ;;
    *)
        echo "Usage: $0 [prepare|train|predict|all]" >&2
        exit 1
        ;;
esac
