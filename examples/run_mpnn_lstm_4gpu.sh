#!/bin/bash
# run_mpnn_lstm_4gpu.sh
# Full MPNN-LSTM training pipeline on rec-amazon-ratings with 4 GPUs.
# Prints per-epoch loss, timing, and final test-set regression metrics (MAE/RMSE/NMAE).
#
# Usage:
#   bash run_mpnn_lstm_4gpu.sh [prepare|train|predict|all]
#
# Default (no arg): runs all three steps in sequence.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$(conda run -n tgnn_3.10 which python 2>/dev/null || echo python)"
ARTIFACT_DIR="/mnt/data/zlj/starrygl-artifacts/rec-amazon-ratings"
LOG_DIR="/mnt/data/zlj/starrygl-logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/mpnn_lstm_4gpu_${TIMESTAMP}.log"

mkdir -p "${ARTIFACT_DIR}" "${LOG_DIR}"

cd "${SCRIPT_DIR}"

MODE="${1:-all}"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

run_prepare() {
    log "=== STEP 1: Data preprocessing (single process) ==="
    conda run -n tgnn_3.10 python train_mpnn_lstm_4gpu.py --mode prepare 2>&1 | tee -a "${LOG_FILE}"
    log "=== Preprocessing done ==="
}

run_train() {
    log "=== STEP 2: Distributed training (4 GPUs) ==="
    conda run -n tgnn_3.10 torchrun \
        --nproc_per_node=4 \
        --master_port=29600 \
        train_mpnn_lstm_4gpu.py --mode train 2>&1 | tee -a "${LOG_FILE}"
    log "=== Training done ==="
}

run_predict() {
    log "=== STEP 3: Prediction + regression metrics (4 GPUs) ==="
    conda run -n tgnn_3.10 torchrun \
        --nproc_per_node=4 \
        --master_port=29601 \
        train_mpnn_lstm_4gpu.py --mode predict 2>&1 | tee -a "${LOG_FILE}"
    log "=== Prediction done ==="
}

log "Starting MPNN-LSTM 4-GPU run | mode=${MODE} | log=${LOG_FILE}"

case "${MODE}" in
    prepare) run_prepare ;;
    train)   run_train   ;;
    predict) run_predict ;;
    all)
        run_prepare
        run_train
        run_predict
        log "=== All steps complete ==="
        ;;
    *)
        echo "Usage: $0 [prepare|train|predict|all]" >&2
        exit 1
        ;;
esac
