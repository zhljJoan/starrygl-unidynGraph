#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/zlj/.miniconda3/envs/tgnn_3.10/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/zlj/.miniconda3/envs/tgnn_3.10/bin/torchrun}"
EPOCHS="${EPOCHS:-50}"
SEED="${SEED:-6773}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-1800}"
PREPARE_TIMEOUT_SECONDS="${PREPARE_TIMEOUT_SECONDS:-1800}"

OUT_DIR="${OUT_DIR:-.artifacts/ctdg_dataset_bench_$(date +%Y%m%d_%H%M%S)}"
CONFIG_DIR="${OUT_DIR}/configs"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${CONFIG_DIR}" "${LOG_DIR}" artifacts

BASE_CONFIG="${BASE_CONFIG:-configs/ctdg_wikitalk_tgn_speed_optimized.json}"

write_config() {
  local name="$1"
  local source="$2"
  local batch_size="$3"
  local fanout="$4"
  local feature_device="$5"
  local output="$6"
  "${PYTHON_BIN}" tools/make_ctdg_benchmark_config.py \
    --base-config "$BASE_CONFIG" \
    --output "$output" \
    --source "$source" \
    --batch-size "$batch_size" \
    --fanout "$fanout" \
    --feature-device "$feature_device" \
    --epochs "$EPOCHS" \
    --dataset-name "$name" \
    --seed "$SEED"
}

run_cmd() {
  local log="$1"
  shift
  echo "[$(date '+%F %T')] $*" | tee -a "$log"
  timeout "$RUN_TIMEOUT_SECONDS" "$@" 2>&1 | tee -a "$log"
  return "${PIPESTATUS[0]}"
}

prepare_cmd() {
  local log="$1"
  shift
  echo "[$(date '+%F %T')] $*" | tee -a "$log"
  timeout "$PREPARE_TIMEOUT_SECONDS" "$@" 2>&1 | tee -a "$log"
  return "${PIPESTATUS[0]}"
}

run_case() {
  local name="$1"
  local source="$2"
  local per_gpu_batch_size="$3"
  local fanout="$4"
  local world_size="$5"
  local feature_device="$6"
  local batch_size=$((per_gpu_batch_size * world_size))
  local tag="${name}_pergpu${per_gpu_batch_size}_bs${batch_size}_fanout${fanout}_${world_size}gpu_feat${feature_device}_edge172_boundary_decay_delta"
  local config="${CONFIG_DIR}/${tag}.json"
  local artifact_root="artifacts/bench_${tag}"
  local log="${LOG_DIR}/${tag}.log"

  write_config "$name" "$source" "$batch_size" "$fanout" "$feature_device" "$config"

  echo "========== ${tag} ==========" | tee -a "${LOG_DIR}/summary.log"
  echo "per_gpu_batch_size=${per_gpu_batch_size}" | tee -a "${LOG_DIR}/summary.log"
  echo "effective_batch_size=${batch_size}" | tee -a "${LOG_DIR}/summary.log"
  echo "config=${config}" | tee -a "${LOG_DIR}/summary.log"
  echo "artifact_root=${artifact_root}" | tee -a "${LOG_DIR}/summary.log"
  echo "log=${log}" | tee -a "${LOG_DIR}/summary.log"

  if [[ ! -f "${artifact_root}/graph.pt" || ! -f "${artifact_root}/rank_000.pt" ]]; then
    if [[ "$world_size" -gt 1 ]]; then
      prepare_cmd "$log" "$TORCHRUN_BIN" --standalone --nproc_per_node="$world_size" tools/atc_run.py prepare --config "$config" --artifact-root "$artifact_root"
    else
      CUDA_VISIBLE_DEVICES=0 prepare_cmd "$log" "$PYTHON_BIN" tools/atc_run.py prepare --config "$config" --artifact-root "$artifact_root"
    fi
    status=$?
    if [[ "$status" -ne 0 ]]; then
      echo "PREPARE_FAILED ${tag} status=${status}" | tee -a "${LOG_DIR}/summary.log"
      return 0
    fi
  else
    echo "[$(date '+%F %T')] reuse existing artifact: ${artifact_root}" | tee -a "$log"
  fi

  if [[ "$world_size" -gt 1 ]]; then
    run_cmd "$log" "$TORCHRUN_BIN" --standalone --nproc_per_node="$world_size" tools/atc_run.py run --no-prepare --config "$config" --artifact-root "$artifact_root" --epochs "$EPOCHS"
  else
    CUDA_VISIBLE_DEVICES=0 run_cmd "$log" "$PYTHON_BIN" tools/atc_run.py run --no-prepare --config "$config" --artifact-root "$artifact_root" --epochs "$EPOCHS"
  fi
  status=$?
  if [[ "$status" -ne 0 ]]; then
    echo "RUN_FAILED ${tag} status=${status}" | tee -a "${LOG_DIR}/summary.log"
  else
    echo "RUN_OK ${tag}" | tee -a "${LOG_DIR}/summary.log"
  fi
}

echo "Output directory: ${OUT_DIR}"
echo "Epochs: ${EPOCHS}"
echo "Seed: ${SEED}"

# First 4-GPU runs. The batch size argument is per GPU.
run_case "LASTFM" "/mnt/data/zlj/tgl_data/DATA/LASTFM" 1000 10 4 "cuda"
run_case "wikitalk" "/mnt/data/zlj/tgl_data/DATA/wikitalk" 3000 20 4 "cuda"
run_case "stackoverflow" "/mnt/data/zlj/tgl_data/DATA/stackoverflow" 3000 20 4 "cuda"
run_case "GDELT" "/mnt/data/zlj/starrygl-data/raw/TGL-DATA/GDELT" 3000 20 4 "cpu"

# Then single-GPU runs. The batch size argument is still per GPU.
run_case "LASTFM" "/mnt/data/zlj/tgl_data/DATA/LASTFM" 1000 10 1 "cuda"
run_case "wikitalk" "/mnt/data/zlj/tgl_data/DATA/wikitalk" 3000 20 1 "cuda"
run_case "stackoverflow" "/mnt/data/zlj/tgl_data/DATA/stackoverflow" 3000 20 1 "cuda"
run_case "GDELT" "/mnt/data/zlj/starrygl-data/raw/TGL-DATA/GDELT" 3000 20 1 "cpu"

echo "All benchmark cases finished. Logs: ${LOG_DIR}" | tee -a "${LOG_DIR}/summary.log"
