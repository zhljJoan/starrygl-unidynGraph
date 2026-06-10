#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/zlj/.miniconda3/envs/tgnn_3.10/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/zlj/.miniconda3/envs/tgnn_3.10/bin/torchrun}"
BASE_CONFIG="${BASE_CONFIG:-configs/ctdg_wikitalk_memshare_public_historical_accuracy.json}"
DATA_ROOT="${DATA_ROOT:-/mnt/nfs/zlj/TGL-DATA}"

find_dataset_source() {
  local name="$1"
  local lower="${name,,}"
  local upper="${name^^}"
  local candidate
  for candidate in \
    "${DATA_ROOT}/${name}" \
    "${DATA_ROOT}/${lower}" \
    "${DATA_ROOT}/${upper}" \
    "${DATA_ROOT}/DATA/${name}" \
    "${DATA_ROOT}/DATA/${lower}" \
    "${DATA_ROOT}/DATA/${upper}"; do
    if [[ -d "${candidate}" || -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  for candidate in "${DATA_ROOT}"/* "${DATA_ROOT}/DATA"/*; do
    local base="${candidate##*/}"
    if [[ -e "${candidate}" && "${base,,}" == "${lower}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  printf '%s\n' "${DATA_ROOT}/${name}"
}

LASTFM_SOURCE="${LASTFM_SOURCE:-$(find_dataset_source LASTFM)}"
WIKITALK_SOURCE="${WIKITALK_SOURCE:-$(find_dataset_source wikitalk)}"
STACKOVERFLOW_SOURCE="${STACKOVERFLOW_SOURCE:-$(find_dataset_source stackoverflow)}"
GDELT_SOURCE="${GDELT_SOURCE:-$(find_dataset_source GDELT)}"

LASTFM_BATCH_SIZE="${LASTFM_BATCH_SIZE:-4000}"
WIKITALK_BATCH_SIZE="${WIKITALK_BATCH_SIZE:-12000}"
STACKOVERFLOW_BATCH_SIZE="${STACKOVERFLOW_BATCH_SIZE:-12000}"
GDELT_BATCH_SIZE="${GDELT_BATCH_SIZE:-12000}"

LASTFM_FANOUT="${LASTFM_FANOUT:-10}"
WIKITALK_FANOUT="${WIKITALK_FANOUT:-20}"
STACKOVERFLOW_FANOUT="${STACKOVERFLOW_FANOUT:-20}"
GDELT_FANOUT="${GDELT_FANOUT:-20}"

LASTFM_EPOCHS="${LASTFM_EPOCHS:-100}"
WIKITALK_EPOCHS="${WIKITALK_EPOCHS:-50}"
STACKOVERFLOW_EPOCHS="${STACKOVERFLOW_EPOCHS:-10}"
GDELT_EPOCHS="${GDELT_EPOCHS:-10}"

SEED="${SEED:-6773}"
ONLY_CASES="${ONLY_CASES:-}"
ONLY_GPUS="${ONLY_GPUS:-}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-0}"
PREPARE_TIMEOUT_SECONDS="${PREPARE_TIMEOUT_SECONDS:-0}"
OUT_DIR="${OUT_DIR:-.artifacts/memshare_public_bench_$(date +%Y%m%d_%H%M%S)}"
CONFIG_DIR="${OUT_DIR}/configs"
LOG_DIR="${OUT_DIR}/logs"
SUMMARY_JSONL="${OUT_DIR}/summary.jsonl"
mkdir -p "${CONFIG_DIR}" "${LOG_DIR}" artifacts
: > "${SUMMARY_JSONL}"
: > "${LOG_DIR}/summary.log"

run_with_optional_timeout() {
  local seconds="$1"
  shift
  if [[ "${seconds}" != "0" ]]; then
    timeout "${seconds}" "$@"
  else
    "$@"
  fi
}

write_config() {
  local name="$1"
  local source="$2"
  local batch_size="$3"
  local fanout="$4"
  local feature_device="$5"
  local epochs="$6"
  local output="$7"
  "${PYTHON_BIN}" tools/make_ctdg_benchmark_config.py \
    --base-config "${BASE_CONFIG}" \
    --output "${output}" \
    --source "${source}" \
    --batch-size "${batch_size}" \
    --fanout "${fanout}" \
    --feature-device "${feature_device}" \
    --epochs "${epochs}" \
    --dataset-name "${name}" \
    --seed "${SEED}" \
    --adaptive-split
}

patch_config() {
  local config="$1"
  local feature_device="$2"
  "${PYTHON_BIN}" - "${config}" "${feature_device}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
feature_device = sys.argv[2]
cfg = json.loads(path.read_text())
runtime = cfg.setdefault("runtime", {})
runtime["memory_sync_mode"] = "memshare_public_historical"
runtime["prefetch_sample_lookahead"] = 3
runtime["prefetch_read_lookahead"] = 2
runtime["dynamic_state_read_after_commit"] = True
runtime["sampler_workers"] = 4
runtime["torch_num_threads"] = 1
runtime["feature_device"] = feature_device
runtime["chunk_feature_layout"] = feature_device != "cpu"
runtime["chunk_feature_sort_gather"] = False
runtime["memory_replica_push"] = False
runtime["mailbox_replica_push"] = False
cfg.setdefault("preprocess", {})["adaptive_split"] = True
cfg["preprocess"]["split_mode"] = "adaptive"
path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
PY
}

extract_summary() {
  local dataset="$1"
  local gpus="$2"
  local config="$3"
  local artifact_root="$4"
  local log="$5"
  "${PYTHON_BIN}" - "${dataset}" "${gpus}" "${config}" "${artifact_root}" "${log}" "${SUMMARY_JSONL}" <<'PY'
import json
import sys
from pathlib import Path

dataset, gpus, config, artifact_root, log, summary_path = sys.argv[1:]
rows = []
for line in Path(log).read_text(errors="ignore").splitlines():
    if not line.startswith("{"):
        continue
    try:
        obj = json.loads(line)
    except Exception:
        continue
    if "epoch" in obj and "train" in obj:
        rows.append(obj)
out = {
    "dataset": dataset,
    "gpus": int(gpus),
    "config": config,
    "artifact_root": artifact_root,
    "log": log,
    "epochs": len(rows),
}
if rows:
    latest = rows[-1]
    best_auc = max(rows, key=lambda r: r.get("test", {}).get("auc", float("-inf")))
    best_ap = max(rows, key=lambda r: r.get("test", {}).get("ap", float("-inf")))
    secs = [float(r["train"]["seconds"]) for r in rows]
    out.update({
        "latest_epoch": int(latest["epoch"]),
        "latest_test_auc": latest["test"]["auc"],
        "latest_test_ap": latest["test"]["ap"],
        "best_auc_epoch": int(best_auc["epoch"]),
        "best_test_auc": best_auc["test"]["auc"],
        "best_test_auc_ap": best_auc["test"]["ap"],
        "best_ap_epoch": int(best_ap["epoch"]),
        "best_test_ap_auc": best_ap["test"]["auc"],
        "best_test_ap": best_ap["test"]["ap"],
        "avg_train_seconds": sum(secs) / len(secs),
        "avg_train_seconds_excl_epoch0": sum(secs[1:]) / max(1, len(secs) - 1),
        "avg_train_seconds_last10": sum(secs[-10:]) / min(10, len(secs)),
    })
with open(summary_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(out, sort_keys=True) + "\n")
print(json.dumps(out, indent=2, sort_keys=True))
PY
}

run_case() {
  local dataset="$1"
  local source="$2"
  local batch_size="$3"
  local fanout="$4"
  local epochs="$5"
  local feature_device="$6"
  local gpus="$7"

  if [[ -n "${ONLY_CASES}" && ",${ONLY_CASES}," != *",${dataset},"* ]]; then
    echo "SKIP ${dataset} ${gpus}gpu: filtered by ONLY_CASES=${ONLY_CASES}" | tee -a "${LOG_DIR}/summary.log"
    return 0
  fi

  if [[ ! -d "${source}" && ! -f "${source}" ]]; then
    echo "SKIP ${dataset} ${gpus}gpu: source not found: ${source}" | tee -a "${LOG_DIR}/summary.log"
    return 0
  fi

  local tag="${dataset}_bs${batch_size}_fanout${fanout}_${epochs}ep_${gpus}gpu_feat${feature_device}_memshare_public_historical"
  local config="${CONFIG_DIR}/${tag}.json"
  local artifact_root="artifacts/bench_${tag}"
  local log="${LOG_DIR}/${tag}.log"

  write_config "${dataset}" "${source}" "${batch_size}" "${fanout}" "${feature_device}" "${epochs}" "${config}"
  patch_config "${config}" "${feature_device}"
  : > "${log}"

  {
    echo "========== ${tag} =========="
    echo "dataset=${dataset}"
    echo "source=${source}"
    echo "batch_size=${batch_size}"
    echo "fanout=${fanout}"
    echo "epochs=${epochs}"
    echo "gpus=${gpus}"
    echo "feature_device=${feature_device}"
    echo "config=${config}"
    echo "artifact_root=${artifact_root}"
    echo "log=${log}"
  } | tee -a "${LOG_DIR}/summary.log"

  if [[ ! -f "${artifact_root}/graph.pt" || ! -f "${artifact_root}/rank_000.pt" ]]; then
    echo "[$(date '+%F %T')] prepare ${tag}" | tee -a "${log}"
    if [[ "${gpus}" -gt 1 ]]; then
      run_with_optional_timeout "${PREPARE_TIMEOUT_SECONDS}" "${TORCHRUN_BIN}" --standalone --nproc_per_node="${gpus}" \
        tools/atc_run.py prepare --config "${config}" --artifact-root "${artifact_root}" 2>&1 | tee -a "${log}"
    else
      CUDA_VISIBLE_DEVICES=0 run_with_optional_timeout "${PREPARE_TIMEOUT_SECONDS}" "${PYTHON_BIN}" \
        tools/atc_run.py prepare --config "${config}" --artifact-root "${artifact_root}" 2>&1 | tee -a "${log}"
    fi
    local status="${PIPESTATUS[0]}"
    if [[ "${status}" -ne 0 ]]; then
      echo "PREPARE_FAILED ${tag} status=${status}" | tee -a "${LOG_DIR}/summary.log"
      return 0
    fi
  else
    echo "[$(date '+%F %T')] reuse artifact ${artifact_root}" | tee -a "${log}"
  fi

  echo "[$(date '+%F %T')] run ${tag}" | tee -a "${log}"
  if [[ "${gpus}" -gt 1 ]]; then
    run_with_optional_timeout "${RUN_TIMEOUT_SECONDS}" "${TORCHRUN_BIN}" --standalone --nproc_per_node="${gpus}" \
      tools/atc_run.py run --no-prepare --config "${config}" --artifact-root "${artifact_root}" --epochs "${epochs}" 2>&1 | tee -a "${log}"
  else
    CUDA_VISIBLE_DEVICES=0 run_with_optional_timeout "${RUN_TIMEOUT_SECONDS}" "${PYTHON_BIN}" \
      tools/atc_run.py run --no-prepare --config "${config}" --artifact-root "${artifact_root}" --epochs "${epochs}" 2>&1 | tee -a "${log}"
  fi
  local status="${PIPESTATUS[0]}"
  if [[ "${status}" -ne 0 ]]; then
    echo "RUN_FAILED ${tag} status=${status}" | tee -a "${LOG_DIR}/summary.log"
    return 0
  fi
  extract_summary "${dataset}" "${gpus}" "${config}" "${artifact_root}" "${log}" | tee -a "${LOG_DIR}/summary.log"
}

run_suite_for_gpus() {
  local gpus="$1"
  run_case "LASTFM" "${LASTFM_SOURCE}" "${LASTFM_BATCH_SIZE}" "${LASTFM_FANOUT}" "${LASTFM_EPOCHS}" "cuda" "${gpus}"
  run_case "wikitalk" "${WIKITALK_SOURCE}" "${WIKITALK_BATCH_SIZE}" "${WIKITALK_FANOUT}" "${WIKITALK_EPOCHS}" "cuda" "${gpus}"
  run_case "stackoverflow" "${STACKOVERFLOW_SOURCE}" "${STACKOVERFLOW_BATCH_SIZE}" "${STACKOVERFLOW_FANOUT}" "${STACKOVERFLOW_EPOCHS}" "cuda" "${gpus}"
  run_case "GDELT" "${GDELT_SOURCE}" "${GDELT_BATCH_SIZE}" "${GDELT_FANOUT}" "${GDELT_EPOCHS}" "cpu" "${gpus}"
}

echo "Output directory: ${OUT_DIR}" | tee -a "${LOG_DIR}/summary.log"
echo "Summary JSONL: ${SUMMARY_JSONL}" | tee -a "${LOG_DIR}/summary.log"
echo "Dataset root: ${DATA_ROOT}" | tee -a "${LOG_DIR}/summary.log"
echo "Running 4-GPU suite first, then single-GPU suite." | tee -a "${LOG_DIR}/summary.log"
if [[ -n "${ONLY_CASES}" ]]; then
  echo "ONLY_CASES=${ONLY_CASES}" | tee -a "${LOG_DIR}/summary.log"
fi
if [[ -n "${ONLY_GPUS}" ]]; then
  echo "ONLY_GPUS=${ONLY_GPUS}" | tee -a "${LOG_DIR}/summary.log"
fi

if [[ -z "${ONLY_GPUS}" || ",${ONLY_GPUS}," == *",4,"* ]]; then
  run_suite_for_gpus 4
fi
if [[ -z "${ONLY_GPUS}" || ",${ONLY_GPUS}," == *",1,"* ]]; then
  run_suite_for_gpus 1
fi

echo "All requested benchmark cases finished. Logs: ${LOG_DIR}" | tee -a "${LOG_DIR}/summary.log"
