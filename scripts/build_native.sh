#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-/tmp/atc_starrygl_native_build}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DPython3_EXECUTABLE="${PYTHON_BIN}"
cmake --build "${BUILD_DIR}" --target libstarrygl_sampler adaptive_split_cpp -j "${JOBS:-$(nproc)}"
