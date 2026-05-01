#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS="${EPOCHS:-3}"
DEFAULT_GPU_ID="${DEFAULT_GPU_ID:-6}"

DATASETS=(${DATASETS:-MMLU GPQA})
DATASET_GPU_MAP="${DATASET_GPU_MAP:-MMLU:5 GPQA:6}"

get_gpu_id_for_dataset() {
  local dataset="$1"
  local pair
  for pair in ${DATASET_GPU_MAP}; do
    if [[ "${pair%%:*}" == "${dataset}" ]]; then
      echo "${pair##*:}"
      return
    fi
  done
  echo "${DEFAULT_GPU_ID}"
}

TOTAL=${#DATASETS[@]}
INDEX=1

for DATASET in "${DATASETS[@]}"; do
  GPU_ID="$(get_gpu_id_for_dataset "${DATASET}")"

  echo "=============================="
  echo "[Dataset ${INDEX}/${TOTAL}] ${DATASET}(gpu_id=${GPU_ID})"
  echo "=============================="

  echo "[1/2] Start training workflow (epochs=${EPOCHS})"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/debate.py" \
    --mode train \
    --resume \
    --epochs "${EPOCHS}" \
    --dataset "${DATASET}"

  echo "[2/2] Start test workflow"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/debate.py" \
    --mode test \
    --dataset "${DATASET}"

  echo "Finished dataset: ${DATASET}"
  echo ""

  INDEX=$((INDEX + 1))
done

echo "All experiments finished."
