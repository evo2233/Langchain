#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS="${EPOCHS:-3}"
DEFAULT_GPU_ID="${DEFAULT_GPU_ID:-6}"

DATASETS=(${DATASETS:-MMLU GPQA})
DATASET_GPU_MAP="${DATASET_GPU_MAP:-MMLU:5 GPQA:6}"

usage() {
  cat <<'USAGE'
Usage:
  run.sh --start   # train + test
  run.sh --resume  # resume train + test
  run.sh --noopt   # test only with --no_opt
USAGE
}

if [[ "$#" -ne 1 ]]; then
  usage
  exit 1
fi

MODE="$1"
case "${MODE}" in
  --start|--resume|--noopt)
    ;;
  *)
    usage
    exit 1
    ;;
esac

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

  if [[ "${MODE}" == "--start" ]]; then
    echo "[1/2] Start training workflow (epochs=${EPOCHS})"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/debate.py" \
      --mode train \
      --epochs "${EPOCHS}" \
      --dataset "${DATASET}"

    echo "[2/2] Start test workflow"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/debate.py" \
      --mode test \
      --dataset "${DATASET}"
  elif [[ "${MODE}" == "--resume" ]]; then
    echo "[1/2] Start training workflow (resume, epochs=${EPOCHS})"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/debate.py" \
      --mode train \
      --resume \
      --epochs "${EPOCHS}" \
      --dataset "${DATASET}"

    echo "[2/2] Start test workflow"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/debate.py" \
      --mode test \
      --dataset "${DATASET}"
  else
    echo "[1/1] Start no-opt test workflow"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/debate.py" \
      --mode test \
      --no_opt \
      --dataset "${DATASET}"
  fi

  echo "Finished dataset: ${DATASET}"
  echo ""

  INDEX=$((INDEX + 1))
done

echo "All experiments finished."