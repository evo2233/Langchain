#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS="${EPOCHS:-3}"

DATASETS=(${DATASETS:-MMLU GPQA})

TOTAL=${#DATASETS[@]}
INDEX=1

for DATASET in "${DATASETS[@]}"; do
  echo "=============================="
  echo "[Dataset ${INDEX}/${TOTAL}] ${DATASET}"
  echo "=============================="

  echo "[1/2] Start training workflow (epochs=${EPOCHS})"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/debate.py" \
    --mode train \
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
