#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS="${EPOCHS:-3}"
PROMPT_PATH="${PROMPT_PATH:-${SCRIPT_DIR}/optimized_prompts.json}"

echo "[1/2] Start training workflow (epochs=${EPOCHS})"
"${PYTHON_BIN}" "${SCRIPT_DIR}/debate.py" \
  --mode train \
  --epochs "${EPOCHS}" \
  --prompt-path "${PROMPT_PATH}"

echo "[2/2] Start test workflow using optimized prompts: ${PROMPT_PATH}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/debate.py" \
  --mode test \
  --prompt-path "${PROMPT_PATH}"

echo "Experiment finished: train -> test"
