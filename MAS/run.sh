#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS="${EPOCHS:-3}"
DEFAULT_GPU_ID="${DEFAULT_GPU_ID:-6}"

DATASETS=(${DATASETS:-MMLU GPQA})
DATASET_GPU_MAP="${DATASET_GPU_MAP:-MMLU:5 GPQA:5}"

# --- 逻辑变量初始化 ---
OPT_MODE=""       # 存储 Temporal 或 Spatial
CONTROL_MODE=""   # 存储 Uniform
RUN_MODE="start"  # 默认为 start (train+test)，对应 --noopt 或 --resume

usage() {
  cat <<'USAGE'
Usage:
  run.sh --noopt                # 仅测试
  run.sh --start [--uniform]    # 全流程 (可选 uniform)
  run.sh --temporal [--uniform] # Temporal 优化 (可选 uniform)
  run.sh --spatial [--uniform]  # Spatial 优化 (可选 uniform)
  run.sh --resume [--uniform]   # 断点续传 (可选 uniform)
USAGE
}

# --- 参数解析循环 ---
if [[ "$#" -eq 0 ]]; then usage; exit 1; fi

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --start)   RUN_MODE="start" ;;
    --resume)  RUN_MODE="resume" ;;
    --noopt)   RUN_MODE="noopt" ;;
    --temporal) OPT_MODE="Temporal" ;;
    --spatial)  OPT_MODE="Spatial" ;;
    --uniform)  CONTROL_MODE="uniform" ;;
    --help|-h)  usage; exit 0 ;;
    *) echo "Unknown parameter: $1"; usage; exit 1 ;;
  esac
  shift
done

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

# 构造动态参数数组
get_python_args() {
  local args=()
  # 如果指定了优化模式 (Temporal/Spatial)
  if [[ -n "${OPT_MODE}" ]]; then
    args+=("--optimization_mode" "${OPT_MODE}")
  fi
  # 如果指定了 uniform
  if [[ -n "${CONTROL_MODE}" ]]; then
    args+=("--control_mode" "${CONTROL_MODE}")
  fi
  echo "${args[@]}"
}

TOTAL=${#DATASETS[@]}
INDEX=1

for DATASET in "${DATASETS[@]}"; do
  GPU_ID="$(get_gpu_id_for_dataset "${DATASET}")"
  DYNAMIC_ARGS=$(get_python_args)

  echo "=============================="
  echo "[Dataset ${INDEX}/${TOTAL}] ${DATASET}"
  echo "Config: GPU=${GPU_ID}, Opt=${OPT_MODE:-None}, Control=${CONTROL_MODE:-Normal}"
  echo "=============================="

  # 根据 RUN_MODE 执行对应的逻辑
  if [[ "${RUN_MODE}" == "start" || "${RUN_MODE}" == "resume" ]]; then
    RESUME_FLAG=""
    [[ "${RUN_MODE}" == "resume" ]] && RESUME_FLAG="--resume"

    echo "[1/2] Training workflow..."
    "${PYTHON_BIN}" "${SCRIPT_DIR}/debate.py" \
      --mode train --gpu_id "${GPU_ID}" --epochs "${EPOCHS}" --dataset "${DATASET}" \
      ${RESUME_FLAG} ${DYNAMIC_ARGS}

    echo "[2/2] Test workflow..."
    "${PYTHON_BIN}" "${SCRIPT_DIR}/debate.py" \
      --mode test --gpu_id "${GPU_ID}" --dataset "${DATASET}" ${DYNAMIC_ARGS}

  elif [[ "${RUN_MODE}" == "noopt" ]]; then
    echo "[1/1] Test workflow (--no_opt)"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/debate.py" \
      --mode test --gpu_id "${GPU_ID}" --no_opt --dataset "${DATASET}" ${DYNAMIC_ARGS}
  fi

  INDEX=$((INDEX + 1))
done

echo "All experiments finished."
