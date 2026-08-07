#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX="${L4STACK_GPU_INDEX:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-log}"

if pgrep -x nvidia-cuda-mps-control >/dev/null 2>&1; then
  echo quit | nvidia-cuda-mps-control
fi
nvidia-smi -i "${GPU_INDEX}" -c DEFAULT >/dev/null 2>&1 || true
rm -rf "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
echo "MPS kapatıldı."
