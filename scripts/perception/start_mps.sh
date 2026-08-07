#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX="${L4STACK_GPU_INDEX:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-log}"

mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
chmod 700 "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"

# GeForce kartlarda compute-mode değişikliği desteklenmeyebilir. Bu optimizasyon
# başarısızsa MPS yine DEFAULT modda başlatılır; hata gizlenmez, uyarı basılır.
if ! nvidia-smi -i "${GPU_INDEX}" -c EXCLUSIVE_PROCESS; then
  echo "UYARI: EXCLUSIVE_PROCESS uygulanamadı; MPS DEFAULT compute mode ile başlatılıyor." >&2
fi

if pgrep -x nvidia-cuda-mps-control >/dev/null 2>&1; then
  echo "MPS control daemon zaten çalışıyor."
  exit 0
fi

nvidia-cuda-mps-control -d
sleep 1
echo get_server_list | nvidia-cuda-mps-control >/dev/null
printf 'MPS hazır: gpu=%s pipe=%s log=%s\n' \
  "${GPU_INDEX}" "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
