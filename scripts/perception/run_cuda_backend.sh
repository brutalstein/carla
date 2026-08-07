#!/usr/bin/env bash
set -euo pipefail

MODEL_KIND="${1:?model kind is required}"
COMMAND_VAR="L4STACK_$(echo "${MODEL_KIND}" | tr '[:lower:]' '[:upper:]')_COMMAND"
COMMAND="${!COMMAND_VAR:-}"

if [[ -z "${COMMAND}" ]]; then
  echo "ERROR: ${COMMAND_VAR} tanımlı değil." >&2
  exit 64
fi
if [[ "${L4STACK_REQUIRE_CUDA:-1}" != "1" ]]; then
  echo "ERROR: CUDA zorunluluğu devre dışı bırakılamaz." >&2
  exit 65
fi
if [[ "${L4STACK_DISABLE_CPU_FALLBACK:-1}" != "1" ]]; then
  echo "ERROR: CPU fallback kapalı olmalıdır." >&2
  exit 66
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi bulunamadı." >&2
  exit 67
fi
if ! nvidia-smi -i "${L4STACK_GPU_INDEX:-0}" >/dev/null 2>&1; then
  echo "ERROR: CUDA GPU görünür değil." >&2
  exit 68
fi
if [[ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ]]; then
  if [[ ! -d "${CUDA_MPS_PIPE_DIRECTORY}" ]]; then
    echo "ERROR: CUDA MPS pipe dizini bulunamadı: ${CUDA_MPS_PIPE_DIRECTORY}" >&2
    exit 69
  fi
  if [[ ! -e "${CUDA_MPS_PIPE_DIRECTORY}/control" ]]; then
    echo "ERROR: CUDA MPS control socket bulunamadı." >&2
    exit 70
  fi
fi

exec bash -lc "${COMMAND}"
