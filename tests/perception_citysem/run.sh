#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="${L4STACK_TRT_ENV:-perception_trt}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found." >&2
  exit 127
fi

exec conda run --no-capture-output -n "${ENV_NAME}" \
  python "${ROOT_DIR}/tests/perception_citysem/opencv_demo.py" "$@"
