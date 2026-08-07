#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec "${ROOT_DIR}/scripts/perception/run_cuda_backend.sh" "bevfusion_segmentation"
