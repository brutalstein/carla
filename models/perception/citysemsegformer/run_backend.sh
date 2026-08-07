#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_KIND="citysemsegformer"

if [[ "${L4STACK_PERCEPTION_MOCK:-0}" == "1" ]]; then
  export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
  exec python "${ROOT_DIR}/scripts/perception/mock_backend.py" --kind "${MODEL_KIND}"
fi

COMMAND_VAR="L4STACK_$(echo "${MODEL_KIND}" | tr '[:lower:]' '[:upper:]')_COMMAND"
COMMAND="${!COMMAND_VAR:-}"
if [[ -z "${COMMAND}" ]]; then
  echo "ERROR: ${COMMAND_VAR} tanımlı değil. Kurulum için README.md dosyasını okuyun." >&2
  exit 64
fi

exec bash -lc "${COMMAND}"
