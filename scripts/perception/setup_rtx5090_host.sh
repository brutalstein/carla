#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERSIONS_FILE="${ROOT_DIR}/infra/perception/versions.env"
# shellcheck disable=SC1090
source "${VERSIONS_FILE}"

APPLY=0
if [[ "${1:-}" == "--apply" ]]; then
  APPLY=1
elif [[ $# -gt 0 ]]; then
  echo "Kullanım: $0 [--apply]" >&2
  exit 64
fi

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: yalnız Linux desteklenir." >&2
  exit 65
fi
if [[ ! -f /etc/os-release ]]; then
  echo "ERROR: /etc/os-release bulunamadı." >&2
  exit 66
fi
# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "24.04" ]]; then
  echo "ERROR: referans host Ubuntu 24.04; bulunan ${ID:-unknown} ${VERSION_ID:-unknown}." >&2
  exit 67
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: NVIDIA driver/nvidia-smi bulunamadı. Driver otomatik kurulmaz." >&2
  exit 68
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)"
DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | xargs)"
if [[ "${GPU_NAME}" != *"RTX 5090"* ]]; then
  echo "ERROR: RTX 5090 bekleniyor; bulunan ${GPU_NAME}." >&2
  exit 69
fi
python3 - "${DRIVER_VERSION}" "${L4STACK_MIN_NVIDIA_DRIVER}" <<'PY'
import sys

def parts(value):
    return tuple(int(''.join(c for c in item if c.isdigit()) or 0) for item in value.split('.'))
a, b = parts(sys.argv[1]), parts(sys.argv[2])
n=max(len(a),len(b)); a += (0,)*(n-len(a)); b += (0,)*(n-len(b))
if a < b:
    raise SystemExit(f"driver {sys.argv[1]} < required {sys.argv[2]}")
PY

echo "Host audit: GPU=${GPU_NAME}, driver=${DRIVER_VERSION}"
if [[ ${APPLY} -eq 0 ]]; then
  echo "Audit tamamlandı. Paket değişikliği için --apply kullanın."
  exit 0
fi

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  ca-certificates curl gnupg2 docker.io docker-compose-v2 python3.12-venv

if [[ ! -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg ]]; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
fi
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
sudo apt-get update
VERSION="${L4STACK_NVIDIA_CONTAINER_TOOLKIT}-1"
sudo apt-get install -y \
  nvidia-container-toolkit="${VERSION}" \
  nvidia-container-toolkit-base="${VERSION}" \
  libnvidia-container-tools="${VERSION}" \
  libnvidia-container1="${VERSION}"
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl enable --now docker
sudo systemctl restart docker
sudo usermod -aG docker "${SUDO_USER:-$USER}"
INSTALL_USER="${SUDO_USER:-$USER}"
INSTALL_UID="$(id -u "${INSTALL_USER}")"
INSTALL_GID="$(id -g "${INSTALL_USER}")"
printf 'L4STACK_UID=%s\nL4STACK_GID=%s\n' "${INSTALL_UID}" "${INSTALL_GID}" \
  > "${ROOT_DIR}/infra/perception/.env"
chown "${INSTALL_UID}:${INSTALL_GID}" "${ROOT_DIR}/infra/perception/.env"

sudo nvidia-smi -pm 1 >/dev/null 2>&1 || true
"${ROOT_DIR}/scripts/perception/start_mps.sh"

python3 "${ROOT_DIR}/scripts/perception/verify_cuda_stack.py" \
  --config-dir "${ROOT_DIR}/config"

echo "Kurulum tamamlandı. Grup üyeliği için oturumu kapatıp yeniden açın."
