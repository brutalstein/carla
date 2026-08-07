from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from l4stack.perception.config import PerceptionConfig


@dataclass(frozen=True, slots=True)
class CudaCheck:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True, slots=True)
class CudaDoctorReport:
    checks: tuple[CudaCheck, ...]

    @property
    def ready(self) -> bool:
        return all(check.ok for check in self.checks)


def run_cuda_doctor(perception: PerceptionConfig) -> CudaDoctorReport:
    """Host GPU, MPS, container runtime, VRAM ve shared-memory kapasitesini doğrular.

    Fonksiyon inference üretmez ve sentetik başarı yolu içermez. Gerçek sistem komutları
    veya dosya sistemi durumu okunamadığında kontrol başarısız olur.
    """

    checks: list[CudaCheck] = []
    checks.append(
        CudaCheck(
            "linux",
            platform.system() == "Linux",
            f"platform={platform.system()} {platform.release()}",
        )
    )
    checks.extend(_gpu_checks(perception))
    checks.append(_binary_check("docker"))
    checks.append(_binary_check("nvidia-ctk"))
    checks.append(_binary_check("nvidia-cuda-mps-control"))
    checks.append(_shared_memory_check(perception))
    checks.append(_mps_check(perception))
    return CudaDoctorReport(tuple(checks))


def _gpu_checks(perception: PerceptionConfig) -> tuple[CudaCheck, ...]:
    if shutil.which("nvidia-smi") is None:
        return (CudaCheck("nvidia-smi", False, "nvidia-smi bulunamadı"),)
    fields = "name,driver_version,memory.total,memory.free,compute_cap"
    command = [
        "nvidia-smi",
        "-i",
        str(perception.gpu.device_index),
        f"--query-gpu={fields}",
        "--format=csv,noheader,nounits",
    ]
    result = _run(command)
    if result.returncode != 0:
        return (CudaCheck("gpu-query", False, result.stderr.strip() or result.stdout.strip()),)
    values = [item.strip() for item in result.stdout.strip().split(",")]
    if len(values) != 5:
        return (CudaCheck("gpu-query", False, f"beklenmeyen çıktı: {result.stdout!r}"),)
    name, driver, total_mib, free_mib, capability = values
    try:
        total_gib = float(total_mib) / 1024.0
        free_gib = float(free_mib) / 1024.0
        capability_value = float(capability)
    except ValueError:
        return (CudaCheck("gpu-query", False, f"sayısal alan çözülemedi: {values}"),)
    return (
        CudaCheck("gpu", "RTX 5090" in name, f"name={name}"),
        CudaCheck(
            "driver",
            _version_at_least(driver, "610.43.02"),
            f"driver={driver}, required>=610.43.02",
        ),
        CudaCheck(
            "compute-capability",
            capability_value >= 12.0,
            f"compute_capability={capability_value:.1f}, required>=12.0",
        ),
        CudaCheck(
            "vram-total",
            total_gib >= 30.0,
            f"total={total_gib:.2f} GiB",
        ),
        CudaCheck(
            "vram-free",
            free_gib >= perception.gpu.minimum_free_vram_gib,
            f"free={free_gib:.2f} GiB, required>={perception.gpu.minimum_free_vram_gib:.2f}",
        ),
    )


def _binary_check(name: str) -> CudaCheck:
    path = shutil.which(name)
    return CudaCheck(name, path is not None, "bulundu" if path else "bulunamadı")


def _shared_memory_check(perception: PerceptionConfig) -> CudaCheck:
    path = Path("/dev/shm")
    if not path.is_dir():
        return CudaCheck("shared-memory", False, "/dev/shm bulunamadı")
    stats = os.statvfs(path)
    total = stats.f_frsize * stats.f_blocks
    free = stats.f_frsize * stats.f_bavail
    enabled_models = [model for model in perception.models.values() if model.enabled]
    cameras = {
        camera
        for model in enabled_models
        for camera in model.cameras
    }
    lidar_names = {model.lidar for model in enabled_models if model.lidar is not None}
    transport = perception.transport
    required = transport.slots_per_sensor * (
        len(cameras) * transport.camera_slot_bytes
        + len(lidar_names) * transport.lidar_slot_bytes
    )
    # Kernel ve diğer IPC kullanıcıları için ring toplamının iki katı boş alan istenir.
    minimum_free = max(required * 2, 512 * 1024 * 1024)
    ok = free >= minimum_free
    return CudaCheck(
        "shared-memory",
        ok,
        f"total={total / 2**20:.1f} MiB free={free / 2**20:.1f} MiB "
        f"required_free={minimum_free / 2**20:.1f} MiB",
    )


def _mps_check(perception: PerceptionConfig) -> CudaCheck:
    if not perception.gpu.mps_required:
        return CudaCheck("cuda-mps", True, "config tarafından zorunlu değil")
    if shutil.which("nvidia-cuda-mps-control") is None:
        return CudaCheck("cuda-mps", False, "nvidia-cuda-mps-control bulunamadı")
    pipe_dir = os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")
    environment = {**os.environ, "CUDA_MPS_PIPE_DIRECTORY": pipe_dir}
    result = _run(
        ["bash", "-lc", "echo get_server_list | nvidia-cuda-mps-control"],
        environment=environment,
    )
    return CudaCheck(
        "cuda-mps",
        result.returncode == 0,
        result.stdout.strip() or result.stderr.strip() or f"pipe={pipe_dir}",
    )


def _run(
    command: list[str],
    *,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(command, 127, "", str(exc))


def _version_at_least(actual: str, required: str) -> bool:
    def parts(value: str) -> tuple[int, ...]:
        result: list[int] = []
        for item in value.split("."):
            digits = "".join(character for character in item if character.isdigit())
            result.append(int(digits or "0"))
        return tuple(result)

    actual_parts = parts(actual)
    required_parts = parts(required)
    length = max(len(actual_parts), len(required_parts))
    return actual_parts + (0,) * (length - len(actual_parts)) >= required_parts + (0,) * (
        length - len(required_parts)
    )
