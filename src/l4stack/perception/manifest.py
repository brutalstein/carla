from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from l4stack.perception.config import ModelArtifactSpec, PerceptionConfig, PerceptionModelConfig


@dataclass(frozen=True, slots=True)
class ArtifactCheck:
    path: Path
    required: bool
    ok: bool
    detail: str


@dataclass(frozen=True, slots=True)
class BackendCheck:
    ok: bool
    detail: str


@dataclass(frozen=True, slots=True)
class ModelInstallationReport:
    model_name: str
    enabled: bool
    ready: bool
    backend: BackendCheck
    artifacts: tuple[ArtifactCheck, ...]


class ModelInstallationError(RuntimeError):
    """Etkin modelin gerekli dosya veya backend komutu hazır olmadığında üretilir."""


def verify_artifact(spec: ModelArtifactSpec) -> ArtifactCheck:
    if not spec.path.is_file():
        return ArtifactCheck(
            path=spec.path,
            required=spec.required,
            ok=not spec.required,
            detail="missing" if spec.required else "optional artifact missing",
        )
    size = spec.path.stat().st_size
    if size < spec.min_size_bytes:
        return ArtifactCheck(
            spec.path,
            spec.required,
            False,
            f"too small: {size} < {spec.min_size_bytes} bytes",
        )
    if spec.sha256:
        digest = _sha256(spec.path)
        if digest.lower() != spec.sha256.lower():
            return ArtifactCheck(
                spec.path,
                spec.required,
                False,
                f"sha256 mismatch: {digest}",
            )
    return ArtifactCheck(spec.path, spec.required, True, f"ok ({size} bytes)")


def verify_model_installation(model: PerceptionModelConfig) -> ModelInstallationReport:
    checks = tuple(verify_artifact(item) for item in model.artifacts)
    command_ok, command_detail = model.backend.validate_command()
    backend = BackendCheck(command_ok, command_detail)
    return ModelInstallationReport(
        model_name=model.name,
        enabled=model.enabled,
        ready=backend.ok and all(item.ok for item in checks),
        backend=backend,
        artifacts=checks,
    )


def require_model_installation(model: PerceptionModelConfig) -> None:
    report = verify_model_installation(model)
    if report.ready:
        return
    failures = [report.backend.detail] if not report.backend.ok else []
    failures.extend(item.detail for item in report.artifacts if not item.ok)
    raise ModelInstallationError(f"{model.name} is not ready: {'; '.join(failures)}")


def verify_installation(config: PerceptionConfig) -> tuple[ModelInstallationReport, ...]:
    return tuple(
        verify_model_installation(model)
        for _, model in sorted(config.models.items())
    )


def required_paths(config: PerceptionConfig) -> tuple[Path, ...]:
    return tuple(
        artifact.path
        for model in config.models.values()
        for artifact in model.artifacts
        if artifact.required
    )


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
