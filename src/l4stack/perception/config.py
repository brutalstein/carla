from __future__ import annotations

import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from l4stack.perception.backend import ProcessBackendConfig


@dataclass(frozen=True, slots=True)
class ModelArtifactSpec:
    path: Path
    required: bool
    min_size_bytes: int
    sha256: str | None = None

    @classmethod
    def from_mapping(cls, root: Path, value: Mapping[str, Any]) -> "ModelArtifactSpec":
        raw_path = Path(str(value["path"]))
        path = raw_path if raw_path.is_absolute() else root / raw_path
        spec = cls(
            path=path.resolve(),
            required=bool(value.get("required", True)),
            min_size_bytes=int(value.get("min_size_bytes", 1)),
            sha256=None if value.get("sha256") in (None, "") else str(value["sha256"]),
        )
        if spec.min_size_bytes <= 0:
            raise ValueError(f"Artifact min_size_bytes must be positive: {spec.path}")
        if spec.sha256 is not None and len(spec.sha256) != 64:
            raise ValueError(
                f"Artifact sha256 must contain 64 hexadecimal characters: {spec.path}"
            )
        return spec


@dataclass(frozen=True, slots=True)
class PerceptionModelConfig:
    name: str
    enabled: bool
    adapter: str
    model_version: str
    executor: str
    runtime_component: str
    target_rate_hz: float
    request_timeout_s: float
    max_sensor_skew_s: float
    cameras: tuple[str, ...]
    lidar: str | None
    backend: ProcessBackendConfig
    artifacts: tuple[ModelArtifactSpec, ...]

    @classmethod
    def from_mapping(
        cls,
        root: Path,
        name: str,
        value: Mapping[str, Any],
    ) -> "PerceptionModelConfig":
        backend_value = value.get("backend", {})
        if not isinstance(backend_value, Mapping):
            raise ValueError(f"Perception backend must be a mapping: {name}")
        command_value = backend_value.get("command", [])
        if isinstance(command_value, str):
            command = tuple(shlex.split(command_value))
        else:
            command = tuple(str(item) for item in command_value)
        cwd_value = backend_value.get("working_directory")
        cwd = None
        if cwd_value not in (None, ""):
            raw_cwd = Path(str(cwd_value))
            cwd = (raw_cwd if raw_cwd.is_absolute() else root / raw_cwd).resolve()
        config = cls(
            name=name,
            enabled=bool(value.get("enabled", False)),
            adapter=str(value["adapter"]),
            model_version=str(value["model_version"]),
            executor=str(value["executor"]),
            runtime_component=str(value["runtime_component"]),
            target_rate_hz=float(value["target_rate_hz"]),
            request_timeout_s=float(value["request_timeout_s"]),
            max_sensor_skew_s=float(value.get("max_sensor_skew_s", 0.10)),
            cameras=tuple(str(item) for item in value.get("cameras", [])),
            lidar=None if value.get("lidar") in (None, "") else str(value["lidar"]),
            backend=ProcessBackendConfig(
                command=command,
                working_directory=cwd,
                environment={
                    str(key): str(item)
                    for key, item in dict(backend_value.get("environment", {})).items()
                },
                required_environment=tuple(
                    str(item) for item in backend_value.get("required_environment", [])
                ),
                startup_timeout_s=float(backend_value.get("startup_timeout_s", 60.0)),
                shutdown_timeout_s=float(backend_value.get("shutdown_timeout_s", 5.0)),
            ),
            artifacts=tuple(
                ModelArtifactSpec.from_mapping(root, item)
                for item in value.get("artifacts", [])
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        identity = {
            "name": self.name,
            "adapter": self.adapter,
            "model_version": self.model_version,
            "executor": self.executor,
            "runtime_component": self.runtime_component,
        }
        if any(not value.strip() for value in identity.values()):
            raise ValueError(f"Perception model identity fields cannot be empty: {self.name}")
        if self.target_rate_hz <= 0.0 or self.request_timeout_s <= 0.0:
            raise ValueError(f"Perception model rates/timeouts must be positive: {self.name}")
        if self.max_sensor_skew_s <= 0.0:
            raise ValueError(f"Perception max_sensor_skew_s must be positive: {self.name}")
        if not self.cameras:
            raise ValueError(f"Perception model requires at least one camera: {self.name}")
        if len(self.cameras) != len(set(self.cameras)):
            raise ValueError(f"Perception model contains duplicate camera names: {self.name}")


@dataclass(frozen=True, slots=True)
class PerceptionConfig:
    enabled: bool
    input_lifespan_s: float
    snapshot_max_age_s: float
    artifact_retention_frames: int
    models: Mapping[str, PerceptionModelConfig]

    @classmethod
    def from_mapping(cls, root: Path, document: Mapping[str, Any]) -> "PerceptionConfig":
        value = document.get("perception", {})
        if not isinstance(value, Mapping):
            raise ValueError("perception YAML node must be a mapping")
        models = {
            str(name): PerceptionModelConfig.from_mapping(root, str(name), model)
            for name, model in dict(value.get("models", {})).items()
        }
        config = cls(
            enabled=bool(value.get("enabled", False)),
            input_lifespan_s=float(value.get("input_lifespan_s", 0.25)),
            snapshot_max_age_s=float(value.get("snapshot_max_age_s", 0.50)),
            artifact_retention_frames=int(value.get("artifact_retention_frames", 32)),
            models=MappingProxyType(models),
        )
        if config.input_lifespan_s <= 0.0 or config.snapshot_max_age_s <= 0.0:
            raise ValueError("Perception lifespan values must be positive")
        if config.artifact_retention_frames <= 0:
            raise ValueError("perception.artifact_retention_frames must be positive")
        return config
