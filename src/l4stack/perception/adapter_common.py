from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from l4stack.perception.protocol import BackendProtocolError, InferenceRequest
from l4stack.perception.types import ArtifactRef, ModelOutput, PerceptionInput, PerceptionOutputKind


class PerceptionInputError(ValueError):
    """Modelin sensör girdileri eksik veya zamansal olarak uyumsuzsa üretilir."""


class ModelAdapter(Protocol):
    name: str
    model_version: str
    kind: PerceptionOutputKind
    coordinate_frame: str

    def validate_input(self, value: PerceptionInput) -> None: ...

    def build_request(self, value: PerceptionInput, request_id: str) -> InferenceRequest: ...

    def parse_response(self, value: PerceptionInput, payload: Mapping[str, Any]) -> ModelOutput: ...


@dataclass(frozen=True, slots=True)
class AdapterRequirements:
    cameras: tuple[str, ...]
    lidar_required: bool
    max_sensor_skew_s: float = 0.10

    def __post_init__(self) -> None:
        if not self.cameras:
            raise ValueError("Adapter requires at least one camera")
        if self.max_sensor_skew_s <= 0.0:
            raise ValueError("max_sensor_skew_s must be positive")


class BaseAdapter:
    name: str
    kind: PerceptionOutputKind
    coordinate_frame: str

    def __init__(self, requirements: AdapterRequirements, model_version: str) -> None:
        if not model_version.strip():
            raise ValueError("model_version cannot be empty")
        self.requirements = requirements
        self.model_version = model_version.strip()

    def validate_input(self, value: PerceptionInput) -> None:
        available = value.cameras_by_name
        missing = [name for name in self.requirements.cameras if name not in available]
        if missing:
            raise PerceptionInputError(f"{self.name} missing cameras: {missing}")
        if self.requirements.lidar_required and value.lidar is None:
            raise PerceptionInputError(f"{self.name} requires LiDAR input")
        if value.calibration.media_type not in {
            "application/json",
            "application/x-npz",
            "application/octet-stream",
        }:
            raise PerceptionInputError(
                f"{self.name} unsupported calibration media type: "
                f"{value.calibration.media_type}"
            )

        selected = [available[name] for name in self.requirements.cameras]
        if self.requirements.lidar_required and value.lidar is not None:
            selected.append(value.lidar)
        for artifact in selected:
            age = artifact.age(value.timestamp)
            if age is not None and age > self.requirements.max_sensor_skew_s:
                raise PerceptionInputError(
                    f"{self.name} sensor artifact is too old: {artifact.name} "
                    f"age={age:.3f}s limit={self.requirements.max_sensor_skew_s:.3f}s"
                )

    def build_request(self, value: PerceptionInput, request_id: str) -> InferenceRequest:
        self.validate_input(value)
        return InferenceRequest(
            request_id=request_id,
            model_name=self.name,
            source_timestamp=value.timestamp,
            payload=value,
        )

    def base_output(self, value: PerceptionInput, **kwargs: Any) -> ModelOutput:
        return ModelOutput(
            model_name=self.name,
            model_version=self.model_version,
            kind=self.kind,
            source_frame=value.frame,
            source_timestamp=value.timestamp,
            **kwargs,
        )


def require_rasters(payload: Mapping[str, Any]) -> tuple[ArtifactRef, ...]:
    raw_many = payload.get("rasters")
    if raw_many is None and payload.get("raster") is not None:
        raw_many = [payload["raster"]]
    if not isinstance(raw_many, list) or not raw_many:
        raise BackendProtocolError("Backend response requires a non-empty rasters list")
    return tuple(ArtifactRef.from_dict(item) for item in raw_many)


def fixed_tuple(value: Any, size: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, list | tuple) or len(value) != size:
        raise BackendProtocolError(f"{name} must contain exactly {size} values")
    return tuple(float(item) for item in value)


def diagnostics(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = payload.get("diagnostics", {})
    if not isinstance(raw, Mapping):
        raise BackendProtocolError("Backend diagnostics must be a mapping")
    return {str(key): value for key, value in raw.items()}
