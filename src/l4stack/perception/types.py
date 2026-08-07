from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any
from urllib.parse import urlparse


class PerceptionOutputKind(str, Enum):
    """Model çıktılarının dünya modeline giden standart görev türleri."""

    OBJECT_DETECTION_3D = "OBJECT_DETECTION_3D"
    BEV_SEGMENTATION = "BEV_SEGMENTATION"
    VECTOR_MAP = "VECTOR_MAP"
    TRAFFIC_LIGHT = "TRAFFIC_LIGHT"
    IMAGE_SEGMENTATION = "IMAGE_SEGMENTATION"


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Ağır sensör veya model verisini kopyalamadan tarif eden immutable referans.

    Başlangıç uygulaması ``file://`` URI kullanır. Sözleşme gelecekte ``shm://`` veya
    CUDA IPC transportuna genişletilebilir. Sensör artifact'larında kaynak frame ve
    zaman tutulduğu için farklı sensör hızları gizlenmez; model adapter'ı gerçek skew
    değerini denetleyebilir.
    """

    name: str
    uri: str
    media_type: str
    shape: tuple[int, ...]
    dtype: str
    byte_size: int
    sha256: str | None = None
    source_frame: int | None = None
    source_timestamp: float | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.uri or not self.media_type or not self.dtype:
            raise ValueError("ArtifactRef text fields cannot be empty")
        scheme = urlparse(self.uri).scheme
        if scheme not in {"file", "shm", "cuda-ipc"}:
            raise ValueError(f"Unsupported artifact URI scheme: {scheme!r}")
        if not self.shape or any(value <= 0 for value in self.shape):
            raise ValueError("ArtifactRef.shape must contain positive dimensions")
        if self.byte_size <= 0:
            raise ValueError("ArtifactRef.byte_size must be positive")
        if self.source_frame is not None and self.source_frame < 0:
            raise ValueError("ArtifactRef.source_frame cannot be negative")
        if self.source_timestamp is not None and (
            not math.isfinite(self.source_timestamp) or self.source_timestamp < 0.0
        ):
            raise ValueError("ArtifactRef.source_timestamp must be finite and non-negative")

    def age(self, reference_timestamp: float) -> float | None:
        if self.source_timestamp is None:
            return None
        return max(0.0, float(reference_timestamp) - self.source_timestamp)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "uri": self.uri,
            "media_type": self.media_type,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
            "source_frame": self.source_frame,
            "source_timestamp": self.source_timestamp,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactRef":
        return cls(
            name=str(value["name"]),
            uri=str(value["uri"]),
            media_type=str(value["media_type"]),
            shape=tuple(int(item) for item in value["shape"]),
            dtype=str(value["dtype"]),
            byte_size=int(value["byte_size"]),
            sha256=None if value.get("sha256") is None else str(value["sha256"]),
            source_frame=(
                None if value.get("source_frame") is None else int(value["source_frame"])
            ),
            source_timestamp=(
                None
                if value.get("source_timestamp") is None
                else float(value["source_timestamp"])
            ),
        )


@dataclass(frozen=True, slots=True)
class PerceptionInput:
    """Bir perception turunun immutable, zaman damgalı sensör görünümü."""

    frame: int
    timestamp: float
    cameras: tuple[ArtifactRef, ...]
    lidar: ArtifactRef | None
    calibration: ArtifactRef
    localization_message_id: str | None = None

    def __post_init__(self) -> None:
        if self.frame < 0 or not math.isfinite(self.timestamp) or self.timestamp < 0.0:
            raise ValueError("PerceptionInput frame/timestamp must be finite and non-negative")
        names = [item.name for item in self.cameras]
        if len(names) != len(set(names)):
            raise ValueError("PerceptionInput contains duplicate camera names")
        if self.lidar is not None and self.lidar.name in names:
            raise ValueError("LiDAR and camera artifact names cannot collide")

    @property
    def cameras_by_name(self) -> Mapping[str, ArtifactRef]:
        return MappingProxyType({camera.name: camera for camera in self.cameras})

    def sensor_artifacts(self) -> tuple[ArtifactRef, ...]:
        lidar = () if self.lidar is None else (self.lidar,)
        return self.cameras + lidar

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "timestamp": self.timestamp,
            "cameras": [item.as_dict() for item in self.cameras],
            "lidar": None if self.lidar is None else self.lidar.as_dict(),
            "calibration": self.calibration.as_dict(),
            "localization_message_id": self.localization_message_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PerceptionInput":
        lidar = value.get("lidar")
        return cls(
            frame=int(value["frame"]),
            timestamp=float(value["timestamp"]),
            cameras=tuple(ArtifactRef.from_dict(item) for item in value.get("cameras", [])),
            lidar=None if lidar is None else ArtifactRef.from_dict(lidar),
            calibration=ArtifactRef.from_dict(value["calibration"]),
            localization_message_id=(
                None
                if value.get("localization_message_id") is None
                else str(value["localization_message_id"])
            ),
        )


@dataclass(frozen=True, slots=True)
class Detection3D:
    class_name: str
    confidence: float
    center_xyz_m: tuple[float, float, float]
    size_wlh_m: tuple[float, float, float]
    yaw_rad: float
    velocity_xy_mps: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not self.class_name:
            raise ValueError("Detection3D.class_name cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Detection3D.confidence must be in [0, 1]")
        if len(self.center_xyz_m) != 3 or len(self.size_wlh_m) != 3:
            raise ValueError("Detection3D vectors must have three elements")
        numeric = self.center_xyz_m + self.size_wlh_m + (self.yaw_rad,)
        if self.velocity_xy_mps is not None:
            if len(self.velocity_xy_mps) != 2:
                raise ValueError("Detection3D velocity must have two elements")
            numeric += self.velocity_xy_mps
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("Detection3D values must be finite")
        if any(value <= 0.0 for value in self.size_wlh_m):
            raise ValueError("Detection3D size must be positive")


@dataclass(frozen=True, slots=True)
class VectorMapElement:
    category: str
    confidence: float
    points_xyz_m: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        if not self.category or len(self.points_xyz_m) < 2:
            raise ValueError("VectorMapElement requires a category and at least two points")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("VectorMapElement.confidence must be in [0, 1]")
        if any(len(point) != 3 for point in self.points_xyz_m):
            raise ValueError("VectorMapElement points must be xyz triplets")
        if not all(math.isfinite(value) for point in self.points_xyz_m for value in point):
            raise ValueError("VectorMapElement points must be finite")


@dataclass(frozen=True, slots=True)
class TrafficLightObservation:
    camera_name: str
    bbox_xyxy: tuple[float, float, float, float]
    state: str
    pictogram: str
    confidence: float
    relevant_to_ego: bool | None

    def __post_init__(self) -> None:
        if not self.camera_name or not self.state or not self.pictogram:
            raise ValueError("TrafficLightObservation text fields cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("TrafficLightObservation.confidence must be in [0, 1]")
        if len(self.bbox_xyxy) != 4:
            raise ValueError("TrafficLightObservation bbox requires four values")
        x1, y1, x2, y2 = self.bbox_xyxy
        if not all(math.isfinite(value) for value in self.bbox_xyxy):
            raise ValueError("TrafficLightObservation bbox must be finite")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("TrafficLightObservation bbox is invalid")


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """Model bağımsız ve normalize edilmiş tek perception model çıktısı."""

    model_name: str
    model_version: str
    kind: PerceptionOutputKind
    source_frame: int
    source_timestamp: float
    detections_3d: tuple[Detection3D, ...] = ()
    vector_map: tuple[VectorMapElement, ...] = ()
    traffic_lights: tuple[TrafficLightObservation, ...] = ()
    rasters: tuple[ArtifactRef, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_name or not self.model_version:
            raise ValueError("ModelOutput model identity cannot be empty")
        if (
            self.source_frame < 0
            or not math.isfinite(self.source_timestamp)
            or self.source_timestamp < 0.0
        ):
            raise ValueError("ModelOutput source metadata must be finite and non-negative")
        raster_names = [item.name for item in self.rasters]
        if len(raster_names) != len(set(raster_names)):
            raise ValueError("ModelOutput contains duplicate raster names")
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "kind": self.kind.value,
            "source_frame": self.source_frame,
            "source_timestamp": self.source_timestamp,
            "detections_3d": [
                {
                    "class_name": item.class_name,
                    "confidence": item.confidence,
                    "center_xyz_m": list(item.center_xyz_m),
                    "size_wlh_m": list(item.size_wlh_m),
                    "yaw_rad": item.yaw_rad,
                    "velocity_xy_mps": (
                        None if item.velocity_xy_mps is None else list(item.velocity_xy_mps)
                    ),
                }
                for item in self.detections_3d
            ],
            "vector_map": [
                {
                    "category": item.category,
                    "confidence": item.confidence,
                    "points_xyz_m": [list(point) for point in item.points_xyz_m],
                }
                for item in self.vector_map
            ],
            "traffic_lights": [
                {
                    "camera_name": item.camera_name,
                    "bbox_xyxy": list(item.bbox_xyxy),
                    "state": item.state,
                    "pictogram": item.pictogram,
                    "confidence": item.confidence,
                    "relevant_to_ego": item.relevant_to_ego,
                }
                for item in self.traffic_lights
            ],
            "rasters": [item.as_dict() for item in self.rasters],
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True, slots=True)
class PerceptionSnapshot:
    """Aynı anda okunabilen son geçerli model çıktılarının atomik görünümü."""

    generated_at: float
    outputs: tuple[ModelOutput, ...]

    @property
    def by_model(self) -> Mapping[str, ModelOutput]:
        return MappingProxyType({item.model_name: item for item in self.outputs})
