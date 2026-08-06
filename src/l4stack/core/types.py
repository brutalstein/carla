from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class HealthState(str, Enum):
    NOMINAL = "NOMINAL"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


class OddState(str, Enum):
    IN_ODD = "IN_ODD"
    ODD_DEGRADED = "ODD_DEGRADED"
    ODD_EXIT_IMMINENT = "ODD_EXIT_IMMINENT"
    OUTSIDE_ODD = "OUTSIDE_ODD"


@dataclass(frozen=True)
class Vector3:
    x: float
    y: float
    z: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class Rotation3:
    roll: float
    pitch: float
    yaw: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class Pose3:
    position: Vector3
    rotation: Rotation3

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LocalizationEstimate:
    frame: int
    timestamp: float
    pose_world: Pose3
    velocity_world_mps: Vector3
    angular_velocity_world_dps: Vector3
    speed_mps: float
    position_std_m: float
    heading_std_deg: float
    state: HealthState
    source: str

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
        return data


@dataclass(frozen=True)
class BoundingBox3D:
    center: Vector3
    size: Vector3


@dataclass(frozen=True)
class Detection3D:
    track_id: int
    semantic_tag: int
    class_name: str
    point_count: int
    confidence: float
    bbox: BoundingBox3D
    centroid: Vector3
    velocity_ego_mps: Vector3
    range_m: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PerceptionFrame:
    frame: int
    timestamp: float
    detections: tuple[Detection3D, ...]
    source: str
    health: HealthState
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["health"] = self.health.value
        return data


@dataclass(frozen=True)
class OddAssessment:
    state: OddState
    reasons: tuple[str, ...]
    checks: dict[str, bool]

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "reasons": list(self.reasons),
            "checks": self.checks,
        }
