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
    pose_local_enu: Pose3
    velocity_local_enu_mps: Vector3
    angular_rate_body_dps: Vector3
    speed_mps: float
    position_std_m: float
    heading_std_deg: float
    state: HealthState
    source: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
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
