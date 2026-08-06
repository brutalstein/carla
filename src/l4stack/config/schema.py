from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from l4stack.errors import ConfigurationError


@dataclass(frozen=True)
class TransformConfig:
    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    @classmethod
    def from_mapping(cls, value: dict[str, Any]):
        required = {"x", "y", "z"}
        missing = required - value.keys()
        if missing:
            raise ConfigurationError(f"Transform is missing keys: {sorted(missing)}")
        return cls(**{key: float(value.get(key, 0.0)) for key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class SensorConfig:
    name: str
    blueprint: str
    group: str
    required: bool
    transform: TransformConfig
    attributes: dict[str, str]

    @classmethod
    def from_mapping(cls, value: dict[str, Any]):
        for key in ("name", "blueprint", "group", "transform"):
            if key not in value:
                raise ConfigurationError(f"Sensor entry is missing '{key}'")
        attrs = {str(key): _to_carla_string(item) for key, item in value.get("attributes", {}).items()}
        return cls(
            name=str(value["name"]),
            blueprint=str(value["blueprint"]),
            group=str(value["group"]),
            required=bool(value.get("required", False)),
            transform=TransformConfig.from_mapping(value["transform"]),
            attributes=attrs,
        )


@dataclass(frozen=True)
class StackConfig:
    root: Path
    simulator: dict[str, Any]
    vehicle: dict[str, Any]
    odd: dict[str, Any]
    sensors: tuple[SensorConfig, ...]
    localization: dict[str, Any]
    logging: dict[str, Any]

    @property
    def required_sensor_names(self) -> tuple[str, ...]:
        return tuple(sensor.name for sensor in self.sensors if sensor.required)

    @property
    def sensors_by_name(self) -> dict[str, SensorConfig]:
        return {sensor.name: sensor for sensor in self.sensors}


def _to_carla_string(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
