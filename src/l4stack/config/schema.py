from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from l4stack.errors import ConfigurationError
from l4stack.perception.config import PerceptionConfig
from l4stack.runtime.contracts import ComponentContract, ExecutorProfile


@dataclass(frozen=True)
class TransformConfig:
    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "TransformConfig":
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
    def from_mapping(cls, value: dict[str, Any]) -> "SensorConfig":
        for key in ("name", "blueprint", "group", "transform"):
            if key not in value:
                raise ConfigurationError(f"Sensor entry is missing '{key}'")
        attrs = {
            str(key): _to_carla_string(item) for key, item in value.get("attributes", {}).items()
        }
        return cls(
            name=str(value["name"]),
            blueprint=str(value["blueprint"]),
            group=str(value["group"]),
            required=bool(value.get("required", False)),
            transform=TransformConfig.from_mapping(value["transform"]),
            attributes=attrs,
        )


@dataclass(frozen=True)
class RuntimeConfig:
    namespace: str
    sensor_bundle_lifespan_s: float
    lineage_max_records: int
    deadline_event_history: int
    executors: dict[str, ExecutorProfile]
    components: dict[str, ComponentContract]

    @classmethod
    def from_mapping(cls, document: dict[str, Any]) -> "RuntimeConfig":
        value = document.get("runtime", {})
        namespace = str(value.get("namespace", "runtime")).strip()
        if not namespace:
            raise ConfigurationError("runtime.namespace cannot be empty")
        try:
            executors = {
                str(name): ExecutorProfile.from_mapping(str(name), dict(profile))
                for name, profile in dict(value.get("executors", {})).items()
            }
            components = {
                str(name): ComponentContract.from_mapping(str(name), dict(contract))
                for name, contract in dict(value.get("components", {})).items()
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Invalid runtime configuration: {exc}") from exc
        sensor_lifespan = float(value.get("sensor_bundle_lifespan_s", 0.0))
        lineage_max_records = int(value.get("lineage_max_records", 0))
        deadline_event_history = int(value.get("deadline_event_history", 0))
        if sensor_lifespan <= 0.0:
            raise ConfigurationError("runtime.sensor_bundle_lifespan_s must be positive")
        if lineage_max_records <= 0:
            raise ConfigurationError("runtime.lineage_max_records must be positive")
        if deadline_event_history <= 0:
            raise ConfigurationError("runtime.deadline_event_history must be positive")
        return cls(
            namespace=namespace,
            sensor_bundle_lifespan_s=sensor_lifespan,
            lineage_max_records=lineage_max_records,
            deadline_event_history=deadline_event_history,
            executors=executors,
            components=components,
        )

    def contract(self, name: str) -> ComponentContract:
        try:
            return self.components[name]
        except KeyError as exc:
            raise ConfigurationError(f"Runtime component contract is missing: {name}") from exc


@dataclass(frozen=True)
class StackConfig:
    root: Path
    simulator: dict[str, Any]
    vehicle: dict[str, Any]
    odd: dict[str, Any]
    sensors: tuple[SensorConfig, ...]
    localization: dict[str, Any]
    runtime: RuntimeConfig
    perception: PerceptionConfig
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
