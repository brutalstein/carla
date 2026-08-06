from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from l4stack.config.schema import SensorConfig, StackConfig
from l4stack.errors import ConfigurationError

_REQUIRED_FILES = (
    "simulator.yaml",
    "vehicle.yaml",
    "odd.yaml",
    "sensors.yaml",
    "perception.yaml",
    "logging.yaml",
)


def load_stack_config(config_dir: str | Path) -> StackConfig:
    root = Path(config_dir).expanduser().resolve()
    if not root.is_dir():
        raise ConfigurationError(f"Configuration directory does not exist: {root}")

    documents = {name: _load_yaml(root / name) for name in _REQUIRED_FILES}
    sensor_document = documents["sensors.yaml"]
    raw_sensors = list(sensor_document.get("sensors", []))
    raw_sensors.extend(_expand_near_field_sensors(sensor_document))
    sensors = tuple(SensorConfig.from_mapping(item) for item in raw_sensors)
    _validate_unique_sensor_names(sensors)

    config = StackConfig(
        root=root,
        simulator=documents["simulator.yaml"],
        vehicle=documents["vehicle.yaml"],
        odd=documents["odd.yaml"],
        sensors=sensors,
        perception=documents["perception.yaml"],
        logging=documents["logging.yaml"],
    )
    validate_stack_config(config)
    return config


def validate_stack_config(config: StackConfig) -> None:
    world = config.simulator.get("world", {})
    if not world.get("synchronous_mode", False):
        raise ConfigurationError("Deterministic stack requires world.synchronous_mode=true")
    fixed_delta = float(world.get("fixed_delta_seconds", 0.0))
    if fixed_delta <= 0.0:
        raise ConfigurationError("world.fixed_delta_seconds must be positive")

    source = config.perception.get("perception", {}).get("source_sensor")
    if source not in config.sensors_by_name:
        raise ConfigurationError(f"Perception source sensor is not configured: {source}")
    if not config.required_sensor_names:
        raise ConfigurationError("At least one required sensor must be configured")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"Required configuration file is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigurationError(f"YAML root must be a mapping: {path}")
    return data


def _expand_near_field_sensors(document: dict[str, Any]) -> list[dict[str, Any]]:
    defaults = document.get("near_field_defaults", {})
    expanded: list[dict[str, Any]] = []
    for raw in document.get("near_field_sensors", []):
        expanded.append(
            {
                "name": raw["name"],
                "blueprint": "sensor.other.obstacle",
                "group": "near_field",
                "required": False,
                "transform": {
                    "x": raw["x"],
                    "y": raw["y"],
                    "z": raw["z"],
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": raw["yaw"],
                },
                "attributes": defaults,
            }
        )
    return expanded


def _validate_unique_sensor_names(sensors: tuple[SensorConfig, ...]) -> None:
    names = [sensor.name for sensor in sensors]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ConfigurationError(f"Duplicate sensor names: {duplicates}")
