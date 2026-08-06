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
    "localization.yaml",
    "logging.yaml",
)
_FORBIDDEN_RUNTIME_BLUEPRINT_FRAGMENTS = (
    "ray_cast_semantic",
    "semantic_segmentation",
    "instance_segmentation",
)


def load_stack_config(config_dir: str | Path) -> StackConfig:
    root = Path(config_dir).expanduser().resolve()
    if not root.is_dir():
        raise ConfigurationError(f"Configuration directory does not exist: {root}")

    documents = {name: _load_yaml(root / name) for name in _REQUIRED_FILES}
    sensor_document = documents["sensors.yaml"]
    sensors = tuple(
        SensorConfig.from_mapping(item) for item in list(sensor_document.get("sensors", []))
    )
    _validate_unique_sensor_names(sensors)

    config = StackConfig(
        root=root,
        simulator=documents["simulator.yaml"],
        vehicle=documents["vehicle.yaml"],
        odd=documents["odd.yaml"],
        sensors=sensors,
        localization=documents["localization.yaml"],
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

    if not config.required_sensor_names:
        raise ConfigurationError("At least one required sensor must be configured")

    for sensor in config.sensors:
        if any(fragment in sensor.blueprint for fragment in _FORBIDDEN_RUNTIME_BLUEPRINT_FRAGMENTS):
            raise ConfigurationError(
                f"Ground-truth sensor blueprint is forbidden at runtime: {sensor.blueprint}"
            )

    localization = config.localization.get("localization", {})
    if localization.get("algorithm") != "planar_error_state_ekf":
        raise ConfigurationError("localization.algorithm must be 'planar_error_state_ekf'")

    for key in ("gnss_sensor", "imu_sensor"):
        sensor_name = str(localization.get(key, ""))
        sensor = config.sensors_by_name.get(sensor_name)
        if sensor is None:
            raise ConfigurationError(f"Localization sensor is not configured: {sensor_name}")
        if not sensor.required:
            raise ConfigurationError(f"Localization sensor must be required: {sensor_name}")
        if sensor.group != "localization":
            raise ConfigurationError(f"Localization sensor has wrong group: {sensor_name}")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"Required configuration file is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigurationError(f"YAML root must be a mapping: {path}")
    return data


def _validate_unique_sensor_names(sensors: tuple[SensorConfig, ...]) -> None:
    names = [sensor.name for sensor in sensors]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ConfigurationError(f"Duplicate sensor names: {duplicates}")
