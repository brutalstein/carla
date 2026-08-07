from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from l4stack.config.schema import RuntimeConfig, SensorConfig, StackConfig
from l4stack.errors import ConfigurationError
from l4stack.perception.config import PerceptionConfig

_REQUIRED_FILES = (
    "simulator.yaml",
    "vehicle.yaml",
    "odd.yaml",
    "sensors.yaml",
    "localization.yaml",
    "runtime.yaml",
    "perception.yaml",
    "logging.yaml",
)
_FORBIDDEN_RUNTIME_BLUEPRINT_FRAGMENTS = (
    "ray_cast_semantic",
    "semantic_segmentation",
    "instance_segmentation",
)
_SUPPORTED_PERCEPTION_ADAPTERS = {
    "bevfusion_detection",
    "bevfusion_segmentation",
    "maptrv2",
    "tld_ready",
    "citysemsegformer",
}


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
    try:
        perception = PerceptionConfig.from_mapping(root.parent, documents["perception.yaml"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid perception configuration: {exc}") from exc

    config = StackConfig(
        root=root,
        simulator=documents["simulator.yaml"],
        vehicle=documents["vehicle.yaml"],
        odd=documents["odd.yaml"],
        sensors=sensors,
        localization=documents["localization.yaml"],
        runtime=RuntimeConfig.from_mapping(documents["runtime.yaml"]),
        perception=perception,
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

    contract = config.runtime.contract("localization")
    if contract.max_input_age_s > config.runtime.sensor_bundle_lifespan_s:
        raise ConfigurationError(
            "localization max_input_age_s cannot exceed sensor bundle lifespan"
        )
    if "localization" not in config.runtime.executors:
        raise ConfigurationError("runtime.executors.localization is required")
    if config.runtime.executors["localization"].priority != contract.priority:
        raise ConfigurationError(
            "runtime localization executor priority must match component contract"
        )

    _validate_perception(config)


def _validate_perception(config: StackConfig) -> None:
    enabled_models = [item for item in config.perception.models.values() if item.enabled]
    if config.perception.enabled and not enabled_models:
        raise ConfigurationError("perception.enabled=true requires at least one enabled model")

    component_names: set[str] = set()
    configured_lidars = {
        model.lidar for model in config.perception.models.values() if model.lidar is not None
    }
    if len(configured_lidars) > 1:
        raise ConfigurationError(
            f"PerceptionInput currently supports one LiDAR stream: {sorted(configured_lidars)}"
        )

    for model in config.perception.models.values():
        if model.adapter not in _SUPPORTED_PERCEPTION_ADAPTERS:
            raise ConfigurationError(f"Unsupported perception adapter: {model.adapter}")
        if model.runtime_component in component_names:
            raise ConfigurationError(
                f"Duplicate perception runtime component: {model.runtime_component}"
            )
        component_names.add(model.runtime_component)

        executor = config.runtime.executors.get(model.executor)
        if executor is None:
            raise ConfigurationError(
                f"Perception executor is not configured: {model.name} -> {model.executor}"
            )
        contract = config.runtime.components.get(model.runtime_component)
        if contract is None:
            raise ConfigurationError(
                f"Perception runtime contract is missing: {model.runtime_component}"
            )
        if executor.priority != contract.priority:
            raise ConfigurationError(
                f"Perception executor/contract priority mismatch: {model.name}"
            )
        if contract.max_input_age_s > config.perception.input_lifespan_s:
            raise ConfigurationError(
                f"Perception max_input_age_s exceeds input lifespan: {model.name}"
            )
        if model.request_timeout_s > contract.execution_budget_s:
            raise ConfigurationError(
                f"Perception request timeout exceeds execution budget: {model.name}"
            )
        minimum_period = 1.0 / model.target_rate_hz
        if contract.expected_output_period_s < minimum_period:
            raise ConfigurationError(
                f"Perception output period is faster than target rate: {model.name}"
            )
        if contract.output_lifespan_s > config.perception.snapshot_max_age_s:
            raise ConfigurationError(
                f"Perception output lifespan exceeds snapshot max age: {model.name}"
            )

        for camera_name in model.cameras:
            sensor = config.sensors_by_name.get(camera_name)
            if sensor is None or sensor.blueprint != "sensor.camera.rgb":
                raise ConfigurationError(
                    f"Perception camera is missing or not RGB: {model.name} -> {camera_name}"
                )
        lidar_required = model.adapter in {"bevfusion_detection", "bevfusion_segmentation"}
        if lidar_required and model.lidar is None:
            raise ConfigurationError(f"Perception model requires LiDAR: {model.name}")
        if not lidar_required and model.lidar is not None:
            raise ConfigurationError(f"Perception model must not declare LiDAR: {model.name}")
        if model.lidar is not None:
            sensor = config.sensors_by_name.get(model.lidar)
            if sensor is None or sensor.blueprint != "sensor.lidar.ray_cast":
                raise ConfigurationError(
                    "Perception LiDAR is missing or not raw ray-cast: "
                    f"{model.name} -> {model.lidar}"
                )


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
