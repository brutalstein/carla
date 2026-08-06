from __future__ import annotations

from dataclasses import asdict
from typing import Any

from l4stack.config.schema import SensorConfig


def export_calibration(sensors: tuple[SensorConfig, ...]) -> dict[str, Any]:
    """Export declared rigid-body extrinsics in the ego coordinate frame."""
    return {
        sensor.name: {
            "blueprint": sensor.blueprint,
            "group": sensor.group,
            "required": sensor.required,
            "extrinsic_ego": asdict(sensor.transform),
            "attributes": sensor.attributes,
        }
        for sensor in sensors
    }
