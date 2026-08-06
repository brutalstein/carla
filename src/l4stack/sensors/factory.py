from __future__ import annotations

import logging
from typing import Any

from l4stack.config.schema import SensorConfig

_LOG = logging.getLogger(__name__)


def spawn_sensor_suite(
    carla: Any,
    world: Any,
    vehicle: Any,
    sensors: tuple[SensorConfig, ...],
    callback_factory: Any,
) -> dict[str, Any]:
    actors: dict[str, Any] = {}
    library = world.get_blueprint_library()
    for sensor in sensors:
        matches = library.filter(sensor.blueprint)
        if not matches:
            if sensor.required:
                raise RuntimeError(f"Required sensor blueprint unavailable: {sensor.blueprint}")
            _LOG.warning("Skipping unsupported optional sensor blueprint %s", sensor.blueprint)
            continue
        blueprint = matches[0]
        for key, value in sensor.attributes.items():
            if blueprint.has_attribute(key):
                blueprint.set_attribute(key, value)
            elif sensor.required:
                raise RuntimeError(
                    f"Required sensor {sensor.name} does not support attribute {key}"
                )
            else:
                _LOG.warning("Ignoring unsupported attribute %s on %s", key, sensor.name)

        t = sensor.transform
        transform = carla.Transform(
            carla.Location(x=t.x, y=t.y, z=t.z),
            carla.Rotation(roll=t.roll, pitch=t.pitch, yaw=t.yaw),
        )
        kwargs: dict[str, Any] = {"attach_to": vehicle}
        if hasattr(carla, "AttachmentType"):
            kwargs["attachment_type"] = carla.AttachmentType.Rigid
        actor = world.spawn_actor(blueprint, transform, **kwargs)
        actor.listen(callback_factory(sensor.name))
        actors[sensor.name] = actor
    return actors
