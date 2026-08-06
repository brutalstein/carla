from __future__ import annotations

from typing import Any

from l4stack.errors import ConfigurationError

_FALLBACK_BLUEPRINTS = (
    "vehicle.lincoln.mkz_2020",
    "vehicle.lincoln.mkz_2017",
)


def spawn_ego_vehicle(carla: Any, world: Any, config: dict[str, Any]) -> Any:
    vehicle_cfg = config["vehicle"]
    library = world.get_blueprint_library()
    blueprint = _select_blueprint(library, str(vehicle_cfg["blueprint"]))
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", str(vehicle_cfg.get("role_name", "ego")))
    if blueprint.has_attribute("color") and vehicle_cfg.get("color"):
        blueprint.set_attribute("color", str(vehicle_cfg["color"]))

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise ConfigurationError("Selected CARLA map has no vehicle spawn points")
    index = int(vehicle_cfg.get("spawn_point_index", 0)) % len(spawn_points)
    vehicle = world.try_spawn_actor(blueprint, spawn_points[index])
    if vehicle is None:
        for transform in spawn_points:
            vehicle = world.try_spawn_actor(blueprint, transform)
            if vehicle is not None:
                break
    if vehicle is None:
        raise RuntimeError("Could not spawn ego vehicle at any map spawn point")

    vehicle.set_autopilot(bool(vehicle_cfg.get("autopilot", False)))
    control_cfg = config.get("initial_control", {})
    vehicle.apply_control(
        carla.VehicleControl(
            throttle=float(control_cfg.get("throttle", 0.0)),
            steer=float(control_cfg.get("steer", 0.0)),
            brake=float(control_cfg.get("brake", 1.0)),
            hand_brake=bool(control_cfg.get("hand_brake", False)),
            reverse=bool(control_cfg.get("reverse", False)),
        )
    )
    return vehicle


def _select_blueprint(library: Any, requested: str) -> Any:
    candidates = (requested,) + tuple(bp for bp in _FALLBACK_BLUEPRINTS if bp != requested)
    for blueprint_id in candidates:
        matches = library.filter(blueprint_id)
        if matches:
            return matches[0]
    raise ConfigurationError(
        f"Vehicle blueprint unavailable: {requested}. Checked fallbacks: {_FALLBACK_BLUEPRINTS}"
    )
