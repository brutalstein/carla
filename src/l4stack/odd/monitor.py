from __future__ import annotations

from typing import Any

from l4stack.config.schema import SensorConfig
from l4stack.core.types import HealthState, LocalizationEstimate, OddAssessment, OddState


class OddMonitor:
    def __init__(self, odd_document: dict[str, Any], sensors: tuple[SensorConfig, ...]) -> None:
        self.config = odd_document["odd"]
        self.sensors = sensors

    def assess(
        self,
        world: Any,
        localization: LocalizationEstimate,
        sensor_bundle: dict[str, Any],
    ) -> OddAssessment:
        weather = world.get_weather()
        map_name = world.get_map().name.rsplit("/", 1)[-1]
        checks: dict[str, bool] = {}
        checks["map"] = map_name in set(self.config["allowed_maps"])
        checks["speed"] = localization.speed_mps <= float(self.config["maximum_speed_mps"])
        checks["localization"] = localization.state == HealthState(
            self.config["required_localization_state"]
        )

        precipitation = float(getattr(weather, "precipitation", 0.0))
        wetness = float(getattr(weather, "wetness", 0.0))
        sun = float(getattr(weather, "sun_altitude_angle", 45.0))
        checks["precipitation"] = _inside(
            precipitation, self.config["allowed_precipitation_percent"]
        )
        checks["wetness"] = _inside(wetness, self.config["allowed_wetness_percent"])
        checks["sun_altitude"] = _inside(sun, self.config["allowed_sun_altitude_deg"])

        fog_density = float(getattr(weather, "fog_density", 0.0))
        fog_distance = float(getattr(weather, "fog_distance", 1000.0))
        minimum_visibility = float(self.config.get("minimum_visibility_m", 0.0))
        checks["visibility"] = fog_density <= 0.0 or fog_distance >= minimum_visibility

        for group in self.config.get("required_sensor_groups", []):
            required_names = [s.name for s in self.sensors if s.group == group and s.required]
            checks[f"sensor_group:{group}"] = bool(required_names) and all(
                name in sensor_bundle for name in required_names
            )

        failed = tuple(name for name, passed in checks.items() if not passed)
        state = OddState.IN_ODD if not failed else OddState.OUTSIDE_ODD
        return OddAssessment(state=state, reasons=failed, checks=checks)


def _inside(value: float, bounds: list[float]) -> bool:
    return float(bounds[0]) <= value <= float(bounds[1])
