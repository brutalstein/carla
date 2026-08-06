from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any


class DeterministicWorld(AbstractContextManager["DeterministicWorld"]):
    def __init__(self, client: Any, world: Any, config: dict[str, Any]) -> None:
        self.client = client
        self.world = world
        self.config = config
        self._original_settings: Any | None = None
        self._traffic_manager: Any | None = None
        self._original_weather: Any | None = None

    def __enter__(self):
        world_cfg = self.config["world"]
        requested_map = str(world_cfg.get("map", "")).strip()
        current_name = self.world.get_map().name.rsplit("/", 1)[-1]
        if requested_map and current_name != requested_map:
            self.world = self.client.load_world(requested_map)

        self._original_settings = self.world.get_settings()
        self._original_weather = self.world.get_weather()
        settings = self.world.get_settings()
        settings.synchronous_mode = bool(world_cfg.get("synchronous_mode", True))
        settings.fixed_delta_seconds = float(world_cfg.get("fixed_delta_seconds", 0.05))
        settings.no_rendering_mode = bool(world_cfg.get("no_rendering_mode", False))
        if hasattr(settings, "substepping"):
            settings.substepping = True
            settings.max_substep_delta_time = 0.01
            settings.max_substeps = max(1, int(settings.fixed_delta_seconds / 0.01 + 0.5))
        self.world.apply_settings(settings)
        weather_cfg = world_cfg.get("weather", {})
        if weather_cfg:
            weather = self.world.get_weather()
            for key, value in weather_cfg.items():
                if hasattr(weather, key):
                    setattr(weather, key, float(value))
            self.world.set_weather(weather)

        tm_cfg = self.config.get("traffic_manager", {})
        if tm_cfg.get("enabled", False):
            self._traffic_manager = self.client.get_trafficmanager(int(tm_cfg.get("port", 8000)))
            self._traffic_manager.set_synchronous_mode(True)
            self._traffic_manager.set_random_device_seed(int(tm_cfg.get("seed", 0)))
        return self

    def tick(self) -> int:
        return int(self.world.tick())

    def __exit__(self, *_: object) -> None:
        if self._traffic_manager is not None:
            self._traffic_manager.set_synchronous_mode(False)
        if self._original_weather is not None:
            self.world.set_weather(self._original_weather)
        if self._original_settings is not None:
            self.world.apply_settings(self._original_settings)
