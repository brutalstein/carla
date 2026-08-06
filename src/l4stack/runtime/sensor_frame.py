from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class SensorFrame:
    """Aynı simülasyon frame'ine ait sensör ölçümlerinin immutable görünümü."""

    frame: int
    timestamp: float
    measurements: Mapping[str, Any]

    def __post_init__(self) -> None:
        # CARLA measurement nesneleri kopyalanmaz; yalnızca isim->ölçüm tablosu sabitlenir.
        object.__setattr__(self, "measurements", MappingProxyType(dict(self.measurements)))

    @property
    def sensor_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.measurements))
