from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_WGS84_A_M = 6_378_137.0
_WGS84_F = 1.0 / 298.257_223_563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)


@dataclass(frozen=True)
class GeodeticPosition:
    latitude_deg: float
    longitude_deg: float
    altitude_m: float


class LocalTangentPlane:
    """WGS-84 geodetic coordinates mapped to a local right-handed ENU frame."""

    def __init__(self, origin: GeodeticPosition) -> None:
        self.origin = origin
        latitude = math.radians(origin.latitude_deg)
        longitude = math.radians(origin.longitude_deg)
        self._origin_ecef = _geodetic_to_ecef(origin)
        sin_lat, cos_lat = math.sin(latitude), math.cos(latitude)
        sin_lon, cos_lon = math.sin(longitude), math.cos(longitude)
        self._ecef_to_enu = np.array(
            [
                [-sin_lon, cos_lon, 0.0],
                [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
                [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
            ],
            dtype=np.float64,
        )

    def to_enu(self, position: GeodeticPosition) -> np.ndarray:
        delta_ecef = _geodetic_to_ecef(position) - self._origin_ecef
        return self._ecef_to_enu @ delta_ecef

    def to_geodetic(self, east_m: float, north_m: float, up_m: float) -> GeodeticPosition:
        enu = np.array([east_m, north_m, up_m], dtype=np.float64)
        ecef = self._origin_ecef + self._ecef_to_enu.T @ enu
        return _ecef_to_geodetic(ecef)


def _geodetic_to_ecef(position: GeodeticPosition) -> np.ndarray:
    latitude = math.radians(position.latitude_deg)
    longitude = math.radians(position.longitude_deg)
    sin_lat = math.sin(latitude)
    cos_lat = math.cos(latitude)
    radius = _WGS84_A_M / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    return np.array(
        [
            (radius + position.altitude_m) * cos_lat * math.cos(longitude),
            (radius + position.altitude_m) * cos_lat * math.sin(longitude),
            (radius * (1.0 - _WGS84_E2) + position.altitude_m) * sin_lat,
        ],
        dtype=np.float64,
    )


def _ecef_to_geodetic(ecef: np.ndarray) -> GeodeticPosition:
    x, y, z = (float(value) for value in ecef)
    longitude = math.atan2(y, x)
    horizontal = math.hypot(x, y)
    latitude = math.atan2(z, horizontal * (1.0 - _WGS84_E2))
    altitude = 0.0
    for _ in range(10):
        sin_lat = math.sin(latitude)
        radius = _WGS84_A_M / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
        cos_lat = max(abs(math.cos(latitude)), 1e-12)
        altitude = horizontal / cos_lat - radius
        latitude_next = math.atan2(
            z,
            horizontal * (1.0 - _WGS84_E2 * radius / (radius + altitude)),
        )
        if abs(latitude_next - latitude) < 1e-13:
            latitude = latitude_next
            break
        latitude = latitude_next
    return GeodeticPosition(
        latitude_deg=math.degrees(latitude),
        longitude_deg=math.degrees(longitude),
        altitude_m=altitude,
    )
