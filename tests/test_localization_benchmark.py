from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from l4stack.config.loader import load_stack_config
from l4stack.localization import PlanarErrorStateEkf
from l4stack.localization.geodesy import GeodeticPosition, LocalTangentPlane

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Vector:
    x: float
    y: float
    z: float


@dataclass
class ImuMeasurement:
    accelerometer: Vector
    gyroscope: Vector
    compass: float


@dataclass
class GnssMeasurement:
    latitude: float
    longitude: float
    altitude: float


def test_planar_eskf_constant_velocity_benchmark_against_test_ground_truth() -> None:
    config = load_stack_config(ROOT / "config")
    document = {"localization": dict(config.localization["localization"])}
    document["localization"].update(
        {
            "gnss_horizontal_std_m": 0.25,
            "compass_std_deg": 0.5,
            "maximum_position_std_m": 10.0,
            "maximum_heading_std_deg": 20.0,
        }
    )
    sensors = dict(config.sensors_by_name)
    gnss_sensor = sensors["gnss_roof"]
    sensors["gnss_roof"] = type(gnss_sensor)(
        name=gnss_sensor.name,
        blueprint=gnss_sensor.blueprint,
        group=gnss_sensor.group,
        required=gnss_sensor.required,
        transform=type(gnss_sensor.transform)(0.0, 0.0, 0.0),
        attributes=gnss_sensor.attributes,
    )
    estimator = PlanarErrorStateEkf(document, sensors)

    origin = GeodeticPosition(40.1950, 29.0600, 120.0)
    plane = LocalTangentPlane(origin)
    dt = 0.05
    velocity_east_mps = 5.0
    errors: list[float] = []
    outlier_was_rejected = False

    for frame in range(1, 301):
        timestamp = frame * dt
        true_east = (frame - 1) * dt * velocity_east_mps
        geodetic = plane.to_geodetic(true_east, 0.0, 0.0)
        # Deterministic bounded measurement disturbance; ground truth stays test-only.
        east_noise = 0.20 * math.sin(frame * 0.17)
        north_noise = 0.15 * math.cos(frame * 0.11)
        if frame == 150:
            noisy_geodetic = plane.to_geodetic(true_east + 100.0, 100.0, 0.0)
        else:
            noisy_geodetic = plane.to_geodetic(true_east + east_noise, north_noise, 0.0)
        bundle = {
            "gnss_roof": GnssMeasurement(
                noisy_geodetic.latitude_deg,
                noisy_geodetic.longitude_deg,
                noisy_geodetic.altitude_m,
            ),
            "imu_center": ImuMeasurement(
                accelerometer=Vector(0.0, 0.0, 0.0),
                gyroscope=Vector(0.0, 0.0, 0.0),
                compass=math.pi / 2.0,
            ),
        }
        estimate = estimator.estimate(frame, timestamp, bundle)
        if frame == 150:
            outlier_was_rejected = not estimate.diagnostics["gnss_update_accepted"]
        if frame > 40:
            position = estimate.pose_local_enu.position
            errors.append(math.hypot(position.x - true_east, position.y))

    rmse = float(np.sqrt(np.mean(np.square(errors))))
    assert rmse < 0.45
    assert outlier_was_rejected
    assert estimate.source == "gnss_imu_planar_error_state_ekf"
    assert estimate.state.value == "NOMINAL"
