from __future__ import annotations

import math
from typing import Any

from l4stack.core.types import (
    HealthState,
    LocalizationEstimate,
    Pose3,
    Rotation3,
    Vector3,
)


class GroundTruthLocalizer:
    """Deterministic CARLA pose adapter with GNSS/IMU availability checks.

    This is intentionally simulation-only. It establishes the localization interface
    before a production GNSS/INS estimator is introduced.
    """

    def __init__(self, gnss_name: str = "gnss_roof", imu_name: str = "imu_center") -> None:
        self.gnss_name = gnss_name
        self.imu_name = imu_name

    def estimate(
        self,
        frame: int,
        timestamp: float,
        vehicle: Any,
        sensor_bundle: dict[str, Any],
    ) -> LocalizationEstimate:
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        angular = vehicle.get_angular_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        sensors_ok = self.gnss_name in sensor_bundle and self.imu_name in sensor_bundle
        state = HealthState.NOMINAL if sensors_ok else HealthState.DEGRADED
        return LocalizationEstimate(
            frame=frame,
            timestamp=timestamp,
            pose_world=Pose3(
                position=Vector3(transform.location.x, transform.location.y, transform.location.z),
                rotation=Rotation3(
                    transform.rotation.roll,
                    transform.rotation.pitch,
                    transform.rotation.yaw,
                ),
            ),
            velocity_world_mps=Vector3(velocity.x, velocity.y, velocity.z),
            angular_velocity_world_dps=Vector3(angular.x, angular.y, angular.z),
            speed_mps=speed,
            position_std_m=0.01 if sensors_ok else 1.0,
            heading_std_deg=0.02 if sensors_ok else 5.0,
            state=state,
            source="carla_ground_truth_with_gnss_imu_health",
        )
