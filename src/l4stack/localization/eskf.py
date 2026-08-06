from __future__ import annotations

import math
from typing import Any

import numpy as np

from l4stack.config.schema import SensorConfig
from l4stack.core.types import (
    HealthState,
    LocalizationEstimate,
    Pose3,
    Rotation3,
    Vector3,
)
from l4stack.errors import ConfigurationError
from l4stack.localization.geodesy import GeodeticPosition, LocalTangentPlane

_STATE_SIZE = 8
_POS = slice(0, 2)
_VEL = slice(2, 4)
_YAW = 4
_ACCEL_BIAS = slice(5, 7)
_GYRO_BIAS = 7


class PlanarErrorStateEkf:
    """Planar GNSS/IMU error-state EKF for a road vehicle.

    Nominal state: east/north position, east/north velocity, ENU yaw,
    two accelerometer biases and yaw-rate gyro bias. IMU propagation runs at the
    simulation tick; GNSS position and compass yaw are independently gated by NIS.
    No CARLA actor pose or velocity is consumed by this estimator.
    """

    def __init__(
        self,
        localization_document: dict[str, Any],
        sensors_by_name: dict[str, SensorConfig],
    ) -> None:
        self.config = localization_document["localization"]
        self.gnss_name = str(self.config["gnss_sensor"])
        self.imu_name = str(self.config["imu_sensor"])
        try:
            gnss_transform = sensors_by_name[self.gnss_name].transform
        except KeyError as exc:
            raise ConfigurationError(f"GNSS sensor not found: {self.gnss_name}") from exc

        # CARLA body is x-forward/y-right/z-up. The filter uses x-forward/y-left/z-up.
        self._gnss_lever_body = np.array(
            [gnss_transform.x, -gnss_transform.y], dtype=np.float64
        )
        self._gnss_lever_up_m = float(gnss_transform.z)
        self._state = np.zeros(_STATE_SIZE, dtype=np.float64)
        self._covariance = self._initial_covariance()
        self._plane: LocalTangentPlane | None = None
        self._last_timestamp: float | None = None
        self._height_m = 0.0
        self._initialized = False
        self._gnss_rejections = 0
        self._compass_rejections = 0
        self._last_gnss_accepted = False
        self._last_compass_accepted = False
        self._last_yaw_rate_rps = 0.0

    def estimate(
        self,
        frame: int,
        timestamp: float,
        sensor_bundle: dict[str, Any],
    ) -> LocalizationEstimate:
        gnss = sensor_bundle[self.gnss_name]
        imu = sensor_bundle[self.imu_name]
        self._validate_measurements(gnss, imu)

        if not self._initialized:
            self._initialize(timestamp, gnss, imu)
        else:
            dt = timestamp - float(self._last_timestamp)
            if dt <= 0.0 or dt > float(self.config["max_dt_seconds"]):
                self._last_timestamp = timestamp
                return self._build_estimate(frame, timestamp, imu, force_degraded=True)
            acceleration_body, yaw_rate = self._read_imu(imu)
            self._propagate(acceleration_body, yaw_rate, dt)
            self._last_compass_accepted = self._update_compass(
                self._compass_to_enu_yaw(imu.compass)
            )
            self._last_gnss_accepted = self._update_gnss(gnss)
            self._compass_rejections = (
                0 if self._last_compass_accepted else self._compass_rejections + 1
            )
            self._gnss_rejections = 0 if self._last_gnss_accepted else self._gnss_rejections + 1
            self._last_timestamp = timestamp

        return self._build_estimate(frame, timestamp, imu)

    def _initialize(self, timestamp: float, gnss: Any, imu: Any) -> None:
        origin = GeodeticPosition(
            latitude_deg=float(gnss.latitude),
            longitude_deg=float(gnss.longitude),
            altitude_m=float(gnss.altitude),
        )
        self._plane = LocalTangentPlane(origin)
        yaw = self._compass_to_enu_yaw(float(imu.compass))
        self._state[_YAW] = yaw
        self._state[_POS] = -_rotation_2d(yaw) @ self._gnss_lever_body
        self._height_m = -self._gnss_lever_up_m
        self._last_timestamp = timestamp
        self._initialized = True
        self._last_gnss_accepted = True
        self._last_compass_accepted = True

    def _propagate(self, acceleration_body: np.ndarray, yaw_rate_rps: float, dt: float) -> None:
        yaw = float(self._state[_YAW])
        accel_unbiased = acceleration_body - self._state[_ACCEL_BIAS]
        yaw_rate_unbiased = yaw_rate_rps - float(self._state[_GYRO_BIAS])
        rotation = _rotation_2d(yaw)
        acceleration_enu = rotation @ accel_unbiased

        self._state[_POS] += self._state[_VEL] * dt + 0.5 * acceleration_enu * dt * dt
        self._state[_VEL] += acceleration_enu * dt
        self._state[_YAW] = _wrap_angle(yaw + yaw_rate_unbiased * dt)
        self._last_yaw_rate_rps = yaw_rate_unbiased

        derivative_rotation = _rotation_2d_derivative(yaw)
        transition = np.zeros((_STATE_SIZE, _STATE_SIZE), dtype=np.float64)
        transition[_POS, _VEL] = np.eye(2)
        transition[_VEL, _YAW] = derivative_rotation @ accel_unbiased
        transition[_VEL, _ACCEL_BIAS] = -rotation
        transition[_YAW, _GYRO_BIAS] = -1.0

        noise_map = np.zeros((_STATE_SIZE, 6), dtype=np.float64)
        noise_map[_VEL, 0:2] = rotation
        noise_map[_YAW, 2] = -1.0
        noise_map[_ACCEL_BIAS, 3:5] = np.eye(2)
        noise_map[_GYRO_BIAS, 5] = 1.0
        continuous_noise = np.diag(
            [
                float(self.config["accel_noise_std_mps2"]) ** 2,
                float(self.config["accel_noise_std_mps2"]) ** 2,
                math.radians(float(self.config["gyro_noise_std_dps"])) ** 2,
                float(self.config["accel_bias_random_walk_std_mps3"]) ** 2,
                float(self.config["accel_bias_random_walk_std_mps3"]) ** 2,
                math.radians(float(self.config["gyro_bias_random_walk_std_dps2"])) ** 2,
            ]
        )

        identity = np.eye(_STATE_SIZE)
        phi = identity + transition * dt + 0.5 * (transition @ transition) * dt * dt
        process_noise = noise_map @ continuous_noise @ noise_map.T * dt
        self._covariance = phi @ self._covariance @ phi.T + process_noise
        self._covariance = _symmetrize(self._covariance)

    def _update_gnss(self, gnss: Any) -> bool:
        assert self._plane is not None
        geodetic = GeodeticPosition(
            latitude_deg=float(gnss.latitude),
            longitude_deg=float(gnss.longitude),
            altitude_m=float(gnss.altitude),
        )
        measurement_enu = self._plane.to_enu(geodetic)
        yaw = float(self._state[_YAW])
        rotation = _rotation_2d(yaw)
        predicted_antenna = self._state[_POS] + rotation @ self._gnss_lever_body
        innovation = measurement_enu[:2] - predicted_antenna

        observation = np.zeros((2, _STATE_SIZE), dtype=np.float64)
        observation[:, _POS] = np.eye(2)
        observation[:, _YAW] = _rotation_2d_derivative(yaw) @ self._gnss_lever_body
        variance = float(self.config["gnss_horizontal_std_m"]) ** 2
        accepted = self._measurement_update(
            innovation,
            observation,
            np.eye(2) * variance,
            float(self.config["gnss_gate_chi2"]),
            angle_innovation=False,
        )
        if accepted:
            self._height_m = float(measurement_enu[2]) - self._gnss_lever_up_m
        return accepted

    def _update_compass(self, measured_yaw: float) -> bool:
        innovation = np.array([_wrap_angle(measured_yaw - float(self._state[_YAW]))])
        observation = np.zeros((1, _STATE_SIZE), dtype=np.float64)
        observation[0, _YAW] = 1.0
        variance = math.radians(float(self.config["compass_std_deg"])) ** 2
        return self._measurement_update(
            innovation,
            observation,
            np.array([[variance]], dtype=np.float64),
            float(self.config["compass_gate_chi2"]),
            angle_innovation=True,
        )

    def _measurement_update(
        self,
        innovation: np.ndarray,
        observation: np.ndarray,
        measurement_noise: np.ndarray,
        gate_chi2: float,
        *,
        angle_innovation: bool,
    ) -> bool:
        innovation_covariance = (
            observation @ self._covariance @ observation.T + measurement_noise
        )
        try:
            solved_innovation = np.linalg.solve(innovation_covariance, innovation)
        except np.linalg.LinAlgError:
            return False
        nis = float(innovation.T @ solved_innovation)
        if not math.isfinite(nis) or nis > gate_chi2:
            return False

        gain = np.linalg.solve(
            innovation_covariance,
            observation @ self._covariance,
        ).T
        correction = gain @ innovation
        self._state += correction
        self._state[_YAW] = _wrap_angle(float(self._state[_YAW]))
        if angle_innovation:
            self._state[_YAW] = _wrap_angle(float(self._state[_YAW]))

        identity = np.eye(_STATE_SIZE)
        residual = identity - gain @ observation
        self._covariance = (
            residual @ self._covariance @ residual.T
            + gain @ measurement_noise @ gain.T
        )
        self._covariance = _symmetrize(self._covariance)
        return True

    def _build_estimate(
        self,
        frame: int,
        timestamp: float,
        imu: Any,
        *,
        force_degraded: bool = False,
    ) -> LocalizationEstimate:
        position_covariance = self._covariance[_POS, _POS]
        position_std = math.sqrt(max(float(np.linalg.eigvalsh(position_covariance)[-1]), 0.0))
        heading_std_deg = math.degrees(
            math.sqrt(max(float(self._covariance[_YAW, _YAW]), 0.0))
        )
        nominal = (
            self._initialized
            and not force_degraded
            and position_std <= float(self.config["maximum_position_std_m"])
            and heading_std_deg <= float(self.config["maximum_heading_std_deg"])
            and self._gnss_rejections <= int(self.config["max_consecutive_rejections"])
            and self._compass_rejections <= int(self.config["max_consecutive_rejections"])
        )
        state = HealthState.NOMINAL if nominal else HealthState.DEGRADED
        velocity = self._state[_VEL]
        acceleration_body, raw_yaw_rate = self._read_imu(imu)
        del acceleration_body
        return LocalizationEstimate(
            frame=frame,
            timestamp=timestamp,
            pose_local_enu=Pose3(
                position=Vector3(
                    x=float(self._state[0]),
                    y=float(self._state[1]),
                    z=float(self._height_m),
                ),
                rotation=Rotation3(
                    roll=0.0,
                    pitch=0.0,
                    yaw=math.degrees(float(self._state[_YAW])),
                ),
            ),
            velocity_local_enu_mps=Vector3(
                x=float(velocity[0]),
                y=float(velocity[1]),
                z=0.0,
            ),
            angular_rate_body_dps=Vector3(
                x=0.0,
                y=0.0,
                z=math.degrees(raw_yaw_rate),
            ),
            speed_mps=float(np.linalg.norm(velocity)),
            position_std_m=position_std,
            heading_std_deg=heading_std_deg,
            state=state,
            source="gnss_imu_planar_error_state_ekf",
            diagnostics={
                "frame": "LOCAL_ENU",
                "gnss_update_accepted": self._last_gnss_accepted,
                "compass_update_accepted": self._last_compass_accepted,
                "gnss_consecutive_rejections": self._gnss_rejections,
                "compass_consecutive_rejections": self._compass_rejections,
                "accel_bias_body_mps2": [
                    float(self._state[5]),
                    float(self._state[6]),
                ],
                "gyro_bias_z_dps": math.degrees(float(self._state[7])),
            },
        )

    def _initial_covariance(self) -> np.ndarray:
        standard_deviations = np.array(
            [
                float(self.config["initial_position_std_m"]),
                float(self.config["initial_position_std_m"]),
                float(self.config["initial_velocity_std_mps"]),
                float(self.config["initial_velocity_std_mps"]),
                math.radians(float(self.config["initial_heading_std_deg"])),
                float(self.config["initial_accel_bias_std_mps2"]),
                float(self.config["initial_accel_bias_std_mps2"]),
                math.radians(float(self.config["initial_gyro_bias_std_dps"])),
            ],
            dtype=np.float64,
        )
        return np.diag(standard_deviations**2)

    @staticmethod
    def _read_imu(imu: Any) -> tuple[np.ndarray, float]:
        acceleration = np.array(
            [float(imu.accelerometer.x), -float(imu.accelerometer.y)],
            dtype=np.float64,
        )
        yaw_rate_rps = -float(imu.gyroscope.z)
        return acceleration, yaw_rate_rps

    @staticmethod
    def _compass_to_enu_yaw(compass_radians: float) -> float:
        return _wrap_angle(math.pi / 2.0 - float(compass_radians))

    @staticmethod
    def _validate_measurements(gnss: Any, imu: Any) -> None:
        values = (
            gnss.latitude,
            gnss.longitude,
            gnss.altitude,
            imu.accelerometer.x,
            imu.accelerometer.y,
            imu.gyroscope.z,
            imu.compass,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("GNSS/IMU measurement contains a non-finite value")


def _rotation_2d(yaw: float) -> np.ndarray:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)


def _rotation_2d_derivative(yaw: float) -> np.ndarray:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.array([[-sine, -cosine], [cosine, -sine]], dtype=np.float64)


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)
