from dataclasses import dataclass
from pathlib import Path

from l4stack.config.loader import load_stack_config
from l4stack.core.types import HealthState, LocalizationEstimate, Pose3, Rotation3, Vector3
from l4stack.odd.monitor import OddMonitor

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class FakeWeather:
    precipitation: float = 0.0
    wetness: float = 0.0
    sun_altitude_angle: float = 60.0
    fog_density: float = 0.0
    fog_distance: float = 1000.0


class FakeMap:
    name = "/Game/Carla/Maps/Town10HD_Opt"


class FakeWorld:
    def get_weather(self):
        return FakeWeather()

    def get_map(self):
        return FakeMap()


def test_nominal_odd() -> None:
    config = load_stack_config(ROOT / "config")
    monitor = OddMonitor(config.odd, config.sensors)
    loc = LocalizationEstimate(
        frame=1,
        timestamp=0.05,
        pose_world=Pose3(Vector3(0, 0, 0), Rotation3(0, 0, 0)),
        velocity_world_mps=Vector3(0, 0, 0),
        angular_velocity_world_dps=Vector3(0, 0, 0),
        speed_mps=0.0,
        position_std_m=0.01,
        heading_std_deg=0.02,
        state=HealthState.NOMINAL,
        source="test",
    )
    bundle = {name: object() for name in config.required_sensor_names}
    assessment = monitor.assess(FakeWorld(), loc, bundle)
    assert assessment.state.value == "IN_ODD"
