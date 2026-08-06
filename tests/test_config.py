from pathlib import Path

from l4stack.config.loader import load_stack_config

ROOT = Path(__file__).resolve().parents[1]


def test_loads_runtime_configuration_without_ground_truth_sensors() -> None:
    config = load_stack_config(ROOT / "config")
    assert config.vehicle["vehicle"]["blueprint"] == "vehicle.lincoln.mkz_2020"
    assert len(config.sensors) == 14
    assert config.required_sensor_names == ("gnss_roof", "imu_center")
    assert config.localization["localization"]["algorithm"] == "planar_error_state_ekf"
    assert all("semantic" not in sensor.blueprint for sensor in config.sensors)
