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
    assert not config.perception.enabled


def test_runtime_contracts_are_loaded_and_consistent() -> None:
    config = load_stack_config(ROOT / "config")
    contract = config.runtime.contract("localization")
    assert contract.priority == config.runtime.executors["localization"].priority
    assert contract.output_lifespan_s > contract.max_input_age_s
    assert config.runtime.sensor_bundle_lifespan_s >= contract.max_input_age_s
    assert config.runtime.deadline_event_history == 10000
    assert "perception_bevfusion_detection" in config.runtime.components
