from pathlib import Path

from l4stack.config.loader import load_stack_config

ROOT = Path(__file__).resolve().parents[1]


def test_loads_all_configuration() -> None:
    config = load_stack_config(ROOT / "config")
    assert config.vehicle["vehicle"]["blueprint"] == "vehicle.lincoln.mkz_2020"
    assert len(config.sensors) == 26
    assert config.required_sensor_names == (
        "lidar_semantic_top",
        "gnss_roof",
        "imu_center",
    )
