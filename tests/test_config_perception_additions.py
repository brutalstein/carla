from pathlib import Path

from l4stack.config.loader import load_stack_config

ROOT = Path(__file__).resolve().parents[1]


def test_perception_models_and_runtime_contracts_are_declared() -> None:
    config = load_stack_config(ROOT / "config")
    assert not config.perception.enabled
    assert set(config.perception.models) == {
        "bevfusion_detection",
        "bevfusion_segmentation",
        "maptrv2",
        "tld_ready",
        "citysemsegformer",
    }
    for model in config.perception.models.values():
        contract = config.runtime.contract(model.runtime_component)
        executor = config.runtime.executors[model.executor]
        assert contract.priority == executor.priority
        assert model.request_timeout_s <= contract.execution_budget_s
        assert contract.expected_output_period_s >= 1.0 / model.target_rate_hz
        assert contract.output_lifespan_s <= config.perception.snapshot_max_age_s


def test_selected_bev_models_use_declared_surround_sensors() -> None:
    config = load_stack_config(ROOT / "config")
    detection = config.perception.models["bevfusion_detection"]
    segmentation = config.perception.models["bevfusion_segmentation"]
    assert len(detection.cameras) == 6
    assert detection.cameras == segmentation.cameras
    assert detection.lidar == segmentation.lidar == "lidar_top"
    assert config.sensors_by_name["lidar_top"].blueprint == "sensor.lidar.ray_cast"
