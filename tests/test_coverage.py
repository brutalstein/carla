from pathlib import Path

from l4stack.config.loader import load_stack_config
from l4stack.sensors.coverage import camera_azimuth_gaps

ROOT = Path(__file__).resolve().parents[1]


def test_camera_coverage_has_no_horizontal_gap() -> None:
    config = load_stack_config(ROOT / "config")
    assert camera_azimuth_gaps(config.sensors) == []
