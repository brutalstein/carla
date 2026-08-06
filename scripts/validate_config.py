from pathlib import Path

from l4stack.config.loader import load_stack_config
from l4stack.sensors.coverage import camera_azimuth_gaps

config = load_stack_config(Path(__file__).resolve().parents[1] / "config")
print(f"sensors={len(config.sensors)} required={config.required_sensor_names}")
print(f"camera_gaps={camera_azimuth_gaps(config.sensors)}")
