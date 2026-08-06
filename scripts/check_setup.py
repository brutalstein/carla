from pathlib import Path

from l4stack.config.loader import load_stack_config
from l4stack.sensors.coverage import camera_azimuth_gaps

root = Path(__file__).resolve().parents[1]
config = load_stack_config(root / "config")
print(f"[OK] config: {len(config.sensors)} sensors")
print(f"[OK] required: {config.required_sensor_names}")
print(f"[OK] camera gaps: {camera_azimuth_gaps(config.sensors)}")
try:
    import carla
except ImportError:
    print("[WARN] CARLA Python API is not importable in this environment")
else:
    print(f"[OK] CARLA Python API: {carla.__file__}")
