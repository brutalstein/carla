from pathlib import Path

import numpy as np

from l4stack.config.loader import load_stack_config
from l4stack.perception.semantic_lidar import SemanticLidarInstanceDetector
from l4stack.sensors.decoders import SEMANTIC_LIDAR_DTYPE

ROOT = Path(__file__).resolve().parents[1]


def _point(x: float, y: float, z: float, object_id: int, tag: int):
    return (x, y, z, 1.0, object_id, tag)


def test_semantic_instance_detection_is_sorted_and_filtered() -> None:
    config = load_stack_config(ROOT / "config")
    detector = SemanticLidarInstanceDetector(config.perception, 0.05)
    points = np.array(
        [
            _point(10.0, 0.0, 0.0, 42, 10),
            _point(10.2, 0.1, 0.1, 42, 10),
            _point(9.8, -0.1, 0.0, 42, 10),
            _point(10.0, 0.2, 1.0, 42, 10),
            _point(4.0, 2.0, 0.0, 7, 4),
            _point(4.1, 2.1, 0.0, 7, 4),
            _point(3.9, 1.9, 0.0, 7, 4),
            _point(4.0, 2.0, 1.7, 7, 4),
            _point(2.0, 0.0, 0.0, 99, 7),
            _point(2.1, 0.0, 0.0, 99, 7),
            _point(2.2, 0.0, 0.0, 99, 7),
            _point(2.3, 0.0, 0.0, 99, 7),
        ],
        dtype=SEMANTIC_LIDAR_DTYPE,
    )
    detections = detector.detect_array(100, points)
    assert [d.track_id for d in detections] == [7, 42]
    assert [d.class_name for d in detections] == ["Pedestrian", "Vehicle"]
    assert detections[1].point_count == 4
