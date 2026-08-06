from __future__ import annotations

import math
from typing import Any

import numpy as np

from l4stack.core.types import (
    BoundingBox3D,
    Detection3D,
    HealthState,
    PerceptionFrame,
    Vector3,
)
from l4stack.perception.tracker import DeterministicInstanceTracker
from l4stack.sensors.decoders import decode_semantic_lidar


class SemanticLidarInstanceDetector:
    """Simulation-only deterministic 3D detector using CARLA semantic LiDAR IDs."""

    def __init__(self, document: dict[str, Any], fixed_delta_seconds: float) -> None:
        cfg = document["perception"]
        self.source_sensor = str(cfg["source_sensor"])
        self.minimum_points = int(cfg["minimum_points_per_object"])
        self.maximum_range = float(cfg["maximum_range_m"])
        self.ignored_tags = {int(v) for v in cfg.get("ignored_semantic_tags", [])}
        self.keep_tags = {int(v) for v in cfg.get("keep_semantic_tags", [])}
        self.class_names = {int(k): str(v) for k, v in document.get("class_names", {}).items()}
        self.tracker = DeterministicInstanceTracker(
            fixed_delta_seconds=fixed_delta_seconds,
            ttl_frames=int(cfg.get("track_ttl_frames", 4)),
            velocity_alpha=float(cfg.get("velocity_alpha", 0.5)),
        )

    def detect(
        self, frame: int, timestamp: float, sensor_bundle: dict[str, Any]
    ) -> PerceptionFrame:
        measurement = sensor_bundle.get(self.source_sensor)
        if measurement is None:
            return PerceptionFrame(
                frame=frame,
                timestamp=timestamp,
                detections=(),
                source=self.source_sensor,
                health=HealthState.FAILED,
                diagnostics={"error": "source sensor missing"},
            )
        points = decode_semantic_lidar(measurement)
        detections = self.detect_array(frame, points)
        return PerceptionFrame(
            frame=frame,
            timestamp=timestamp,
            detections=detections,
            source=self.source_sensor,
            health=HealthState.NOMINAL,
            diagnostics={"input_points": int(points.size), "objects": len(detections)},
        )

    def detect_array(self, frame: int, points: np.ndarray) -> tuple[Detection3D, ...]:
        if points.size == 0:
            return ()
        result: list[Detection3D] = []
        for object_id in sorted(int(v) for v in np.unique(points["object_idx"]) if int(v) != 0):
            object_points = points[points["object_idx"] == object_id]
            tag = int(object_points["object_tag"][0])
            if tag in self.ignored_tags or (self.keep_tags and tag not in self.keep_tags):
                continue
            xyz = np.column_stack((object_points["x"], object_points["y"], object_points["z"]))
            ranges = np.linalg.norm(xyz, axis=1)
            xyz = xyz[ranges <= self.maximum_range]
            if xyz.shape[0] < self.minimum_points:
                continue
            minimum = xyz.min(axis=0)
            maximum = xyz.max(axis=0)
            center = (minimum + maximum) * 0.5
            size = maximum - minimum
            centroid_raw = xyz.mean(axis=0)
            centroid = Vector3(*(float(v) for v in centroid_raw))
            velocity = self.tracker.update(object_id, frame, centroid)
            distance = math.sqrt(centroid.x**2 + centroid.y**2 + centroid.z**2)
            result.append(
                Detection3D(
                    track_id=object_id,
                    semantic_tag=tag,
                    class_name=self.class_names.get(tag, f"SemanticTag{tag}"),
                    point_count=int(xyz.shape[0]),
                    confidence=min(1.0, xyz.shape[0] / max(20.0, float(self.minimum_points))),
                    bbox=BoundingBox3D(
                        center=Vector3(*(float(v) for v in center)),
                        size=Vector3(*(float(v) for v in size)),
                    ),
                    centroid=centroid,
                    velocity_ego_mps=velocity,
                    range_m=distance,
                )
            )
        return tuple(result)
