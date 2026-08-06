from __future__ import annotations

from dataclasses import dataclass

from l4stack.core.types import Vector3


@dataclass
class _TrackState:
    centroid: Vector3
    frame: int
    velocity: Vector3


class DeterministicInstanceTracker:
    def __init__(self, fixed_delta_seconds: float, ttl_frames: int, velocity_alpha: float) -> None:
        self.dt = fixed_delta_seconds
        self.ttl_frames = ttl_frames
        self.alpha = velocity_alpha
        self._tracks: dict[int, _TrackState] = {}

    def update(self, object_id: int, frame: int, centroid: Vector3) -> Vector3:
        previous = self._tracks.get(object_id)
        velocity = Vector3(0.0, 0.0, 0.0)
        if previous is not None and frame > previous.frame:
            dt = (frame - previous.frame) * self.dt
            measured = Vector3(
                (centroid.x - previous.centroid.x) / dt,
                (centroid.y - previous.centroid.y) / dt,
                (centroid.z - previous.centroid.z) / dt,
            )
            velocity = Vector3(
                self.alpha * measured.x + (1.0 - self.alpha) * previous.velocity.x,
                self.alpha * measured.y + (1.0 - self.alpha) * previous.velocity.y,
                self.alpha * measured.z + (1.0 - self.alpha) * previous.velocity.z,
            )
        self._tracks[object_id] = _TrackState(centroid, frame, velocity)
        self._expire(frame)
        return velocity

    def _expire(self, frame: int) -> None:
        for object_id, track in tuple(self._tracks.items()):
            if frame - track.frame > self.ttl_frames:
                self._tracks.pop(object_id, None)
