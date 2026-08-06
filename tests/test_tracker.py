from l4stack.core.types import Vector3
from l4stack.perception.tracker import DeterministicInstanceTracker


def test_tracker_velocity_is_deterministic() -> None:
    tracker = DeterministicInstanceTracker(0.05, ttl_frames=4, velocity_alpha=1.0)
    assert tracker.update(1, 10, Vector3(0.0, 0.0, 0.0)) == Vector3(0.0, 0.0, 0.0)
    velocity = tracker.update(1, 11, Vector3(0.5, 0.0, 0.0))
    assert velocity == Vector3(10.0, 0.0, 0.0)
