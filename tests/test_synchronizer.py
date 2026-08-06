from dataclasses import dataclass

from l4stack.sensors.synchronizer import SensorSynchronizer


@dataclass
class FakeMeasurement:
    frame: int


def test_exact_frame_barrier() -> None:
    sync = SensorSynchronizer()
    sync.callback("a")(FakeMeasurement(10))
    sync.callback("b")(FakeMeasurement(10))
    bundle = sync.wait_for_frame(10, ("a", "b"), 0.01)
    assert sorted(bundle) == ["a", "b"]
    assert bundle["a"].frame == 10
