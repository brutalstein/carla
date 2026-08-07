from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any

from l4stack.errors import SensorTimeoutError


class SensorSynchronizer:
    """Thread-safe exact-frame ve latest-at-or-before sensör bariyeri."""

    def __init__(self, retention_frames: int = 8) -> None:
        self._condition = threading.Condition()
        self._data: dict[int, dict[str, Any]] = defaultdict(dict)
        self._latest: dict[str, Any] = {}
        self._retention_frames = retention_frames

    def callback(self, sensor_name: str):
        def receive(data: Any) -> None:
            frame = int(data.frame)
            with self._condition:
                self._data[frame][sensor_name] = data
                self._latest[sensor_name] = data
                self._purge(frame)
                self._condition.notify_all()

        return receive

    def wait_for_frame(
        self,
        frame: int,
        required_sensor_names: tuple[str, ...],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while True:
                frame_data = self._data.get(frame, {})
                missing = [name for name in required_sensor_names if name not in frame_data]
                if not missing:
                    result = dict(frame_data)
                    for name, data in self._latest.items():
                        if name not in result and int(getattr(data, "frame", -1)) <= frame:
                            result[name] = data
                    return result
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise SensorTimeoutError(
                        f"Frame {frame} timed out; missing required sensors: {missing}"
                    )
                self._condition.wait(remaining)

    def wait_for_latest_at_or_before(
        self,
        frame: int,
        sensor_names: tuple[str, ...],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Her sensörün hedef frame'den ileri olmayan son ölçümünü döndürür.

        Kamera 10 Hz, dünya ve LiDAR 20 Hz çalışırken her sensörün aynı frame numarasına
        sahip olması beklenmez. Bu bariyer ilk geçerli callback gelene kadar uyur ve
        sonrasında model adapter'ının gerçek timestamp skew kontrolü yapacağı ölçümleri
        döndürür. Busy-spin veya gelecekteki bir frame'in yanlışlıkla kullanımı yoktur.
        """

        if timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        names = tuple(dict.fromkeys(sensor_names))
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while True:
                result = {
                    name: data
                    for name in names
                    if (data := self._latest.get(name)) is not None
                    and int(getattr(data, "frame", -1)) <= frame
                }
                missing = [name for name in names if name not in result]
                if not missing:
                    return result
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise SensorTimeoutError(
                        f"Latest sensor wait for frame {frame} timed out; missing: {missing}"
                    )
                self._condition.wait(remaining)

    def _purge(self, current_frame: int) -> None:
        oldest = current_frame - self._retention_frames
        for frame in tuple(self._data):
            if frame < oldest:
                self._data.pop(frame, None)
