from __future__ import annotations

import math
import threading
import time
from typing import Protocol


class Clock(Protocol):
    """Runtime bileşenlerinin kullandığı ortak saat arayüzü."""

    def now(self) -> float:
        """Saniye cinsinden monoton zaman döndürür."""


class SteadyClock:
    """İşlem süresi ve timeout ölçümleri için işletim sistemi monoton saati."""

    def now(self) -> float:
        return time.monotonic()


class WallClock:
    """İnsan okunabilir kayıtlar için duvar saati."""

    def now(self) -> float:
        return time.time()


class SimulationClock:
    """CARLA simülasyon zamanını thread-safe biçimde yayınlayan saat.

    Bu saat yalnızca ``update`` çağrılarıyla ilerler. Böylece replay ve deterministik
    testlerde aynı zaman dizisi tekrar kullanılabilir. Geriye giden zaman kabul edilmez;
    aksi hâlde freshness ve deadline hesapları anlamsızlaşır.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._timestamp: float | None = None

    def update(self, timestamp: float) -> None:
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("Simulation timestamp must be finite")
        with self._condition:
            if self._timestamp is not None and timestamp < self._timestamp:
                raise ValueError(
                    f"Simulation clock cannot move backwards: {timestamp} < {self._timestamp}"
                )
            self._timestamp = timestamp
            self._condition.notify_all()

    def now(self) -> float:
        with self._condition:
            if self._timestamp is None:
                raise RuntimeError("Simulation clock has not been initialized")
            return self._timestamp

    def wait_until(self, timestamp: float, timeout: float | None = None) -> bool:
        """Saat hedef zamana ulaşana kadar busy-spin yapmadan bekler."""

        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._timestamp is None or self._timestamp < timestamp:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0.0:
                    return False
                self._condition.wait(remaining)
            return True


class ManualClock:
    """Unit test ve deterministic replay için elle ilerletilen saat."""

    def __init__(self, initial_time: float = 0.0) -> None:
        self._lock = threading.Lock()
        self._time = float(initial_time)

    def now(self) -> float:
        with self._lock:
            return self._time

    def set(self, timestamp: float) -> None:
        with self._lock:
            timestamp = float(timestamp)
            if timestamp < self._time:
                raise ValueError("Manual clock cannot move backwards")
            self._time = timestamp

    def advance(self, seconds: float) -> float:
        if seconds < 0.0:
            raise ValueError("Clock advance must be non-negative")
        with self._lock:
            self._time += float(seconds)
            return self._time
