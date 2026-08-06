from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

T = TypeVar("T")


class OverflowPolicy(str, Enum):
    """Bounded channel dolduğunda uygulanacak deterministik politika."""

    BLOCK = "BLOCK"
    DROP_OLDEST = "DROP_OLDEST"
    DROP_NEWEST = "DROP_NEWEST"
    LATEST_ONLY = "LATEST_ONLY"


class ChannelClosed(RuntimeError):
    """Kapatılmış bir channel üzerinde okuma/yazma yapılmak istendiğinde üretilir."""


@dataclass(frozen=True, slots=True)
class ChannelStats:
    published: int
    received: int
    dropped: int
    rejected: int
    current_depth: int
    capacity: int
    closed: bool


class BoundedChannel(Generic[T]):
    """Busy-spin kullanmayan, thread-safe ve bounded mesaj kanalı.

    Channel, producer ile consumer arasında sahipliği ayrıştırır. Consumer veri yokken
    ``Condition.wait`` üzerinde uyur; boş tur dönerek CPU tüketmez.
    """

    def __init__(
        self,
        name: str,
        capacity: int,
        overflow_policy: OverflowPolicy = OverflowPolicy.DROP_OLDEST,
    ) -> None:
        if capacity <= 0:
            raise ValueError("Channel capacity must be positive")
        self.name = name
        self.capacity = int(capacity)
        self.overflow_policy = OverflowPolicy(overflow_policy)
        self._items: deque[T] = deque()
        self._condition = threading.Condition()
        self._closed = False
        self._published = 0
        self._received = 0
        self._dropped = 0
        self._rejected = 0

    def publish(self, item: T, timeout: float | None = None) -> bool:
        """Mesajı yayınlar; DROP_NEWEST politikasında reddedilirse ``False`` döner."""

        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            if self._closed:
                raise ChannelClosed(f"Channel is closed: {self.name}")

            if self.overflow_policy is OverflowPolicy.LATEST_ONLY:
                self._dropped += len(self._items)
                self._items.clear()
            elif (
                self.overflow_policy is OverflowPolicy.DROP_OLDEST
                and len(self._items) >= self.capacity
            ):
                self._items.popleft()
                self._dropped += 1
            elif (
                self.overflow_policy is OverflowPolicy.DROP_NEWEST
                and len(self._items) >= self.capacity
            ):
                self._rejected += 1
                return False
            elif self.overflow_policy is OverflowPolicy.BLOCK:
                while len(self._items) >= self.capacity and not self._closed:
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0.0:
                        self._rejected += 1
                        return False
                    self._condition.wait(remaining)
                if self._closed:
                    raise ChannelClosed(f"Channel is closed: {self.name}")

            self._items.append(item)
            self._published += 1
            self._condition.notify_all()
            return True

    def receive(self, timeout: float | None = None) -> T:
        """Sıradaki mesajı döndürür; veri gelene kadar thread'i uyutur."""

        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while not self._items:
                if self._closed:
                    raise ChannelClosed(f"Channel is closed and empty: {self.name}")
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0.0:
                    raise TimeoutError(f"Channel receive timed out: {self.name}")
                self._condition.wait(remaining)
            item = self._items.popleft()
            self._received += 1
            self._condition.notify_all()
            return item

    def receive_latest(self, timeout: float | None = None) -> T:
        """En güncel mesajı döndürür ve aradaki eski mesajları kontrollü düşürür."""

        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while not self._items:
                if self._closed:
                    raise ChannelClosed(f"Channel is closed and empty: {self.name}")
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0.0:
                    raise TimeoutError(f"Channel receive timed out: {self.name}")
                self._condition.wait(remaining)
            item = self._items.pop()
            self._dropped += len(self._items)
            self._items.clear()
            self._received += 1
            self._condition.notify_all()
            return item

    def try_receive(self) -> T | None:
        with self._condition:
            if not self._items:
                return None
            item = self._items.popleft()
            self._received += 1
            self._condition.notify_all()
            return item

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def stats(self) -> ChannelStats:
        with self._condition:
            return ChannelStats(
                published=self._published,
                received=self._received,
                dropped=self._dropped,
                rejected=self._rejected,
                current_depth=len(self._items),
                capacity=self.capacity,
                closed=self._closed,
            )
