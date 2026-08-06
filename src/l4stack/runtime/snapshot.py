from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Snapshot(Generic[T]):
    version: int
    published_at: float
    value: T


class AtomicSnapshotStore(Generic[T]):
    """Okuyuculara yalnızca tamamlanmış eski veya yeni snapshot gösterir.

    Yeni değer ayrı bir nesne olarak hazırlanır ve ``publish`` çağrısında tek kilitli
    işlemle aktif hâle gelir. Okuyucu hiçbir zaman yarım güncellenmiş state görmez.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._condition = threading.Condition()
        self._snapshot: Snapshot[T] | None = None
        self._version = 0

    def publish(self, value: T, published_at: float) -> Snapshot[T]:
        with self._condition:
            self._version += 1
            snapshot = Snapshot(self._version, float(published_at), value)
            self._snapshot = snapshot
            self._condition.notify_all()
            return snapshot

    def get(self) -> Snapshot[T] | None:
        with self._condition:
            return self._snapshot

    def require(self) -> Snapshot[T]:
        snapshot = self.get()
        if snapshot is None:
            raise LookupError(f"Snapshot is not available: {self.name}")
        return snapshot

    def wait_for_newer(self, version: int, timeout: float | None = None) -> Snapshot[T]:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._snapshot is None or self._snapshot.version <= version:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0.0:
                    raise TimeoutError(f"Snapshot wait timed out: {self.name}")
                self._condition.wait(remaining)
            return self._snapshot
