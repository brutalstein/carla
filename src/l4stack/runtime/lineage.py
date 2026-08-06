from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

from l4stack.runtime.message import MessageEnvelope


@dataclass(frozen=True, slots=True)
class LineageRecord:
    message_id: str
    source: str
    sequence_id: int
    source_timestamp: float
    publish_timestamp: float
    parents: tuple[str, ...]


class LineageStore:
    """Mesajlar arası parent ilişkilerini bounded bellek içinde takip eder."""

    def __init__(self, maximum_records: int = 100_000) -> None:
        if maximum_records <= 0:
            raise ValueError("maximum_records must be positive")
        self._maximum_records = maximum_records
        self._records: dict[str, LineageRecord] = {}
        self._order: deque[str] = deque()
        self._lock = threading.Lock()

    def record(self, envelope: MessageEnvelope[Any]) -> None:
        record = LineageRecord(
            message_id=envelope.message_id,
            source=envelope.source,
            sequence_id=envelope.sequence_id,
            source_timestamp=envelope.source_timestamp,
            publish_timestamp=envelope.publish_timestamp,
            parents=envelope.parents,
        )
        with self._lock:
            if record.message_id in self._records:
                raise ValueError(f"Duplicate message id in lineage: {record.message_id}")
            self._records[record.message_id] = record
            self._order.append(record.message_id)
            while len(self._order) > self._maximum_records:
                oldest = self._order.popleft()
                self._records.pop(oldest, None)

    def get(self, message_id: str) -> LineageRecord | None:
        with self._lock:
            return self._records.get(message_id)

    def trace(self, message_id: str, maximum_depth: int = 32) -> tuple[LineageRecord, ...]:
        """Çıktıdan sensör girdilerine doğru breadth-first soy ağacı döndürür."""

        if maximum_depth <= 0:
            return ()
        with self._lock:
            records = dict(self._records)
        result: list[LineageRecord] = []
        queue: deque[tuple[str, int]] = deque([(message_id, 0)])
        visited: set[str] = set()
        while queue:
            current, depth = queue.popleft()
            if current in visited or depth >= maximum_depth:
                continue
            visited.add(current)
            record = records.get(current)
            if record is None:
                continue
            result.append(record)
            queue.extend((parent, depth + 1) for parent in record.parents)
        return tuple(result)
