from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from urllib.parse import urlencode

from l4stack.perception.types import ArtifactRef

class SharedMemoryTransportError(RuntimeError):
    """Shared-memory taşıma sözleşmesi ihlal edildiğinde üretilir."""


@dataclass(slots=True)
class _SlotState:
    generation: int = 0
    readers: int = 0
    byte_size: int = 0


@dataclass(frozen=True, slots=True)
class SlotToken:
    """Producer tarafında bir ring slotunun yaşam süresini temsil eder."""

    ring_name: str
    slot: int
    generation: int


class SharedMemoryRing:
    """Linux POSIX shared-memory üzerinde bounded ve lease korumalı byte ring'i.

    Segment tek kez oluşturulur. Her slot sabit boyutludur; çalışma sırasında dosya
    sistemi yazımı, yeniden allocation veya SHA-256 hesabı yapılmaz. Model process'i
    ``shm://`` URI içindeki segment, offset, slot ve generation bilgisiyle veriyi açar.
    """

    def __init__(
        self,
        *,
        name: str,
        slot_count: int,
        slot_size: int,
        allocation_timeout_s: float,
    ) -> None:
        if slot_count < 2:
            raise ValueError("slot_count must be at least 2")
        if slot_size <= 0:
            raise ValueError("slot_size must be positive")
        if allocation_timeout_s <= 0.0:
            raise ValueError("allocation_timeout_s must be positive")
        self.name = name
        self.slot_count = int(slot_count)
        self.slot_size = int(slot_size)
        self.allocation_timeout_s = float(allocation_timeout_s)
        self._condition = threading.Condition()
        self._slots = [_SlotState() for _ in range(self.slot_count)]
        self._cursor = 0
        self._closed = False
        self._shm = shared_memory.SharedMemory(
            name=self.name,
            create=True,
            size=self.slot_count * self.slot_size,
        )

    def write(
        self,
        raw: bytes | bytearray | memoryview,
        *,
        artifact_name: str,
        media_type: str,
        shape: tuple[int, ...],
        dtype: str,
        source_frame: int,
        source_timestamp: float,
    ) -> tuple[ArtifactRef, SlotToken]:
        view = memoryview(raw)
        byte_size = view.nbytes
        if byte_size <= 0 or byte_size > self.slot_size:
            raise SharedMemoryTransportError(
                f"Artifact {artifact_name} size {byte_size} exceeds slot size {self.slot_size}"
            )
        deadline = time.monotonic() + self.allocation_timeout_s
        with self._condition:
            slot_index = self._wait_for_free_slot(deadline)
            slot = self._slots[slot_index]
            slot.generation += 1
            slot.readers = 1  # Producer rezervasyonu; commit sırasında consumer sayısına çevrilir.
            slot.byte_size = byte_size
            generation = slot.generation
            self._cursor = (slot_index + 1) % self.slot_count

        offset = slot_index * self.slot_size
        target = self._shm.buf[offset : offset + byte_size]
        try:
            target[:] = view.cast("B")
        except Exception:
            # Copy başarısız olursa producer rezervasyonu kaybolmamalıdır; aksi hâlde
            # ring slotu sonsuza kadar busy kalır.
            with self._condition:
                current = self._slots[slot_index]
                if current.generation == generation and current.readers == 1:
                    current.readers = 0
                    current.byte_size = 0
                    self._condition.notify_all()
            raise
        finally:
            target.release()
        uri = _build_shm_uri(
            segment=self.name,
            offset=offset,
            slot=slot_index,
            generation=generation,
            capacity=self.slot_size,
        )
        artifact = ArtifactRef(
            name=artifact_name,
            uri=uri,
            media_type=media_type,
            shape=shape,
            dtype=dtype,
            byte_size=byte_size,
            sha256=None,
            source_frame=source_frame,
            source_timestamp=source_timestamp,
        )
        return artifact, SlotToken(self.name, slot_index, generation)

    def commit(self, token: SlotToken, consumers: int) -> None:
        with self._condition:
            slot = self._checked_slot(token)
            if slot.readers != 1:
                raise SharedMemoryTransportError("Slot is not in producer-reserved state")
            slot.readers = consumers
            if consumers == 0:
                self._condition.notify_all()

    def release(self, token: SlotToken) -> None:
        with self._condition:
            slot = self._checked_slot(token)
            if slot.readers <= 0:
                raise SharedMemoryTransportError("Slot reader count underflow")
            slot.readers -= 1
            if slot.readers == 0:
                self._condition.notify_all()

    def abort(self, token: SlotToken) -> None:
        with self._condition:
            slot = self._checked_slot(token)
            if slot.readers != 1:
                raise SharedMemoryTransportError("Only producer-reserved slots can be aborted")
            slot.readers = 0
            self._condition.notify_all()

    def close(self, *, unlink: bool = True) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        self._shm.close()
        if unlink:
            with contextlib.suppress(FileNotFoundError):
                self._shm.unlink()

    def stats(self) -> dict[str, int]:
        with self._condition:
            return {
                "slot_count": self.slot_count,
                "slot_size": self.slot_size,
                "busy_slots": sum(1 for slot in self._slots if slot.readers > 0),
                "allocated_bytes": self.slot_count * self.slot_size,
            }

    def _wait_for_free_slot(self, deadline: float) -> int:
        while True:
            if self._closed:
                raise SharedMemoryTransportError(f"Shared-memory ring is closed: {self.name}")
            for step in range(self.slot_count):
                index = (self._cursor + step) % self.slot_count
                if self._slots[index].readers == 0:
                    return index
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(f"No free shared-memory slot in ring {self.name}")
            self._condition.wait(remaining)

    def _checked_slot(self, token: SlotToken) -> _SlotState:
        if token.ring_name != self.name:
            raise SharedMemoryTransportError("Slot token belongs to another ring")
        if token.slot < 0 or token.slot >= self.slot_count:
            raise SharedMemoryTransportError("Slot token index is invalid")
        slot = self._slots[token.slot]
        if slot.generation != token.generation:
            raise SharedMemoryTransportError("Slot token generation is stale")
        return slot




def _build_shm_uri(
    *,
    segment: str,
    offset: int,
    slot: int,
    generation: int,
    capacity: int,
) -> str:
    query = urlencode(
        {
            "offset": offset,
            "slot": slot,
            "generation": generation,
            "capacity": capacity,
        }
    )
    return f"shm://{segment}?{query}"
