from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass

from l4stack.perception.shared_memory import SharedMemoryRing, SlotToken
from l4stack.perception.types import ArtifactRef


@dataclass(frozen=True, slots=True)
class WorkerOutputConfig:
    namespace: str
    slot_count: int = 4
    slot_size: int = 4_194_304
    allocation_timeout_s: float = 0.05


class WorkerOutputStore:
    """Raster üreten CUDA worker'lar için lease kontrollü output ring yöneticisi.

    Host, ModelOutput snapshot'ı değiştirdiğinde protokol v2 ``release`` mesajı gönderir.
    Store URI→token eşlemesini kullanarak ilgili slotu serbest bırakır. Böylece worker
    maskeyi host/downstream okurken overwrite etmez ve request başına shm oluşturmaz.
    """

    def __init__(self, config: WorkerOutputConfig) -> None:
        if not config.namespace:
            raise ValueError("worker output namespace cannot be empty")
        self._config = config
        self._instance = uuid.uuid4().hex[:8]
        self._rings: dict[str, SharedMemoryRing] = {}
        self._leases: dict[str, tuple[SharedMemoryRing, SlotToken]] = {}
        self._lock = threading.RLock()

    def publish(
        self,
        raw: bytes | bytearray | memoryview,
        *,
        name: str,
        media_type: str,
        shape: tuple[int, ...],
        dtype: str,
        source_frame: int,
        source_timestamp: float,
    ) -> ArtifactRef:
        ring = self._ring(name)
        artifact, token = ring.write(
            raw,
            artifact_name=name,
            media_type=media_type,
            shape=shape,
            dtype=dtype,
            source_frame=source_frame,
            source_timestamp=source_timestamp,
        )
        ring.commit(token, 1)
        with self._lock:
            self._leases[artifact.uri] = (ring, token)
        return artifact

    def release(self, artifact_uris: tuple[str, ...]) -> None:
        missing: list[str] = []
        with self._lock:
            leases = []
            for uri in artifact_uris:
                lease = self._leases.pop(uri, None)
                if lease is None:
                    missing.append(uri)
                else:
                    leases.append(lease)
        for ring, token in leases:
            ring.release(token)
        if missing:
            raise KeyError(f"Unknown or already released output artifacts: {missing}")

    def close(self) -> None:
        with self._lock:
            rings = tuple(self._rings.values())
            self._rings.clear()
            self._leases.clear()
        for ring in rings:
            ring.close(unlink=True)

    def stats(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {name: ring.stats() for name, ring in self._rings.items()}

    def _ring(self, output_name: str) -> SharedMemoryRing:
        with self._lock:
            existing = self._rings.get(output_name)
            if existing is not None:
                return existing
            safe = "".join(
                character if character.isalnum() or character == "_" else "_"
                for character in output_name
            )
            name = f"{self._config.namespace}_{self._instance}_{safe}"[:200]
            ring = SharedMemoryRing(
                name=name,
                slot_count=self._config.slot_count,
                slot_size=self._config.slot_size,
                allocation_timeout_s=self._config.allocation_timeout_s,
            )
            self._rings[output_name] = ring
            return ring
