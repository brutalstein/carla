from __future__ import annotations

import contextlib
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from l4stack.perception.shared_memory_ring import (
    SharedMemoryRing,
    SharedMemoryTransportError,
    SlotToken,
)
from l4stack.perception.types import ArtifactRef


def _sanitize_name(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in value
    )
    if not cleaned:
        raise ValueError("Shared-memory name cannot be empty")
    return cleaned


@dataclass(slots=True)
class PreparedArtifacts:
    """Bir perception input'u için hazırlanan artifact ve slot rezervasyonları.

    Slotlar ilk aşamada producer rezervasyonu ile tutulur. Model submit sonucuna göre
    ``commit`` çağrısı gerçek consumer sayısını kaydeder. Her tamamlanan model için
    ``release_one`` çağrılmalıdır. Böylece producer, model okurken slotu yeniden yazmaz.
    """

    store: "SharedMemoryArtifactStore"
    cameras: tuple[ArtifactRef, ...]
    lidar: ArtifactRef | None
    tokens: tuple[SlotToken, ...]
    _message_id: str | None = None
    _remaining: int = 0
    _closed: bool = False

    def commit(self, message_id: str, consumers: int) -> None:
        if self._closed:
            raise SharedMemoryTransportError("Prepared artifact lease is already closed")
        if self._message_id is not None:
            raise SharedMemoryTransportError("Prepared artifact lease was already committed")
        if consumers < 0:
            raise ValueError("consumers cannot be negative")
        self._message_id = message_id
        self._remaining = consumers
        self.store._commit_tokens(self.tokens, consumers)
        if consumers == 0:
            self._closed = True

    def release_one(self) -> None:
        if self._closed:
            return
        if self._message_id is None:
            raise SharedMemoryTransportError("Lease must be committed before release")
        if self._remaining <= 0:
            raise SharedMemoryTransportError("Lease release underflow")
        self.store._release_tokens(self.tokens)
        self._remaining -= 1
        if self._remaining == 0:
            self._closed = True

    def abort(self) -> None:
        """Submit öncesi hata oluşursa producer rezervasyonunu serbest bırakır."""

        if self._closed:
            return
        if self._message_id is None:
            self.store._abort_tokens(self.tokens)
        else:
            while self._remaining > 0:
                self.release_one()
        self._closed = True


class SharedMemoryArtifactStore:
    """CARLA kamera/LiDAR ölçümlerini kopyasız process paylaşımına hazırlar.

    CARLA callback belleği Python/CARLA tarafından yönetildiği için en az bir host copy
    zorunludur. Bu copy doğrudan önceden ayrılmış POSIX shared-memory slotuna yapılır;
    disk I/O ve frame başına allocation yoktur. Container'lar ``ipc: host`` ile aynı
    segmentleri açar ve kendi CUDA stream'lerine async H2D copy yapar.
    """

    def __init__(
        self,
        *,
        namespace: str,
        slot_count: int,
        camera_slot_bytes: int,
        lidar_slot_bytes: int,
        allocation_timeout_s: float,
    ) -> None:
        if not namespace:
            raise ValueError("namespace cannot be empty")
        self.namespace = _sanitize_name(namespace)
        self.slot_count = slot_count
        self.camera_slot_bytes = camera_slot_bytes
        self.lidar_slot_bytes = lidar_slot_bytes
        self.allocation_timeout_s = allocation_timeout_s
        self._rings: dict[str, SharedMemoryRing] = {}
        self._lock = threading.RLock()
        self._instance = uuid.uuid4().hex[:8]
        self._calibration_ref: ArtifactRef | None = None

    def prepare(
        self,
        *,
        camera_names: tuple[str, ...],
        lidar_name: str | None,
        bundle: dict[str, Any],
    ) -> PreparedArtifacts:
        artifacts: list[ArtifactRef] = []
        tokens: list[SlotToken] = []
        try:
            for name in camera_names:
                measurement = bundle[name]
                width = int(measurement.width)
                height = int(measurement.height)
                raw = memoryview(measurement.raw_data)
                expected = width * height * 4
                if raw.nbytes != expected:
                    raise SharedMemoryTransportError(
                        f"Unexpected CARLA BGRA size for {name}: {raw.nbytes} != {expected}"
                    )
                ring = self._ring(name, self.camera_slot_bytes)
                artifact, token = ring.write(
                    raw,
                    artifact_name=name,
                    media_type="application/x-carla-bgra8",
                    shape=(height, width, 4),
                    dtype="uint8",
                    source_frame=int(measurement.frame),
                    source_timestamp=float(measurement.timestamp),
                )
                artifacts.append(artifact)
                tokens.append(token)

            lidar_artifact: ArtifactRef | None = None
            if lidar_name is not None:
                measurement = bundle[lidar_name]
                raw = memoryview(measurement.raw_data)
                point_width = 16
                if raw.nbytes <= 0 or raw.nbytes % point_width != 0:
                    raise SharedMemoryTransportError(
                        f"Unexpected CARLA LiDAR byte count for {lidar_name}: {raw.nbytes}"
                    )
                ring = self._ring(lidar_name, self.lidar_slot_bytes)
                lidar_artifact, token = ring.write(
                    raw,
                    artifact_name=lidar_name,
                    media_type="application/x-carla-lidar-f32",
                    shape=(raw.nbytes // point_width, 4),
                    dtype="float32",
                    source_frame=int(measurement.frame),
                    source_timestamp=float(measurement.timestamp),
                )
                tokens.append(token)
            return PreparedArtifacts(
                store=self,
                cameras=tuple(artifacts),
                lidar=lidar_artifact,
                tokens=tuple(tokens),
            )
        except Exception:
            self._abort_tokens(tuple(tokens))
            raise

    def calibration_ref(self, path: Path) -> ArtifactRef:
        """Kalibrasyonu container-bağımsız immutable shared-memory artifact'ına taşır."""

        with self._lock:
            if self._calibration_ref is not None:
                return self._calibration_ref
        resolved = path.resolve()
        if not resolved.is_file():
            raise SharedMemoryTransportError(f"Calibration file does not exist: {resolved}")
        raw = resolved.read_bytes()
        if not raw:
            raise SharedMemoryTransportError("Calibration file cannot be empty")
        ring = self._ring("__calibration__", max(65_536, len(raw)))
        artifact, _token = ring.write(
            raw,
            artifact_name="calibration",
            media_type="application/json",
            shape=(len(raw),),
            dtype="uint8",
            source_frame=0,
            source_timestamp=0.0,
        )
        # Kalibrasyon stack yaşam süresince immutable kalır. Producer rezervasyonu
        # bilinçli olarak tutulur; ring yalnız store.close() sırasında kaldırılır.
        with self._lock:
            if self._calibration_ref is None:
                self._calibration_ref = artifact
            return self._calibration_ref

    def close(self) -> None:
        with self._lock:
            rings = tuple(self._rings.values())
            self._rings.clear()
        for ring in rings:
            ring.close(unlink=True)

    def stats(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {name: ring.stats() for name, ring in self._rings.items()}

    def _ring(self, sensor_name: str, slot_size: int) -> SharedMemoryRing:
        with self._lock:
            existing = self._rings.get(sensor_name)
            if existing is not None:
                return existing
            ring_name = _sanitize_name(
                f"{self.namespace}_{self._instance}_{sensor_name}"
            )[:200]
            ring = SharedMemoryRing(
                name=ring_name,
                slot_count=self.slot_count,
                slot_size=slot_size,
                allocation_timeout_s=self.allocation_timeout_s,
            )
            self._rings[sensor_name] = ring
            return ring

    def _commit_tokens(self, tokens: tuple[SlotToken, ...], consumers: int) -> None:
        for token in tokens:
            self._rings_by_token(token).commit(token, consumers)

    def _release_tokens(self, tokens: tuple[SlotToken, ...]) -> None:
        for token in tokens:
            self._rings_by_token(token).release(token)

    def _abort_tokens(self, tokens: tuple[SlotToken, ...]) -> None:
        for token in tokens:
            with contextlib.suppress(Exception):
                self._rings_by_token(token).abort(token)

    def _rings_by_token(self, token: SlotToken) -> SharedMemoryRing:
        with self._lock:
            for ring in self._rings.values():
                if ring.name == token.ring_name:
                    return ring
        raise SharedMemoryTransportError(f"Unknown shared-memory ring: {token.ring_name}")
