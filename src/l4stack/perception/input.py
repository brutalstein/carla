from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from l4stack.perception.types import ArtifactRef, PerceptionInput
from l4stack.runtime.clock import Clock
from l4stack.runtime.message import MessageEnvelope, MessageFactory


class PerceptionArtifactError(RuntimeError):
    """CARLA sensör ölçümü artifact dosyasına çevrilemediğinde üretilir."""


class PerceptionArtifactStore:
    """CARLA callback verilerini bounded disk artifact'larına yazar.

    Bu ilk transport uygulaması debug ve replay odaklıdır. Üretim hızında disk yerine
    aynı ``ArtifactRef`` sözleşmesini kullanan shared-memory/CUDA IPC transportu
    eklenmelidir.
    """

    def __init__(self, root: Path, retention_frames: int = 32) -> None:
        if retention_frames <= 0:
            raise ValueError("retention_frames must be positive")
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._retention_frames = retention_frames
        self._cache: dict[tuple[str, int], ArtifactRef] = {}

    def write_camera(self, sensor_name: str, measurement: Any) -> ArtifactRef:
        frame = int(measurement.frame)
        cached = self._cache.get((sensor_name, frame))
        if cached is not None:
            return cached
        timestamp = float(measurement.timestamp)
        width = int(measurement.width)
        height = int(measurement.height)
        raw = bytes(measurement.raw_data)
        expected = width * height * 4
        if len(raw) != expected:
            raise PerceptionArtifactError(
                f"Unexpected CARLA BGRA size for {sensor_name}: {len(raw)} != {expected}"
            )
        path = self._frame_dir(frame) / f"{sensor_name}.bgra8"
        path.write_bytes(raw)
        self._purge(frame)
        artifact = ArtifactRef(
            name=sensor_name,
            uri=path.as_uri(),
            media_type="application/x-carla-bgra8",
            shape=(height, width, 4),
            dtype="uint8",
            byte_size=len(raw),
            sha256=_sha256_bytes(raw),
            source_frame=frame,
            source_timestamp=timestamp,
        )
        self._cache[(sensor_name, frame)] = artifact
        return artifact

    def write_lidar(self, sensor_name: str, measurement: Any) -> ArtifactRef:
        frame = int(measurement.frame)
        cached = self._cache.get((sensor_name, frame))
        if cached is not None:
            return cached
        timestamp = float(measurement.timestamp)
        raw = bytes(measurement.raw_data)
        point_width = 4 * 4
        if not raw or len(raw) % point_width != 0:
            raise PerceptionArtifactError(
                f"Unexpected CARLA LiDAR byte count for {sensor_name}: {len(raw)}"
            )
        point_count = len(raw) // point_width
        path = self._frame_dir(frame) / f"{sensor_name}.f32"
        path.write_bytes(raw)
        self._purge(frame)
        artifact = ArtifactRef(
            name=sensor_name,
            uri=path.as_uri(),
            media_type="application/x-carla-lidar-f32",
            shape=(point_count, 4),
            dtype="float32",
            byte_size=len(raw),
            sha256=_sha256_bytes(raw),
            source_frame=frame,
            source_timestamp=timestamp,
        )
        self._cache[(sensor_name, frame)] = artifact
        return artifact

    def calibration_ref(self, path: Path) -> ArtifactRef:
        resolved = path.resolve()
        if not resolved.is_file():
            raise PerceptionArtifactError(f"Calibration file does not exist: {resolved}")
        size = resolved.stat().st_size
        return ArtifactRef(
            name="calibration",
            uri=resolved.as_uri(),
            media_type="application/json",
            shape=(1,),
            dtype="json",
            byte_size=size,
            sha256=_sha256_file(resolved),
        )

    def _frame_dir(self, frame: int) -> Path:
        path = self.root / f"frame_{frame:08d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _purge(self, current_frame: int) -> None:
        oldest_allowed = current_frame - self._retention_frames
        for path in self.root.glob("frame_*"):
            try:
                frame = int(path.name.removeprefix("frame_"))
            except ValueError:
                continue
            if frame < oldest_allowed and path.is_dir():
                shutil.rmtree(path)
                stale_keys = [key for key in self._cache if key[1] == frame]
                for key in stale_keys:
                    self._cache.pop(key, None)


class PerceptionInputPublisher:
    """Artifact referanslarını runtime ``MessageEnvelope`` girdisine dönüştürür."""

    def __init__(self, clock: Clock, namespace: str, lifespan_s: float) -> None:
        if lifespan_s <= 0.0:
            raise ValueError("lifespan_s must be positive")
        self._lifespan_s = lifespan_s
        self._factory = MessageFactory[PerceptionInput](
            source="perception_input",
            clock=clock,
            coordinate_frame="CARLA_SENSOR_ARTIFACTS",
            namespace=namespace,
        )

    def publish(
        self,
        value: PerceptionInput,
        *,
        localization_message_id: str | None = None,
    ) -> MessageEnvelope[PerceptionInput]:
        parents = () if localization_message_id is None else (localization_message_id,)
        return self._factory.create(
            value,
            source_timestamp=value.timestamp,
            lifespan_s=self._lifespan_s,
            parents=parents,
        )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
