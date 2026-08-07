from __future__ import annotations

import contextlib
import mmap
import os
from typing import Iterator
from urllib.parse import parse_qs, urlparse

from l4stack.perception.shared_memory_ring import SharedMemoryTransportError
from l4stack.perception.types import ArtifactRef

@contextlib.contextmanager
def open_shared_artifact(artifact: ArtifactRef) -> Iterator[memoryview]:
    """Linux worker tarafında ``shm://`` artifact'ını tracker olmadan açar.

    ``multiprocessing.shared_memory`` consumer process'te resource tracker kaydı
    oluşturabildiği için worker çıkışında producer segmentini yanlışlıkla unlink etme
    riski taşır. Linux hedefinde segment doğrudan ``/dev/shm`` üzerinden read-only
    mmap edilir; böylece sahiplik yalnız producer process'te kalır.
    """

    parsed = parse_shared_memory_uri(artifact.uri)
    path = f"/dev/shm/{parsed['segment']}"
    descriptor = os.open(path, os.O_RDONLY)
    mapping: mmap.mmap | None = None
    view: memoryview | None = None
    try:
        segment_size = os.fstat(descriptor).st_size
        offset = int(parsed["offset"])
        slot = int(parsed["slot"])
        capacity = int(parsed["capacity"])
        if artifact.byte_size > capacity or offset != slot * capacity:
            raise SharedMemoryTransportError(
                "Artifact metadata does not match shared-memory slot geometry"
            )
        end = offset + artifact.byte_size
        if end > segment_size:
            raise SharedMemoryTransportError(
                "Artifact range exceeds shared-memory segment"
            )
        mapping = mmap.mmap(descriptor, segment_size, access=mmap.ACCESS_READ)
        view = memoryview(mapping)[offset:end].toreadonly()
        yield view
    finally:
        if view is not None:
            view.release()
        if mapping is not None:
            mapping.close()
        os.close(descriptor)


def parse_shared_memory_uri(uri: str) -> dict[str, int | str]:
    parsed = urlparse(uri)
    if parsed.scheme != "shm" or not parsed.netloc:
        raise SharedMemoryTransportError(f"Invalid shared-memory URI: {uri}")
    query = parse_qs(parsed.query)
    required = ("offset", "slot", "generation", "capacity")
    missing = [key for key in required if key not in query]
    if missing:
        raise SharedMemoryTransportError(f"Shared-memory URI is missing: {missing}")
    values: dict[str, int | str] = {"segment": parsed.netloc}
    for key in required:
        value = int(query[key][0])
        if value < 0:
            raise SharedMemoryTransportError(f"Negative shared-memory URI value: {key}")
        values[key] = value
    return values


