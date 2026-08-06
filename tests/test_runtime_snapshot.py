from __future__ import annotations

import threading

from l4stack.runtime import AtomicSnapshotStore


def test_snapshot_versions_are_atomic_and_waitable() -> None:
    store = AtomicSnapshotStore[dict]("world_model")
    first = store.publish({"id": 1}, published_at=1.0)
    result = []

    def reader() -> None:
        result.append(store.wait_for_newer(first.version, timeout=1.0))

    thread = threading.Thread(target=reader)
    thread.start()
    second = store.publish({"id": 2}, published_at=1.1)
    thread.join(timeout=1.0)

    assert first.version == 1
    assert second.version == 2
    assert result[0].value == {"id": 2}
    assert store.require() is second
