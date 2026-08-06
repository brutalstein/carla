from __future__ import annotations

import threading

from l4stack.runtime import PriorityExecutor


def test_priority_executor_runs_queued_high_priority_task_first() -> None:
    executor = PriorityExecutor("test", workers=1, queue_capacity=8)
    blocker_started = threading.Event()
    release_blocker = threading.Event()
    order: list[str] = []

    def blocker() -> None:
        blocker_started.set()
        release_blocker.wait(1.0)
        order.append("blocker")

    first = executor.submit("blocker", blocker, priority=0)
    assert blocker_started.wait(0.5)
    low = executor.submit("low", lambda: order.append("low"), priority=100)
    high = executor.submit("high", lambda: order.append("high"), priority=10)
    release_blocker.set()

    first.result(timeout=1.0)
    high.result(timeout=1.0)
    low.result(timeout=1.0)
    executor.shutdown()

    assert order == ["blocker", "high", "low"]
