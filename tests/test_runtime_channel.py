from __future__ import annotations

import threading
import time

import pytest

from l4stack.runtime import BoundedChannel, ChannelClosed, OverflowPolicy


def test_latest_only_channel_keeps_only_newest_message() -> None:
    channel = BoundedChannel[int]("camera", 1, OverflowPolicy.LATEST_ONLY)
    channel.publish(1)
    channel.publish(2)
    channel.publish(3)

    assert channel.receive() == 3
    stats = channel.stats()
    assert stats.dropped == 2
    assert stats.published == 3


def test_blocking_channel_waits_without_busy_spin_and_wakes_on_publish() -> None:
    channel = BoundedChannel[str]("events", 1, OverflowPolicy.BLOCK)
    result: list[str] = []
    started = threading.Event()

    def consumer() -> None:
        started.set()
        result.append(channel.receive(timeout=1.0))

    thread = threading.Thread(target=consumer)
    thread.start()
    assert started.wait(0.2)
    time.sleep(0.02)
    channel.publish("ready")
    thread.join(timeout=1.0)

    assert result == ["ready"]
    channel.close()
    with pytest.raises(ChannelClosed):
        channel.receive(timeout=0.01)


def test_drop_newest_channel_reports_rejection() -> None:
    channel = BoundedChannel[int]("audit", 1, OverflowPolicy.DROP_NEWEST)
    assert channel.publish(1)
    assert not channel.publish(2)
    assert channel.receive() == 1
    assert channel.stats().rejected == 1
