from __future__ import annotations

import pytest

from l4stack.runtime import ManualClock, MessageFactory
from l4stack.runtime.message import MessageExpiredError


def test_message_factory_produces_deterministic_sequence_and_validity() -> None:
    clock = ManualClock(10.0)
    factory = MessageFactory[dict]("perception", clock, namespace="test")

    first = factory.create(
        {"objects": [1, 2]},
        source_timestamp=9.95,
        lifespan_s=0.20,
        freeze_payload=True,
    )
    second = factory.create({}, source_timestamp=10.0, lifespan_s=0.20)

    assert first.message_id == "test/perception/1"
    assert second.message_id == "test/perception/2"
    assert first.sequence_id == 1
    assert first.age(10.05) == pytest.approx(0.10)
    assert first.is_valid(10.14)
    with pytest.raises(TypeError):
        first.payload["objects"] = ()

    clock.set(10.16)
    with pytest.raises(MessageExpiredError):
        first.require_valid(clock.now())
