from __future__ import annotations

from l4stack.perception.shared_memory import SharedMemoryArtifactStore
from l4stack.perception.types import PerceptionInput
from l4stack.runtime.clock import Clock
from l4stack.runtime.message import MessageEnvelope, MessageFactory


class PerceptionInputPublisher:
    """Sensör referanslarını zaman damgalı runtime mesajına dönüştürür."""

    def __init__(self, clock: Clock, namespace: str, lifespan_s: float) -> None:
        if lifespan_s <= 0.0:
            raise ValueError("lifespan_s must be positive")
        self._lifespan_s = float(lifespan_s)
        self._factory = MessageFactory[PerceptionInput](
            source="perception_input",
            clock=clock,
            coordinate_frame="CARLA_SENSOR_SHARED_MEMORY",
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


__all__ = ["PerceptionInputPublisher", "SharedMemoryArtifactStore"]
