from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Generic, TypeVar

from l4stack.runtime.clock import Clock

T = TypeVar("T")


class MessageExpiredError(RuntimeError):
    """Geçerlilik süresi dolmuş bir mesaj kullanılmak istendiğinde üretilir."""


@dataclass(frozen=True, slots=True)
class MessageEnvelope(Generic[T]):
    """Katmanlar arasında taşınan immutable mesaj zarfı.

    ``source_timestamp`` ölçümün temsil ettiği zamanı, ``publish_timestamp`` ise
    çıktının runtime tarafından yayımlandığı zamanı belirtir. ``parents`` alanı veri
    soy ağacını kurar; örneğin bir lokalizasyon çıktısı, kullandığı sensör bundle
    mesajının kimliğini parent olarak taşır.
    """

    message_id: str
    source: str
    sequence_id: int
    source_timestamp: float
    publish_timestamp: float
    valid_until: float
    coordinate_frame: str
    parents: tuple[str, ...]
    payload: T

    def age(self, now: float) -> float:
        return max(0.0, float(now) - self.source_timestamp)

    def processing_latency(self) -> float:
        return max(0.0, self.publish_timestamp - self.source_timestamp)

    def is_valid(self, now: float) -> bool:
        return float(now) <= self.valid_until

    def require_valid(self, now: float) -> None:
        if not self.is_valid(now):
            raise MessageExpiredError(
                f"Message expired: id={self.message_id} now={now:.6f} "
                f"valid_until={self.valid_until:.6f}"
            )

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "source": self.source,
            "sequence_id": self.sequence_id,
            "source_timestamp": self.source_timestamp,
            "publish_timestamp": self.publish_timestamp,
            "valid_until": self.valid_until,
            "coordinate_frame": self.coordinate_frame,
            "parents": list(self.parents),
            "processing_latency_s": self.processing_latency(),
        }

    def as_dict(self) -> dict[str, Any]:
        data = self.metadata_dict()
        payload = self.payload
        if hasattr(payload, "as_dict"):
            data["payload"] = payload.as_dict()
        elif is_dataclass(payload):
            data["payload"] = asdict(payload)
        else:
            data["payload"] = thaw_value(payload)
        return data


class MessageFactory(Generic[T]):
    """Tek bir yayıncı için monoton sequence ve deterministik message id üretir."""

    def __init__(
        self,
        source: str,
        clock: Clock,
        *,
        coordinate_frame: str = "UNSPECIFIED",
        namespace: str = "runtime",
    ) -> None:
        if not source:
            raise ValueError("Message source cannot be empty")
        self._source = source
        self._clock = clock
        self._coordinate_frame = coordinate_frame
        self._namespace = namespace
        self._sequence = 0
        self._lock = threading.Lock()

    @property
    def last_sequence(self) -> int:
        with self._lock:
            return self._sequence

    def create(
        self,
        payload: T,
        *,
        source_timestamp: float,
        lifespan_s: float,
        parents: tuple[str, ...] = (),
        coordinate_frame: str | None = None,
        freeze_payload: bool = False,
    ) -> MessageEnvelope[T]:
        if lifespan_s <= 0.0:
            raise ValueError("Message lifespan must be positive")
        # Sequence ve publish zamanı aynı kilit altında alınır. Böylece aynı factory'yi
        # kullanan eşzamanlı publisher çağrılarında sıra ile timestamp tutarlı kalır.
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
            publish_timestamp = float(self._clock.now())
            source_timestamp = float(source_timestamp)
            frozen_payload = freeze_value(payload) if freeze_payload else payload
            message_id = f"{self._namespace}/{self._source}/{sequence}"
            return MessageEnvelope(
                message_id=message_id,
                source=self._source,
                sequence_id=sequence,
                source_timestamp=source_timestamp,
                publish_timestamp=publish_timestamp,
                valid_until=source_timestamp + float(lifespan_s),
                coordinate_frame=coordinate_frame or self._coordinate_frame,
                parents=tuple(parents),
                payload=frozen_payload,
            )


def freeze_value(value: Any) -> Any:
    """Standart mutable konteynerleri recursive immutable karşılıklarına çevirir.

    CARLA ölçüm nesneleri gibi dış kütüphane nesneleri dönüştürülmez. Fonksiyon özellikle
    dict/list tabanlı dünya modeli ve planlama mesajlarının yayın sonrası değiştirilmesini
    engellemek için kullanılacaktır.
    """

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): freeze_value(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(freeze_value(item) for item in value)
    if isinstance(value, Enum):
        return value
    return value


def thaw_value(value: Any) -> Any:
    """Immutable runtime konteynerlerini JSON uyumlu standart yapılara çevirir."""

    if isinstance(value, Mapping):
        return {str(key): thaw_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [thaw_value(item) for item in value]
    if isinstance(value, frozenset | set):
        return sorted((thaw_value(item) for item in value), key=repr)
    if isinstance(value, Enum):
        return value.value
    return value
