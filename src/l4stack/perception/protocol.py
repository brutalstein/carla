from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from l4stack.perception.types import PerceptionInput


PROTOCOL_VERSION = 2


class BackendProtocolError(RuntimeError):
    """Model süreci ortak JSONL protokolünü ihlal ettiğinde üretilir."""


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    request_id: str
    model_name: str
    source_timestamp: float
    payload: PerceptionInput

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "type": "infer",
            "request_id": self.request_id,
            "model_name": self.model_name,
            "source_timestamp": self.source_timestamp,
            "payload": self.payload.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class InferenceResponse:
    request_id: str
    ok: bool
    payload: Mapping[str, Any]
    error: str = ""

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InferenceResponse":
        if int(value.get("protocol_version", -1)) != PROTOCOL_VERSION:
            raise BackendProtocolError("Unsupported perception backend protocol version")
        if value.get("type") != "result":
            raise BackendProtocolError("Backend response type must be 'result'")
        request_id = str(value.get("request_id", ""))
        if not request_id:
            raise BackendProtocolError("Backend response request_id is missing")
        ok = value.get("ok")
        if not isinstance(ok, bool):
            raise BackendProtocolError("Backend response ok must be a boolean")
        payload = value.get("payload", {})
        if not isinstance(payload, Mapping):
            raise BackendProtocolError("Backend response payload must be a mapping")
        return cls(
            request_id=request_id,
            ok=ok,
            payload=dict(payload),
            error=str(value.get("error", "")),
        )
