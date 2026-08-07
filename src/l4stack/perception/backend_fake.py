from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from l4stack.perception.backend_contracts import BackendHealth, BackendUnavailable
from l4stack.perception.protocol import InferenceRequest


class FakeBackend:
    """Unit ve smoke testlerde GPU modeli yerine kullanılan deterministik backend."""

    def __init__(self, responses: Sequence[Mapping[str, Any]] | None = None) -> None:
        self._responses = list([{}] if responses is None else responses)
        self._started = False
        self.calls: list[InferenceRequest] = []

    def start(self) -> None:
        self._started = True

    def infer(self, request: InferenceRequest, timeout_s: float) -> Mapping[str, Any]:
        if not self._started:
            raise BackendUnavailable("Fake backend is not started")
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        self.calls.append(request)
        if not self._responses:
            raise BackendUnavailable("Fake backend has no response")
        return self._responses.pop(0)

    def health(self) -> BackendHealth:
        return BackendHealth(self._started, "ready" if self._started else "not started")

    def stop(self) -> None:
        self._started = False
