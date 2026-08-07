from __future__ import annotations

import threading
from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass

from l4stack.perception.component import PerceptionModelComponent
from l4stack.perception.types import ModelOutput, PerceptionInput, PerceptionSnapshot
from l4stack.runtime.executor import PriorityExecutor
from l4stack.runtime.lifecycle import LifecycleState
from l4stack.runtime.message import MessageEnvelope


@dataclass(frozen=True, slots=True)
class ModelRoute:
    component: PerceptionModelComponent
    executor: PriorityExecutor
    target_rate_hz: float

    def __post_init__(self) -> None:
        if self.target_rate_hz <= 0.0:
            raise ValueError("target_rate_hz must be positive")


@dataclass(frozen=True, slots=True)
class PipelineMetrics:
    submitted: Mapping[str, int]
    skipped_by_rate: Mapping[str, int]
    rejected_by_backpressure: Mapping[str, int]
    completed: Mapping[str, int]
    failed: Mapping[str, int]


class PerceptionPipeline:
    """Model component'lerini paralel ve bağımsız yöneten fan-out katmanı."""

    def __init__(self, routes: Mapping[str, ModelRoute], snapshot_max_age_s: float) -> None:
        if snapshot_max_age_s <= 0.0:
            raise ValueError("snapshot_max_age_s must be positive")
        self._routes = dict(routes)
        self._snapshot_max_age_s = snapshot_max_age_s
        self._last_submit_timestamp: dict[str, float] = {}
        self._latest: dict[str, MessageEnvelope[ModelOutput]] = {}
        self._lock = threading.RLock()
        self._submitted = {name: 0 for name in self._routes}
        self._skipped = {name: 0 for name in self._routes}
        self._backpressure = {name: 0 for name in self._routes}
        self._completed = {name: 0 for name in self._routes}
        self._failed = {name: 0 for name in self._routes}
        self._enabled = {name: True for name in self._routes}

    def due_models(self, source_timestamp: float) -> tuple[str, ...]:
        """Source-time anında yeni girdi kabul etmeye hazır route'ları döndürür."""

        with self._lock:
            due: list[str] = []
            for name, route in self._routes.items():
                if not self._enabled.get(name, False):
                    continue
                previous = self._last_submit_timestamp.get(name)
                period_s = 1.0 / route.target_rate_hz
                if previous is None or source_timestamp - previous + 1e-9 >= period_s:
                    due.append(name)
            return tuple(due)

    def submit(
        self,
        input_message: MessageEnvelope[PerceptionInput],
        *,
        model_names: tuple[str, ...] | None = None,
    ) -> Mapping[str, Future[MessageEnvelope[ModelOutput]]]:
        futures: dict[str, Future[MessageEnvelope[ModelOutput]]] = {}
        with self._lock:
            selected = set(self._routes) if model_names is None else set(model_names)
            unknown = selected - self._routes.keys()
            if unknown:
                raise KeyError(f"Unknown perception routes: {sorted(unknown)}")
            routes = tuple(
                (name, route) for name, route in self._routes.items() if name in selected
            )

        for name, route in routes:
            with self._lock:
                if not self._enabled.get(name, False):
                    continue
                period_s = 1.0 / route.target_rate_hz
                previous = self._last_submit_timestamp.get(name)
                if (
                    previous is not None
                    and input_message.source_timestamp - previous + 1e-9 < period_s
                ):
                    self._skipped[name] += 1
                    continue
            try:
                future = route.executor.submit(
                    name=f"{name}:{input_message.sequence_id}",
                    callback=route.component.process,
                    input_message=input_message,
                    priority=route.component.priority,
                    deadline_at=input_message.valid_until,
                    drop_if_expired=True,
                    timeout=0.0,
                )
            except TimeoutError:
                with self._lock:
                    self._backpressure[name] += 1
                continue
            with self._lock:
                self._last_submit_timestamp[name] = input_message.source_timestamp
                self._submitted[name] += 1
            future.add_done_callback(lambda item, model=name: self._record_result(model, item))
            futures[name] = future
        return futures

    def set_enabled(self, model_name: str, enabled: bool) -> None:
        with self._lock:
            if model_name not in self._routes:
                raise KeyError(f"Unknown perception route: {model_name}")
            self._enabled[model_name] = bool(enabled)

    def snapshot(self, now: float) -> PerceptionSnapshot:
        with self._lock:
            valid: list[ModelOutput] = []
            for name, envelope in list(self._latest.items()):
                age = max(0.0, now - envelope.source_timestamp)
                if envelope.is_valid(now) and age <= self._snapshot_max_age_s:
                    valid.append(envelope.payload)
                else:
                    self._latest.pop(name, None)
            valid.sort(key=lambda item: item.model_name)
            return PerceptionSnapshot(generated_at=now, outputs=tuple(valid))

    def metrics(self) -> PipelineMetrics:
        with self._lock:
            return PipelineMetrics(
                submitted=dict(self._submitted),
                skipped_by_rate=dict(self._skipped),
                rejected_by_backpressure=dict(self._backpressure),
                completed=dict(self._completed),
                failed=dict(self._failed),
            )

    def _record_result(
        self,
        model_name: str,
        future: Future[MessageEnvelope[ModelOutput]],
    ) -> None:
        with self._lock:
            if future.cancelled() or future.exception() is not None:
                self._failed[model_name] += 1
                component = self._routes[model_name].component
                if component.state in {LifecycleState.ERROR, LifecycleState.FINALIZED}:
                    self._enabled[model_name] = False
                return
            envelope = future.result()
            self._latest[model_name] = envelope
            self._completed[model_name] += 1
