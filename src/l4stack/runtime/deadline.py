from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

from l4stack.runtime.clock import Clock
from l4stack.runtime.contracts import ComponentContract
from l4stack.runtime.message import MessageEnvelope

T = TypeVar("T")


class DeadlineViolation(str, Enum):
    INPUT_STALE = "INPUT_STALE"
    INPUT_EXPIRED = "INPUT_EXPIRED"
    EXECUTION_BUDGET = "EXECUTION_BUDGET"
    OUTPUT_PERIOD = "OUTPUT_PERIOD"


@dataclass(frozen=True, slots=True)
class DeadlineEvent:
    component: str
    violation: DeadlineViolation
    observed_s: float
    limit_s: float
    timestamp: float
    message_id: str | None = None


@dataclass(frozen=True, slots=True)
class DeadlineStats:
    executions: int
    violations: int
    consecutive_violations: int
    last_execution_s: float | None
    max_execution_s: float
    last_output_timestamp: float | None


class DeadlineMonitor:
    """Freshness, execution budget ve output period ihlallerini merkezi toplar."""

    def __init__(
        self,
        clock: Clock,
        processing_clock: Clock,
        maximum_events: int = 10_000,
    ) -> None:
        if maximum_events <= 0:
            raise ValueError("maximum_events must be positive")
        self._clock = clock
        self._processing_clock = processing_clock
        self._contracts: dict[str, ComponentContract] = {}
        self._events: deque[DeadlineEvent] = deque(maxlen=maximum_events)
        self._stats: dict[str, dict[str, float | int | None]] = {}
        self._lock = threading.Lock()

    def register(self, contract: ComponentContract) -> None:
        with self._lock:
            existing = self._contracts.get(contract.name)
            if existing is not None:
                if existing == contract:
                    return
                raise ValueError(f"Deadline contract already registered: {contract.name}")
            self._contracts[contract.name] = contract
            self._stats[contract.name] = {
                "executions": 0,
                "violations": 0,
                "consecutive": 0,
                "last_execution": None,
                "max_execution": 0.0,
                "last_output": None,
            }

    def contract(self, component: str) -> ComponentContract:
        try:
            return self._contracts[component]
        except KeyError as exc:
            raise KeyError(f"Deadline contract is not registered: {component}") from exc

    def validate_input(
        self,
        component: str,
        envelope: MessageEnvelope[T],
    ) -> tuple[DeadlineEvent, ...]:
        contract = self.contract(component)
        now = self._clock.now()
        violations: list[DeadlineEvent] = []
        age = envelope.age(now)
        if not envelope.is_valid(now):
            violations.append(
                self._record(
                    component,
                    DeadlineViolation.INPUT_EXPIRED,
                    age,
                    envelope.valid_until - envelope.source_timestamp,
                    envelope.message_id,
                )
            )
        elif age > contract.max_input_age_s:
            violations.append(
                self._record(
                    component,
                    DeadlineViolation.INPUT_STALE,
                    age,
                    contract.max_input_age_s,
                    envelope.message_id,
                )
            )
        return tuple(violations)

    def start_execution(self) -> float:
        return self._processing_clock.now()

    def finish_execution(
        self,
        component: str,
        started_at: float,
        *,
        output_timestamp: float,
        message_id: str | None = None,
    ) -> tuple[DeadlineEvent, ...]:
        contract = self.contract(component)
        duration = max(0.0, self._processing_clock.now() - started_at)
        events: list[DeadlineEvent] = []
        with self._lock:
            stats = self._stats[component]
            last_output = stats["last_output"]
        if duration > contract.execution_budget_s:
            events.append(
                self._record(
                    component,
                    DeadlineViolation.EXECUTION_BUDGET,
                    duration,
                    contract.execution_budget_s,
                    message_id,
                )
            )
        if last_output is not None:
            period = float(output_timestamp) - float(last_output)
            if period > contract.expected_output_period_s:
                events.append(
                    self._record(
                        component,
                        DeadlineViolation.OUTPUT_PERIOD,
                        period,
                        contract.expected_output_period_s,
                        message_id,
                    )
                )
        with self._lock:
            stats = self._stats[component]
            stats["executions"] = int(stats["executions"]) + 1
            stats["last_execution"] = duration
            stats["max_execution"] = max(float(stats["max_execution"]), duration)
            stats["last_output"] = float(output_timestamp)
            if not events:
                stats["consecutive"] = 0
        return tuple(events)

    def events(self) -> tuple[DeadlineEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def stats(self, component: str) -> DeadlineStats:
        with self._lock:
            values = dict(self._stats[component])
        return DeadlineStats(
            executions=int(values["executions"]),
            violations=int(values["violations"]),
            consecutive_violations=int(values["consecutive"]),
            last_execution_s=(
                None if values["last_execution"] is None else float(values["last_execution"])
            ),
            max_execution_s=float(values["max_execution"]),
            last_output_timestamp=(
                None if values["last_output"] is None else float(values["last_output"])
            ),
        )

    def _record(
        self,
        component: str,
        violation: DeadlineViolation,
        observed: float,
        limit: float,
        message_id: str | None,
    ) -> DeadlineEvent:
        event = DeadlineEvent(
            component=component,
            violation=violation,
            observed_s=float(observed),
            limit_s=float(limit),
            timestamp=float(self._clock.now()),
            message_id=message_id,
        )
        with self._lock:
            self._events.append(event)
            stats = self._stats[component]
            stats["violations"] = int(stats["violations"]) + 1
            stats["consecutive"] = int(stats["consecutive"]) + 1
        return event
