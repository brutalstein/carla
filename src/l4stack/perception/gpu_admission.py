from __future__ import annotations

import math
import statistics
import threading
import uuid
from collections import deque
from dataclasses import dataclass
from enum import Enum


class GpuExecutionClass(str, Enum):
    """GPU baskısı altında hangi modelin önce korunacağını belirtir."""

    CRITICAL = "CRITICAL"
    REQUIRED = "REQUIRED"
    OPPORTUNISTIC = "OPPORTUNISTIC"


@dataclass(frozen=True, slots=True)
class GpuModelPolicy:
    name: str
    priority: int
    execution_class: GpuExecutionClass
    estimated_gpu_ms: float
    deadline_ms: float
    max_inflight: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("GPU model policy name cannot be empty")
        if self.priority < 0:
            raise ValueError("GPU model priority cannot be negative")
        if self.estimated_gpu_ms <= 0.0 or self.deadline_ms <= 0.0:
            raise ValueError("GPU model timing values must be positive")
        if self.max_inflight <= 0:
            raise ValueError("GPU model max_inflight must be positive")


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    selected: tuple[str, ...]
    rejected_budget: tuple[str, ...]
    rejected_inflight: tuple[str, ...]
    predicted_gpu_ms: float


@dataclass(frozen=True, slots=True)
class GpuReservation:
    model_name: str
    reservation_id: str
    started_at: float


@dataclass(frozen=True, slots=True)
class GpuModelRuntimeStats:
    samples: int
    inflight: int
    estimated_ms: float
    p50_ms: float | None
    p95_ms: float | None
    maximum_ms: float | None


class GpuAdmissionController:
    """Tek GPU için deadline-aware ve ölçüm geri beslemeli admission controller.

    Bu sınıf GPU kernel'lerini kendisi çalıştırmaz. Model süreçlerine iş gönderilmeden
    önce hangi modellerin bu release penceresine kabul edileceğine karar verir. Böylece
    beş model aynı anda RTX 5090'a yığılmaz; kritik işler korunur, düşük öncelikli işler
    kontrollü atlanır ve backlog oluşmaz.
    """

    def __init__(
        self,
        policies: tuple[GpuModelPolicy, ...],
        *,
        frame_budget_ms: float,
        max_concurrent: int,
        history_size: int = 128,
        safety_margin: float = 1.15,
    ) -> None:
        if frame_budget_ms <= 0.0:
            raise ValueError("frame_budget_ms must be positive")
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be positive")
        if history_size < 8:
            raise ValueError("history_size must be at least 8")
        if safety_margin < 1.0:
            raise ValueError("safety_margin must be at least 1.0")
        by_name = {policy.name: policy for policy in policies}
        if len(by_name) != len(policies):
            raise ValueError("GPU model policy names must be unique")
        self._policies = by_name
        self._frame_budget_ms = float(frame_budget_ms)
        self._max_concurrent = int(max_concurrent)
        self._safety_margin = float(safety_margin)
        self._history = {name: deque(maxlen=history_size) for name in by_name}
        self._inflight = {name: 0 for name in by_name}
        self._reservations: dict[str, dict[str, float]] = {
            name: {} for name in by_name
        }
        self._lock = threading.RLock()

    def select(self, candidates: tuple[str, ...]) -> AdmissionDecision:
        with self._lock:
            unknown = set(candidates) - self._policies.keys()
            if unknown:
                raise KeyError(f"Unknown GPU model policies: {sorted(unknown)}")
            available_slots = self._max_concurrent - sum(self._inflight.values())
            if available_slots <= 0:
                return AdmissionDecision((), (), tuple(candidates), 0.0)

            rejected_inflight: list[str] = []
            eligible: list[GpuModelPolicy] = []
            for name in candidates:
                policy = self._policies[name]
                if self._inflight[name] >= policy.max_inflight:
                    rejected_inflight.append(name)
                else:
                    eligible.append(policy)

            eligible.sort(
                key=lambda policy: (
                    _class_rank(policy.execution_class),
                    policy.priority,
                    policy.deadline_ms,
                    self._prediction_ms(policy.name),
                    policy.name,
                )
            )
            selected: list[str] = []
            rejected_budget: list[str] = []
            predicted_total = 0.0
            for policy in eligible:
                predicted = self._prediction_ms(policy.name) * self._safety_margin
                critical = policy.execution_class is GpuExecutionClass.CRITICAL
                fits_slot = len(selected) < available_slots
                fits_budget = predicted_total + predicted <= self._frame_budget_ms
                if fits_slot and (fits_budget or critical):
                    selected.append(policy.name)
                    predicted_total += predicted
                else:
                    rejected_budget.append(policy.name)
            return AdmissionDecision(
                selected=tuple(selected),
                rejected_budget=tuple(rejected_budget),
                rejected_inflight=tuple(rejected_inflight),
                predicted_gpu_ms=predicted_total,
            )

    def reserve(self, model_name: str, started_at: float) -> GpuReservation:
        if not math.isfinite(started_at):
            raise ValueError("started_at must be finite")
        with self._lock:
            policy = self._policies[model_name]
            if self._inflight[model_name] >= policy.max_inflight:
                raise RuntimeError(f"GPU max_inflight exceeded: {model_name}")
            if sum(self._inflight.values()) >= self._max_concurrent:
                raise RuntimeError("Global GPU max_concurrent exceeded")
            reservation_id = uuid.uuid4().hex
            self._inflight[model_name] += 1
            self._reservations[model_name][reservation_id] = float(started_at)
            return GpuReservation(model_name, reservation_id, float(started_at))

    def finish(self, reservation: GpuReservation, finished_at: float) -> float:
        if not math.isfinite(finished_at):
            raise ValueError("finished_at must be finite")
        with self._lock:
            starts = self._reservations[reservation.model_name]
            try:
                started_at = starts.pop(reservation.reservation_id)
            except KeyError as exc:
                raise RuntimeError(
                    f"GPU reservation missing: {reservation.model_name}"
                ) from exc
            if self._inflight[reservation.model_name] <= 0:
                raise RuntimeError(f"GPU inflight underflow: {reservation.model_name}")
            duration_ms = max(0.0, float(finished_at) - started_at) * 1000.0
            self._inflight[reservation.model_name] -= 1
            self._history[reservation.model_name].append(duration_ms)
            return duration_ms

    def cancel(self, reservation: GpuReservation) -> None:
        with self._lock:
            starts = self._reservations[reservation.model_name]
            if starts.pop(reservation.reservation_id, None) is None:
                return
            if self._inflight[reservation.model_name] > 0:
                self._inflight[reservation.model_name] -= 1

    def stats(self) -> dict[str, GpuModelRuntimeStats]:
        with self._lock:
            result: dict[str, GpuModelRuntimeStats] = {}
            for name, history in self._history.items():
                values = tuple(history)
                result[name] = GpuModelRuntimeStats(
                    samples=len(values),
                    inflight=self._inflight[name],
                    estimated_ms=self._prediction_ms(name),
                    p50_ms=(None if not values else statistics.median(values)),
                    p95_ms=(None if not values else _percentile(values, 0.95)),
                    maximum_ms=(None if not values else max(values)),
                )
            return result

    def _prediction_ms(self, model_name: str) -> float:
        history = self._history[model_name]
        if len(history) < 4:
            return self._policies[model_name].estimated_gpu_ms
        return max(
            self._policies[model_name].estimated_gpu_ms * 0.50,
            _percentile(tuple(history), 0.95),
        )


def _class_rank(value: GpuExecutionClass) -> int:
    if value is GpuExecutionClass.CRITICAL:
        return 0
    if value is GpuExecutionClass.REQUIRED:
        return 1
    return 2


def _percentile(values: tuple[float, ...], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = quantile * (len(ordered) - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
