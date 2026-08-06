from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any


class RuntimeHealth(str, Enum):
    NOMINAL = "NOMINAL"
    DEGRADED = "DEGRADED"
    STALE = "STALE"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"


_SEVERITY = {
    RuntimeHealth.NOMINAL: 0,
    RuntimeHealth.DEGRADED: 1,
    RuntimeHealth.STALE: 2,
    RuntimeHealth.UNAVAILABLE: 3,
    RuntimeHealth.FAILED: 4,
}


@dataclass(frozen=True, slots=True)
class HealthReport:
    component: str
    state: RuntimeHealth
    timestamp: float
    reason: str = ""
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Dışarı verilen metrics sözlüğünün sonradan değiştirilmesini engelliyoruz.
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "state": self.state.value,
            "timestamp": self.timestamp,
            "reason": self.reason,
            "metrics": dict(self.metrics),
        }


class HealthRegistry:
    """Bütün runtime bileşenlerinin son sağlık raporlarını atomik olarak saklar."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reports: dict[str, HealthReport] = {}

    def report(self, report: HealthReport) -> None:
        with self._lock:
            previous = self._reports.get(report.component)
            if previous is not None and report.timestamp < previous.timestamp:
                raise ValueError(f"Health timestamp moved backwards: {report.component}")
            self._reports[report.component] = report

    def get(self, component: str) -> HealthReport | None:
        with self._lock:
            return self._reports.get(component)

    def snapshot(self) -> dict[str, HealthReport]:
        with self._lock:
            return dict(self._reports)

    def aggregate(self) -> RuntimeHealth:
        with self._lock:
            if not self._reports:
                return RuntimeHealth.UNAVAILABLE
            return max((report.state for report in self._reports.values()), key=_SEVERITY.get)

    def mark_stale(self, now: float, maximum_ages: Mapping[str, float]) -> tuple[HealthReport, ...]:
        stale_reports: list[HealthReport] = []
        with self._lock:
            for component, max_age in maximum_ages.items():
                report = self._reports.get(component)
                if report is None:
                    continue
                age = float(now) - report.timestamp
                if age > float(max_age) and report.state is not RuntimeHealth.FAILED:
                    stale = HealthReport(
                        component=component,
                        state=RuntimeHealth.STALE,
                        timestamp=float(now),
                        reason=f"health report age {age:.3f}s exceeded {max_age:.3f}s",
                        metrics={**dict(report.metrics), "health_age_s": age},
                    )
                    self._reports[component] = stale
                    stale_reports.append(stale)
        return tuple(stale_reports)
