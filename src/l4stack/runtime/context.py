from __future__ import annotations

from dataclasses import dataclass

from l4stack.runtime.clock import Clock
from l4stack.runtime.deadline import DeadlineMonitor
from l4stack.runtime.health import HealthRegistry
from l4stack.runtime.lineage import LineageStore


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    """Bileşenlere dependency injection ile verilen ortak runtime servisleri."""

    clock: Clock
    processing_clock: Clock
    deadlines: DeadlineMonitor
    health: HealthRegistry
    lineage: LineageStore
