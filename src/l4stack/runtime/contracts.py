from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any

from l4stack.runtime.channel import OverflowPolicy


class ExecutionCriticality(str, Enum):
    """Deadline kaçırıldığında uygulanacak sistem yaklaşımını sınıflandırır."""

    SAFETY_CRITICAL = "SAFETY_CRITICAL"
    FIRM_REALTIME = "FIRM_REALTIME"
    SOFT_REALTIME = "SOFT_REALTIME"


class TaskPriority(IntEnum):
    """Küçük sayının daha yüksek öncelik olduğu runtime priority sınıfları."""

    SAFETY = 0
    CONTROL = 10
    LOCALIZATION = 20
    WORLD_MODEL = 30
    PLANNING = 40
    PERCEPTION = 50
    BACKGROUND = 100


@dataclass(frozen=True, slots=True)
class ExecutorProfile:
    """Bir executor havuzunun thread ve bounded queue yapılandırması."""

    name: str
    workers: int
    queue_capacity: int
    priority: int

    @classmethod
    def from_mapping(cls, name: str, value: dict[str, Any]) -> "ExecutorProfile":
        profile = cls(
            name=name,
            workers=int(value["workers"]),
            queue_capacity=int(value["queue_capacity"]),
            priority=int(value["priority"]),
        )
        if profile.workers <= 0 or profile.queue_capacity <= 0:
            raise ValueError(f"Executor profile must be positive: {name}")
        if profile.priority < 0:
            raise ValueError(f"Executor priority must be non-negative: {name}")
        return profile


@dataclass(frozen=True, slots=True)
class ComponentContract:
    """Bir bileşenin zamanlama, freshness ve channel sözleşmesi."""

    name: str
    criticality: ExecutionCriticality
    priority: int
    max_input_age_s: float
    execution_budget_s: float
    expected_output_period_s: float
    output_lifespan_s: float
    channel_capacity: int
    overflow_policy: OverflowPolicy
    drop_expired_inputs: bool = True

    @classmethod
    def from_mapping(cls, name: str, value: dict[str, Any]) -> "ComponentContract":
        contract = cls(
            name=name,
            criticality=ExecutionCriticality(value["criticality"]),
            priority=int(value["priority"]),
            max_input_age_s=float(value["max_input_age_s"]),
            execution_budget_s=float(value["execution_budget_s"]),
            expected_output_period_s=float(value["expected_output_period_s"]),
            output_lifespan_s=float(value["output_lifespan_s"]),
            channel_capacity=int(value.get("channel_capacity", 1)),
            overflow_policy=OverflowPolicy(value.get("overflow_policy", "LATEST_ONLY")),
            drop_expired_inputs=bool(value.get("drop_expired_inputs", True)),
        )
        contract.validate()
        return contract

    def validate(self) -> None:
        positive = {
            "max_input_age_s": self.max_input_age_s,
            "execution_budget_s": self.execution_budget_s,
            "expected_output_period_s": self.expected_output_period_s,
            "output_lifespan_s": self.output_lifespan_s,
        }
        for key, value in positive.items():
            if value <= 0.0:
                raise ValueError(f"{self.name}.{key} must be positive")
        if self.channel_capacity <= 0:
            raise ValueError(f"{self.name}.channel_capacity must be positive")
        if self.priority < 0:
            raise ValueError(f"{self.name}.priority must be non-negative")
