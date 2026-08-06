from __future__ import annotations

import pytest

from l4stack.runtime import HealthRegistry, LifecycleState, ManagedComponent, RuntimeSupervisor
from l4stack.runtime.lifecycle import LifecycleError


class RecordingComponent(ManagedComponent):
    def __init__(self, name: str, events: list[str], dependencies: tuple[str, ...] = ()) -> None:
        super().__init__(name, dependencies)
        self.events = events

    def on_configure(self) -> None:
        self.events.append(f"configure:{self.name}")

    def on_activate(self) -> None:
        self.events.append(f"activate:{self.name}")

    def on_deactivate(self) -> None:
        self.events.append(f"deactivate:{self.name}")

    def on_cleanup(self) -> None:
        self.events.append(f"cleanup:{self.name}")

    def on_shutdown(self) -> None:
        self.events.append(f"shutdown:{self.name}")


def test_supervisor_respects_dependencies_and_reverse_shutdown_order() -> None:
    events: list[str] = []
    supervisor = RuntimeSupervisor(HealthRegistry())
    sensors = RecordingComponent("sensors", events)
    localization = RecordingComponent("localization", events, ("sensors",))
    planning = RecordingComponent("planning", events, ("localization",))
    supervisor.register(planning)
    supervisor.register(localization)
    supervisor.register(sensors)

    supervisor.start_all()
    assert planning.state is LifecycleState.ACTIVE
    supervisor.stop_all()

    assert events[:6] == [
        "configure:sensors",
        "configure:localization",
        "configure:planning",
        "activate:sensors",
        "activate:localization",
        "activate:planning",
    ]
    assert events[-3:] == ["deactivate:sensors", "cleanup:sensors", "shutdown:sensors"]


def test_invalid_lifecycle_transition_is_rejected() -> None:
    component = RecordingComponent("component", [])
    with pytest.raises(LifecycleError):
        component.activate()


def test_supervisor_rejects_dependency_cycle() -> None:
    supervisor = RuntimeSupervisor(HealthRegistry())
    supervisor.register(RecordingComponent("a", [], ("b",)))
    supervisor.register(RecordingComponent("b", [], ("a",)))
    with pytest.raises(ValueError, match="cycle"):
        supervisor.start_all()


class FailingConfigureComponent(RecordingComponent):
    def on_configure(self) -> None:
        super().on_configure()
        raise RuntimeError("configure failed")


def test_supervisor_rolls_back_partial_startup() -> None:
    events: list[str] = []
    supervisor = RuntimeSupervisor(HealthRegistry())
    supervisor.register(RecordingComponent("a", events))
    failing = FailingConfigureComponent("b", events, ("a",))
    supervisor.register(failing)

    with pytest.raises(Exception, match="configure failed"):
        supervisor.start_all()

    assert failing.state is LifecycleState.FINALIZED
    assert "shutdown:a" in events
    assert "shutdown:b" in events
