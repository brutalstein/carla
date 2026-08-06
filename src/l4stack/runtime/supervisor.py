from __future__ import annotations

from dataclasses import dataclass

from l4stack.runtime.health import HealthRegistry, HealthReport, RuntimeHealth
from l4stack.runtime.lifecycle import LifecycleState, ManagedComponent


@dataclass(frozen=True, slots=True)
class SupervisorSnapshot:
    component_states: dict[str, LifecycleState]
    health: dict[str, HealthReport]


class RuntimeSupervisor:
    """Bağımlılık sırasına göre lifecycle başlatma ve ters sırada kapatma yöneticisi."""

    def __init__(self, health_registry: HealthRegistry) -> None:
        self._health = health_registry
        self._components: dict[str, ManagedComponent] = {}
        self._activation_order: tuple[str, ...] = ()

    def register(self, component: ManagedComponent) -> None:
        if component.name in self._components:
            raise ValueError(f"Component already registered: {component.name}")
        self._components[component.name] = component

    def start_all(self) -> None:
        order = self._topological_order()
        configured: list[ManagedComponent] = []
        activated: list[ManagedComponent] = []
        try:
            for name in order:
                component = self._components[name]
                component.configure()
                configured.append(component)
            for name in order:
                component = self._components[name]
                component.activate()
                activated.append(component)
        except Exception:
            # Yarım başlatılmış sistem bırakmamak için deterministik rollback uygulanır.
            for component in reversed(activated):
                if component.state is LifecycleState.ACTIVE:
                    try:
                        component.deactivate()
                    except Exception:
                        pass
            for component in reversed(configured):
                if component.state is LifecycleState.INACTIVE:
                    try:
                        component.cleanup()
                    except Exception:
                        pass
            # Hata veren component dahil tüm kaynaklar finalized duruma taşınmaya çalışılır.
            for name in reversed(order):
                component = self._components[name]
                if component.state is not LifecycleState.FINALIZED:
                    try:
                        component.shutdown()
                    except Exception:
                        pass
            self._activation_order = ()
            raise
        self._activation_order = order

    def stop_all(self) -> None:
        order = self._activation_order or self._topological_order()
        errors: list[str] = []
        for name in reversed(order):
            component = self._components[name]
            try:
                if component.state is LifecycleState.ACTIVE:
                    component.deactivate()
                if component.state is LifecycleState.INACTIVE:
                    component.cleanup()
            except Exception as exc:
                errors.append(f"{name}: {exc}")
            finally:
                if component.state is not LifecycleState.FINALIZED:
                    try:
                        component.shutdown()
                    except Exception as exc:
                        errors.append(f"{name} shutdown: {exc}")
        self._activation_order = ()
        if errors:
            raise RuntimeError("Runtime shutdown errors: " + "; ".join(errors))

    def fail_component(self, name: str, reason: str, timestamp: float) -> None:
        component = self._components[name]
        component.fail(reason)
        self._health.report(
            HealthReport(
                component=name,
                state=RuntimeHealth.FAILED,
                timestamp=timestamp,
                reason=reason,
            )
        )

    def snapshot(self) -> SupervisorSnapshot:
        return SupervisorSnapshot(
            component_states={
                name: component.state for name, component in self._components.items()
            },
            health=self._health.snapshot(),
        )

    def _topological_order(self) -> tuple[str, ...]:
        missing = {
            dependency
            for component in self._components.values()
            for dependency in component.dependencies
            if dependency not in self._components
        }
        if missing:
            raise ValueError(f"Unregistered component dependencies: {sorted(missing)}")

        temporary: set[str] = set()
        permanent: set[str] = set()
        result: list[str] = []

        def visit(name: str) -> None:
            if name in permanent:
                return
            if name in temporary:
                raise ValueError(f"Lifecycle dependency cycle detected at: {name}")
            temporary.add(name)
            for dependency in self._components[name].dependencies:
                visit(dependency)
            temporary.remove(name)
            permanent.add(name)
            result.append(name)

        for name in sorted(self._components):
            visit(name)
        return tuple(result)
