from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from l4stack.perception.adapters import create_adapter
from l4stack.perception.backend import JsonlProcessBackend, ModelBackend
from l4stack.perception.component import PerceptionModelComponent
from l4stack.perception.config import PerceptionConfig, PerceptionModelConfig
from l4stack.perception.manifest import require_model_installation
from l4stack.perception.orchestrator import ModelRoute, PerceptionPipeline
from l4stack.runtime.context import RuntimeContext
from l4stack.runtime.executor import ExecutorRegistry
from l4stack.runtime.health import HealthReport, RuntimeHealth
from l4stack.runtime.lifecycle import LifecycleState
from l4stack.runtime.supervisor import RuntimeSupervisor

BackendFactory = Callable[[PerceptionModelConfig], ModelBackend]


@dataclass(frozen=True, slots=True)
class ComponentStartResult:
    model_name: str
    active: bool
    detail: str


@dataclass(slots=True)
class PerceptionRuntime:
    """İzole model supervisor'ları ve fan-out pipeline bütünü."""

    supervisors: Mapping[str, RuntimeSupervisor]
    pipeline: PerceptionPipeline
    components: Mapping[str, PerceptionModelComponent]

    def start(self, *, strict: bool = False) -> tuple[ComponentStartResult, ...]:
        results: list[ComponentStartResult] = []
        started: list[str] = []
        for name, supervisor in self.supervisors.items():
            try:
                supervisor.start_all()
            except Exception as exc:
                results.append(ComponentStartResult(name, False, str(exc)))
                self.pipeline.set_enabled(name, False)
                if strict:
                    for started_name in reversed(started):
                        self.supervisors[started_name].stop_all()
                    raise RuntimeError(f"Perception startup failed at {name}: {exc}") from exc
                continue
            self.pipeline.set_enabled(name, True)
            started.append(name)
            results.append(ComponentStartResult(name, True, "active"))
        return tuple(results)

    def stop(self) -> None:
        errors: list[str] = []
        for name, supervisor in reversed(tuple(self.supervisors.items())):
            component = self.components[name]
            if component.state is LifecycleState.FINALIZED:
                continue
            try:
                supervisor.stop_all()
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        if errors:
            raise RuntimeError("Perception shutdown errors: " + "; ".join(errors))


def build_perception_runtime(
    *,
    perception: PerceptionConfig,
    runtime_config,
    runtime_context: RuntimeContext,
    executors: ExecutorRegistry,
    backend_factory: BackendFactory | None = None,
    verify_artifacts: bool = True,
) -> PerceptionRuntime:
    """YAML model tanımlarından izole model component'lerini deterministik kurar."""

    create_backend = backend_factory or (lambda model: JsonlProcessBackend(model.backend))
    supervisors: dict[str, RuntimeSupervisor] = {}
    components: dict[str, PerceptionModelComponent] = {}
    routes: dict[str, ModelRoute] = {}

    if perception.enabled:
        for name, model in sorted(perception.models.items()):
            if not model.enabled:
                continue
            adapter = create_adapter(
                model.adapter,
                model.cameras,
                model.model_version,
                model.max_sensor_skew_s,
            )
            preflight = (
                (lambda selected=model: require_model_installation(selected))
                if verify_artifacts
                else None
            )
            component = PerceptionModelComponent(
                adapter=adapter,
                backend=create_backend(model),
                runtime=runtime_context,
                contract=runtime_config.contract(model.runtime_component),
                request_timeout_s=model.request_timeout_s,
                namespace=runtime_config.namespace,
                preflight=preflight,
            )
            supervisor = RuntimeSupervisor(runtime_context.health)
            supervisor.register(component)
            supervisors[name] = supervisor
            components[name] = component
            routes[name] = ModelRoute(
                component=component,
                executor=executors.get(model.executor),
                target_rate_hz=model.target_rate_hz,
            )

    if perception.enabled and not components:
        runtime_context.health.report(
            HealthReport(
                component="perception",
                state=RuntimeHealth.UNAVAILABLE,
                timestamp=runtime_context.clock.now(),
                reason="perception is enabled but no model component is enabled",
            )
        )

    return PerceptionRuntime(
        supervisors=MappingProxyType(supervisors),
        pipeline=PerceptionPipeline(routes, perception.snapshot_max_age_s),
        components=MappingProxyType(components),
    )
