from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from l4stack.perception.adapters import ModelAdapter, PerceptionInputError
from l4stack.perception.backend import ModelBackend
from l4stack.perception.protocol import InferenceRequest
from l4stack.perception.types import ModelOutput, PerceptionInput
from l4stack.runtime.channel import BoundedChannel
from l4stack.runtime.context import RuntimeContext
from l4stack.runtime.contracts import ComponentContract
from l4stack.runtime.health import HealthReport, RuntimeHealth
from l4stack.runtime.lifecycle import ManagedComponent
from l4stack.runtime.message import MessageEnvelope, MessageFactory
from l4stack.runtime.snapshot import AtomicSnapshotStore


class StalePerceptionInput(RuntimeError):
    """Model girdisi freshness sözleşmesini karşılamadığında üretilir."""


@dataclass(frozen=True, slots=True)
class PerceptionComponentMetrics:
    processed: int
    failed: int
    last_request_id: str | None


class PerceptionModelComponent(ManagedComponent):
    """Tek bir model backend'ini ortak runtime sözleşmelerine bağlar.

    Her model ayrı lifecycle component'i ve ayrı süreçtir. Bir modelin bağımlılık,
    timeout veya inference hatası yalnızca o component'i ``ERROR`` durumuna taşır.
    """

    def __init__(
        self,
        *,
        adapter: ModelAdapter,
        backend: ModelBackend,
        runtime: RuntimeContext,
        contract: ComponentContract,
        request_timeout_s: float,
        namespace: str,
        preflight: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(name=contract.name)
        if request_timeout_s <= 0.0:
            raise ValueError("request_timeout_s must be positive")
        self.adapter = adapter
        self.backend = backend
        self._runtime = runtime
        self._contract = contract
        self._request_timeout_s = request_timeout_s
        self._preflight = preflight
        self._output_factory = MessageFactory[ModelOutput](
            source=self.name,
            clock=runtime.clock,
            coordinate_frame=adapter.coordinate_frame,
            namespace=namespace,
        )
        self.output_channel = BoundedChannel[MessageEnvelope[ModelOutput]](
            name=f"{self.name}.output",
            capacity=contract.channel_capacity,
            overflow_policy=contract.overflow_policy,
        )
        self.output_snapshot = AtomicSnapshotStore[MessageEnvelope[ModelOutput]](
            f"{self.name}.latest"
        )
        self._processed = 0
        self._failed = 0
        self._last_request_id: str | None = None

    def on_configure(self) -> None:
        self._runtime.deadlines.register(self._contract)
        if self._preflight is not None:
            self._preflight()
        self.backend.start()
        health = self.backend.health()
        if not health.ready:
            raise RuntimeError(f"Backend is not ready: {health.detail}")
        self._runtime.health.report(
            HealthReport(
                component=self.name,
                state=RuntimeHealth.UNAVAILABLE,
                timestamp=self._runtime.clock.now(),
                reason="configured; waiting for activation",
                metrics={"backend_pid": health.pid, "model": self.adapter.name},
            )
        )

    def on_activate(self) -> None:
        self._runtime.health.report(
            HealthReport(
                component=self.name,
                state=RuntimeHealth.DEGRADED,
                timestamp=self._runtime.clock.now(),
                reason="active; waiting for first perception input",
                metrics={"model": self.adapter.name},
            )
        )

    def on_deactivate(self) -> None:
        self._runtime.health.report(
            HealthReport(
                component=self.name,
                state=RuntimeHealth.UNAVAILABLE,
                timestamp=self._runtime.clock.now(),
                reason="component deactivated",
                metrics={"model": self.adapter.name},
            )
        )

    def on_cleanup(self) -> None:
        self.backend.stop()

    def on_shutdown(self) -> None:
        self.backend.stop()
        self.output_channel.close()

    def process(
        self,
        input_message: MessageEnvelope[PerceptionInput],
    ) -> MessageEnvelope[ModelOutput]:
        self.require_active()
        input_events = self._runtime.deadlines.validate_input(self.name, input_message)
        if input_events and self._contract.drop_expired_inputs:
            self._runtime.health.report(
                HealthReport(
                    component=self.name,
                    state=RuntimeHealth.STALE,
                    timestamp=self._runtime.clock.now(),
                    reason=input_events[-1].violation.value,
                    metrics={"input_message_id": input_message.message_id},
                )
            )
            raise StalePerceptionInput(input_events[-1].violation.value)

        try:
            self.adapter.validate_input(input_message.payload)
        except PerceptionInputError as exc:
            self._runtime.health.report(
                HealthReport(
                    component=self.name,
                    state=RuntimeHealth.DEGRADED,
                    timestamp=self._runtime.clock.now(),
                    reason=f"invalid perception input: {exc}",
                    metrics={"input_message_id": input_message.message_id},
                )
            )
            raise

        request_id = f"{input_message.message_id}:{self.adapter.name}"
        request: InferenceRequest = self.adapter.build_request(input_message.payload, request_id)
        self._last_request_id = request_id
        started_at = self._runtime.deadlines.start_execution()
        try:
            raw_output = self.backend.infer(request, timeout_s=self._request_timeout_s)
            model_output = self.adapter.parse_response(input_message.payload, raw_output)
        except Exception as exc:
            self._failed += 1
            reason = f"perception model failed: {type(exc).__name__}: {exc}"
            self._runtime.health.report(
                HealthReport(
                    component=self.name,
                    state=RuntimeHealth.FAILED,
                    timestamp=self._runtime.clock.now(),
                    reason=reason,
                    metrics={
                        "input_message_id": input_message.message_id,
                        "request_id": request_id,
                        "model": self.adapter.name,
                    },
                )
            )
            self.fail(reason)
            raise

        output = self._output_factory.create(
            model_output,
            source_timestamp=input_message.source_timestamp,
            lifespan_s=self._contract.output_lifespan_s,
            parents=(input_message.message_id,),
            coordinate_frame=self.adapter.coordinate_frame,
        )
        self._runtime.lineage.record(output)
        self.output_snapshot.publish(output, published_at=output.publish_timestamp)
        accepted = self.output_channel.publish(output)
        execution_events = self._runtime.deadlines.finish_execution(
            self.name,
            started_at,
            output_timestamp=output.publish_timestamp,
            message_id=output.message_id,
        )
        self._processed += 1

        state = RuntimeHealth.NOMINAL
        reasons: list[str] = []
        if input_events or execution_events or not accepted:
            state = RuntimeHealth.DEGRADED
            reasons.extend(event.violation.value for event in input_events)
            reasons.extend(event.violation.value for event in execution_events)
            if not accepted:
                reasons.append("OUTPUT_CHANNEL_REJECTED")

        backend_health = self.backend.health()
        if not backend_health.ready:
            state = RuntimeHealth.DEGRADED
            reasons.append("BACKEND_NOT_READY_AFTER_OUTPUT")

        self._runtime.health.report(
            HealthReport(
                component=self.name,
                state=state,
                timestamp=output.publish_timestamp,
                reason=",".join(reasons),
                metrics={
                    "model_name": self.adapter.name,
                    "model_version": model_output.model_version,
                    "frame": model_output.source_frame,
                    "input_message_id": input_message.message_id,
                    "output_message_id": output.message_id,
                    "channel_depth": self.output_channel.stats().current_depth,
                    "processed": self._processed,
                    "failed": self._failed,
                    "backend_pid": backend_health.pid,
                },
            )
        )
        return output

    @property
    def priority(self) -> int:
        return self._contract.priority

    def metrics(self) -> PerceptionComponentMetrics:
        return PerceptionComponentMetrics(
            processed=self._processed,
            failed=self._failed,
            last_request_id=self._last_request_id,
        )
