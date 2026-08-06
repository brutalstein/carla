from __future__ import annotations

from typing import Any

from l4stack.config.schema import SensorConfig
from l4stack.core.types import HealthState, LocalizationEstimate
from l4stack.localization.eskf import PlanarErrorStateEkf
from l4stack.runtime.channel import BoundedChannel
from l4stack.runtime.context import RuntimeContext
from l4stack.runtime.contracts import ComponentContract
from l4stack.runtime.health import HealthReport, RuntimeHealth
from l4stack.runtime.lifecycle import ManagedComponent
from l4stack.runtime.message import MessageEnvelope, MessageFactory
from l4stack.runtime.sensor_frame import SensorFrame
from l4stack.runtime.snapshot import AtomicSnapshotStore


class StaleLocalizationInput(RuntimeError):
    """Lokalizasyon girdisi freshness sözleşmesini karşılamadığında üretilir."""


class LocalizationRuntimeComponent(ManagedComponent):
    """Planar ESKF'yi ortak runtime sözleşmelerine bağlayan managed component.

    Input:
      ``MessageEnvelope[SensorFrame]``. SensorFrame içinde aynı CARLA frame'ine ait
      GNSS ve IMU ölçümleri bulunur.

    Output:
      ``MessageEnvelope[LocalizationEstimate]``. Çıktı LOCAL_ENU frame'indedir,
      source sensor frame zamanını korur ve parent olarak input message id taşır.
    """

    def __init__(
        self,
        localization_document: dict[str, Any],
        sensors_by_name: dict[str, SensorConfig],
        runtime: RuntimeContext,
        contract: ComponentContract,
        *,
        namespace: str,
    ) -> None:
        super().__init__(name=contract.name)
        self._runtime = runtime
        self._contract = contract
        # Estimator lifecycle configure aşamasında oluşturulur. Böylece cleanup sonrası
        # yeniden configure edilen component eski filtre durumunu taşımaz.
        self._localization_document = localization_document
        self._sensors_by_name = sensors_by_name
        self._estimator: PlanarErrorStateEkf | None = None
        self._output_factory = MessageFactory[LocalizationEstimate](
            source=self.name,
            clock=runtime.clock,
            coordinate_frame="LOCAL_ENU",
            namespace=namespace,
        )
        self.output_channel = BoundedChannel[MessageEnvelope[LocalizationEstimate]](
            name="localization.output",
            capacity=contract.channel_capacity,
            overflow_policy=contract.overflow_policy,
        )
        self.output_snapshot = AtomicSnapshotStore[MessageEnvelope[LocalizationEstimate]](
            "localization.latest"
        )

    def on_configure(self) -> None:
        self._runtime.deadlines.register(self._contract)
        self._estimator = PlanarErrorStateEkf(
            self._localization_document,
            self._sensors_by_name,
        )
        self._runtime.health.report(
            HealthReport(
                component=self.name,
                state=RuntimeHealth.UNAVAILABLE,
                timestamp=self._runtime.clock.now(),
                reason="configured; waiting for activation",
            )
        )

    def on_activate(self) -> None:
        self._runtime.health.report(
            HealthReport(
                component=self.name,
                state=RuntimeHealth.DEGRADED,
                timestamp=self._runtime.clock.now(),
                reason="active; waiting for first synchronized sensor frame",
            )
        )

    def on_deactivate(self) -> None:
        self._runtime.health.report(
            HealthReport(
                component=self.name,
                state=RuntimeHealth.UNAVAILABLE,
                timestamp=self._runtime.clock.now(),
                reason="component deactivated",
            )
        )

    def on_cleanup(self) -> None:
        # Yeniden configure edilirse filtre sıfır ve deterministik state ile başlar.
        self._estimator = None

    def on_shutdown(self) -> None:
        self._estimator = None
        self.output_channel.close()

    def process(
        self,
        input_message: MessageEnvelope[SensorFrame],
    ) -> MessageEnvelope[LocalizationEstimate]:
        """Tek bir synchronized sensör frame'ini deterministik olarak işler."""

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
            raise StaleLocalizationInput(input_events[-1].violation.value)

        estimator = self._estimator
        if estimator is None:
            raise RuntimeError("Localization estimator is not configured")

        started_at = self._runtime.deadlines.start_execution()
        frame = input_message.payload
        try:
            estimate = estimator.estimate(
                frame=frame.frame,
                timestamp=frame.timestamp,
                sensor_bundle=dict(frame.measurements),
            )
        except Exception as exc:
            # Algoritmik hata stale input gibi geçici bir QoS olayı değildir. Component
            # ERROR durumuna alınır ve supervisor kontrollü kapanış/recovery yapabilir.
            reason = f"localization processing failed: {exc}"
            self._runtime.health.report(
                HealthReport(
                    component=self.name,
                    state=RuntimeHealth.FAILED,
                    timestamp=self._runtime.clock.now(),
                    reason=reason,
                    metrics={"input_message_id": input_message.message_id},
                )
            )
            self.fail(reason)
            raise
        output = self._output_factory.create(
            estimate,
            source_timestamp=input_message.source_timestamp,
            lifespan_s=self._contract.output_lifespan_s,
            parents=(input_message.message_id,),
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

        state = _runtime_health(estimate.state)
        reasons: list[str] = []
        if input_events:
            state = RuntimeHealth.DEGRADED
            reasons.extend(event.violation.value for event in input_events)
        if execution_events:
            state = RuntimeHealth.DEGRADED
            reasons.extend(event.violation.value for event in execution_events)
        if not accepted:
            state = RuntimeHealth.DEGRADED
            reasons.append("OUTPUT_CHANNEL_REJECTED")

        self._runtime.health.report(
            HealthReport(
                component=self.name,
                state=state,
                timestamp=output.publish_timestamp,
                reason=",".join(reasons),
                metrics={
                    "frame": frame.frame,
                    "position_std_m": estimate.position_std_m,
                    "heading_std_deg": estimate.heading_std_deg,
                    "input_message_id": input_message.message_id,
                    "output_message_id": output.message_id,
                    "channel_depth": self.output_channel.stats().current_depth,
                },
            )
        )
        return output


def _runtime_health(state: HealthState) -> RuntimeHealth:
    if state is HealthState.NOMINAL:
        return RuntimeHealth.NOMINAL
    if state is HealthState.DEGRADED:
        return RuntimeHealth.DEGRADED
    return RuntimeHealth.FAILED
