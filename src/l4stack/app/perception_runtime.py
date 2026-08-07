from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from l4stack.config.schema import StackConfig
from l4stack.errors import SensorTimeoutError
from l4stack.perception import (
    PerceptionArtifactStore,
    PerceptionInput,
    PerceptionInputPublisher,
    build_perception_runtime,
)
from l4stack.perception.factory import PerceptionRuntime
from l4stack.runtime.context import RuntimeContext
from l4stack.runtime.executor import ExecutorRegistry
from l4stack.runtime.health import HealthRegistry, HealthReport, RuntimeHealth
from l4stack.runtime.lineage import LineageStore
from l4stack.runtime.message import MessageEnvelope
from l4stack.sensors.synchronizer import SensorSynchronizer

_LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class PerceptionLoop:
    """CARLA ana döngüsü ile asenkron perception pipeline arasındaki ince köprü."""

    runtime: PerceptionRuntime
    active_models: tuple[str, ...]
    input_publisher: PerceptionInputPublisher | None
    artifact_store: PerceptionArtifactStore | None
    calibration_ref: object | None

    @classmethod
    def start(
        cls,
        config: StackConfig,
        runtime_context: RuntimeContext,
        executors: ExecutorRegistry,
        output_dir: Path,
        calibration_path: Path,
    ) -> "PerceptionLoop":
        perception_runtime = build_perception_runtime(
            perception=config.perception,
            runtime_config=config.runtime,
            runtime_context=runtime_context,
            executors=executors,
        )
        active_models: tuple[str, ...] = ()
        publisher = None
        store = None
        calibration_ref = None
        if config.perception.enabled:
            start_results = perception_runtime.start(strict=False)
            active_models = tuple(item.model_name for item in start_results if item.active)
            for item in start_results:
                log = _LOG.info if item.active else _LOG.error
                log(
                    "perception model=%s active=%s detail=%s",
                    item.model_name,
                    item.active,
                    item.detail,
                )
        if active_models:
            publisher = PerceptionInputPublisher(
                runtime_context.clock,
                config.runtime.namespace,
                config.perception.input_lifespan_s,
            )
            store = PerceptionArtifactStore(
                output_dir / "perception_artifacts",
                retention_frames=config.perception.artifact_retention_frames,
            )
            calibration_ref = store.calibration_ref(calibration_path)
        return cls(perception_runtime, active_models, publisher, store, calibration_ref)

    def submit_frame(
        self,
        *,
        config: StackConfig,
        synchronizer: SensorSynchronizer,
        health: HealthRegistry,
        lineage: LineageStore,
        frame: int,
        timestamp: float,
        timeout: float,
        localization_message_id: str,
    ) -> tuple[MessageEnvelope[PerceptionInput] | None, tuple[str, ...]]:
        due_models = self.runtime.pipeline.due_models(timestamp)
        if not due_models:
            return None, ()
        if (
            self.artifact_store is None
            or self.calibration_ref is None
            or self.input_publisher is None
        ):
            raise RuntimeError("Perception input services are not initialized")

        due_cameras, due_lidar = _active_perception_sensors(config, due_models)
        required_artifacts = due_cameras + (() if due_lidar is None else (due_lidar,))
        try:
            bundle = synchronizer.wait_for_latest_at_or_before(
                frame,
                required_artifacts,
                timeout,
            )
        except SensorTimeoutError as exc:
            for name in due_models:
                health.report(
                    HealthReport(
                        component=config.perception.models[name].runtime_component,
                        state=RuntimeHealth.STALE,
                        timestamp=timestamp,
                        reason=f"perception sensor input timeout: {exc}",
                    )
                )
            return None, ()

        camera_refs = tuple(
            self.artifact_store.write_camera(name, bundle[name]) for name in due_cameras
        )
        lidar_ref = (
            None
            if due_lidar is None
            else self.artifact_store.write_lidar(due_lidar, bundle[due_lidar])
        )
        value = PerceptionInput(
            frame=frame,
            timestamp=timestamp,
            cameras=camera_refs,
            lidar=lidar_ref,
            calibration=self.calibration_ref,
            localization_message_id=localization_message_id,
        )
        message = self.input_publisher.publish(
            value,
            localization_message_id=localization_message_id,
        )
        lineage.record(message)
        submitted = tuple(
            self.runtime.pipeline.submit(message, model_names=due_models)
        )
        return message, submitted

    def diagnostics(self, config: StackConfig, health: HealthRegistry, now: float) -> dict:
        snapshot = self.runtime.pipeline.snapshot(now)
        metrics = self.runtime.pipeline.metrics()
        model_health = {
            name: report.as_dict()
            for name in self.active_models
            if (
                report := health.get(config.perception.models[name].runtime_component)
            )
            is not None
        }
        return {
            "snapshot": {
                "generated_at": snapshot.generated_at,
                "outputs": [item.as_dict() for item in snapshot.outputs],
            },
            "metrics": {
                "submitted": dict(metrics.submitted),
                "skipped_by_rate": dict(metrics.skipped_by_rate),
                "rejected_by_backpressure": dict(metrics.rejected_by_backpressure),
                "completed": dict(metrics.completed),
                "failed": dict(metrics.failed),
            },
            "health": model_health,
        }

    def stop(self) -> None:
        if self.active_models:
            self.runtime.stop()


def _active_perception_sensors(
    config: StackConfig,
    model_names: tuple[str, ...],
) -> tuple[tuple[str, ...], str | None]:
    cameras: list[str] = []
    lidar: str | None = None
    for name in model_names:
        model = config.perception.models[name]
        for camera in model.cameras:
            if camera not in cameras:
                cameras.append(camera)
        if model.lidar is not None:
            if lidar is not None and lidar != model.lidar:
                raise RuntimeError("Only one perception LiDAR stream is supported")
            lidar = model.lidar
    return tuple(cameras), lidar
