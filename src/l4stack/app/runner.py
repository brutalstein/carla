from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from l4stack.config.schema import StackConfig
from l4stack.core.actors import destroy_actors
from l4stack.core.jsonlog import JsonlWriter
from l4stack.localization import LocalizationRuntimeComponent
from l4stack.odd.monitor import OddMonitor
from l4stack.runtime import (
    DeadlineMonitor,
    ExecutorRegistry,
    HealthRegistry,
    LineageStore,
    MessageFactory,
    RuntimeContext,
    RuntimeSupervisor,
    SensorFrame,
    SimulationClock,
    SteadyClock,
)
from l4stack.sensors.calibration import export_calibration
from l4stack.sensors.factory import spawn_sensor_suite
from l4stack.sensors.synchronizer import SensorSynchronizer
from l4stack.simulation.client import connect
from l4stack.simulation.deterministic_world import DeterministicWorld
from l4stack.simulation.vehicle import spawn_ego_vehicle

_LOG = logging.getLogger(__name__)


def run_stack(config: StackConfig, frames_override: int | None = None) -> Path:
    """CARLA stack'i ortak runtime altyapısı üzerinde çalıştırır.

    Bu sürümde runtime'a taşınan ilk fonksiyonel bileşen lokalizasyondur. Sensör bundle
    mesajı zaman damgalı bir envelope olarak üretilir; lokalizasyon çıktısı parent id,
    freshness, deadline, health ve lineage bilgileriyle birlikte yayınlanır.
    """

    carla, client, initial_world = connect(config.simulator)
    run_cfg = config.simulator["run"]
    frame_count = int(frames_override if frames_override is not None else run_cfg["frames"])
    timeout = float(run_cfg.get("sensor_timeout_seconds", 2.0))
    output_dir = (config.root.parent / str(run_cfg.get("output_dir", "output"))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "calibration.json").open("w", encoding="utf-8") as handle:
        json.dump(export_calibration(config.sensors), handle, ensure_ascii=False, indent=2)

    writer_path = output_dir / config.logging["logging"].get("jsonl_filename", "frames.jsonl")
    synchronizer = SensorSynchronizer()
    odd_monitor = OddMonitor(config.odd, config.sensors)
    actors: list[Any] = []

    # Simülasyon zamanı bütün fonksiyonel katmanların ortak source-time eksenidir.
    simulation_clock = SimulationClock()
    simulation_clock.update(0.0)
    processing_clock = SteadyClock()
    health_registry = HealthRegistry()
    deadline_monitor = DeadlineMonitor(
        simulation_clock,
        processing_clock,
        maximum_events=config.runtime.deadline_event_history,
    )
    lineage = LineageStore(config.runtime.lineage_max_records)
    runtime = RuntimeContext(
        clock=simulation_clock,
        processing_clock=processing_clock,
        deadlines=deadline_monitor,
        health=health_registry,
        lineage=lineage,
    )
    localization_component = LocalizationRuntimeComponent(
        config.localization,
        config.sensors_by_name,
        runtime,
        config.runtime.contract("localization"),
        namespace=config.runtime.namespace,
    )
    executor_registry = ExecutorRegistry(config.runtime.executors, processing_clock)
    supervisor = RuntimeSupervisor(health_registry)
    supervisor.register(localization_component)
    supervisor_started = False

    try:
        localization_executor = executor_registry.get("localization")
        supervisor.start_all()
        supervisor_started = True
        sensor_message_factory = MessageFactory[SensorFrame](
            source="sensor_synchronizer",
            clock=simulation_clock,
            coordinate_frame="CARLA_SENSOR_BUNDLE",
            namespace=config.runtime.namespace,
        )

        with DeterministicWorld(client, initial_world, config.simulator) as deterministic:
            world = deterministic.world
            ego = spawn_ego_vehicle(carla, world, config.vehicle)
            actors.append(ego)
            sensors = spawn_sensor_suite(
                carla,
                world,
                ego,
                config.sensors,
                synchronizer.callback,
            )
            actors.extend(sensors.values())
            _LOG.info("Spawned ego and %d sensors", len(sensors))

            with JsonlWriter(writer_path) as writer:
                for step in range(frame_count):
                    frame = deterministic.tick()
                    bundle = synchronizer.wait_for_frame(
                        frame,
                        config.required_sensor_names,
                        timeout,
                    )
                    snapshot = world.get_snapshot()
                    timestamp = float(snapshot.timestamp.elapsed_seconds)
                    simulation_clock.update(timestamp)

                    # Aynı frame'e ait sensörler tek immutable SensorFrame görünümüne alınır.
                    sensor_frame = SensorFrame(
                        frame=frame,
                        timestamp=timestamp,
                        measurements=bundle,
                    )
                    sensor_message = sensor_message_factory.create(
                        sensor_frame,
                        source_timestamp=timestamp,
                        lifespan_s=config.runtime.sensor_bundle_lifespan_s,
                    )
                    lineage.record(sensor_message)
                    localization_future = localization_executor.submit(
                        "localization.process",
                        localization_component.process,
                        sensor_message,
                        priority=config.runtime.contract("localization").priority,
                    )
                    localization_message = localization_future.result(timeout=timeout)
                    localization = localization_message.payload
                    odd = odd_monitor.assess(world, localization, bundle)
                    deadline_stats = deadline_monitor.stats("localization")
                    health = health_registry.get("localization")

                    writer.write(
                        {
                            "frame": frame,
                            "timestamp": timestamp,
                            "odd": odd.as_dict(),
                            "localization": localization.as_dict(),
                            "runtime": {
                                "sensor_message": sensor_message.metadata_dict(),
                                "localization_message": localization_message.metadata_dict(),
                                "localization_snapshot_version": (
                                    localization_component.output_snapshot.require().version
                                ),
                                "localization_health": None if health is None else health.as_dict(),
                                "localization_deadline": {
                                    "executions": deadline_stats.executions,
                                    "violations": deadline_stats.violations,
                                    "consecutive_violations": (
                                        deadline_stats.consecutive_violations
                                    ),
                                    "last_execution_s": deadline_stats.last_execution_s,
                                    "max_execution_s": deadline_stats.max_execution_s,
                                },
                            },
                        }
                    )
                    every = int(config.logging["logging"].get("console_every_n_frames", 10))
                    if every > 0 and step % every == 0:
                        _LOG.info(
                            "frame=%d odd=%s loc=%s pos_std=%.2fm heading_std=%.2fdeg "
                            "runtime_health=%s deadline_miss=%d",
                            frame,
                            odd.state.value,
                            localization.state.value,
                            localization.position_std_m,
                            localization.heading_std_deg,
                            "UNKNOWN" if health is None else health.state.value,
                            deadline_stats.violations,
                        )
    finally:
        try:
            if supervisor_started:
                supervisor.stop_all()
        finally:
            # Lifecycle kapanışı hata verse bile worker ve CARLA actor kaynakları sızmaz.
            executor_registry.shutdown_all(wait=True, cancel_pending=True)
            destroy_actors(actors)
    return writer_path
