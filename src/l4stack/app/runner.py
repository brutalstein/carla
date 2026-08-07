from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from l4stack.app.perception_runtime import PerceptionLoop
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
    """Lokalizasyonu senkron, perception modellerini asenkron runtime'da çalıştırır."""

    carla, client, initial_world = connect(config.simulator)
    run_cfg = config.simulator["run"]
    frame_count = int(frames_override if frames_override is not None else run_cfg["frames"])
    timeout = float(run_cfg.get("sensor_timeout_seconds", 2.0))
    output_dir = (config.root.parent / str(run_cfg.get("output_dir", "output"))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration_path = output_dir / "calibration.json"
    with calibration_path.open("w", encoding="utf-8") as handle:
        json.dump(export_calibration(config.sensors), handle, ensure_ascii=False, indent=2)

    writer_path = output_dir / config.logging["logging"].get("jsonl_filename", "frames.jsonl")
    synchronizer = SensorSynchronizer()
    odd_monitor = OddMonitor(config.odd, config.sensors)
    actors: list[Any] = []

    simulation_clock = SimulationClock()
    simulation_clock.update(0.0)
    processing_clock = SteadyClock()
    health = HealthRegistry()
    deadlines = DeadlineMonitor(
        simulation_clock,
        processing_clock,
        maximum_events=config.runtime.deadline_event_history,
    )
    lineage = LineageStore(config.runtime.lineage_max_records)
    runtime = RuntimeContext(
        clock=simulation_clock,
        processing_clock=processing_clock,
        deadlines=deadlines,
        health=health,
        lineage=lineage,
    )
    localization_component = LocalizationRuntimeComponent(
        config.localization,
        config.sensors_by_name,
        runtime,
        config.runtime.contract("localization"),
        namespace=config.runtime.namespace,
    )
    executors = ExecutorRegistry(config.runtime.executors, processing_clock)
    localization_supervisor = RuntimeSupervisor(health)
    localization_supervisor.register(localization_component)
    localization_started = False
    perception_loop: PerceptionLoop | None = None

    try:
        localization_executor = executors.get("localization")
        localization_supervisor.start_all()
        localization_started = True
        perception_loop = PerceptionLoop.start(
            config,
            runtime,
            executors,
            output_dir,
            calibration_path,
        )
        sensor_messages = MessageFactory[SensorFrame](
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
                    timestamp = float(world.get_snapshot().timestamp.elapsed_seconds)
                    simulation_clock.update(timestamp)

                    sensor_message = sensor_messages.create(
                        SensorFrame(frame=frame, timestamp=timestamp, measurements=bundle),
                        source_timestamp=timestamp,
                        lifespan_s=config.runtime.sensor_bundle_lifespan_s,
                    )
                    lineage.record(sensor_message)
                    localization_message = localization_executor.submit(
                        "localization.process",
                        localization_component.process,
                        sensor_message,
                        priority=config.runtime.contract("localization").priority,
                    ).result(timeout=timeout)
                    localization = localization_message.payload
                    odd = odd_monitor.assess(world, localization, bundle)

                    perception_input, submitted = perception_loop.submit_frame(
                        config=config,
                        synchronizer=synchronizer,
                        health=health,
                        lineage=lineage,
                        frame=frame,
                        timestamp=timestamp,
                        timeout=timeout,
                        localization_message_id=localization_message.message_id,
                    )
                    perception = perception_loop.diagnostics(config, health, timestamp)
                    perception["input_message"] = (
                        None if perception_input is None else perception_input.metadata_dict()
                    )
                    perception["submitted_models"] = list(submitted)

                    localization_deadline = deadlines.stats("localization")
                    localization_health = health.get("localization")
                    writer.write(
                        {
                            "frame": frame,
                            "timestamp": timestamp,
                            "odd": odd.as_dict(),
                            "localization": localization.as_dict(),
                            "perception": perception,
                            "runtime": {
                                "sensor_message": sensor_message.metadata_dict(),
                                "localization_message": localization_message.metadata_dict(),
                                "localization_snapshot_version": (
                                    localization_component.output_snapshot.require().version
                                ),
                                "localization_health": (
                                    None
                                    if localization_health is None
                                    else localization_health.as_dict()
                                ),
                                "localization_deadline": {
                                    "executions": localization_deadline.executions,
                                    "violations": localization_deadline.violations,
                                    "consecutive_violations": (
                                        localization_deadline.consecutive_violations
                                    ),
                                    "last_execution_s": localization_deadline.last_execution_s,
                                    "max_execution_s": localization_deadline.max_execution_s,
                                },
                            },
                        }
                    )
                    every = int(config.logging["logging"].get("console_every_n_frames", 10))
                    if every > 0 and step % every == 0:
                        output_count = len(perception["snapshot"]["outputs"])
                        _LOG.info(
                            "frame=%d odd=%s loc=%s pos_std=%.2fm heading_std=%.2fdeg "
                            "deadline_miss=%d perception_outputs=%d",
                            frame,
                            odd.state.value,
                            localization.state.value,
                            localization.position_std_m,
                            localization.heading_std_deg,
                            localization_deadline.violations,
                            output_count,
                        )
    finally:
        try:
            if perception_loop is not None:
                perception_loop.stop()
        finally:
            try:
                if localization_started:
                    localization_supervisor.stop_all()
            finally:
                executors.shutdown_all(wait=True, cancel_pending=True)
                destroy_actors(actors)

    return writer_path
