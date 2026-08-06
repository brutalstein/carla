from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from l4stack.config.schema import StackConfig
from l4stack.core.jsonlog import JsonlWriter
from l4stack.core.lifecycle import destroy_actors
from l4stack.localization.estimator import GroundTruthLocalizer
from l4stack.odd.monitor import OddMonitor
from l4stack.perception.semantic_lidar import SemanticLidarInstanceDetector
from l4stack.sensors.calibration import export_calibration
from l4stack.sensors.factory import spawn_sensor_suite
from l4stack.sensors.synchronizer import SensorSynchronizer
from l4stack.simulation.client import connect
from l4stack.simulation.deterministic_world import DeterministicWorld
from l4stack.simulation.vehicle import spawn_ego_vehicle

_LOG = logging.getLogger(__name__)


def run_stack(config: StackConfig, frames_override: int | None = None) -> Path:
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
    localizer = GroundTruthLocalizer()
    fixed_delta = float(config.simulator["world"]["fixed_delta_seconds"])
    detector = SemanticLidarInstanceDetector(config.perception, fixed_delta)
    odd_monitor = OddMonitor(config.odd, config.sensors)
    actors: list[Any] = []

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

        try:
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
                    localization = localizer.estimate(frame, timestamp, ego, bundle)
                    odd = odd_monitor.assess(world, localization, bundle)
                    perception = detector.detect(frame, timestamp, bundle)
                    record = {
                        "frame": frame,
                        "timestamp": timestamp,
                        "odd": odd.as_dict(),
                        "localization": localization.as_dict(),
                        "perception": perception.as_dict(),
                    }
                    writer.write(record)
                    every = int(config.logging["logging"].get("console_every_n_frames", 10))
                    if every > 0 and step % every == 0:
                        _LOG.info(
                            "frame=%d odd=%s loc=%s detections=%d",
                            frame,
                            odd.state.value,
                            localization.state.value,
                            len(perception.detections),
                        )
        finally:
            destroy_actors(actors)
    return writer_path
