from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from l4stack.config.loader import load_stack_config
from l4stack.localization import LocalizationRuntimeComponent
from l4stack.localization.runtime_component import StaleLocalizationInput
from l4stack.runtime import (
    DeadlineMonitor,
    HealthRegistry,
    LineageStore,
    ManualClock,
    MessageFactory,
    RuntimeContext,
    RuntimeSupervisor,
    SensorFrame,
)
from l4stack.runtime.health import RuntimeHealth

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Vector:
    x: float
    y: float
    z: float


@dataclass
class ImuMeasurement:
    accelerometer: Vector
    gyroscope: Vector
    compass: float


@dataclass
class GnssMeasurement:
    latitude: float
    longitude: float
    altitude: float


def _bundle() -> dict[str, object]:
    return {
        "gnss_roof": GnssMeasurement(40.1950, 29.0600, 120.0),
        "imu_center": ImuMeasurement(
            accelerometer=Vector(0.0, 0.0, 0.0),
            gyroscope=Vector(0.0, 0.0, 0.0),
            compass=math.pi / 2.0,
        ),
    }


def _component(clock: ManualClock):
    config = load_stack_config(ROOT / "config")
    processing = ManualClock(0.0)
    health = HealthRegistry()
    runtime = RuntimeContext(
        clock=clock,
        processing_clock=processing,
        deadlines=DeadlineMonitor(clock, processing),
        health=health,
        lineage=LineageStore(100),
    )
    component = LocalizationRuntimeComponent(
        config.localization,
        config.sensors_by_name,
        runtime,
        config.runtime.contract("localization"),
        namespace="test",
    )
    supervisor = RuntimeSupervisor(health)
    supervisor.register(component)
    supervisor.start_all()
    return config, runtime, health, component, supervisor


def test_localization_component_publishes_versioned_lineage_output() -> None:
    clock = ManualClock(0.05)
    config, runtime, health, component, supervisor = _component(clock)
    factory = MessageFactory[SensorFrame]("sensor_synchronizer", clock, namespace="test")
    sensor_message = factory.create(
        SensorFrame(1, 0.05, _bundle()),
        source_timestamp=0.05,
        lifespan_s=config.runtime.sensor_bundle_lifespan_s,
    )
    runtime.lineage.record(sensor_message)

    output = component.process(sensor_message)
    queued = component.output_channel.receive(timeout=0.1)

    assert output.parents == (sensor_message.message_id,)
    assert output.coordinate_frame == "LOCAL_ENU"
    assert output.payload.frame == 1
    assert queued.message_id == output.message_id
    assert component.output_snapshot.require().version == 1
    assert health.get("localization").state is RuntimeHealth.NOMINAL
    assert [item.message_id for item in runtime.lineage.trace(output.message_id)] == [
        output.message_id,
        sensor_message.message_id,
    ]
    supervisor.stop_all()


def test_localization_component_rejects_expired_sensor_message() -> None:
    clock = ManualClock(0.0)
    config, runtime, health, component, supervisor = _component(clock)
    factory = MessageFactory[SensorFrame]("sensor_synchronizer", clock, namespace="test")
    sensor_message = factory.create(
        SensorFrame(1, 0.0, _bundle()),
        source_timestamp=0.0,
        lifespan_s=config.runtime.sensor_bundle_lifespan_s,
    )
    clock.set(0.2)

    with pytest.raises(StaleLocalizationInput):
        component.process(sensor_message)
    assert health.get("localization").state is RuntimeHealth.STALE
    supervisor.stop_all()


def test_localization_processing_failure_moves_component_to_error() -> None:
    from l4stack.runtime.lifecycle import LifecycleState

    clock = ManualClock(0.05)
    config, runtime, health, component, supervisor = _component(clock)
    factory = MessageFactory[SensorFrame]("sensor_synchronizer", clock, namespace="test")
    broken = _bundle()
    broken["gnss_roof"] = GnssMeasurement(float("nan"), 29.0600, 120.0)
    sensor_message = factory.create(
        SensorFrame(1, 0.05, broken),
        source_timestamp=0.05,
        lifespan_s=config.runtime.sensor_bundle_lifespan_s,
    )

    with pytest.raises(ValueError, match="non-finite"):
        component.process(sensor_message)
    assert component.state is LifecycleState.ERROR
    assert health.get("localization").state is RuntimeHealth.FAILED
    supervisor.stop_all()
