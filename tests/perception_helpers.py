# ruff: noqa: F401
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from l4stack.perception.adapters import (
    AdapterRequirements,
    BevFusionDetectionAdapter,
    BevFusionSegmentationAdapter,
    CitySemSegFormerAdapter,
    MapTRv2Adapter,
    PerceptionInputError,
    TldReadyAdapter,
)
from l4stack.perception.backend import (
    BackendUnavailable,
    FakeBackend,
    JsonlProcessBackend,
    ProcessBackendConfig,
)
from l4stack.perception.component import PerceptionModelComponent, StalePerceptionInput
from l4stack.perception.config import PerceptionConfig
from l4stack.perception.factory import build_perception_runtime
from l4stack.perception.input import PerceptionArtifactStore, PerceptionInputPublisher
from l4stack.perception.manifest import verify_installation
from l4stack.perception.orchestrator import ModelRoute, PerceptionPipeline
from l4stack.perception.protocol import BackendProtocolError, InferenceResponse
from l4stack.perception.types import ArtifactRef, PerceptionInput, PerceptionOutputKind
from l4stack.sensors.synchronizer import SensorSynchronizer
from l4stack.runtime.channel import OverflowPolicy
from l4stack.runtime.clock import ManualClock
from l4stack.runtime.context import RuntimeContext
from l4stack.runtime.contracts import (
    ComponentContract,
    ExecutionCriticality,
    ExecutorProfile,
)
from l4stack.runtime.deadline import DeadlineMonitor
from l4stack.runtime.executor import ExecutorRegistry, PriorityExecutor
from l4stack.runtime.health import HealthRegistry, RuntimeHealth
from l4stack.runtime.lineage import LineageStore
from l4stack.runtime.message import MessageFactory

ROOT = Path(__file__).resolve().parents[1]

CAMERAS = (
    "camera_front",
    "camera_front_right",
    "camera_back_right",
    "camera_back",
    "camera_back_left",
    "camera_front_left",
)


def artifact(
    name: str,
    media_type: str = "image/jpeg",
    *,
    timestamp: float | None = 1.0,
) -> ArtifactRef:
    return ArtifactRef(
        name=name,
        uri=f"file:///tmp/{name}.bin",
        media_type=media_type,
        shape=(10, 10, 3),
        dtype="uint8",
        byte_size=300,
        source_frame=None if timestamp is None else 10,
        source_timestamp=timestamp,
    )


def input_value(timestamp: float = 1.0, with_lidar: bool = True) -> PerceptionInput:
    return PerceptionInput(
        frame=10,
        timestamp=timestamp,
        cameras=tuple(artifact(name, timestamp=timestamp) for name in CAMERAS),
        lidar=(
            ArtifactRef(
                name="lidar_top",
                uri="file:///tmp/lidar.bin",
                media_type="application/x-pointcloud-f32",
                shape=(100, 5),
                dtype="float32",
                byte_size=2000,
                source_frame=10,
                source_timestamp=timestamp,
            )
            if with_lidar
            else None
        ),
        calibration=artifact("calibration", "application/json", timestamp=None),
        localization_message_id="loc/10",
    )


def runtime(clock: ManualClock) -> RuntimeContext:
    return RuntimeContext(
        clock=clock,
        processing_clock=clock,
        deadlines=DeadlineMonitor(clock, clock),
        health=HealthRegistry(),
        lineage=LineageStore(),
    )


def contract(name: str, max_age: float = 0.5) -> ComponentContract:
    return ComponentContract(
        name=name,
        criticality=ExecutionCriticality.FIRM_REALTIME,
        priority=50,
        max_input_age_s=max_age,
        execution_budget_s=0.2,
        expected_output_period_s=0.5,
        output_lifespan_s=0.4,
        channel_capacity=1,
        overflow_policy=OverflowPolicy.LATEST_ONLY,
    )


def envelope(clock: ManualClock, value: PerceptionInput, lifespan: float = 0.5):
    return MessageFactory[PerceptionInput]("sensor", clock, namespace="test").create(
        value,
        source_timestamp=value.timestamp,
        lifespan_s=lifespan,
    )


def detection_payload() -> dict:
    return {
        "detections_3d": [
            {
                "class_name": "car",
                "confidence": 0.9,
                "center_xyz_m": [10.0, 1.0, 0.5],
                "size_wlh_m": [1.8, 4.5, 1.6],
                "yaw_rad": 0.1,
                "velocity_xy_mps": [2.0, 0.0],
            }
        ],
        "diagnostics": {"inference_ms": 20.0},
    }


def raster_payload(name: str = "bev_semantic") -> dict:
    return {
        "rasters": [
            ArtifactRef(
                name=name,
                uri=f"file:///tmp/{name}.npy",
                media_type="application/x-npy",
                shape=(100, 100),
                dtype="uint8",
                byte_size=10_000,
            ).as_dict()
        ]
    }


def adapter_requirements(cameras: tuple[str, ...], lidar: bool):
    return AdapterRequirements(cameras, lidar, max_sensor_skew_s=0.1)
