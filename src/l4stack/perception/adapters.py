from __future__ import annotations

from l4stack.perception.adapter_common import (
    AdapterRequirements,
    ModelAdapter,
    PerceptionInputError,
)
from l4stack.perception.adapters_bevfusion import (
    BevFusionDetectionAdapter,
    BevFusionSegmentationAdapter,
)
from l4stack.perception.adapters_map import MapTRv2Adapter
from l4stack.perception.adapters_vision import CitySemSegFormerAdapter, TldReadyAdapter


def create_adapter(
    name: str,
    cameras: tuple[str, ...],
    model_version: str,
    max_sensor_skew_s: float,
) -> ModelAdapter:
    """YAML adapter adını somut ve strict model adapter'ına dönüştürür."""

    mapping = {
        "bevfusion_detection": (BevFusionDetectionAdapter, True),
        "bevfusion_segmentation": (BevFusionSegmentationAdapter, True),
        "maptrv2": (MapTRv2Adapter, False),
        "tld_ready": (TldReadyAdapter, False),
        "citysemsegformer": (CitySemSegFormerAdapter, False),
    }
    try:
        adapter_type, lidar_required = mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unknown perception adapter: {name}") from exc
    requirements = AdapterRequirements(
        cameras=cameras,
        lidar_required=lidar_required,
        max_sensor_skew_s=max_sensor_skew_s,
    )
    return adapter_type(requirements, model_version)


__all__ = [
    "AdapterRequirements",
    "BevFusionDetectionAdapter",
    "BevFusionSegmentationAdapter",
    "CitySemSegFormerAdapter",
    "MapTRv2Adapter",
    "ModelAdapter",
    "PerceptionInputError",
    "TldReadyAdapter",
    "create_adapter",
]
