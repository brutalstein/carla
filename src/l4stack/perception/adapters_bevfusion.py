from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from l4stack.perception.adapter_common import (
    BaseAdapter,
    diagnostics,
    fixed_tuple,
    require_rasters,
)
from l4stack.perception.protocol import BackendProtocolError
from l4stack.perception.types import Detection3D, ModelOutput, PerceptionInput, PerceptionOutputKind


class BevFusionDetectionAdapter(BaseAdapter):
    name = "bevfusion_detection"
    kind = PerceptionOutputKind.OBJECT_DETECTION_3D
    coordinate_frame = "EGO_LOCAL"

    def parse_response(self, value: PerceptionInput, payload: Mapping[str, Any]) -> ModelOutput:
        detections: list[Detection3D] = []
        raw_items = payload.get("detections_3d", [])
        if not isinstance(raw_items, list):
            raise BackendProtocolError("BEVFusion detections_3d must be a list")
        for item in raw_items:
            if not isinstance(item, Mapping):
                raise BackendProtocolError("BEVFusion detection item must be a mapping")
            velocity = item.get("velocity_xy_mps")
            detections.append(
                Detection3D(
                    class_name=str(item["class_name"]),
                    confidence=float(item["confidence"]),
                    center_xyz_m=fixed_tuple(item["center_xyz_m"], 3, "center_xyz_m"),
                    size_wlh_m=fixed_tuple(item["size_wlh_m"], 3, "size_wlh_m"),
                    yaw_rad=float(item["yaw_rad"]),
                    velocity_xy_mps=(
                        None
                        if velocity is None
                        else fixed_tuple(velocity, 2, "velocity_xy_mps")
                    ),
                )
            )
        return self.base_output(
            value,
            detections_3d=tuple(detections),
            diagnostics=diagnostics(payload),
        )


class BevFusionSegmentationAdapter(BaseAdapter):
    name = "bevfusion_segmentation"
    kind = PerceptionOutputKind.BEV_SEGMENTATION
    coordinate_frame = "EGO_BEV_RASTER"

    def parse_response(self, value: PerceptionInput, payload: Mapping[str, Any]) -> ModelOutput:
        rasters = require_rasters(payload)
        if len(rasters) != 1:
            raise BackendProtocolError(
                "BEVFusion segmentation requires exactly one BEV raster"
            )
        return self.base_output(
            value,
            rasters=rasters,
            diagnostics=diagnostics(payload),
        )
