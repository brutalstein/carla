from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from l4stack.perception.adapter_common import BaseAdapter, diagnostics, fixed_tuple, require_rasters
from l4stack.perception.protocol import BackendProtocolError
from l4stack.perception.types import (
    ModelOutput,
    PerceptionInput,
    PerceptionOutputKind,
    TrafficLightObservation,
)


class TldReadyAdapter(BaseAdapter):
    name = "tld_ready"
    kind = PerceptionOutputKind.TRAFFIC_LIGHT
    coordinate_frame = "CAMERA_PIXEL"

    def parse_response(self, value: PerceptionInput, payload: Mapping[str, Any]) -> ModelOutput:
        observations: list[TrafficLightObservation] = []
        raw_items = payload.get("traffic_lights", [])
        if not isinstance(raw_items, list):
            raise BackendProtocolError("TLD-READY traffic_lights must be a list")
        allowed_cameras = set(self.requirements.cameras)
        for item in raw_items:
            if not isinstance(item, Mapping):
                raise BackendProtocolError("TLD-READY observation must be a mapping")
            camera_name = str(item["camera_name"])
            if camera_name not in allowed_cameras:
                raise BackendProtocolError(
                    f"TLD-READY returned an unconfigured camera: {camera_name}"
                )
            observations.append(
                TrafficLightObservation(
                    camera_name=camera_name,
                    bbox_xyxy=fixed_tuple(item["bbox_xyxy"], 4, "bbox_xyxy"),
                    state=str(item["state"]).upper(),
                    pictogram=str(item.get("pictogram", "circle")),
                    confidence=float(item["confidence"]),
                    relevant_to_ego=(
                        None
                        if item.get("relevant_to_ego") is None
                        else bool(item["relevant_to_ego"])
                    ),
                )
            )
        return self.base_output(
            value,
            traffic_lights=tuple(observations),
            diagnostics=diagnostics(payload),
        )


class CitySemSegFormerAdapter(BaseAdapter):
    name = "citysemsegformer"
    kind = PerceptionOutputKind.IMAGE_SEGMENTATION
    coordinate_frame = "CAMERA_PIXEL"

    def parse_response(self, value: PerceptionInput, payload: Mapping[str, Any]) -> ModelOutput:
        rasters = require_rasters(payload)
        expected = set(self.requirements.cameras)
        actual = {item.name.removesuffix("_semantic") for item in rasters}
        if actual != expected:
            raise BackendProtocolError(
                "CitySemSegFormer camera masks do not match the configured set: "
                f"missing={sorted(expected - actual)} unknown={sorted(actual - expected)}"
            )
        return self.base_output(
            value,
            rasters=rasters,
            diagnostics=diagnostics(payload),
        )
