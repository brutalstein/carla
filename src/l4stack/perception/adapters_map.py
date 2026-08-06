from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from l4stack.perception.adapter_common import BaseAdapter, diagnostics, fixed_tuple
from l4stack.perception.protocol import BackendProtocolError
from l4stack.perception.types import (
    ModelOutput,
    PerceptionInput,
    PerceptionOutputKind,
    VectorMapElement,
)


class MapTRv2Adapter(BaseAdapter):
    name = "maptrv2"
    kind = PerceptionOutputKind.VECTOR_MAP
    coordinate_frame = "EGO_LOCAL"

    def parse_response(self, value: PerceptionInput, payload: Mapping[str, Any]) -> ModelOutput:
        elements: list[VectorMapElement] = []
        raw_items = payload.get("vector_map", [])
        if not isinstance(raw_items, list):
            raise BackendProtocolError("MapTRv2 vector_map must be a list")
        for item in raw_items:
            if not isinstance(item, Mapping):
                raise BackendProtocolError("MapTRv2 vector element must be a mapping")
            points = tuple(
                fixed_tuple(point, 3, "points_xyz_m") for point in item["points_xyz_m"]
            )
            elements.append(
                VectorMapElement(
                    category=str(item["category"]),
                    confidence=float(item["confidence"]),
                    points_xyz_m=points,
                )
            )
        return self.base_output(
            value,
            vector_map=tuple(elements),
            diagnostics=diagnostics(payload),
        )
