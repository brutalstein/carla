#!/usr/bin/env python3
"""Gerçek model kurulmadan JSONL perception protokolünü smoke-test eder."""

from __future__ import annotations

import argparse
from pathlib import Path

from l4stack.perception.server import JsonlBackendServer
from l4stack.perception.types import PerceptionInput


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind",
        required=True,
        choices=(
            "bevfusion_detection",
            "bevfusion_segmentation",
            "maptrv2",
            "tld_ready",
            "citysemsegformer",
        ),
    )
    parser.add_argument("--output-dir", default="output/perception-mock")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    def infer(value: PerceptionInput):
        diagnostics = {"backend": "mock", "frame": value.frame, "inference_ms": 0.0}
        if args.kind == "bevfusion_detection":
            return {"detections_3d": [], "diagnostics": diagnostics}
        if args.kind == "maptrv2":
            return {"vector_map": [], "diagnostics": diagnostics}
        if args.kind == "tld_ready":
            return {"traffic_lights": [], "diagnostics": diagnostics}
        raster_names = (
            [f"{camera.name}_semantic" for camera in value.cameras]
            if args.kind == "citysemsegformer"
            else ["bev_semantic"]
        )
        rasters = []
        for name in raster_names:
            mask = output_dir / f"{args.kind}-{name}-{value.frame}.bin"
            mask.write_bytes(b"\x00")
            rasters.append(
                {
                    "name": name,
                    "uri": mask.as_uri(),
                    "media_type": "application/x-uint8-mask",
                    "shape": [1, 1],
                    "dtype": "uint8",
                    "byte_size": 1,
                    "sha256": None,
                }
            )
        return {"rasters": rasters, "diagnostics": diagnostics}

    JsonlBackendServer(infer).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
