from __future__ import annotations

from typing import Any

import numpy as np

SEMANTIC_LIDAR_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("cos_incidence", "<f4"),
        ("object_idx", "<u4"),
        ("object_tag", "<u4"),
    ]
)


def decode_semantic_lidar(measurement: Any) -> np.ndarray:
    return decode_semantic_lidar_bytes(measurement.raw_data)


def decode_semantic_lidar_bytes(raw_data: bytes | bytearray | memoryview) -> np.ndarray:
    points = np.frombuffer(raw_data, dtype=SEMANTIC_LIDAR_DTYPE)
    return points.copy()


def decode_rgb_image(image: Any) -> np.ndarray:
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    return array.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1].copy()
