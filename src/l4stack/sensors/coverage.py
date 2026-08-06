from __future__ import annotations

from l4stack.config.schema import SensorConfig


def camera_azimuth_gaps(sensors: tuple[SensorConfig, ...]) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for sensor in sensors:
        if sensor.blueprint != "sensor.camera.rgb":
            continue
        half = float(sensor.attributes.get("fov", 0.0)) * 0.5
        center = sensor.transform.yaw % 360.0
        start = (center - half) % 360.0
        end = (center + half) % 360.0
        if start <= end:
            intervals.append((start, end))
        else:
            intervals.append((start, 360.0))
            intervals.append((0.0, end))
    if not intervals:
        return [(0.0, 360.0)]
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + 1e-9:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    gaps: list[tuple[float, float]] = []
    if merged[0][0] > 0.0:
        gaps.append((0.0, merged[0][0]))
    for left, right in zip(merged, merged[1:], strict=False):
        if right[0] > left[1]:
            gaps.append((left[1], right[0]))
    if merged[-1][1] < 360.0:
        gaps.append((merged[-1][1], 360.0))
    return gaps
