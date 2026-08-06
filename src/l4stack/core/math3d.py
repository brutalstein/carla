from __future__ import annotations

import math

from l4stack.core.types import Vector3


def norm3(value: Vector3) -> float:
    return math.sqrt(value.x * value.x + value.y * value.y + value.z * value.z)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
