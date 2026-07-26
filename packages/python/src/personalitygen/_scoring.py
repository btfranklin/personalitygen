"""Internal score validation and vector helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence

from personalitygen.constants import UNIT_RANGE_MAX, UNIT_RANGE_MIN

SIGNED_RANGE_MIN = -1.0
SIGNED_RANGE_MAX = 1.0


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def validate_unit_range(*values: float) -> None:
    for value in values:
        if not (UNIT_RANGE_MIN <= value <= UNIT_RANGE_MAX):
            raise ValueError(
                "All trait components must be in the range 0.0...1.0"
            )


def validate_signed_range(*values: float) -> None:
    for value in values:
        if not (SIGNED_RANGE_MIN <= value <= SIGNED_RANGE_MAX):
            raise ValueError(
                "All signed scores must be in the range -1.0...1.0"
            )


def component_average_score(*values: float) -> float:
    validate_unit_range(*values)
    return sum(values) / len(values)


def unit_to_signed(value: float) -> float:
    validate_unit_range(value)
    return (value * 2.0) - 1.0


def weighted_signed_average(*weighted_values: tuple[float, float]) -> float:
    if not weighted_values:
        raise ValueError("weighted_values must be non-empty")

    total_weight = sum(abs(weight) for _, weight in weighted_values)
    if total_weight <= 0.0:
        raise ValueError("weighted_values must include a non-zero weight")

    for value, _ in weighted_values:
        validate_signed_range(value)

    score = sum(value * weight for value, weight in weighted_values)
    return _clamp(
        score / total_weight,
        SIGNED_RANGE_MIN,
        SIGNED_RANGE_MAX,
    )


def dot_product(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same length")
    validate_signed_range(*left)
    validate_signed_range(*right)
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same length")
    validate_signed_range(*left)
    validate_signed_range(*right)

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot_product(left, right) / (left_norm * right_norm)
