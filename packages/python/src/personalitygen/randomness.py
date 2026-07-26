"""Random utilities used by personalitygen."""

from __future__ import annotations

import math
import random
import statistics
from typing import Protocol


_CDF_EPSILON = 1e-12


class RandomSource(Protocol):
    """Minimal interface needed for deterministic sampling."""

    def uniform(self, a: float, b: float) -> float: ...


def _coerce_rng(rng: RandomSource | None) -> RandomSource:
    return rng if rng is not None else random


def random_gaussian(
    *,
    mean: float,
    stddev: float,
    min_value: float,
    max_value: float,
    rng: RandomSource | None = None,
) -> float:
    """Draw a truncated Gaussian sample within the provided bounds."""
    parameters = (mean, stddev, min_value, max_value)
    if not all(math.isfinite(value) for value in parameters):
        raise ValueError("Gaussian parameters must be finite")
    if stddev <= 0:
        raise ValueError("stddev must be positive")
    if min_value > max_value:
        raise ValueError("min_value must be <= max_value")

    source = _coerce_rng(rng)
    distribution = statistics.NormalDist(mean, stddev)
    lower = distribution.cdf(min_value)
    upper = distribution.cdf(max_value)
    if lower >= upper:
        return max(min_value, min(max_value, mean))

    lower = max(lower, _CDF_EPSILON)
    upper = min(upper, 1.0 - _CDF_EPSILON)
    if lower >= upper:
        return max(min_value, min(max_value, mean))

    probability = source.uniform(lower, upper)
    if (
        not math.isfinite(probability)
        or probability < lower
        or probability > upper
    ):
        raise ValueError("RandomSource.uniform returned a value out of range")
    if probability == lower:
        return min_value
    if probability == upper:
        return max_value

    sample = distribution.inv_cdf(probability)
    return max(min_value, min(max_value, sample))
