import statistics

import pytest

from personalitygen.randomness import random_gaussian


class FixedRandom:
    def __init__(self, value: float) -> None:
        self._value = value

    def uniform(self, a: float, b: float) -> float:
        if not (a <= self._value <= b):
            raise AssertionError(
                f"Uniform value {self._value} not in {a}..{b}"
            )
        return self._value

    def gauss(self, mu: float, sigma: float) -> float:
        raise AssertionError("gauss is not expected in this test")


class LowerBoundRandom:
    def uniform(self, a: float, b: float) -> float:
        return a

    def gauss(self, mu: float, sigma: float) -> float:
        raise AssertionError("gauss is not expected in this test")


class UpperBoundRandom:
    def uniform(self, a: float, b: float) -> float:
        return b

    def gauss(self, mu: float, sigma: float) -> float:
        raise AssertionError("gauss is not expected in this test")


def test_random_gaussian_rejects_nonpositive_stddev() -> None:
    # stddev must be positive to define a valid Gaussian.
    with pytest.raises(ValueError):
        random_gaussian(
            mean=0.5, stddev=0.0, min_value=0.0, max_value=1.0
        )
    with pytest.raises(ValueError):
        random_gaussian(
            mean=0.5, stddev=-0.1, min_value=0.0, max_value=1.0
        )


def test_random_gaussian_rejects_inverted_bounds() -> None:
    # min_value must not exceed max_value.
    with pytest.raises(ValueError):
        random_gaussian(
            mean=0.5, stddev=0.2, min_value=1.0, max_value=0.0
        )


def test_random_gaussian_midpoint_matches_mean() -> None:
    # A uniform draw at 0.5 maps to the mean of a normal distribution.
    rng = FixedRandom(0.5)
    value = random_gaussian(
        mean=0.5, stddev=0.2, min_value=0.0, max_value=1.0, rng=rng
    )

    assert value == pytest.approx(0.5)


def test_random_gaussian_respects_bounds() -> None:
    # Lower and upper CDF bounds should return the truncation limits.
    lower_value = random_gaussian(
        mean=0.5,
        stddev=0.2,
        min_value=0.0,
        max_value=1.0,
        rng=LowerBoundRandom(),
    )
    upper_value = random_gaussian(
        mean=0.5,
        stddev=0.2,
        min_value=0.0,
        max_value=1.0,
        rng=UpperBoundRandom(),
    )

    assert lower_value == pytest.approx(0.0)
    assert upper_value == pytest.approx(1.0)


def test_random_gaussian_matches_inverse_cdf() -> None:
    # Use a fixed uniform value to confirm inverse-CDF sampling.
    u = 0.42
    rng = FixedRandom(u)
    value = random_gaussian(
        mean=0.5, stddev=0.2, min_value=0.0, max_value=1.0, rng=rng
    )

    dist = statistics.NormalDist(0.5, 0.2)
    expected = dist.inv_cdf(u)
    assert value == pytest.approx(expected)
