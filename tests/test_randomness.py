import pytest

from personalitygen.randomness import random_gaussian


class MidpointRandom:
    def uniform(self, a: float, b: float) -> float:
        return (a + b) / 2.0

    def gauss(self, mu: float, sigma: float) -> float:
        raise AssertionError("gauss is not expected in this test")


def test_random_gaussian_returns_value_within_bounds() -> None:
    rng = MidpointRandom()
    value = random_gaussian(
        mean=0.6,
        stddev=0.1,
        min_value=0.01,
        max_value=1.0,
        rng=rng,
    )
    assert 0.01 <= value <= 1.0


def test_random_gaussian_clamps_when_bounds_collapse() -> None:
    value = random_gaussian(
        mean=0.8,
        stddev=0.1,
        min_value=0.25,
        max_value=0.25,
    )
    assert value == 0.25


def test_random_gaussian_rejects_non_positive_stddev() -> None:
    with pytest.raises(ValueError, match="stddev must be positive"):
        random_gaussian(
            mean=0.5,
            stddev=0.0,
            min_value=0.0,
            max_value=1.0,
        )


def test_random_gaussian_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="min_value must be <= max_value"):
        random_gaussian(
            mean=0.5,
            stddev=0.1,
            min_value=0.9,
            max_value=0.1,
        )
