import pytest

from personalitygen.randomness import random_gaussian


class FractionalRandom:
    def __init__(self, fraction: float) -> None:
        self.fraction = fraction
        self.calls: list[tuple[float, float]] = []

    def uniform(self, a: float, b: float) -> float:
        self.calls.append((a, b))
        return a + ((b - a) * self.fraction)

    def gauss(self, mu: float, sigma: float) -> float:
        raise AssertionError("gauss is not expected in this test")


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


def test_random_gaussian_midpoint_returns_center_for_symmetric_bounds() -> None:
    value = random_gaussian(
        mean=0.5,
        stddev=0.1,
        min_value=0.0,
        max_value=1.0,
        rng=MidpointRandom(),
    )

    assert value == pytest.approx(0.5)


def test_random_gaussian_uses_truncated_cdf_bounds() -> None:
    rng = FractionalRandom(0.25)

    value = random_gaussian(
        mean=0.5,
        stddev=0.1,
        min_value=0.25,
        max_value=0.75,
        rng=rng,
    )

    assert 0.25 <= value <= 0.75
    assert len(rng.calls) == 1
    lower, upper = rng.calls[0]
    assert 0.0 < lower < upper < 1.0


def test_random_gaussian_clamps_when_bounds_collapse() -> None:
    value = random_gaussian(
        mean=0.8,
        stddev=0.1,
        min_value=0.25,
        max_value=0.25,
    )
    assert value == 0.25


@pytest.mark.parametrize(
    ("mean", "expected"),
    [
        (-100.0, 0.0),
        (100.0, 1.0),
    ],
)
def test_random_gaussian_clamps_mean_when_distribution_misses_bounds(
    mean: float, expected: float
) -> None:
    value = random_gaussian(
        mean=mean,
        stddev=0.1,
        min_value=0.0,
        max_value=1.0,
    )

    assert value == expected


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
