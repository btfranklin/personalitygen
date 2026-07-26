from __future__ import annotations

import math
import random

import pytest

from personalitygen.adaptive import (
    AdaptiveBifurcatedDomain,
    AdaptiveBifurcatedPole,
    AdaptiveBifurcatedProfile,
)
from personalitygen.personality import BigFiveTraitConfiguration
from personalitygen.traits import (
    BigFiveAgreeableness,
    BigFiveConscientiousness,
    BigFiveExtraversion,
    BigFiveNeuroticism,
    BigFiveOpenness,
)


class MidpointRandom:
    def uniform(self, a: float, b: float) -> float:
        return (a + b) / 2.0


def test_adaptive_profile_accepts_signed_boundaries() -> None:
    profile = AdaptiveBifurcatedProfile(
        order_score=-1.0,
        chaos_score=1.0,
        cooperation_score=0.0,
        conflict_score=0.5,
        competition_score=-0.5,
    )

    assert profile.vector == (-1.0, 1.0, 0.0, 0.5, -0.5)


@pytest.mark.parametrize("component_index", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("invalid_value", [-1.001, 1.001])
def test_adaptive_scores_must_stay_in_signed_range(
    component_index: int, invalid_value: float
) -> None:
    values = [0.0, 0.0, 0.0, 0.0, 0.0]
    values[component_index] = invalid_value

    with pytest.raises(
        ValueError,
        match="All signed scores must be in the range -1.0...1.0",
    ):
        AdaptiveBifurcatedProfile(*values)


def test_adaptive_random_generation_samples_signed_scores() -> None:
    profile = AdaptiveBifurcatedProfile.random(rng=MidpointRandom())

    assert profile.vector == pytest.approx((0.0, 0.0, 0.0, 0.0, 0.0))
    assert all(-1.0 <= value <= 1.0 for value in profile.vector)


def test_adaptive_random_generation_is_deterministic_for_seed() -> None:
    rng_a = random.Random(123)
    rng_b = random.Random(123)

    assert AdaptiveBifurcatedProfile.random(
        rng=rng_a
    ) == AdaptiveBifurcatedProfile.random(rng=rng_b)


def test_adaptive_profile_vector_uses_canonical_chart_order() -> None:
    profile = AdaptiveBifurcatedProfile(0.1, 0.2, 0.3, 0.4, 0.5)

    assert profile.vector == (0.1, 0.2, 0.3, 0.4, 0.5)


def test_adaptive_axes_expose_domain_and_pole_metadata() -> None:
    profile = AdaptiveBifurcatedProfile(0.1, -0.2, 0.3, -0.4, 0.5)

    assert [
        (
            axis.domain,
            axis.positive_pole,
            axis.negative_pole,
            axis.score,
        )
        for axis in profile.axes
    ] == [
        (
            AdaptiveBifurcatedDomain.ORDER,
            AdaptiveBifurcatedPole.STRATEGIZING,
            AdaptiveBifurcatedPole.IMPROVISATION,
            0.1,
        ),
        (
            AdaptiveBifurcatedDomain.CHAOS,
            AdaptiveBifurcatedPole.ETHICALITY,
            AdaptiveBifurcatedPole.INSTINCTIVITY,
            -0.2,
        ),
        (
            AdaptiveBifurcatedDomain.COOPERATION,
            AdaptiveBifurcatedPole.COLLABORATION,
            AdaptiveBifurcatedPole.INDEPENDENCE,
            0.3,
        ),
        (
            AdaptiveBifurcatedDomain.CONFLICT,
            AdaptiveBifurcatedPole.HARMONIZING,
            AdaptiveBifurcatedPole.UTILITARIANISM,
            -0.4,
        ),
        (
            AdaptiveBifurcatedDomain.COMPETITION,
            AdaptiveBifurcatedPole.PROFICIENCY,
            AdaptiveBifurcatedPole.DOMINANCY,
            0.5,
        ),
    ]


def test_adaptive_dominant_poles_respect_sign_and_threshold() -> None:
    profile = AdaptiveBifurcatedProfile(0.2, -0.3, 0.0, 0.05, -0.8)

    assert profile.dominant_poles() == {
        AdaptiveBifurcatedDomain.ORDER: (
            AdaptiveBifurcatedPole.STRATEGIZING
        ),
        AdaptiveBifurcatedDomain.CHAOS: (
            AdaptiveBifurcatedPole.INSTINCTIVITY
        ),
        AdaptiveBifurcatedDomain.CONFLICT: (
            AdaptiveBifurcatedPole.HARMONIZING
        ),
        AdaptiveBifurcatedDomain.COMPETITION: (
            AdaptiveBifurcatedPole.DOMINANCY
        ),
    }
    assert profile.dominant_poles(threshold=0.1) == {
        AdaptiveBifurcatedDomain.ORDER: (
            AdaptiveBifurcatedPole.STRATEGIZING
        ),
        AdaptiveBifurcatedDomain.CHAOS: (
            AdaptiveBifurcatedPole.INSTINCTIVITY
        ),
        AdaptiveBifurcatedDomain.COMPETITION: (
            AdaptiveBifurcatedPole.DOMINANCY
        ),
    }


def test_adaptive_dominant_poles_reject_invalid_thresholds() -> None:
    profile = AdaptiveBifurcatedProfile(0.0, 0.0, 0.0, 0.0, 0.0)

    with pytest.raises(
        ValueError, match="threshold must be in the range 0.0...1.0"
    ):
        profile.dominant_poles(threshold=-0.1)


def test_adaptive_dot_product_and_cosine_similarity() -> None:
    left = AdaptiveBifurcatedProfile(1.0, 0.0, 0.0, 0.0, 0.0)
    right = AdaptiveBifurcatedProfile(0.5, 0.5, 0.0, 0.0, 0.0)

    assert left.dot_product(right) == pytest.approx(0.5)
    assert left.cosine_similarity(right) == pytest.approx(1 / math.sqrt(2))


def test_adaptive_cosine_similarity_returns_zero_for_zero_vector() -> None:
    zero = AdaptiveBifurcatedProfile(0.0, 0.0, 0.0, 0.0, 0.0)
    profile = AdaptiveBifurcatedProfile(1.0, 0.0, 0.0, 0.0, 0.0)

    assert zero.cosine_similarity(profile) == 0.0
    assert profile.cosine_similarity(zero) == 0.0


def test_adaptive_profile_projects_from_big_five_traits() -> None:
    traits = BigFiveTraitConfiguration(
        openness=BigFiveOpenness(0.75, 0.75, 0.75),
        conscientiousness=BigFiveConscientiousness(0.25, 0.25, 0.25),
        extraversion=BigFiveExtraversion(0.60, 0.60, 0.60),
        agreeableness=BigFiveAgreeableness(0.80, 0.80, 0.80),
        neuroticism=BigFiveNeuroticism(0.10, 0.10, 0.10),
    )

    profile = AdaptiveBifurcatedProfile.from_big_five(traits)

    assert profile.vector == pytest.approx(
        (
            0.50,
            0.255,
            0.38,
            0.15,
            -0.11,
        )
    )
