import random

import pytest

from personalitygen.constants import UNIT_RANGE_MAX, UNIT_RANGE_MIN
from personalitygen.enums import LifeStage
from personalitygen.traits import (
    BigFiveAgreeableness,
    BigFiveConscientiousness,
    BigFiveExtraversion,
    BigFiveNeuroticism,
    BigFiveOpenness,
)


class MeanRandom:
    def uniform(self, a: float, b: float) -> float:
        return 0.5

    def gauss(self, mu: float, sigma: float) -> float:
        raise AssertionError("gauss is not expected in this test")


class LowerBoundRandom:
    def uniform(self, a: float, b: float) -> float:
        return a

    def gauss(self, mu: float, sigma: float) -> float:
        raise AssertionError("gauss is not expected in this test")


def _assert_unit_range(value: float) -> None:
    assert UNIT_RANGE_MIN <= value <= UNIT_RANGE_MAX


def test_openness_score() -> None:
    trait = BigFiveOpenness(1.0, 1.0, 1.0)
    assert trait.score == pytest.approx(1.0)
    trait = BigFiveOpenness(0.0, 0.0, 0.0)
    assert trait.score == pytest.approx(0.0)
    trait = BigFiveOpenness(0.5, 0.5, 0.5)
    assert trait.score == pytest.approx(0.5, abs=0.01)


def test_conscientiousness_score() -> None:
    trait = BigFiveConscientiousness(1.0, 1.0, 1.0)
    assert trait.score == pytest.approx(1.0)
    trait = BigFiveConscientiousness(0.0, 0.0, 0.0)
    assert trait.score == pytest.approx(0.0)
    trait = BigFiveConscientiousness(0.5, 0.5, 0.5)
    assert trait.score == pytest.approx(0.5, abs=0.01)


def test_extraversion_score() -> None:
    trait = BigFiveExtraversion(1.0, 1.0, 1.0)
    assert trait.score == pytest.approx(1.0)
    trait = BigFiveExtraversion(0.0, 0.0, 0.0)
    assert trait.score == pytest.approx(0.0)
    trait = BigFiveExtraversion(0.5, 0.5, 0.5)
    assert trait.score == pytest.approx(0.5, abs=0.01)


def test_agreeableness_score() -> None:
    trait = BigFiveAgreeableness(1.0, 1.0, 1.0)
    assert trait.score == pytest.approx(1.0)
    trait = BigFiveAgreeableness(0.0, 0.0, 0.0)
    assert trait.score == pytest.approx(0.0)
    trait = BigFiveAgreeableness(0.5, 0.5, 0.5)
    assert trait.score == pytest.approx(0.5, abs=0.01)


def test_neuroticism_score() -> None:
    trait = BigFiveNeuroticism(1.0, 1.0, 1.0)
    assert trait.score == pytest.approx(1.0)
    trait = BigFiveNeuroticism(0.0, 0.0, 0.0)
    assert trait.score == pytest.approx(0.0)
    trait = BigFiveNeuroticism(0.5, 0.5, 0.5)
    assert trait.score == pytest.approx(0.5, abs=0.01)


def test_random_traits_are_in_range() -> None:
    rng = random.Random(42)
    for stage in LifeStage:
        openness = BigFiveOpenness.random(stage, rng=rng)
        _assert_unit_range(openness.score)
        _assert_unit_range(openness.aesthetic_sensitivity_score)
        _assert_unit_range(openness.creative_imagination_score)
        _assert_unit_range(openness.intellectual_curiosity_score)

        conscientiousness = BigFiveConscientiousness.random(stage, rng=rng)
        _assert_unit_range(conscientiousness.score)
        _assert_unit_range(conscientiousness.organization_score)
        _assert_unit_range(conscientiousness.responsibility_score)
        _assert_unit_range(conscientiousness.productivity_score)

        extraversion = BigFiveExtraversion.random(stage, rng=rng)
        _assert_unit_range(extraversion.score)
        _assert_unit_range(extraversion.assertiveness_score)
        _assert_unit_range(extraversion.sociability_score)
        _assert_unit_range(extraversion.energy_level_score)

        agreeableness = BigFiveAgreeableness.random(stage, rng=rng)
        _assert_unit_range(agreeableness.score)
        _assert_unit_range(agreeableness.compassion_score)
        _assert_unit_range(agreeableness.respectfulness_score)
        _assert_unit_range(agreeableness.trust_score)

        neuroticism = BigFiveNeuroticism.random(stage, rng=rng)
        _assert_unit_range(neuroticism.score)
        _assert_unit_range(neuroticism.anxiety_score)
        _assert_unit_range(neuroticism.emotional_volatility_score)
        _assert_unit_range(neuroticism.depression_score)


def test_random_traits_match_configured_means() -> None:
    # MeanRandom returns the midpoint CDF, which maps back to the mean.
    rng = MeanRandom()
    cases: list[
        tuple[
            type,
            tuple[str, str, str],
            dict[LifeStage, tuple[float, float, float]],
        ]
    ] = [
        (
            BigFiveOpenness,
            (
                "aesthetic_sensitivity_score",
                "creative_imagination_score",
                "intellectual_curiosity_score",
            ),
            {
                LifeStage.CHILD: (0.80, 0.85, 0.85),
                LifeStage.YOUNG_ADULT: (0.70, 0.75, 0.75),
                LifeStage.ADULT: (0.60, 0.65, 0.65),
            },
        ),
        (
            BigFiveConscientiousness,
            (
                "organization_score",
                "responsibility_score",
                "productivity_score",
            ),
            {
                LifeStage.CHILD: (0.50, 0.55, 0.50),
                LifeStage.YOUNG_ADULT: (0.60, 0.65, 0.60),
                LifeStage.ADULT: (0.70, 0.75, 0.70),
            },
        ),
        (
            BigFiveExtraversion,
            (
                "assertiveness_score",
                "sociability_score",
                "energy_level_score",
            ),
            {
                LifeStage.CHILD: (0.72, 0.70, 0.72),
                LifeStage.YOUNG_ADULT: (0.62, 0.60, 0.62),
                LifeStage.ADULT: (0.52, 0.50, 0.52),
            },
        ),
        (
            BigFiveAgreeableness,
            (
                "compassion_score",
                "respectfulness_score",
                "trust_score",
            ),
            {
                LifeStage.CHILD: (0.55, 0.55, 0.40),
                LifeStage.YOUNG_ADULT: (0.65, 0.65, 0.50),
                LifeStage.ADULT: (0.75, 0.75, 0.60),
            },
        ),
        (
            BigFiveNeuroticism,
            (
                "anxiety_score",
                "emotional_volatility_score",
                "depression_score",
            ),
            {
                LifeStage.CHILD: (0.70, 0.60, 0.55),
                LifeStage.YOUNG_ADULT: (0.60, 0.50, 0.45),
                LifeStage.ADULT: (0.50, 0.40, 0.35),
            },
        ),
    ]

    for trait_class, attributes, expected_means in cases:
        for stage, means in expected_means.items():
            # Deterministic draws verify the data-driven life-stage mapping.
            trait = trait_class.random(stage, rng=rng)
            for attribute, mean in zip(attributes, means):
                assert getattr(trait, attribute) == pytest.approx(mean)
            assert trait.score == pytest.approx(sum(means) / 3, abs=1e-4)


def test_random_trait_respects_nonzero_minimum() -> None:
    # For a truncated normal, the lower CDF bound maps to the minimum value.
    rng = LowerBoundRandom()
    trait = BigFiveOpenness.random(LifeStage.CHILD, rng=rng)

    assert trait.aesthetic_sensitivity_score == pytest.approx(0.01)
    assert trait.creative_imagination_score == pytest.approx(0.01)
    assert trait.intellectual_curiosity_score == pytest.approx(0.01)


def test_trait_component_range_validation() -> None:
    # Any component outside 0.0..1.0 should raise immediately.
    def invalid_openness() -> BigFiveOpenness:
        return BigFiveOpenness(-0.1, 0.5, 0.5)

    def invalid_conscientiousness() -> BigFiveConscientiousness:
        return BigFiveConscientiousness(0.5, 1.1, 0.5)

    def invalid_extraversion() -> BigFiveExtraversion:
        return BigFiveExtraversion(0.5, 0.5, -0.1)

    def invalid_agreeableness() -> BigFiveAgreeableness:
        return BigFiveAgreeableness(1.1, 0.5, 0.5)

    def invalid_neuroticism() -> BigFiveNeuroticism:
        return BigFiveNeuroticism(0.5, 0.5, 1.5)

    invalid_constructors = [
        invalid_openness,
        invalid_conscientiousness,
        invalid_extraversion,
        invalid_agreeableness,
        invalid_neuroticism,
    ]
    for constructor in invalid_constructors:
        with pytest.raises(ValueError):
            constructor()
