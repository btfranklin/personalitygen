from __future__ import annotations

import random

from personalitygen.enums import LifeStage, PriorityLevel
from personalitygen.personality import (
    BigFiveConflictResolutionConfiguration,
    BigFiveConflictResolutionStyle,
    BigFivePersonality,
    BigFiveTraitConfiguration,
)
from personalitygen.traits import (
    BigFiveAgreeableness,
    BigFiveConscientiousness,
    BigFiveExtraversion,
    BigFiveNeuroticism,
    BigFiveOpenness,
)


CONCERN_BY_STYLE = {
    BigFiveConflictResolutionStyle.AVOIDING: (
        PriorityLevel.LOW,
        PriorityLevel.LOW,
    ),
    BigFiveConflictResolutionStyle.OBLIGING: (
        PriorityLevel.LOW,
        PriorityLevel.HIGH,
    ),
    BigFiveConflictResolutionStyle.INTEGRATING: (
        PriorityLevel.HIGH,
        PriorityLevel.HIGH,
    ),
    BigFiveConflictResolutionStyle.DOMINATING: (
        PriorityLevel.HIGH,
        PriorityLevel.LOW,
    ),
    BigFiveConflictResolutionStyle.COMPROMISING: (
        PriorityLevel.MODERATE,
        PriorityLevel.MODERATE,
    ),
}


class SequenceRandom:
    def __init__(self, uniforms: list[float]) -> None:
        self._uniforms = iter(uniforms)

    def uniform(self, a: float, b: float) -> float:
        value = next(self._uniforms)
        if not (a <= value <= b):
            raise AssertionError(f"Uniform value {value} not in {a}..{b}")
        return value

    def gauss(self, mu: float, sigma: float) -> float:
        raise AssertionError("gauss is not expected in this test")


def flat_trait_configuration(score: float) -> BigFiveTraitConfiguration:
    return BigFiveTraitConfiguration(
        openness=BigFiveOpenness(score, score, score),
        conscientiousness=BigFiveConscientiousness(score, score, score),
        extraversion=BigFiveExtraversion(score, score, score),
        agreeableness=BigFiveAgreeableness(score, score, score),
        neuroticism=BigFiveNeuroticism(score, score, score),
    )


def test_personality_random_configuration() -> None:
    rng = random.Random(7)
    personality = BigFivePersonality.random(LifeStage.ADULT, rng=rng)
    conflict = personality.conflict_resolution_configuration

    expected = CONCERN_BY_STYLE[conflict.conflict_resolution_style]
    assert (conflict.concern_for_self, conflict.concern_for_others) == expected


def test_full_personality_generation_is_deterministic_for_seed() -> None:
    rng_a = random.Random(7)
    rng_b = random.Random(7)

    assert BigFivePersonality.random(
        LifeStage.ADULT, rng=rng_a
    ) == BigFivePersonality.random(LifeStage.ADULT, rng=rng_b)


def test_conflict_configuration_derives_concerns_for_every_style() -> None:
    trait_configuration = flat_trait_configuration(0.0)
    rng = SequenceRandom([0.05, 0.15, 0.25, 0.35, 0.45])

    results = [
        BigFiveConflictResolutionConfiguration.random(
            trait_configuration, rng=rng
        )
        for _ in CONCERN_BY_STYLE
    ]

    assert [
        result.conflict_resolution_style for result in results
    ] == list(CONCERN_BY_STYLE)
    assert [
        (result.concern_for_self, result.concern_for_others)
        for result in results
    ] == list(CONCERN_BY_STYLE.values())


def test_conflict_style_floors_negative_weights() -> None:
    # All maxed scores make DOMINATING negative and COMPROMISING zero.
    trait_configuration = flat_trait_configuration(1.0)
    rng = SequenceRandom([1.25, 1.35])

    # Negative/zero levels are floored to allow rare counter-indicated picks.
    results = [
        BigFiveConflictResolutionStyle.random(
            trait_configuration, rng=rng
        )
        for _ in range(2)
    ]
    assert results == [
        BigFiveConflictResolutionStyle.DOMINATING,
        BigFiveConflictResolutionStyle.COMPROMISING,
    ]
