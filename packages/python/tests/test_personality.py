from __future__ import annotations

import random

import pytest

from personalitygen.enums import LifeStage, PriorityLevel
from personalitygen.personality import (
    BigFiveConflictResolution,
    BigFiveConflictResolutionStyle,
    BigFivePersonality,
    BigFiveTraits,
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


class UpperEndpointRandom:
    def uniform(self, a: float, b: float) -> float:
        return b


class InvalidRandom:
    def uniform(self, a: float, b: float) -> float:
        return float("nan")


def flat_traits(score: float) -> BigFiveTraits:
    return BigFiveTraits(
        openness=BigFiveOpenness(score, score, score),
        conscientiousness=BigFiveConscientiousness(score, score, score),
        extraversion=BigFiveExtraversion(score, score, score),
        agreeableness=BigFiveAgreeableness(score, score, score),
        neuroticism=BigFiveNeuroticism(score, score, score),
    )


def test_personality_random() -> None:
    rng = random.Random(7)
    personality = BigFivePersonality.random(LifeStage.ADULT, rng=rng)
    conflict = personality.conflict_resolution

    expected = CONCERN_BY_STYLE[conflict.style]
    assert (conflict.concern_for_self, conflict.concern_for_others) == expected


def test_full_personality_generation_is_deterministic_for_seed() -> None:
    rng_a = random.Random(7)
    rng_b = random.Random(7)

    assert BigFivePersonality.random(
        LifeStage.ADULT, rng=rng_a
    ) == BigFivePersonality.random(LifeStage.ADULT, rng=rng_b)


def test_conflict_resolution_derives_concerns_for_every_style() -> None:
    traits = flat_traits(0.0)
    rng = SequenceRandom([0.05, 0.15, 0.25, 0.35, 0.45])

    results = [
        BigFiveConflictResolution.random(traits, rng=rng)
        for _ in CONCERN_BY_STYLE
    ]

    assert [
        result.style for result in results
    ] == list(CONCERN_BY_STYLE)
    assert [
        (result.concern_for_self, result.concern_for_others)
        for result in results
    ] == list(CONCERN_BY_STYLE.values())


def test_conflict_resolution_derives_concerns_when_authored() -> None:
    resolutions = [
        BigFiveConflictResolution(style=style)
        for style in BigFiveConflictResolutionStyle
    ]

    assert [
        (resolution.concern_for_self, resolution.concern_for_others)
        for resolution in resolutions
    ] == list(CONCERN_BY_STYLE.values())


def test_conflict_style_floors_negative_weights() -> None:
    # All maxed scores make DOMINATING negative and COMPROMISING zero.
    traits = flat_traits(1.0)
    rng = SequenceRandom([1.25, 1.35])

    # Negative/zero levels are floored to allow rare counter-indicated picks.
    results = [
        BigFiveConflictResolutionStyle.random(
            traits, rng=rng
        )
        for _ in range(2)
    ]
    assert results == [
        BigFiveConflictResolutionStyle.DOMINATING,
        BigFiveConflictResolutionStyle.COMPROMISING,
    ]


def test_conflict_style_assigns_upper_endpoint_to_final_bucket() -> None:
    traits = flat_traits(1.0)

    assert BigFiveConflictResolutionStyle.random(
        traits,
        rng=UpperEndpointRandom(),
    ) is BigFiveConflictResolutionStyle.COMPROMISING


def test_conflict_style_rejects_invalid_uniform_draw() -> None:
    with pytest.raises(
        ValueError,
        match="RandomSource.uniform returned a value out of range",
    ):
        BigFiveConflictResolutionStyle.random(
            flat_traits(0.5),
            rng=InvalidRandom(),
        )
