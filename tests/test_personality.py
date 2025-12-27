import random

from personalitygen.enums import LifeStage, PriorityLevel
from personalitygen.personality import (
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


def test_personality_random_configuration() -> None:
    rng = random.Random(7)
    personality = BigFivePersonality.random(LifeStage.ADULT, rng=rng)
    conflict = personality.conflict_resolution_configuration

    mapping = {
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

    expected = mapping[conflict.conflict_resolution_style]
    assert (conflict.concern_for_self, conflict.concern_for_others) == expected


def test_conflict_style_clamps_negative_weights() -> None:
    # All maxed scores make DOMINATING negative and COMPROMISING zero.
    trait_configuration = BigFiveTraitConfiguration(
        openness=BigFiveOpenness(1.0, 1.0, 1.0),
        conscientiousness=BigFiveConscientiousness(1.0, 1.0, 1.0),
        extraversion=BigFiveExtraversion(1.0, 1.0, 1.0),
        agreeableness=BigFiveAgreeableness(1.0, 1.0, 1.0),
        neuroticism=BigFiveNeuroticism(1.0, 1.0, 1.0),
    )
    rng = SequenceRandom([0.1, 0.7, 1.1])

    # The only reachable styles are the ones with positive weights.
    results = [
        BigFiveConflictResolutionStyle.random(
            trait_configuration, rng=rng
        )
        for _ in range(3)
    ]
    assert results == [
        BigFiveConflictResolutionStyle.AVOIDING,
        BigFiveConflictResolutionStyle.OBLIGING,
        BigFiveConflictResolutionStyle.INTEGRATING,
    ]
    assert BigFiveConflictResolutionStyle.DOMINATING not in results
    assert BigFiveConflictResolutionStyle.COMPROMISING not in results


def test_conflict_style_falls_back_to_uniform_when_all_zero() -> None:
    # Zeroed trait scores mean every style weight is zero.
    trait_configuration = BigFiveTraitConfiguration(
        openness=BigFiveOpenness(0.0, 0.0, 0.0),
        conscientiousness=BigFiveConscientiousness(0.0, 0.0, 0.0),
        extraversion=BigFiveExtraversion(0.0, 0.0, 0.0),
        agreeableness=BigFiveAgreeableness(0.0, 0.0, 0.0),
        neuroticism=BigFiveNeuroticism(0.0, 0.0, 0.0),
    )
    rng = SequenceRandom([0.1, 1.1, 2.1, 3.1, 4.1])

    # Uniform fallback should step through the enum order predictably.
    results = [
        BigFiveConflictResolutionStyle.random(
            trait_configuration, rng=rng
        )
        for _ in range(5)
    ]
    assert results == [
        BigFiveConflictResolutionStyle.AVOIDING,
        BigFiveConflictResolutionStyle.OBLIGING,
        BigFiveConflictResolutionStyle.INTEGRATING,
        BigFiveConflictResolutionStyle.DOMINATING,
        BigFiveConflictResolutionStyle.COMPROMISING,
    ]
