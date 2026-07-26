"""Top-level personality configuration models."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Self

from personalitygen.enums import LifeStage, PriorityLevel
from personalitygen.randomness import RandomSource
from personalitygen.traits import (
    BigFiveAgreeableness,
    BigFiveConscientiousness,
    BigFiveExtraversion,
    BigFiveNeuroticism,
    BigFiveOpenness,
)


def _weighted_choice(
    weights: dict["BigFiveConflictResolutionStyle", float],
    *,
    rng: RandomSource | None = None,
) -> "BigFiveConflictResolutionStyle":
    if not weights:
        raise ValueError("weights must be non-empty")
    if any(weight < 0.0 for weight in weights.values()):
        raise ValueError("weights must be non-negative")
    source = rng if rng is not None else random
    total = sum(weights.values())
    if total <= 0.0:
        weights = {style: 1.0 for style in weights}
        total = float(len(weights))

    threshold = source.uniform(0.0, total)
    for style, weight in weights.items():
        threshold -= weight
        if threshold <= 0.0:
            return style
    return next(iter(weights))


class BigFiveConflictResolutionStyle(str, Enum):
    # Concern for self: low. Concern for others: low. Tries to avoid conflict.
    AVOIDING = "avoiding"
    # Concern for self: low. Concern for others: high. Accommodates others.
    OBLIGING = "obliging"
    # Concern for self: high. Concern for others: high. Collaborates.
    INTEGRATING = "integrating"
    # Concern for self: high. Concern for others: low. Competes to win.
    DOMINATING = "dominating"
    # Concern for self: moderate. Concern for others: moderate. Trades off.
    COMPROMISING = "compromising"

    @classmethod
    def random(
        cls,
        trait_configuration: BigFiveTraitConfiguration,
        *,
        rng: RandomSource | None = None,
    ) -> Self:
        return _weighted_choice(
            _conflict_style_weights(trait_configuration),
            rng=rng,
        )


@dataclass(frozen=True, slots=True)
class BigFiveTraitConfiguration:
    # Appreciation for art, emotion, adventure, and curiosity.
    # Opposite: closedness.
    openness: BigFiveOpenness
    # Self-discipline, dutifulness, and achievement orientation.
    # Opposite: undisciplined.
    conscientiousness: BigFiveConscientiousness
    # Energy, sociability, and stimulation-seeking.
    # Opposite: introversion.
    extraversion: BigFiveExtraversion
    # Compassion and cooperation toward others.
    # Opposite: antagonism.
    agreeableness: BigFiveAgreeableness
    # Tendency toward unpleasant emotions and instability.
    # Opposite: emotional stability.
    neuroticism: BigFiveNeuroticism

    @classmethod
    def random(
        cls, life_stage: LifeStage, *, rng: RandomSource | None = None
    ) -> Self:
        return cls(
            openness=BigFiveOpenness.random(life_stage, rng=rng),
            conscientiousness=BigFiveConscientiousness.random(
                life_stage, rng=rng
            ),
            extraversion=BigFiveExtraversion.random(life_stage, rng=rng),
            agreeableness=BigFiveAgreeableness.random(life_stage, rng=rng),
            neuroticism=BigFiveNeuroticism.random(life_stage, rng=rng),
        )

    def __str__(self) -> str:
        return (
            "openness: "
            f"{self.openness}\n"
            "conscientiousness: "
            f"{self.conscientiousness}\n"
            "extraversion: "
            f"{self.extraversion}\n"
            "agreeableness: "
            f"{self.agreeableness}\n"
            "neuroticism: "
            f"{self.neuroticism}"
        )


# These weights are loosely based on:
# Priyadarshini, S. (2017). Effect of Personality on Conflict Resolution
# Styles. IRA-International Journal of Management & Social Sciences, 7(2),
# 196-207.
_CONFLICT_STYLE_COEFFICIENTS: dict[
    BigFiveConflictResolutionStyle, dict[str, float]
] = {
    BigFiveConflictResolutionStyle.AVOIDING: {
        "neuroticism": 0.7,
        "openness": -0.1,
        "agreeableness": 0.2,
        "conscientiousness": -0.2,
        "extraversion": 0.0,
    },
    BigFiveConflictResolutionStyle.OBLIGING: {
        "neuroticism": 0.2,
        "extraversion": -0.2,
        "openness": -0.1,
        "agreeableness": 0.3,
        "conscientiousness": 0.0,
    },
    BigFiveConflictResolutionStyle.INTEGRATING: {
        "openness": 0.1,
        "agreeableness": 0.2,
        "conscientiousness": 0.1,
        "neuroticism": 0.0,
        "extraversion": 0.0,
    },
    BigFiveConflictResolutionStyle.DOMINATING: {
        "neuroticism": -0.2,
        "extraversion": 0.2,
        "openness": -0.2,
        "agreeableness": -0.4,
        "conscientiousness": 0.2,
    },
    BigFiveConflictResolutionStyle.COMPROMISING: {
        "neuroticism": 0.1,
        "extraversion": 0.1,
        "conscientiousness": -0.2,
        "openness": 0.0,
        "agreeableness": 0.0,
    },
}
_CONFLICT_MINIMUM_WEIGHT = 0.1


def _conflict_style_weights(
    trait_configuration: BigFiveTraitConfiguration,
) -> dict[BigFiveConflictResolutionStyle, float]:
    trait_scores = {
        "openness": trait_configuration.openness.score,
        "conscientiousness": trait_configuration.conscientiousness.score,
        "extraversion": trait_configuration.extraversion.score,
        "agreeableness": trait_configuration.agreeableness.score,
        "neuroticism": trait_configuration.neuroticism.score,
    }
    coefficients = _CONFLICT_STYLE_COEFFICIENTS
    avoiding = coefficients[BigFiveConflictResolutionStyle.AVOIDING]
    obliging = coefficients[BigFiveConflictResolutionStyle.OBLIGING]
    integrating = coefficients[BigFiveConflictResolutionStyle.INTEGRATING]
    dominating = coefficients[BigFiveConflictResolutionStyle.DOMINATING]
    compromising = coefficients[BigFiveConflictResolutionStyle.COMPROMISING]
    style_levels = {
        BigFiveConflictResolutionStyle.AVOIDING: (
            trait_scores["neuroticism"] * avoiding["neuroticism"]
            + trait_scores["openness"] * avoiding["openness"]
            + trait_scores["agreeableness"] * avoiding["agreeableness"]
            + trait_scores["conscientiousness"]
            * avoiding["conscientiousness"]
        ),
        BigFiveConflictResolutionStyle.OBLIGING: (
            trait_scores["neuroticism"] * obliging["neuroticism"]
            + trait_scores["extraversion"] * obliging["extraversion"]
            + trait_scores["openness"] * obliging["openness"]
            + trait_scores["agreeableness"] * obliging["agreeableness"]
        ),
        BigFiveConflictResolutionStyle.INTEGRATING: (
            trait_scores["openness"] * integrating["openness"]
            + trait_scores["agreeableness"] * integrating["agreeableness"]
            + trait_scores["conscientiousness"]
            * integrating["conscientiousness"]
        ),
        BigFiveConflictResolutionStyle.DOMINATING: (
            trait_scores["neuroticism"] * dominating["neuroticism"]
            + trait_scores["extraversion"] * dominating["extraversion"]
            + trait_scores["openness"] * dominating["openness"]
            + trait_scores["agreeableness"] * dominating["agreeableness"]
            + trait_scores["conscientiousness"]
            * dominating["conscientiousness"]
        ),
        BigFiveConflictResolutionStyle.COMPROMISING: (
            trait_scores["neuroticism"] * compromising["neuroticism"]
            + trait_scores["extraversion"] * compromising["extraversion"]
            + trait_scores["conscientiousness"]
            * compromising["conscientiousness"]
        ),
    }
    return {
        style: max(level, _CONFLICT_MINIMUM_WEIGHT)
        for style, level in style_levels.items()
    }


_STYLE_TO_CONCERNS: dict[
    BigFiveConflictResolutionStyle, tuple[PriorityLevel, PriorityLevel]
] = {
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


def _validate_style_concerns() -> None:
    expected = set(BigFiveConflictResolutionStyle)
    actual = set(_STYLE_TO_CONCERNS)
    if expected != actual:
        missing = {style.value for style in expected - actual}
        extra = {style.value for style in actual - expected}
        raise ValueError(
            "Conflict resolution styles and concern mapping are out of sync. "
            f"Missing: {sorted(missing)}. Extra: {sorted(extra)}."
        )


_validate_style_concerns()


@dataclass(frozen=True, slots=True)
class BigFiveConflictResolutionConfiguration:
    conflict_resolution_style: BigFiveConflictResolutionStyle
    concern_for_self: PriorityLevel
    concern_for_others: PriorityLevel

    @classmethod
    def random(
        cls,
        trait_configuration: BigFiveTraitConfiguration,
        *,
        rng: RandomSource | None = None,
    ) -> Self:
        style = BigFiveConflictResolutionStyle.random(
            trait_configuration, rng=rng
        )
        concern_for_self, concern_for_others = _STYLE_TO_CONCERNS[style]
        return cls(
            conflict_resolution_style=style,
            concern_for_self=concern_for_self,
            concern_for_others=concern_for_others,
        )


@dataclass(frozen=True, slots=True)
class BigFivePersonality:
    trait_configuration: BigFiveTraitConfiguration
    conflict_resolution_configuration: BigFiveConflictResolutionConfiguration

    @classmethod
    def random(
        cls, life_stage: LifeStage, *, rng: RandomSource | None = None
    ) -> Self:
        trait_configuration = BigFiveTraitConfiguration.random(
            life_stage, rng=rng
        )
        conflict_configuration = BigFiveConflictResolutionConfiguration.random(
            trait_configuration, rng=rng
        )
        return cls(
            trait_configuration=trait_configuration,
            conflict_resolution_configuration=conflict_configuration,
        )
