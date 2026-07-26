"""Top-level personality configuration models."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
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
    if any(
        not math.isfinite(weight) or weight < 0.0
        for weight in weights.values()
    ):
        raise ValueError("weights must be finite and non-negative")
    source = rng if rng is not None else random
    total = sum(weights.values())
    if total <= 0.0:
        weights = {style: 1.0 for style in weights}
        total = float(len(weights))

    threshold = source.uniform(0.0, total)
    if (
        not math.isfinite(threshold)
        or threshold < 0.0
        or threshold > total
    ):
        raise ValueError("RandomSource.uniform returned a value out of range")
    for style, weight in weights.items():
        threshold -= weight
        if threshold <= 0.0:
            return style
    return next(reversed(weights))


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
        traits: BigFiveTraits,
        *,
        rng: RandomSource | None = None,
    ) -> Self:
        return _weighted_choice(
            _conflict_style_weights(traits),
            rng=rng,
        )


@dataclass(frozen=True, slots=True)
class BigFiveTraits:
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
    traits: BigFiveTraits,
) -> dict[BigFiveConflictResolutionStyle, float]:
    trait_scores = {
        "openness": traits.openness.score,
        "conscientiousness": traits.conscientiousness.score,
        "extraversion": traits.extraversion.score,
        "agreeableness": traits.agreeableness.score,
        "neuroticism": traits.neuroticism.score,
    }
    return {
        style: max(
            sum(
                trait_scores[trait_name] * coefficient
                for trait_name, coefficient in coefficients.items()
            ),
            _CONFLICT_MINIMUM_WEIGHT,
        )
        for style, coefficients in _CONFLICT_STYLE_COEFFICIENTS.items()
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
class BigFiveConflictResolution:
    style: BigFiveConflictResolutionStyle
    concern_for_self: PriorityLevel = field(init=False)
    concern_for_others: PriorityLevel = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(
            self.style,
            BigFiveConflictResolutionStyle,
        ):
            raise TypeError(
                "style must be a BigFiveConflictResolutionStyle"
            )
        concern_for_self, concern_for_others = _STYLE_TO_CONCERNS[
            self.style
        ]
        object.__setattr__(self, "concern_for_self", concern_for_self)
        object.__setattr__(self, "concern_for_others", concern_for_others)

    @classmethod
    def random(
        cls,
        traits: BigFiveTraits,
        *,
        rng: RandomSource | None = None,
    ) -> Self:
        style = BigFiveConflictResolutionStyle.random(
            traits, rng=rng
        )
        return cls(style=style)


@dataclass(frozen=True, slots=True)
class BigFivePersonality:
    traits: BigFiveTraits
    conflict_resolution: BigFiveConflictResolution

    @classmethod
    def random(
        cls, life_stage: LifeStage, *, rng: RandomSource | None = None
    ) -> Self:
        traits = BigFiveTraits.random(life_stage, rng=rng)
        conflict_resolution = BigFiveConflictResolution.random(
            traits, rng=rng
        )
        return cls(
            traits=traits,
            conflict_resolution=conflict_resolution,
        )
