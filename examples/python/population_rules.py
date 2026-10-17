"""Population generation, reports, and case selection for this example."""

import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean

from personalitygen import (
    BigFiveConflictResolutionStyle,
    BigFivePersonality,
    LifeStage,
)


TRAIT_NAMES = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)


@dataclass(frozen=True)
class TestPersona:
    identifier: str
    life_stage: LifeStage
    personality: BigFivePersonality


def generate_population(*, seed: int = 42, per_stage: int = 30) -> list[TestPersona]:
    """Use a fresh random source for each complete generation run."""
    if per_stage < 1:
        raise ValueError("Each life stage needs at least one persona")
    rng = random.Random(seed)
    return [
        TestPersona(
            f"{stage.value}-{index + 1:02d}",
            stage,
            BigFivePersonality.random(stage, rng=rng),
        )
        for stage in LifeStage
        for index in range(per_stage)
    ]


def trait_means(population: Sequence[TestPersona]) -> dict[str, float]:
    """Report all five aggregate trait means for a supplied group."""
    if not population:
        raise ValueError("Cannot report means for an empty population")
    return {
        name: mean(
            getattr(person.personality.traits, name).score for person in population
        )
        for name in TRAIT_NAMES
    }


def style_counts(population: Sequence[TestPersona]) -> dict[str, int]:
    """Include every style, including styles with no members."""
    counts = Counter(
        person.personality.conflict_resolution.style for person in population
    )
    return {style.value: counts[style] for style in BigFiveConflictResolutionStyle}


def select_cases(population: Sequence[TestPersona]) -> dict[str, list[TestPersona]]:
    """Filter component scores with example-owned test thresholds."""
    return {
        "high anxiety": [
            person
            for person in population
            if person.personality.traits.neuroticism.anxiety_score >= 0.7
        ],
        "low sociability": [
            person
            for person in population
            if person.personality.traits.extraversion.sociability_score <= 0.3
        ],
    }
