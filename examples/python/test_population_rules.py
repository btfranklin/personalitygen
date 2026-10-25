"""Check complete repeatability, report totals, and component filtering."""

from dataclasses import replace
from statistics import mean

import pytest
from personalitygen import BigFiveConflictResolutionStyle, LifeStage

from population_rules import (
    generate_population,
    select_cases,
    style_counts,
    trait_means,
)


def test_population_is_repeatable_and_covers_each_stage() -> None:
    population = generate_population(seed=42, per_stage=3)
    repeated = generate_population(seed=42, per_stage=3)
    assert population == repeated
    assert population != generate_population(seed=43, per_stage=3)
    assert len({p.identifier for p in population}) == len(population) == 9
    for stage in LifeStage:
        assert sum(p.life_stage == stage for p in population) == 3
    counts = style_counts(population)
    assert set(counts) == {style.value for style in BigFiveConflictResolutionStyle}
    assert sum(counts.values()) == len(population)
    assert trait_means(population)["openness"] == pytest.approx(
        mean(p.personality.traits.openness.score for p in population)
    )


def test_filter_uses_components_instead_of_aggregate_scores() -> None:
    person = generate_population(per_stage=1)[0]
    traits = person.personality.traits
    traits = replace(
        traits,
        neuroticism=replace(
            traits.neuroticism,
            anxiety_score=0.7,
            emotional_volatility_score=0.1,
            depression_score=0.1,
        ),
        extraversion=replace(
            traits.extraversion,
            sociability_score=0.3,
            assertiveness_score=0.9,
            energy_level_score=0.9,
        ),
    )
    selected = replace(person, personality=replace(person.personality, traits=traits))
    assert traits.neuroticism.score < 0.7
    assert traits.extraversion.score > 0.3
    assert select_cases([selected]) == {
        "high anxiety": [selected],
        "low sociability": [selected],
    }
    assert select_cases([]) == {"high anxiety": [], "low sociability": []}


def test_report_handles_missing_styles_and_rejects_invalid_sizes() -> None:
    assert all(count == 0 for count in style_counts([]).values())
    for size in (0, -1):
        with pytest.raises(ValueError, match="at least one"):
            generate_population(per_stage=size)
    with pytest.raises(ValueError, match="empty"):
        trait_means([])
