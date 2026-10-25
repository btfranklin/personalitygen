from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

import pytest

from personalitygen.enums import LifeStage
from personalitygen.personality import BigFiveTraits
from personalitygen.traits import (
    BigFiveAgreeableness,
    BigFiveConscientiousness,
    BigFiveExtraversion,
    BigFiveNeuroticism,
    BigFiveOpenness,
)


@dataclass(frozen=True, slots=True)
class TraitSpec:
    name: str
    make: Callable[[float, float, float], Trait]
    sample: Callable[[LifeStage], Trait]
    component_values: Callable[[Trait], tuple[float, float, float]]
    formatted_midpoint: str
    stage_direction: int


Trait = (
    BigFiveOpenness
    | BigFiveConscientiousness
    | BigFiveExtraversion
    | BigFiveAgreeableness
    | BigFiveNeuroticism
)


def openness_components(trait: Trait) -> tuple[float, float, float]:
    assert isinstance(trait, BigFiveOpenness)
    return (
        trait.aesthetic_sensitivity_score,
        trait.creative_imagination_score,
        trait.intellectual_curiosity_score,
    )


def conscientiousness_components(trait: Trait) -> tuple[float, float, float]:
    assert isinstance(trait, BigFiveConscientiousness)
    return (
        trait.organization_score,
        trait.responsibility_score,
        trait.productivity_score,
    )


def extraversion_components(trait: Trait) -> tuple[float, float, float]:
    assert isinstance(trait, BigFiveExtraversion)
    return trait.assertiveness_score, trait.sociability_score, trait.energy_level_score


def agreeableness_components(trait: Trait) -> tuple[float, float, float]:
    assert isinstance(trait, BigFiveAgreeableness)
    return trait.compassion_score, trait.respectfulness_score, trait.trust_score


def neuroticism_components(trait: Trait) -> tuple[float, float, float]:
    assert isinstance(trait, BigFiveNeuroticism)
    return trait.anxiety_score, trait.emotional_volatility_score, trait.depression_score


TRAIT_SPECS = (
    TraitSpec(
        name="openness",
        make=BigFiveOpenness,
        sample=lambda stage: BigFiveOpenness.random(stage, rng=MedianRandom()),
        component_values=openness_components,
        formatted_midpoint="0.5 {A:0.2 C:0.5 I:0.8}",
        stage_direction=-1,
    ),
    TraitSpec(
        name="conscientiousness",
        make=BigFiveConscientiousness,
        sample=lambda stage: BigFiveConscientiousness.random(stage, rng=MedianRandom()),
        component_values=conscientiousness_components,
        formatted_midpoint="0.5 {O:0.2 R:0.5 P:0.8}",
        stage_direction=1,
    ),
    TraitSpec(
        name="extraversion",
        make=BigFiveExtraversion,
        sample=lambda stage: BigFiveExtraversion.random(stage, rng=MedianRandom()),
        component_values=extraversion_components,
        formatted_midpoint="0.5 {A:0.2 S:0.5 E:0.8}",
        stage_direction=-1,
    ),
    TraitSpec(
        name="agreeableness",
        make=BigFiveAgreeableness,
        sample=lambda stage: BigFiveAgreeableness.random(stage, rng=MedianRandom()),
        component_values=agreeableness_components,
        formatted_midpoint="0.5 {C:0.2 R:0.5 T:0.8}",
        stage_direction=1,
    ),
    TraitSpec(
        name="neuroticism",
        make=BigFiveNeuroticism,
        sample=lambda stage: BigFiveNeuroticism.random(stage, rng=MedianRandom()),
        component_values=neuroticism_components,
        formatted_midpoint="0.5 {A:0.2 E:0.5 D:0.8}",
        stage_direction=-1,
    ),
)


def trait_spec_name(spec: object) -> str:
    assert isinstance(spec, TraitSpec)
    return spec.name


class MedianRandom:
    def uniform(self, a: float, b: float) -> float:
        if not (a <= 0.5 <= b):
            raise AssertionError(f"0.5 is outside requested range {a}..{b}")
        return 0.5


def make_trait(
    spec: TraitSpec,
    values: tuple[float, float, float] = (0.2, 0.5, 0.8),
) -> Trait:
    return spec.make(*values)


@pytest.mark.parametrize("spec", TRAIT_SPECS, ids=trait_spec_name)
def test_trait_score_is_component_average(spec: TraitSpec) -> None:
    trait = make_trait(spec, (0.1, 0.4, 1.0))

    assert trait.score == pytest.approx(0.5)


@pytest.mark.parametrize("spec", TRAIT_SPECS, ids=trait_spec_name)
@pytest.mark.parametrize("component_index", [0, 1, 2])
@pytest.mark.parametrize("invalid_value", [-0.001, 1.001])
def test_trait_components_must_stay_in_unit_range(
    spec: TraitSpec, component_index: int, invalid_value: float
) -> None:
    values = [0.5, 0.5, 0.5]
    values[component_index] = invalid_value

    with pytest.raises(
        ValueError,
        match="All trait components must be in the range 0.0...1.0",
    ):
        make_trait(spec, (values[0], values[1], values[2]))


@pytest.mark.parametrize("spec", TRAIT_SPECS, ids=trait_spec_name)
def test_trait_string_representation_exposes_score_and_components(
    spec: TraitSpec,
) -> None:
    assert str(make_trait(spec)) == spec.formatted_midpoint


@pytest.mark.parametrize("spec", TRAIT_SPECS, ids=trait_spec_name)
@pytest.mark.parametrize("life_stage", list(LifeStage))
def test_every_trait_samples_every_life_stage(
    spec: TraitSpec, life_stage: LifeStage
) -> None:
    trait = spec.sample(life_stage)
    values = spec.component_values(trait)

    assert all(0.01 <= value <= 1.0 for value in values)
    assert trait.score == pytest.approx(sum(values) / 3)


@pytest.mark.parametrize("spec", TRAIT_SPECS, ids=trait_spec_name)
def test_life_stage_bias_keeps_documented_direction(
    spec: TraitSpec,
) -> None:
    child = spec.component_values(spec.sample(LifeStage.CHILD))
    young_adult = spec.component_values(spec.sample(LifeStage.YOUNG_ADULT))
    adult = spec.component_values(spec.sample(LifeStage.ADULT))

    for child_value, young_adult_value, adult_value in zip(
        child, young_adult, adult, strict=True
    ):
        if spec.stage_direction > 0:
            assert child_value < young_adult_value < adult_value
        else:
            assert child_value > young_adult_value > adult_value


@pytest.mark.parametrize("spec", TRAIT_SPECS, ids=trait_spec_name)
def test_unknown_life_stage_is_rejected(spec: TraitSpec) -> None:
    with pytest.raises(ValueError, match="Unsupported life stage"):
        spec.sample("elder")  # type: ignore[arg-type]  # Test invalid runtime input.


@pytest.mark.parametrize("life_stage", list(LifeStage))
def test_traits_sample_all_big_five_traits(
    life_stage: LifeStage,
) -> None:
    traits = BigFiveTraits.random(
        life_stage, rng=MedianRandom()
    )

    assert isinstance(traits.openness, BigFiveOpenness)
    assert isinstance(traits.conscientiousness, BigFiveConscientiousness)
    assert isinstance(traits.extraversion, BigFiveExtraversion)
    assert isinstance(traits.agreeableness, BigFiveAgreeableness)
    assert isinstance(traits.neuroticism, BigFiveNeuroticism)
    assert all(
        0.01 <= trait.score <= 1.0
        for trait in (
            traits.openness,
            traits.conscientiousness,
            traits.extraversion,
            traits.agreeableness,
            traits.neuroticism,
        )
    )


def test_traits_are_deterministic_for_seed() -> None:
    rng_a = random.Random(123)
    rng_b = random.Random(123)

    traits_a = BigFiveTraits.random(LifeStage.ADULT, rng=rng_a)
    traits_b = BigFiveTraits.random(LifeStage.ADULT, rng=rng_b)

    assert traits_a == traits_b
