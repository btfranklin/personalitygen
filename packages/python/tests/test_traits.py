from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

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
    trait_type: type[Any]
    component_names: tuple[str, str, str]
    formatted_midpoint: str
    stage_direction: int


TRAIT_SPECS = (
    TraitSpec(
        name="openness",
        trait_type=BigFiveOpenness,
        component_names=(
            "aesthetic_sensitivity_score",
            "creative_imagination_score",
            "intellectual_curiosity_score",
        ),
        formatted_midpoint="0.5 {A:0.2 C:0.5 I:0.8}",
        stage_direction=-1,
    ),
    TraitSpec(
        name="conscientiousness",
        trait_type=BigFiveConscientiousness,
        component_names=(
            "organization_score",
            "responsibility_score",
            "productivity_score",
        ),
        formatted_midpoint="0.5 {O:0.2 R:0.5 P:0.8}",
        stage_direction=1,
    ),
    TraitSpec(
        name="extraversion",
        trait_type=BigFiveExtraversion,
        component_names=(
            "assertiveness_score",
            "sociability_score",
            "energy_level_score",
        ),
        formatted_midpoint="0.5 {A:0.2 S:0.5 E:0.8}",
        stage_direction=-1,
    ),
    TraitSpec(
        name="agreeableness",
        trait_type=BigFiveAgreeableness,
        component_names=(
            "compassion_score",
            "respectfulness_score",
            "trust_score",
        ),
        formatted_midpoint="0.5 {C:0.2 R:0.5 T:0.8}",
        stage_direction=1,
    ),
    TraitSpec(
        name="neuroticism",
        trait_type=BigFiveNeuroticism,
        component_names=(
            "anxiety_score",
            "emotional_volatility_score",
            "depression_score",
        ),
        formatted_midpoint="0.5 {A:0.2 E:0.5 D:0.8}",
        stage_direction=-1,
    ),
)


class MedianRandom:
    def uniform(self, a: float, b: float) -> float:
        if not (a <= 0.5 <= b):
            raise AssertionError(f"0.5 is outside requested range {a}..{b}")
        return 0.5


def make_trait(
    spec: TraitSpec,
    values: tuple[float, float, float] = (0.2, 0.5, 0.8),
) -> Any:
    kwargs = dict(zip(spec.component_names, values, strict=True))
    return spec.trait_type(**kwargs)


def component_values(spec: TraitSpec, trait: Any) -> tuple[float, float, float]:
    return tuple(getattr(trait, name) for name in spec.component_names)


@pytest.mark.parametrize("spec", TRAIT_SPECS, ids=lambda spec: spec.name)
def test_trait_score_is_component_average(spec: TraitSpec) -> None:
    trait = make_trait(spec, (0.1, 0.4, 1.0))

    assert trait.score == pytest.approx(0.5)


@pytest.mark.parametrize("spec", TRAIT_SPECS, ids=lambda spec: spec.name)
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
        make_trait(spec, tuple(values))


@pytest.mark.parametrize("spec", TRAIT_SPECS, ids=lambda spec: spec.name)
def test_trait_string_representation_exposes_score_and_components(
    spec: TraitSpec,
) -> None:
    assert str(make_trait(spec)) == spec.formatted_midpoint


@pytest.mark.parametrize("spec", TRAIT_SPECS, ids=lambda spec: spec.name)
@pytest.mark.parametrize("life_stage", list(LifeStage))
def test_every_trait_samples_every_life_stage(
    spec: TraitSpec, life_stage: LifeStage
) -> None:
    trait = spec.trait_type.random(life_stage, rng=MedianRandom())
    values = component_values(spec, trait)

    assert all(0.01 <= value <= 1.0 for value in values)
    assert trait.score == pytest.approx(sum(values) / 3)


@pytest.mark.parametrize("spec", TRAIT_SPECS, ids=lambda spec: spec.name)
def test_life_stage_bias_keeps_documented_direction(
    spec: TraitSpec,
) -> None:
    child = component_values(
        spec,
        spec.trait_type.random(LifeStage.CHILD, rng=MedianRandom()),
    )
    young_adult = component_values(
        spec,
        spec.trait_type.random(LifeStage.YOUNG_ADULT, rng=MedianRandom()),
    )
    adult = component_values(
        spec,
        spec.trait_type.random(LifeStage.ADULT, rng=MedianRandom()),
    )

    for child_value, young_adult_value, adult_value in zip(
        child, young_adult, adult, strict=True
    ):
        if spec.stage_direction > 0:
            assert child_value < young_adult_value < adult_value
        else:
            assert child_value > young_adult_value > adult_value


@pytest.mark.parametrize("spec", TRAIT_SPECS, ids=lambda spec: spec.name)
def test_unknown_life_stage_is_rejected(spec: TraitSpec) -> None:
    with pytest.raises(ValueError, match="Unsupported life stage"):
        spec.trait_type.random("elder", rng=MedianRandom())


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
