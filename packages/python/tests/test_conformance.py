from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from personalitygen import (
    AdaptiveBifurcatedProfile,
    BigFiveAgreeableness,
    BigFiveConflictResolutionStyle,
    BigFiveConscientiousness,
    BigFiveExtraversion,
    BigFiveNeuroticism,
    BigFiveOpenness,
    BigFiveTraitConfiguration,
    LifeStage,
    PriorityLevel,
)
from personalitygen._scoring import (
    SIGNED_RANGE_MAX,
    SIGNED_RANGE_MIN,
    component_average_score,
    validate_signed_range,
    validate_unit_range,
)
from personalitygen.adaptive import (
    _ABBF_RANDOM_STDDEV,
    _ABBF_PROJECTION_COEFFICIENTS,
    _AXIS_DEFINITIONS,
)
from personalitygen.constants import UNIT_RANGE_MAX, UNIT_RANGE_MIN
from personalitygen.personality import (
    _CONFLICT_MINIMUM_WEIGHT,
    _CONFLICT_STYLE_COEFFICIENTS,
    _STYLE_TO_CONCERNS,
    _conflict_style_weights,
)
from personalitygen.randomness import _CDF_EPSILON, random_gaussian
from personalitygen.traits import (
    _AGREEABLENESS_CONFIG,
    _CONSCIENTIOUSNESS_CONFIG,
    _EXTRAVERSION_CONFIG,
    _NEUROTICISM_CONFIG,
    _OPENNESS_CONFIG,
    _TRAIT_SAMPLE_MIN,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SPEC_ROOT = REPOSITORY_ROOT / "spec"
CONFORMANCE_ROOT = SPEC_ROOT / "conformance"
EXPECTED_FIXTURES = {
    "adaptive.json",
    "aggregate-scoring.json",
    "conflict-resolution.json",
    "gaussian-sampling.json",
    "life-stage-sampling.json",
    "validation.json",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


MODEL = load_json(SPEC_ROOT / "model.json")
ARITHMETIC_TOLERANCE = MODEL["tolerances"]["arithmetic"]
GAUSSIAN_TOLERANCE = MODEL["tolerances"]["gaussian"]


class FractionRandom:
    def __init__(self, *fractions: float) -> None:
        self._fractions: Iterator[float] = iter(fractions)

    def uniform(self, minimum: float, maximum: float) -> float:
        fraction = next(self._fractions)
        return minimum + ((maximum - minimum) * fraction)


def flat_traits(scores: dict[str, float]) -> BigFiveTraitConfiguration:
    return BigFiveTraitConfiguration(
        openness=BigFiveOpenness(*(scores["openness"],) * 3),
        conscientiousness=BigFiveConscientiousness(
            *(scores["conscientiousness"],) * 3
        ),
        extraversion=BigFiveExtraversion(*(scores["extraversion"],) * 3),
        agreeableness=BigFiveAgreeableness(*(scores["agreeableness"],) * 3),
        neuroticism=BigFiveNeuroticism(*(scores["neuroticism"],) * 3),
    )


def trait_components(
    traits: BigFiveTraitConfiguration,
) -> dict[str, tuple[float, float, float]]:
    return {
        "openness": (
            traits.openness.aesthetic_sensitivity_score,
            traits.openness.creative_imagination_score,
            traits.openness.intellectual_curiosity_score,
        ),
        "conscientiousness": (
            traits.conscientiousness.organization_score,
            traits.conscientiousness.responsibility_score,
            traits.conscientiousness.productivity_score,
        ),
        "extraversion": (
            traits.extraversion.assertiveness_score,
            traits.extraversion.sociability_score,
            traits.extraversion.energy_level_score,
        ),
        "agreeableness": (
            traits.agreeableness.compassion_score,
            traits.agreeableness.respectfulness_score,
            traits.agreeableness.trust_score,
        ),
        "neuroticism": (
            traits.neuroticism.anxiety_score,
            traits.neuroticism.emotional_volatility_score,
            traits.neuroticism.depression_score,
        ),
    }


def test_conformance_fixture_set_is_explicit() -> None:
    actual = {path.name for path in CONFORMANCE_ROOT.glob("*.json")}
    assert actual == EXPECTED_FIXTURES


def test_model_matches_python_configuration() -> None:
    assert MODEL["schemaVersion"] == 1
    assert MODEL["ranges"]["unit"] == [UNIT_RANGE_MIN, UNIT_RANGE_MAX]
    assert MODEL["ranges"]["traitSample"] == [
        _TRAIT_SAMPLE_MIN,
        UNIT_RANGE_MAX,
    ]
    assert MODEL["ranges"]["signed"] == [
        SIGNED_RANGE_MIN,
        SIGNED_RANGE_MAX,
    ]
    assert MODEL["randomness"] == {
        "cdfEpsilon": _CDF_EPSILON,
        "abbfStandardDeviation": _ABBF_RANDOM_STDDEV,
    }
    assert MODEL["lifeStages"] == [stage.value for stage in LifeStage]
    assert MODEL["priorityLevels"] == [level.value for level in PriorityLevel]

    trait_configs = {
        "openness": _OPENNESS_CONFIG,
        "conscientiousness": _CONSCIENTIOUSNESS_CONFIG,
        "extraversion": _EXTRAVERSION_CONFIG,
        "agreeableness": _AGREEABLENESS_CONFIG,
        "neuroticism": _NEUROTICISM_CONFIG,
    }
    component_names = {
        "openness": [
            "aesthetic_sensitivity",
            "creative_imagination",
            "intellectual_curiosity",
        ],
        "conscientiousness": [
            "organization",
            "responsibility",
            "productivity",
        ],
        "extraversion": [
            "assertiveness",
            "sociability",
            "energy_level",
        ],
        "agreeableness": ["compassion", "respectfulness", "trust"],
        "neuroticism": ["anxiety", "emotional_volatility", "depression"],
    }
    for trait_name, config in trait_configs.items():
        expected = MODEL["traits"][trait_name]
        assert expected["components"] == component_names[trait_name]
        assert config.stddev == expected["standardDeviation"]
        assert {
            stage.value: list(means)
            for stage, means in config.means_by_stage.items()
        } == expected["means"]

    conflict_model = MODEL["conflictResolution"]
    assert conflict_model["minimumWeight"] == _CONFLICT_MINIMUM_WEIGHT
    assert list(conflict_model["styles"]) == [
        style.value for style in BigFiveConflictResolutionStyle
    ]
    for style, coefficients in _CONFLICT_STYLE_COEFFICIENTS.items():
        expected = conflict_model["styles"][style.value]
        assert coefficients == expected["coefficients"]
        assert [priority.value for priority in _STYLE_TO_CONCERNS[style]] == (
            expected["concerns"]
        )

    assert [
        {
            "domain": domain.value,
            "positivePole": positive.value,
            "negativePole": negative.value,
        }
        for _, domain, positive, negative in _AXIS_DEFINITIONS
    ] == MODEL["adaptive"]["axes"]
    assert _ABBF_PROJECTION_COEFFICIENTS == MODEL["adaptive"]["projection"]


def test_validation_conformance() -> None:
    fixture = load_json(CONFORMANCE_ROOT / "validation.json")

    for score in fixture["unitScores"]["valid"]:
        validate_unit_range(score)
    for score in fixture["unitScores"]["invalid"]:
        with pytest.raises(ValueError):
            validate_unit_range(score)

    for score in fixture["signedScores"]["valid"]:
        validate_signed_range(score)
    for score in fixture["signedScores"]["invalid"]:
        with pytest.raises(ValueError):
            validate_signed_range(score)

    zero_profile = AdaptiveBifurcatedProfile(*(0.0,) * 5)
    for threshold in fixture["dominantPoleThresholds"]["valid"]:
        zero_profile.dominant_poles(threshold=threshold)
    for threshold in fixture["dominantPoleThresholds"]["invalid"]:
        with pytest.raises(ValueError):
            zero_profile.dominant_poles(threshold=threshold)

    for parameters in fixture["gaussianParameters"]["invalid"]:
        with pytest.raises(ValueError):
            random_gaussian(
                mean=parameters["mean"],
                stddev=parameters["standardDeviation"],
                min_value=parameters["minimum"],
                max_value=parameters["maximum"],
            )


def test_aggregate_scoring_conformance() -> None:
    fixture = load_json(CONFORMANCE_ROOT / "aggregate-scoring.json")
    for case in fixture["cases"]:
        assert component_average_score(*case["components"]) == pytest.approx(
            case["expected"],
            abs=ARITHMETIC_TOLERANCE,
        )


def test_gaussian_sampling_conformance() -> None:
    fixture = load_json(CONFORMANCE_ROOT / "gaussian-sampling.json")
    for case in fixture["cases"]:
        actual = random_gaussian(
            mean=case["mean"],
            stddev=case["standardDeviation"],
            min_value=case["minimum"],
            max_value=case["maximum"],
            rng=FractionRandom(case["uniformFraction"]),
        )
        assert actual == pytest.approx(
            case["expected"],
            abs=GAUSSIAN_TOLERANCE,
        )


def test_life_stage_sampling_conformance() -> None:
    fixture = load_json(CONFORMANCE_ROOT / "life-stage-sampling.json")
    for case in fixture["cases"]:
        traits = BigFiveTraitConfiguration.random(
            LifeStage(case["lifeStage"]),
            rng=FractionRandom(*case["uniformFractions"]),
        )
        for trait_name, actual in trait_components(traits).items():
            assert actual == pytest.approx(
                case["expected"][trait_name],
                abs=GAUSSIAN_TOLERANCE,
            )


def test_conflict_resolution_conformance() -> None:
    fixture = load_json(CONFORMANCE_ROOT / "conflict-resolution.json")
    for case in fixture["cases"]:
        traits = flat_traits(case["traits"])
        actual_weights = {
            style.value: weight
            for style, weight in _conflict_style_weights(traits).items()
        }
        assert actual_weights == pytest.approx(
            case["weights"],
            abs=ARITHMETIC_TOLERANCE,
        )
        for selection in case["selections"]:
            actual = BigFiveConflictResolutionStyle.random(
                traits,
                rng=FractionRandom(selection["uniformFraction"]),
            )
            assert actual.value == selection["expected"]


def test_adaptive_projection_conformance() -> None:
    fixture = load_json(CONFORMANCE_ROOT / "adaptive.json")
    for case in fixture["projectionCases"]:
        signed = {
            trait: (score * 2.0) - 1.0
            for trait, score in case["bigFive"].items()
        }
        model_projection = []
        for coefficients in MODEL["adaptive"]["projection"].values():
            weighted_sum = sum(
                signed[trait] * weight
                for trait, weight in coefficients.items()
            )
            total_weight = sum(abs(weight) for weight in coefficients.values())
            model_projection.append(
                max(-1.0, min(1.0, weighted_sum / total_weight))
            )
        assert model_projection == pytest.approx(
            case["expected"],
            abs=ARITHMETIC_TOLERANCE,
        )

        actual = AdaptiveBifurcatedProfile.from_big_five(
            flat_traits(case["bigFive"])
        )
        assert actual.vector == pytest.approx(
            case["expected"],
            abs=ARITHMETIC_TOLERANCE,
        )


def test_adaptive_dominant_pole_conformance() -> None:
    fixture = load_json(CONFORMANCE_ROOT / "adaptive.json")
    for case in fixture["dominantPoleCases"]:
        profile = AdaptiveBifurcatedProfile(*case["vector"])
        actual = {
            domain.value: pole.value
            for domain, pole in profile.dominant_poles(
                threshold=case["threshold"]
            ).items()
        }
        assert actual == case["expected"]


def test_adaptive_vector_math_conformance() -> None:
    fixture = load_json(CONFORMANCE_ROOT / "adaptive.json")
    for case in fixture["vectorCases"]:
        left = AdaptiveBifurcatedProfile(*case["left"])
        right = AdaptiveBifurcatedProfile(*case["right"])
        assert left.dot_product(right) == pytest.approx(
            case["dotProduct"],
            abs=ARITHMETIC_TOLERANCE,
        )
        assert left.cosine_similarity(right) == pytest.approx(
            case["cosineSimilarity"],
            abs=ARITHMETIC_TOLERANCE,
        )
