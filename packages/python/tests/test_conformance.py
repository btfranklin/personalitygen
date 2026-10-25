from __future__ import annotations

from pathlib import Path

import pytest

from fixture_data import load_fixture
from personalitygen import (
    AdaptiveBifurcatedProfile,
    BigFiveAgreeableness,
    BigFiveConflictResolutionStyle,
    BigFiveConscientiousness,
    BigFiveExtraversion,
    BigFiveNeuroticism,
    BigFiveOpenness,
    BigFivePersonality,
    BigFiveTraits,
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


MODEL = load_fixture(SPEC_ROOT / "model.json")
ARITHMETIC_TOLERANCE = MODEL.mapping("tolerances").number("arithmetic")
GAUSSIAN_TOLERANCE = MODEL.mapping("tolerances").number("gaussian")


class FractionRandom:
    def __init__(self, *fractions: float) -> None:
        self._fractions = fractions
        self._index = 0

    def uniform(self, minimum: float, maximum: float) -> float:
        fraction = self._fractions[self._index]
        self._index += 1
        return minimum + ((maximum - minimum) * fraction)

    def assert_consumed(self) -> None:
        assert self._index == len(self._fractions)


def flat_traits(scores: dict[str, float]) -> BigFiveTraits:
    return BigFiveTraits(
        openness=BigFiveOpenness(*(scores["openness"],) * 3),
        conscientiousness=BigFiveConscientiousness(
            *(scores["conscientiousness"],) * 3
        ),
        extraversion=BigFiveExtraversion(*(scores["extraversion"],) * 3),
        agreeableness=BigFiveAgreeableness(*(scores["agreeableness"],) * 3),
        neuroticism=BigFiveNeuroticism(*(scores["neuroticism"],) * 3),
    )


def trait_components(
    traits: BigFiveTraits,
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
    assert MODEL.value("schemaVersion") == 1
    assert MODEL.mapping("ranges").value("unit") == [UNIT_RANGE_MIN, UNIT_RANGE_MAX]
    assert MODEL.mapping("ranges").value("traitSample") == [
        _TRAIT_SAMPLE_MIN,
        UNIT_RANGE_MAX,
    ]
    assert MODEL.mapping("ranges").value("signed") == [
        SIGNED_RANGE_MIN,
        SIGNED_RANGE_MAX,
    ]
    assert MODEL.value("randomness") == {
        "cdfEpsilon": _CDF_EPSILON,
        "abbfStandardDeviation": _ABBF_RANDOM_STDDEV,
    }
    assert MODEL.value("lifeStages") == [stage.value for stage in LifeStage]
    assert MODEL.value("priorityLevels") == [level.value for level in PriorityLevel]

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
        expected = MODEL.mapping("traits").mapping(trait_name)
        assert expected.value("components") == component_names[trait_name]
        assert config.stddev == expected.value("standardDeviation")
        assert {
            stage.value: list(means)
            for stage, means in config.means_by_stage.items()
        } == expected.value("means")

    conflict_model = MODEL.mapping("conflictResolution")
    assert conflict_model.value("minimumWeight") == _CONFLICT_MINIMUM_WEIGHT
    assert conflict_model.mapping("styles").keys() == [
        style.value for style in BigFiveConflictResolutionStyle
    ]
    for style, coefficients in _CONFLICT_STYLE_COEFFICIENTS.items():
        expected = conflict_model.mapping("styles").mapping(style.value)
        assert coefficients == expected.value("coefficients")
        assert [priority.value for priority in _STYLE_TO_CONCERNS[style]] == (
            expected.value("concerns")
        )

    assert [
        {
            "domain": domain.value,
            "positivePole": positive.value,
            "negativePole": negative.value,
        }
        for domain, positive, negative in _AXIS_DEFINITIONS
    ] == MODEL.mapping("adaptive").value("axes")
    assert _ABBF_PROJECTION_COEFFICIENTS == MODEL.mapping("adaptive").value(
        "projection"
    )


def test_validation_conformance() -> None:
    fixture = load_fixture(CONFORMANCE_ROOT / "validation.json")

    for score in fixture.mapping("unitScores").numbers("valid"):
        validate_unit_range(score)
    for score in fixture.mapping("unitScores").numbers("invalid"):
        with pytest.raises(ValueError):
            validate_unit_range(score)

    for score in fixture.mapping("signedScores").numbers("valid"):
        validate_signed_range(score)
    for score in fixture.mapping("signedScores").numbers("invalid"):
        with pytest.raises(ValueError):
            validate_signed_range(score)

    zero_profile = AdaptiveBifurcatedProfile(*(0.0,) * 5)
    for threshold in fixture.mapping("dominantPoleThresholds").numbers("valid"):
        zero_profile.dominant_poles(threshold=threshold)
    for threshold in fixture.mapping("dominantPoleThresholds").numbers("invalid"):
        with pytest.raises(ValueError):
            zero_profile.dominant_poles(threshold=threshold)

    for parameters in fixture.mapping("gaussianParameters").objects("invalid"):
        with pytest.raises(ValueError):
            random_gaussian(
                mean=parameters.number("mean"),
                stddev=parameters.number("standardDeviation"),
                min_value=parameters.number("minimum"),
                max_value=parameters.number("maximum"),
            )


def test_aggregate_scoring_conformance() -> None:
    fixture = load_fixture(CONFORMANCE_ROOT / "aggregate-scoring.json")
    for case in fixture.objects("cases"):
        assert component_average_score(*case.numbers("components")) == pytest.approx(
            case.number("expected"),
            abs=ARITHMETIC_TOLERANCE,
        )


def test_gaussian_sampling_conformance() -> None:
    fixture = load_fixture(CONFORMANCE_ROOT / "gaussian-sampling.json")
    for case in fixture.objects("cases"):
        actual = random_gaussian(
            mean=case.number("mean"),
            stddev=case.number("standardDeviation"),
            min_value=case.number("minimum"),
            max_value=case.number("maximum"),
            rng=FractionRandom(case.number("uniformFraction")),
        )
        assert actual == pytest.approx(
            case.number("expected"),
            abs=GAUSSIAN_TOLERANCE,
        )


def test_life_stage_sampling_conformance() -> None:
    fixture = load_fixture(CONFORMANCE_ROOT / "life-stage-sampling.json")
    for case in fixture.objects("cases"):
        rng = FractionRandom(*case.numbers("uniformFractions"))
        traits = BigFiveTraits.random(LifeStage(case.string("lifeStage")), rng=rng)
        rng.assert_consumed()
        for trait_name, actual in trait_components(traits).items():
            assert actual == pytest.approx(
                case.mapping("expected").numbers(trait_name),
                abs=GAUSSIAN_TOLERANCE,
            )


def test_full_personality_sampling_conformance() -> None:
    fixture = load_fixture(CONFORMANCE_ROOT / "life-stage-sampling.json")
    for case in fixture.objects("personalityCases"):
        rng = FractionRandom(*case.numbers("uniformFractions"))
        personality = BigFivePersonality.random(
            LifeStage(case.string("lifeStage")), rng=rng
        )
        rng.assert_consumed()
        for name, actual in trait_components(personality.traits).items():
            assert actual == pytest.approx(
                case.mapping("expected").numbers(name), abs=GAUSSIAN_TOLERANCE
            )
        conflict = personality.conflict_resolution
        assert {
            "style": conflict.style.value,
            "concernForSelf": conflict.concern_for_self.value,
            "concernForOthers": conflict.concern_for_others.value,
        } == case.value("expectedConflictResolution")


def test_conflict_resolution_conformance() -> None:
    fixture = load_fixture(CONFORMANCE_ROOT / "conflict-resolution.json")
    for case in fixture.objects("cases"):
        traits = flat_traits(case.number_map("traits"))
        actual_weights = {
            style.value: weight
            for style, weight in _conflict_style_weights(traits).items()
        }
        assert actual_weights == pytest.approx(
            case.number_map("weights"),
            abs=ARITHMETIC_TOLERANCE,
        )
        for selection in case.objects("selections"):
            actual = BigFiveConflictResolutionStyle.random(
                traits,
                rng=FractionRandom(selection.number("uniformFraction")),
            )
            assert actual.value == selection.string("expected")


def test_adaptive_projection_conformance() -> None:
    fixture = load_fixture(CONFORMANCE_ROOT / "adaptive.json")
    for case in fixture.objects("projectionCases"):
        signed = {
            trait: (score * 2.0) - 1.0
            for trait, score in case.number_map("bigFive").items()
        }
        model_projection = []
        projection = MODEL.mapping("adaptive").mapping("projection")
        for axis in projection.object_values():
            coefficients = axis.number_values()
            weighted_sum = sum(
                signed[trait] * weight
                for trait, weight in coefficients.items()
            )
            total_weight = sum(abs(weight) for weight in coefficients.values())
            model_projection.append(
                max(-1.0, min(1.0, weighted_sum / total_weight))
            )
        assert model_projection == pytest.approx(
            case.numbers("expected"),
            abs=ARITHMETIC_TOLERANCE,
        )

        actual = AdaptiveBifurcatedProfile.from_big_five(
            flat_traits(case.number_map("bigFive"))
        )
        assert actual.vector == pytest.approx(
            case.numbers("expected"),
            abs=ARITHMETIC_TOLERANCE,
        )


def test_adaptive_dominant_pole_conformance() -> None:
    fixture = load_fixture(CONFORMANCE_ROOT / "adaptive.json")
    for case in fixture.objects("dominantPoleCases"):
        profile = AdaptiveBifurcatedProfile(*case.numbers("vector"))
        actual = {
            domain.value: pole.value
            for domain, pole in profile.dominant_poles(
                threshold=case.number("threshold")
            ).items()
        }
        assert actual == case.value("expected")


def test_adaptive_vector_math_conformance() -> None:
    fixture = load_fixture(CONFORMANCE_ROOT / "adaptive.json")
    for case in fixture.objects("vectorCases"):
        left = AdaptiveBifurcatedProfile(*case.numbers("left"))
        right = AdaptiveBifurcatedProfile(*case.numbers("right"))
        assert left.dot_product(right) == pytest.approx(
            case.number("dotProduct"),
            abs=ARITHMETIC_TOLERANCE,
        )
        assert left.cosine_similarity(right) == pytest.approx(
            case.number("cosineSimilarity"),
            abs=ARITHMETIC_TOLERANCE,
        )
