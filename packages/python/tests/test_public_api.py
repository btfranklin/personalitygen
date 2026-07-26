from __future__ import annotations

import personalitygen
from personalitygen.adaptive import (
    AdaptiveBifurcatedAxis,
    AdaptiveBifurcatedDomain,
    AdaptiveBifurcatedPole,
    AdaptiveBifurcatedProfile,
)
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


EXPECTED_PUBLIC_API = {
    "AdaptiveBifurcatedAxis": AdaptiveBifurcatedAxis,
    "AdaptiveBifurcatedDomain": AdaptiveBifurcatedDomain,
    "AdaptiveBifurcatedPole": AdaptiveBifurcatedPole,
    "AdaptiveBifurcatedProfile": AdaptiveBifurcatedProfile,
    "BigFiveAgreeableness": BigFiveAgreeableness,
    "BigFiveConscientiousness": BigFiveConscientiousness,
    "BigFiveConflictResolution": BigFiveConflictResolution,
    "BigFiveConflictResolutionStyle": BigFiveConflictResolutionStyle,
    "BigFiveExtraversion": BigFiveExtraversion,
    "BigFiveNeuroticism": BigFiveNeuroticism,
    "BigFiveOpenness": BigFiveOpenness,
    "BigFivePersonality": BigFivePersonality,
    "BigFiveTraits": BigFiveTraits,
    "LifeStage": LifeStage,
    "PriorityLevel": PriorityLevel,
}


def test_public_api_exports_are_explicit_and_importable() -> None:
    assert set(personalitygen.__all__) == set(EXPECTED_PUBLIC_API)
    for name, expected_object in EXPECTED_PUBLIC_API.items():
        assert getattr(personalitygen, name) is expected_object
