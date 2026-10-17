"""Check the example's game rules without changing the personality model."""

import random
from dataclasses import replace

import pytest
from personalitygen import (
    BigFiveConflictResolution,
    BigFiveConflictResolutionStyle,
    BigFivePersonality,
    LifeStage,
)

from rpg_rules import (
    PartyMember,
    preferred_plan,
    response_to,
    settle_disagreement,
    trust_change,
)


def wounded_member(style: BigFiveConflictResolutionStyle) -> PartyMember:
    generated = BigFivePersonality.random(LifeStage.ADULT, rng=random.Random(42))
    personality = replace(
        generated, conflict_resolution=BigFiveConflictResolution(style)
    )
    return PartyMember("Bram", personality, "protect the party", wounded=True)


@pytest.mark.parametrize(
    ("style", "response"),
    [
        (BigFiveConflictResolutionStyle.AVOIDING, "stand aside"),
        (BigFiveConflictResolutionStyle.OBLIGING, "yield"),
        (BigFiveConflictResolutionStyle.INTEGRATING, "seek a joint plan"),
        (BigFiveConflictResolutionStyle.DOMINATING, "refuse"),
        (BigFiveConflictResolutionStyle.COMPROMISING, "offer a limited visit"),
    ],
)
def test_opposition_uses_saved_style(
    style: BigFiveConflictResolutionStyle,
    response: str,
) -> None:
    member = wounded_member(style)
    assert response_to(member, "enter the ruins") == response
    assert response_to(member, "return to camp") == "accept"


def test_injury_changes_preference_without_changing_personality() -> None:
    member = wounded_member(BigFiveConflictResolutionStyle.INTEGRATING)
    healed = replace(member, wounded=False)
    assert preferred_plan(member) == "return to camp"
    assert preferred_plan(healed) == "enter the ruins"
    assert member.personality is healed.personality


def test_settlement_and_relationship_effects() -> None:
    assert (
        settle_disagreement(["accept", "yield"], "enter the ruins") == "enter the ruins"
    )
    assert (
        settle_disagreement(["accept", "seek a joint plan"], "enter the ruins")
        == "scout the entrance"
    )
    assert (
        settle_disagreement(["refuse", "offer a limited visit"], "enter the ruins")
        == "return to camp"
    )
    member = wounded_member(BigFiveConflictResolutionStyle.OBLIGING)
    assert trust_change(member, "scout the entrance") == 1
    assert trust_change(member, "enter the ruins") == -1
    assert trust_change(member, "return to camp") == 0
