"""Game rules for the RPG example. These rules are not package behavior."""

from dataclasses import dataclass
from typing import Literal

from personalitygen import BigFivePersonality


Plan = Literal["enter the ruins", "return to camp", "scout the entrance"]


@dataclass(frozen=True)
class PartyMember:
    name: str
    personality: BigFivePersonality
    goal: Literal["find the relic", "protect the party"]
    wounded: bool = False


def preferred_plan(member: PartyMember) -> Plan:
    """Combine a personal goal, current injury, and generated traits."""
    traits = member.personality.traits
    goal_bias = 0.3 if member.goal == "find the relic" else -0.3
    interest = (
        traits.openness.intellectual_curiosity_score
        - traits.neuroticism.anxiety_score
        + goal_bias
        - (0.4 if member.wounded else 0.0)
    )
    return "enter the ruins" if interest > 0 else "return to camp"


def response_to(member: PartyMember, proposal: Plan) -> str:
    """Use the saved conflict style to respond to an unwanted plan."""
    if preferred_plan(member) == proposal:
        return "accept"
    return {
        "avoiding": "stand aside",
        "obliging": "yield",
        "integrating": "seek a joint plan",
        "dominating": "refuse",
        "compromising": "offer a limited visit",
    }[member.personality.conflict_resolution.style.value]


def settle_disagreement(responses: list[str], proposal: Plan) -> Plan:
    """Let a refusal stop the visit; otherwise accept a compromise."""
    if "refuse" in responses:
        return "return to camp"
    if any(
        response in {"seek a joint plan", "offer a limited visit"}
        for response in responses
    ):
        return "scout the entrance"
    return proposal


def trust_change(member: PartyMember, decision: Plan) -> int:
    """Reward a compromise; penalize a visit against a member's preference."""
    if decision == "scout the entrance":
        return 1
    if decision == "enter the ruins" and preferred_plan(member) == "return to camp":
        return -1
    return 0
