"""Generate an RPG party and run one disagreement with separate game rules."""

import random

from personalitygen import BigFivePersonality, LifeStage

from rpg_rules import (
    PartyMember,
    preferred_plan,
    response_to,
    settle_disagreement,
    trust_change,
)


def main() -> None:
    rng = random.Random(42)
    party = [
        PartyMember(
            "Asha",
            BigFivePersonality.random(LifeStage.ADULT, rng=rng),
            "find the relic",
        ),
        PartyMember(
            "Bram",
            BigFivePersonality.random(LifeStage.ADULT, rng=rng),
            "protect the party",
            wounded=True,
        ),
        PartyMember(
            "Cato",
            BigFivePersonality.random(LifeStage.ADULT, rng=rng),
            "find the relic",
        ),
    ]
    proposal = preferred_plan(party[0])
    print("Scene: The party reaches ruins before nightfall. Bram is wounded.")
    print(f"Leader's proposal: {proposal}.")
    responses = []
    for member in party:
        response = response_to(member, proposal)
        responses.append(response)
        traits = member.personality.traits
        print(
            f"{member.name}: goal={member.goal}; "
            f"curiosity={traits.openness.intellectual_curiosity_score:.2f}; "
            f"anxiety={traits.neuroticism.anxiety_score:.2f}; "
            f"style={member.personality.conflict_resolution.style.value}"
        )
        print(f"  Preference: {preferred_plan(member)}. Response: {response}.")
    decision = settle_disagreement(responses, proposal)
    print(f"Party decision: {decision}.")
    for member in party[1:]:
        before = 2
        after = before + trust_change(member, decision)
        print(f"{member.name}: trust in leader {before} -> {after}")
    print("The game owns goals, injuries, plans, and trust. Profiles stay fixed.")


if __name__ == "__main__":
    main()
