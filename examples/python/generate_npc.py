"""Generate one reproducible NPC profile."""

from __future__ import annotations

import random

from personalitygen import (
    AdaptiveBifurcatedProfile,
    BigFivePersonality,
    LifeStage,
)


def _format_poles(profile: AdaptiveBifurcatedProfile) -> str:
    poles = profile.dominant_poles(threshold=0.25)
    if not poles:
        return "balanced"
    return ", ".join(
        f"{domain.value}: {pole.value}" for domain, pole in poles.items()
    )


def main() -> None:
    rng = random.Random(7)
    personality = BigFivePersonality.random(LifeStage.ADULT, rng=rng)
    abb_five = AdaptiveBifurcatedProfile.from_big_five(
        personality.traits
    )
    conflict = personality.conflict_resolution

    print("NPC: Quartermaster Ilya")
    print(f"Conflict style: {conflict.style.value}")
    print(
        "Concern priorities: "
        f"self={conflict.concern_for_self.value}, "
        f"others={conflict.concern_for_others.value}"
    )
    print(f"ABBF vector: {tuple(round(value, 2) for value in abb_five.vector)}")
    print(f"Dominant ABBF poles: {_format_poles(abb_five)}")


if __name__ == "__main__":
    main()
