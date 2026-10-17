"""Generate a candidate pool and apply separate story casting rules."""

import random
from itertools import combinations

from personalitygen import AdaptiveBifurcatedProfile, BigFivePersonality, LifeStage

from mystery_rules import Candidate, cast_mystery


def main() -> None:
    rng = random.Random(42)
    candidates = [
        Candidate(
            f"Candidate-{index + 1:02d}",
            AdaptiveBifurcatedProfile.from_big_five(
                BigFivePersonality.random(LifeStage.ADULT, rng=rng).traits
            ),
        )
        for index in range(40)
    ]
    cast = cast_mystery(candidates)
    print(f"Mystery casting: {len(candidates)} generated candidates")
    selections = [
        ("Protagonist", cast.protagonist, "highest proficiency above 0.1"),
        ("Ally", cast.ally, "most similar remaining profile to the protagonist"),
        ("Rival", cast.rival, "least similar remaining profile to the protagonist"),
    ]
    for role, candidate, reason in selections:
        poles = (
            ", ".join(
                f"{domain.value}={pole.value}"
                for domain, pole in candidate.profile.dominant_poles(
                    threshold=0.1
                ).items()
            )
            or "balanced"
        )
        print(f"{role}: {candidate.name}; selected for {reason}")
        print(f"  Proficiency score: {candidate.profile.competition_score:.2f}")
        print(f"  Dominant poles: {poles}")
    for left, right in combinations([cast.protagonist, cast.ally, cast.rival], 2):
        similarity = left.profile.cosine_similarity(right.profile)
        print(f"{left.name} / {right.name}: cosine similarity={similarity:.2f}")
    print("Story rules assign roles. Similarity is not a conflict probability.")
    print("A contrasting rival is not necessarily a villain.")


if __name__ == "__main__":
    main()
