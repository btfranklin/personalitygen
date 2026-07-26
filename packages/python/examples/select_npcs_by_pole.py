"""Filter NPCs by dominant ABBF poles."""

from __future__ import annotations

from personalitygen import (
    AdaptiveBifurcatedDomain,
    AdaptiveBifurcatedPole,
    AdaptiveBifurcatedProfile,
)


CAST = {
    "Asha": AdaptiveBifurcatedProfile(0.7, 0.5, 0.8, 0.4, 0.3),
    "Bram": AdaptiveBifurcatedProfile(-0.4, -0.7, -0.6, -0.5, -0.8),
    "Cato": AdaptiveBifurcatedProfile(0.2, 0.1, -0.2, -0.7, 0.8),
    "Dima": AdaptiveBifurcatedProfile(-0.6, 0.6, 0.5, 0.7, 0.1),
}


def select_by_pole(
    domain: AdaptiveBifurcatedDomain,
    pole: AdaptiveBifurcatedPole,
    *,
    threshold: float = 0.4,
) -> list[str]:
    matches = []
    for name, profile in CAST.items():
        poles = profile.dominant_poles(threshold=threshold)
        if poles.get(domain) == pole:
            matches.append(name)
    return matches


def main() -> None:
    collaborators = select_by_pole(
        AdaptiveBifurcatedDomain.COOPERATION,
        AdaptiveBifurcatedPole.COLLABORATION,
    )
    dominancy = select_by_pole(
        AdaptiveBifurcatedDomain.COMPETITION,
        AdaptiveBifurcatedPole.DOMINANCY,
    )

    print(f"Collaborative NPCs: {', '.join(collaborators)}")
    print(f"Dominancy-leaning NPCs: {', '.join(dominancy)}")


if __name__ == "__main__":
    main()
