"""Casting rules for the mystery example, separate from personality behavior."""

from collections.abc import Sequence
from dataclasses import dataclass

from personalitygen import (
    AdaptiveBifurcatedDomain,
    AdaptiveBifurcatedPole,
    AdaptiveBifurcatedProfile,
)


@dataclass(frozen=True)
class Candidate:
    name: str
    profile: AdaptiveBifurcatedProfile


@dataclass(frozen=True)
class MysteryCast:
    protagonist: Candidate
    ally: Candidate
    rival: Candidate


def cast_mystery(candidates: Sequence[Candidate]) -> MysteryCast:
    """Select three distinct roles. Equal scores keep input order."""
    if len(candidates) < 3:
        raise ValueError("Casting needs at least three candidates")
    if len({candidate.name for candidate in candidates}) != len(candidates):
        raise ValueError("Candidate names must be unique")
    eligible = [
        candidate
        for candidate in candidates
        if candidate.profile.dominant_poles(threshold=0.1).get(
            AdaptiveBifurcatedDomain.COMPETITION
        )
        == AdaptiveBifurcatedPole.PROFICIENCY
    ]
    if not eligible:
        raise ValueError("No candidate has a proficiency score above 0.1")
    def competition_score(candidate: Candidate) -> float:
        return candidate.profile.competition_score

    protagonist = max(eligible, key=competition_score)
    remaining = [
        candidate for candidate in candidates if candidate.name != protagonist.name
    ]

    def similarity(candidate: Candidate) -> float:
        return protagonist.profile.cosine_similarity(candidate.profile)

    ally = max(remaining, key=similarity)
    remaining = [candidate for candidate in remaining if candidate.name != ally.name]
    rival = min(remaining, key=similarity)
    return MysteryCast(protagonist, ally, rival)
