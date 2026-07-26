"""Adaptive Bifurcated Big Five profile models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Self

from personalitygen._scoring import (
    SIGNED_RANGE_MAX,
    SIGNED_RANGE_MIN,
    cosine_similarity,
    dot_product,
    unit_to_signed,
    validate_signed_range,
    weighted_signed_average,
)
from personalitygen.randomness import RandomSource, random_gaussian

if TYPE_CHECKING:
    from personalitygen.personality import BigFiveTraitConfiguration


class AdaptiveBifurcatedDomain(str, Enum):
    """The five ABBF vector dimensions in canonical chart order."""

    ORDER = "order"
    CHAOS = "chaos"
    COOPERATION = "cooperation"
    CONFLICT = "conflict"
    COMPETITION = "competition"


class AdaptiveBifurcatedPole(str, Enum):
    """The ten ABBF poles used by the five signed axes."""

    STRATEGIZING = "strategizing"
    IMPROVISATION = "improvisation"
    ETHICALITY = "ethicality"
    INSTINCTIVITY = "instinctivity"
    COLLABORATION = "collaboration"
    INDEPENDENCE = "independence"
    HARMONIZING = "harmonizing"
    UTILITARIANISM = "utilitarianism"
    PROFICIENCY = "proficiency"
    DOMINANCY = "dominancy"


@dataclass(frozen=True, slots=True)
class AdaptiveBifurcatedAxis:
    """One signed ABBF axis with its domain, poles, and score."""

    domain: AdaptiveBifurcatedDomain
    positive_pole: AdaptiveBifurcatedPole
    negative_pole: AdaptiveBifurcatedPole
    score: float

    def __post_init__(self) -> None:
        validate_signed_range(self.score)

    @property
    def dominant_pole(self) -> AdaptiveBifurcatedPole | None:
        if self.score > 0.0:
            return self.positive_pole
        if self.score < 0.0:
            return self.negative_pole
        return None


_AXIS_DEFINITIONS = (
    (
        "order_score",
        AdaptiveBifurcatedDomain.ORDER,
        AdaptiveBifurcatedPole.STRATEGIZING,
        AdaptiveBifurcatedPole.IMPROVISATION,
    ),
    (
        "chaos_score",
        AdaptiveBifurcatedDomain.CHAOS,
        AdaptiveBifurcatedPole.ETHICALITY,
        AdaptiveBifurcatedPole.INSTINCTIVITY,
    ),
    (
        "cooperation_score",
        AdaptiveBifurcatedDomain.COOPERATION,
        AdaptiveBifurcatedPole.COLLABORATION,
        AdaptiveBifurcatedPole.INDEPENDENCE,
    ),
    (
        "conflict_score",
        AdaptiveBifurcatedDomain.CONFLICT,
        AdaptiveBifurcatedPole.HARMONIZING,
        AdaptiveBifurcatedPole.UTILITARIANISM,
    ),
    (
        "competition_score",
        AdaptiveBifurcatedDomain.COMPETITION,
        AdaptiveBifurcatedPole.PROFICIENCY,
        AdaptiveBifurcatedPole.DOMINANCY,
    ),
)

_ABBF_RANDOM_STDDEV = 0.35


def _sample_signed_score(*, rng: RandomSource | None = None) -> float:
    return random_gaussian(
        mean=0.0,
        stddev=_ABBF_RANDOM_STDDEV,
        min_value=SIGNED_RANGE_MIN,
        max_value=SIGNED_RANGE_MAX,
        rng=rng,
    )


@dataclass(frozen=True, slots=True)
class AdaptiveBifurcatedProfile:
    """A signed 5D ABBF profile for simulation character systems."""

    order_score: float
    chaos_score: float
    cooperation_score: float
    conflict_score: float
    competition_score: float

    def __post_init__(self) -> None:
        validate_signed_range(*self.vector)

    @classmethod
    def random(cls, *, rng: RandomSource | None = None) -> Self:
        """Generate a symmetric random ABBF profile in signed range."""

        return cls(
            order_score=_sample_signed_score(rng=rng),
            chaos_score=_sample_signed_score(rng=rng),
            cooperation_score=_sample_signed_score(rng=rng),
            conflict_score=_sample_signed_score(rng=rng),
            competition_score=_sample_signed_score(rng=rng),
        )

    @classmethod
    def from_big_five(cls, traits: BigFiveTraitConfiguration) -> Self:
        """Project a Big Five trait configuration into an ABBF profile."""

        openness = unit_to_signed(traits.openness.score)
        conscientiousness = unit_to_signed(traits.conscientiousness.score)
        extraversion = unit_to_signed(traits.extraversion.score)
        agreeableness = unit_to_signed(traits.agreeableness.score)
        neuroticism = unit_to_signed(traits.neuroticism.score)

        return cls(
            order_score=openness,
            chaos_score=weighted_signed_average(
                (agreeableness, 0.45),
                (conscientiousness, 0.35),
                (neuroticism, -0.20),
            ),
            cooperation_score=weighted_signed_average(
                (extraversion, 0.55),
                (agreeableness, 0.45),
            ),
            conflict_score=weighted_signed_average(
                (agreeableness, 0.40),
                (neuroticism, 0.30),
                (openness, 0.20),
                (conscientiousness, -0.10),
            ),
            competition_score=weighted_signed_average(
                (conscientiousness, 0.50),
                (agreeableness, 0.30),
                (extraversion, -0.20),
            ),
        )

    @property
    def vector(self) -> tuple[float, float, float, float, float]:
        """Return scores in order, chaos, cooperation, conflict, competition order."""

        return (
            self.order_score,
            self.chaos_score,
            self.cooperation_score,
            self.conflict_score,
            self.competition_score,
        )

    @property
    def axes(self) -> tuple[AdaptiveBifurcatedAxis, ...]:
        """Return axis metadata paired with this profile's scores."""

        return tuple(
            AdaptiveBifurcatedAxis(
                domain=domain,
                positive_pole=positive_pole,
                negative_pole=negative_pole,
                score=getattr(self, field_name),
            )
            for (
                field_name,
                domain,
                positive_pole,
                negative_pole,
            ) in _AXIS_DEFINITIONS
        )

    def dominant_poles(
        self, *, threshold: float = 0.0
    ) -> dict[AdaptiveBifurcatedDomain, AdaptiveBifurcatedPole]:
        """Return poles whose absolute score is greater than threshold."""

        if not (0.0 <= threshold <= SIGNED_RANGE_MAX):
            raise ValueError("threshold must be in the range 0.0...1.0")

        poles: dict[AdaptiveBifurcatedDomain, AdaptiveBifurcatedPole] = {}
        for axis in self.axes:
            if axis.score > threshold:
                poles[axis.domain] = axis.positive_pole
            elif axis.score < -threshold:
                poles[axis.domain] = axis.negative_pole
        return poles

    def dot_product(self, other: Self) -> float:
        """Return vector dot product with another ABBF profile."""

        return dot_product(self.vector, other.vector)

    def cosine_similarity(self, other: Self) -> float:
        """Return cosine similarity with another ABBF profile."""

        return cosine_similarity(self.vector, other.vector)
