"""Project fixed Big Five traits into an ABBF vector."""

from __future__ import annotations

from personalitygen import (
    AdaptiveBifurcatedProfile,
    BigFiveAgreeableness,
    BigFiveConscientiousness,
    BigFiveExtraversion,
    BigFiveNeuroticism,
    BigFiveOpenness,
    BigFiveTraits,
)


def main() -> None:
    traits = BigFiveTraits(
        openness=BigFiveOpenness(0.75, 0.70, 0.80),
        conscientiousness=BigFiveConscientiousness(0.65, 0.70, 0.75),
        extraversion=BigFiveExtraversion(0.40, 0.45, 0.50),
        agreeableness=BigFiveAgreeableness(0.80, 0.75, 0.70),
        neuroticism=BigFiveNeuroticism(0.25, 0.30, 0.20),
    )
    profile = AdaptiveBifurcatedProfile.from_big_five(traits)

    print("Projected ABBF axes:")
    for axis in profile.axes:
        pole = axis.dominant_pole.value if axis.dominant_pole else "balanced"
        print(f"{axis.domain.value}: {axis.score:.2f} ({pole})")


if __name__ == "__main__":
    main()
