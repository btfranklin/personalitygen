# 0002: Adaptive Bifurcated Model

## Status

Accepted

## Decision

ABBF is a first-class public model beside the existing Big Five API. It uses a signed 5D vector in the chart's top-to-bottom order:

1. order: strategizing vs improvisation
2. chaos: ethicality vs instinctivity
3. cooperation: collaboration vs independence
4. conflict: harmonizing vs utilitarianism
5. competition: proficiency vs dominancy

Positive values mean the left pole. Negative values mean the right pole. Zero means equal bias on that axis.

## Sources

- User-provided ABBF chart.
- ABBF author comment on bifurcated axes, signed bounds, orthogonal domains, 5D vectors, and dot-product comparison: <https://www.vectorsofmind.com/p/the-big-five-are-word-vectors/comments>
- ABBF Quora space referenced by the source and chart: <https://abbf.quora.com/>
- Big Five Aspect Scale context: <https://pubmed.ncbi.nlm.nih.gov/17983306/>
- HEXACO context: <https://hexaco.org/scaledescriptions>

## Big Five Projection

`AdaptiveBifurcatedProfile.from_big_five()` is an explicit generator heuristic. Each Big Five aggregate is converted from unit range to signed range with `signed = (score * 2) - 1`, then projected as:

- order = openness
- chaos = `0.45 * agreeableness + 0.35 * conscientiousness - 0.20 * neuroticism`
- cooperation = `0.55 * extraversion + 0.45 * agreeableness`
- conflict = `0.40 * agreeableness + 0.30 * neuroticism + 0.20 * openness - 0.10 * conscientiousness`
- competition = `0.50 * conscientiousness + 0.30 * agreeableness - 0.20 * extraversion`

Each weighted projection is normalized by absolute weight sum and clamped to `-1.0..1.0`.

## Consequences

- Keep ABBF separate from the `BigFive*` classes so both models have clear score semantics.
- Keep ABBF source/rationale in docs when changing axes, poles, or projection coefficients.
- Tests must lock vector order, signed bounds, pole metadata, vector math, and Big Five projection.
