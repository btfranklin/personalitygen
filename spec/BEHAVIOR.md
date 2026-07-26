# personalitygen Behavioral Contract

## Purpose

This specification defines behavior shared by every first-class
`personalitygen` implementation. Language packages own their runtime code and
public ergonomics; the files in `spec/` are test-time contract artifacts and
must not become runtime dependencies.

`model.json` records canonical names, bounds, sampling parameters, conflict
weights, ABBF axes, and projection coefficients. `conformance/` contains
language-neutral input and expected-output cases. A behavior change is
incomplete until the model, fixtures, implementations, tests, and user-facing
documentation agree.

## Numeric Semantics

- Big Five component and aggregate scores are finite numbers in `0.0..1.0`.
- Every Big Five trait has exactly three components. Its aggregate score is
  their arithmetic mean.
- ABBF scores are finite numbers in `-1.0..1.0`.
- Weighted ABBF projections divide by the sum of absolute coefficient weights
  and clamp the result to the signed range.
- A zero-length vector is not part of the public ABBF model. Cosine similarity
  with a zero-valued vector returns `0.0`.
- Conformance comparisons use absolute tolerance `1e-12` for normal arithmetic
  and `1e-6` for truncated-Gaussian results.

## Randomness

Random sampling uses an injectable source capable of drawing a finite value
uniformly between two inclusive bounds. Implementations reject values outside
the requested bounds and may adapt their ecosystem's native random source.

Life-stage components use inverse-CDF sampling from a normal distribution
truncated to `0.01..1.0`. ABBF random profiles use the same approach with mean
`0.0`, standard deviation `0.35`, and bounds `-1.0..1.0`. CDF probabilities
are constrained to `1e-12..(1 - 1e-12)` before inverse projection.

The contract guarantees semantic parity, not identical outputs from the same
integer seed in different runtimes. Conformance fixtures provide explicit
uniform fractions so each implementation can test the same draws without
standardizing a cross-language PRNG.

## Big Five And Conflict Resolution

The canonical Big Five order is openness, conscientiousness, extraversion,
agreeableness, and neuroticism. Supported life stages are child, young adult,
and adult. Component names, standard deviations, and per-stage means are
defined in `model.json`.

Conflict styles are evaluated in the order recorded in `model.json`. Each raw
style level is a weighted sum of the five aggregate trait scores, then floored
to `0.1` so counter-indicated styles remain reachable. Weighted selection maps
a uniform draw across the sum of the final weights. Concern-for-self and
concern-for-others priorities are fixed mappings.

## Adaptive Bifurcated Big Five

ABBF vector order is order, chaos, cooperation, conflict, and competition.
Positive scores select each axis's positive pole; negative scores select its
negative pole; zero selects neither. Dominant-pole filtering uses a strict
comparison, so a score whose absolute value equals the threshold is omitted.

Big Five-to-ABBF projection first maps aggregate unit scores with
`signed = (score * 2) - 1`, then applies the coefficients in `model.json`.
