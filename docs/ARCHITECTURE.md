# Architecture

## Intent

`personalitygen` generates character personality profiles for games, storytelling, simulations, and tests. Its models are generator-friendly value objects with explicit score contracts and deterministic random sampling when callers supply a seeded random source.

The repository is deliberately small. Both implementations have:

- no runtime dependencies
- immutable value objects
- caller-controlled random generation
- public imports from `personalitygen`

Python supports 3.11–3.14 and uses frozen dataclasses. TypeScript publishes ESM
targeting ES2022 and uses frozen classes plus frozen const objects.

## Repository Map

- `packages/python/`: independently buildable Python distribution, tests, and
  examples.
- `packages/typescript/`: independently buildable npm distribution, tests, and
  examples.
- `spec/`: language-neutral behavioral model and conformance fixtures consumed
  by implementation test suites.
- `docs/`: repository-wide architecture, quality, usage, and decisions.

## Python Module Map

- `packages/python/src/personalitygen/enums.py`: public enum types such as `LifeStage` and `PriorityLevel`.
- `packages/python/src/personalitygen/constants.py`: shared numeric bounds for unit-range scores.
- `packages/python/src/personalitygen/randomness.py`: the minimal random-source protocol and truncated Gaussian helper.
- `packages/python/src/personalitygen/_scoring.py`: internal score validation, signed projection, and vector math helpers.
- `packages/python/src/personalitygen/traits.py`: Big Five trait value objects, life-stage sampling configuration, unit-range validation, and aggregate score calculation.
- `packages/python/src/personalitygen/personality.py`: full trait configurations, conflict-resolution style derivation, and full personality generation.
- `packages/python/src/personalitygen/adaptive.py`: Adaptive Bifurcated Big Five signed-vector profiles, axis metadata, and Big Five projection.
- `packages/python/src/personalitygen/__init__.py`: the stable public import surface.

## TypeScript Module Map

- `packages/typescript/src/enums.ts`: public frozen categorical values and
  their string-union types.
- `packages/typescript/src/randomness.ts`: the structural `RandomSource` and
  truncated-Gaussian helper.
- `packages/typescript/src/scoring.ts`: internal score validation, projection,
  and vector math.
- `packages/typescript/src/traits.ts`: Big Five value objects and life-stage
  sampling.
- `packages/typescript/src/personality.ts`: trait configurations, weighted
  conflict resolution, and complete personality generation.
- `packages/typescript/src/adaptive.ts`: ABBF profiles, axes, projection, and
  vector operations.
- `packages/typescript/src/index.ts`: the stable public export surface.

## Dependency Direction

The core direction is intentionally simple:

```text
constants/enums/randomness/_scoring
        -> traits
        -> personality
        -> __init__ public exports

constants/enums/randomness/_scoring
        -> adaptive
        -> __init__ public exports
```

Within either package, keep randomness and scoring helpers free of model
knowledge. Public index modules should collect exports, not own behavior.

No language package imports from another language package. Implementations own
their runtime constants and algorithms; their tests consume `spec/` to detect
semantic drift. The specification is never loaded by published packages at
runtime.

## Domain Contracts

[`spec/BEHAVIOR.md`](../spec/BEHAVIOR.md) and `spec/model.json` are the
canonical cross-language contract. The notes below summarize that contract for
repository orientation.

### Big Five

Trait component scores are unit-range floats from `0.0` through `1.0`.

Each Big Five trait has exactly three component scores and one aggregate `score`. The aggregate score is the arithmetic mean of those three components. The shared helper in `_scoring.py` owns range validation and average calculation so every trait class follows the same rule.

Random trait generation is life-stage biased. The current supported stages are `child`, `young_adult`, and `adult`. Stage configuration belongs in `traits.py` because it is part of the package's domain model, not an external adapter.

Conflict-resolution style is derived from aggregate trait scores and mapped to concern-for-self and concern-for-others priorities. Every `BigFiveConflictResolutionStyle` must have a matching concern tuple.

### Adaptive Bifurcated Big Five

ABBF profile scores are signed floats from `-1.0` through `1.0`.

The canonical vector order follows the chart's top-to-bottom axis order: order, chaos, cooperation, conflict, competition.

Positive ABBF scores select the chart's left pole: strategizing, ethicality, collaboration, harmonizing, and proficiency. Negative ABBF scores select the chart's right pole: improvisation, instinctivity, independence, utilitarianism, and dominancy. A zero score means neither pole dominates that axis.

ABBF random generation samples all five axes symmetrically around zero. ABBF Big Five projection is an explicit package heuristic and belongs in `adaptive.py`, not in the Big Five trait classes.

## Extension Rules

When adding a new life stage:

1. Update `spec/model.json` and its conformance fixtures.
2. Add the categorical value and means in both implementations.
3. Update both test suites.
4. Update examples or docs if the stage is user-facing.

When adding a new trait component:

1. Update the shared model.
2. Update the owning value object and random constructor in each language.
3. Revisit the aggregate score contract in this document.
4. Update conformance and package tests.

When adding a new public type:

1. Add idiomatic public types in both packages when the concept is shared.
2. Export from the Python `__init__.py` and TypeScript `index.ts`.
3. Add tests for both public surfaces.

When changing ABBF axis semantics:

1. Update the shared model and fixtures.
2. Update the pole/domain metadata in both language packages.
3. Update the ABBF decision record.
4. Update both suites for vector order, metadata, poles, and projection.
