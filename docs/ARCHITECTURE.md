# Architecture

## Intent

`personalitygen` generates simulated, human-like personality profiles for storytelling, simulations, and tests. It is not a clinical assessment package and does not implement survey scoring rubrics.

The package is deliberately small:

- no runtime dependencies
- Python 3.11+ support
- immutable dataclass value objects
- deterministic generation when a seeded random source is supplied
- public imports from `personalitygen`

## Module Map

- `src/personalitygen/enums.py`: public enum types such as `LifeStage` and `PriorityLevel`.
- `src/personalitygen/constants.py`: shared numeric bounds for unit-range scores.
- `src/personalitygen/randomness.py`: the minimal random-source protocol and truncated Gaussian helper.
- `src/personalitygen/traits.py`: Big Five trait value objects, life-stage sampling configuration, unit-range validation, and aggregate score calculation.
- `src/personalitygen/personality.py`: full trait configurations, conflict-resolution style derivation, and full personality generation.
- `src/personalitygen/__init__.py`: the stable public import surface.

## Dependency Direction

The core direction is intentionally simple:

```text
constants/enums/randomness
        -> traits
        -> personality
        -> __init__ public exports
```

Do not make lower-level modules import from `personality.py` or `__init__.py`. Keep `randomness.py` generic and free of trait or personality knowledge.

## Domain Contracts

Trait component scores are unit-range floats from `0.0` through `1.0`.

Each Big Five trait has exactly three component scores and one aggregate `score`. The aggregate score is the arithmetic mean of those three components. The shared helper in `traits.py` owns range validation and average calculation so every trait class follows the same rule.

Random trait generation is life-stage biased. The current supported stages are `child`, `young_adult`, and `adult`. Stage configuration belongs in `traits.py` because it is part of the package's domain model, not an external adapter.

Conflict-resolution style is derived from aggregate trait scores and mapped to concern-for-self and concern-for-others priorities. Every `BigFiveConflictResolutionStyle` must have a matching concern tuple.

## Extension Rules

When adding a new life stage:

1. Add the enum member in `enums.py`.
2. Add means for every trait configuration in `traits.py`.
3. Update tests so every trait samples the new stage and preserves the intended stage-bias contract.
4. Update README examples or docs if the new stage is user-facing.

When adding a new trait component:

1. Update the owning trait dataclass and its random constructor.
2. Revisit the aggregate score contract in this document.
3. Update parameterized trait tests so validation, aggregate scoring, string formatting, and random sampling all cover the new component.

When adding a new public type:

1. Export it from `src/personalitygen/__init__.py`.
2. Add it to `__all__`.
3. Add tests for the public import surface.
