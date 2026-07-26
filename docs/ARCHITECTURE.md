# Architecture

## Intent

`personalitygen` generates character personality profiles for games, storytelling, simulations, and tests. Its models are generator-friendly value objects with explicit score contracts and deterministic random sampling when callers supply a seeded random source.

The repository is deliberately small. Its current Python implementation has:

- no runtime dependencies
- Python 3.11+ support
- immutable dataclass value objects
- deterministic generation when a seeded random source is supplied
- public imports from `personalitygen`

## Repository Map

- `packages/python/`: independently buildable Python distribution, tests, and
  examples.
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

Do not make lower-level modules import from `personality.py` or `__init__.py`. Keep `randomness.py` generic and free of trait or personality knowledge. Keep `_scoring.py` free of model-specific names so Big Five and ABBF can share it without coupling their public types.

## Domain Contracts

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

1. Add the enum member in `enums.py`.
2. Add means for every trait configuration in `traits.py`.
3. Update tests so every trait samples the new stage and preserves the intended stage-bias contract.
4. Update README examples or docs if the new stage is user-facing.

When adding a new trait component:

1. Update the owning trait dataclass and its random constructor.
2. Revisit the aggregate score contract in this document.
3. Update parameterized trait tests so validation, aggregate scoring, string formatting, and random sampling all cover the new component.

When adding a new public type:

1. Export it from `packages/python/src/personalitygen/__init__.py`.
2. Add it to `__all__`.
3. Add tests for the public import surface.

When changing ABBF axis semantics:

1. Update the pole/domain metadata in `adaptive.py`.
2. Update the ABBF decision record.
3. Update tests for vector order, axis metadata, dominant poles, and Big Five projection.
