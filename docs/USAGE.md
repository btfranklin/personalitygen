# Usage

`personalitygen` is meant for generating and comparing character profiles in
games, simulations, stories, and tests. The public API gives you two model
shapes:

- Big Five profiles for familiar OCEAN-style traits, life-stage bias, and
  conflict-resolution style.
- ABBF profiles for signed 5D vectors that are easy to compare, filter, and
  use as gameplay or simulation parameters.

## Generate A Character

Use `BigFivePersonality.random()` when you want a complete character profile
with traits and a derived conflict style.

```python
import random

from personalitygen import BigFivePersonality, LifeStage

rng = random.Random(42)
personality = BigFivePersonality.random(LifeStage.ADULT, rng=rng)

print(personality.trait_configuration)
print(personality.conflict_resolution_configuration.conflict_resolution_style)
```

Pass a seeded random source when a world, save file, test fixture, or content
pipeline needs reproducible output.

## Use ABBF As A Character Vector

Use `AdaptiveBifurcatedProfile.random()` when a system needs compact signed
values for behavior knobs.

```python
import random

from personalitygen import AdaptiveBifurcatedProfile

rng = random.Random(17)
profile = AdaptiveBifurcatedProfile.random(rng=rng)

print(profile.vector)
print(profile.dominant_poles(threshold=0.25))
```

ABBF scores range from `-1.0` through `1.0`. Positive values mean the left pole
from the ABBF chart, negative values mean the right pole, and zero means neither
side dominates. The canonical vector order is:

1. order: strategizing vs improvisation
2. chaos: ethicality vs instinctivity
3. cooperation: collaboration vs independence
4. conflict: harmonizing vs utilitarianism
5. competition: proficiency vs dominancy

## Project Big Five Into ABBF

Use `AdaptiveBifurcatedProfile.from_big_five()` when an existing generator or
save format already uses Big Five traits, but a gameplay system wants ABBF
vectors.

```python
from personalitygen import AdaptiveBifurcatedProfile, BigFiveTraitConfiguration

traits = BigFiveTraitConfiguration.random(LifeStage.YOUNG_ADULT, rng=rng)
profile = AdaptiveBifurcatedProfile.from_big_five(traits)

for axis in profile.axes:
    print(axis.domain.value, axis.score, axis.dominant_pole)
```

Projection is a stable generator heuristic documented in
[`decisions/0002-adaptive-bifurcated-model.md`](decisions/0002-adaptive-bifurcated-model.md).

## Compare And Select Characters

Use `.dot_product()` or `.cosine_similarity()` for character matching,
clustering, relationship systems, or content selection.

```python
from personalitygen import AdaptiveBifurcatedProfile

merchant = AdaptiveBifurcatedProfile(0.6, 0.4, 0.7, 0.2, 0.5)
raider = AdaptiveBifurcatedProfile(-0.5, -0.6, -0.7, -0.4, -0.8)

print(merchant.cosine_similarity(raider))
```

Use `.dominant_poles(threshold=...)` when you need readable tags for filters,
dialogue conditions, faction membership, procedural quest roles, or debugging
tools.

## Runnable Examples

The `packages/python/examples/` directory contains small scripts that
demonstrate typical simulation workflows:

- `generate_npc.py`: generate a full character and project it into ABBF.
- `project_big_five_to_abbf.py`: inspect a fixed Big Five-to-ABBF projection.
- `compare_characters.py`: compare authored ABBF vectors.
- `select_npcs_by_pole.py`: filter a small cast by dominant ABBF poles.
