# personalitygen

![personalitygen social preview](https://raw.githubusercontent.com/btfranklin/personalitygen/main/.github/social%20preview/personalitygen_social_preview.jpg "personalitygen")

[![Build Status](https://github.com/btfranklin/personalitygen/actions/workflows/python-package.yml/badge.svg)](https://github.com/btfranklin/personalitygen/actions/workflows/python-package.yml) [![Supports Python versions 3.11+](https://img.shields.io/pypi/pyversions/personalitygen.svg)](https://pypi.python.org/pypi/personalitygen)

`personalitygen` generates simulated character personalities for games, storytelling, simulations, and tests. It supports
conventional Big Five (OCEAN) profiles and Adaptive Bifurcated Big Five (ABBF) signed-vector profiles.

## Intent and scope

- Generate full Big Five profiles with sub-trait components and aggregate scores.
- Generate ABBF profiles as signed 5D vectors with dominant poles.
- Bias outputs by life stage using tuned Gaussian distributions (child, young adult, adult).
- Derive a conflict-resolution style from trait weights, plus mapped concern-for-self/others.
- Project Big Five profiles into ABBF vectors for systems that want both model shapes.
- Support deterministic generation by accepting a seeded random source.
- Stay lightweight and dependency-free (pure Python).

## Model overview

- Big Five traits: openness, conscientiousness, extraversion, agreeableness, neuroticism.
- Each trait is composed of three sub-traits and an aggregate score.
- Life stage influences distribution means and standard deviations for sampling.
- Conflict-resolution style is selected from avoiding, obliging, integrating, dominating, or compromising based on trait scores.
- ABBF profiles use five signed axes in chart order: order, chaos, cooperation, conflict, and competition.
- Positive ABBF values select the chart's left pole; negative values select the chart's right pole.

## Usage

```python
from personalitygen import BigFivePersonality, LifeStage

personality = BigFivePersonality.random(LifeStage.ADULT)
print(personality.trait_configuration)
print(personality.conflict_resolution_configuration)
```

If you want deterministic output, pass a seeded random number generator:

```python
import random
from personalitygen import BigFiveTraitConfiguration, LifeStage

rng = random.Random(42)
traits = BigFiveTraitConfiguration.random(LifeStage.YOUNG_ADULT, rng=rng)
print(traits)
```

ABBF profiles can be generated directly or projected from Big Five traits:

```python
from personalitygen import AdaptiveBifurcatedProfile

profile = AdaptiveBifurcatedProfile.random()
print(profile.vector)
print(profile.dominant_poles(threshold=0.2))

projected = AdaptiveBifurcatedProfile.from_big_five(traits)
print(projected.cosine_similarity(profile))
```

## Development

This package targets Python 3.11+.

```bash
pdm install --group dev
pdm run test
pdm run lint
```

Deeper architecture, quality, and maintenance guidance lives in the
[repository documentation](https://github.com/btfranklin/personalitygen/tree/main/docs).
Simulation recipes live in the
[usage guide](https://github.com/btfranklin/personalitygen/blob/main/docs/USAGE.md),
and runnable examples live in [`examples/`](examples/).
