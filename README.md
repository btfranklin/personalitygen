# personalitygen

![personalitygen banner](https://raw.githubusercontent.com/btfranklin/personalitygen/main/.github/social%20preview/personalitygen_social_preview.jpg "personalitygen")

`personalitygen` generates simulated character personalities for games,
storytelling, simulations, and tests. The project models conventional Big Five
(OCEAN) profiles and Adaptive Bifurcated Big Five (ABBF) signed vectors.

## Implementations

The repository contains equal, dependency-free language packages with the same
version and shared behavior:

- [Python](packages/python/README.md): Python 3.11–3.14, published on
  [PyPI](https://pypi.org/project/personalitygen/).
- [TypeScript](packages/typescript/README.md): ESM and ES2022 for browsers,
  bundlers, and the current Node.js LTS and latest release lines, published on
  [npm](https://www.npmjs.com/package/personalitygen).

```shell
pip install personalitygen
npm install personalitygen
```

Python:

```python
from personalitygen import AdaptiveBifurcatedProfile, BigFivePersonality, LifeStage

personality = BigFivePersonality.random(LifeStage.ADULT)
profile = AdaptiveBifurcatedProfile.from_big_five(
    personality.traits
)
```

TypeScript:

```typescript
import {
  AdaptiveBifurcatedProfile,
  BigFivePersonality,
  LifeStage,
} from "personalitygen";

const personality = BigFivePersonality.random(LifeStage.Adult);
const profile = AdaptiveBifurcatedProfile.fromBigFive(
  personality.traits,
);
```

The APIs are idiomatic peers: Python uses `snake_case` and enums; TypeScript
uses `camelCase`, object-parameter constructors, and frozen const objects with
string-union types. Both implement the same ranges, vector order, sampling
parameters, projections, and discrete outcomes from `spec/`.

## Repository layout

```text
packages/
  python/        Python package, tests, and examples
  typescript/    TypeScript package, tests, and examples
spec/            Shared behavioral model and conformance fixtures
docs/            Architecture, quality, usage, and decisions
```

## Development

```bash
pdm install -p packages/python --group dev
pdm run -p packages/python check

cd packages/typescript
npm ci
npm run check
```

See the [documentation index](docs/README.md) for architecture, quality, and
maintenance guidance. The
[behavioral contract](spec/BEHAVIOR.md) is the source of truth shared by every
language implementation. Python and TypeScript release in version lockstep.
