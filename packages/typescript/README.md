# personalitygen for TypeScript

Generate simulated character personalities for games, stories, and
simulations using the Big Five (OCEAN) and Adaptive Bifurcated Big Five
(ABBF) models.

The package is dependency-free, ESM-only, browser/bundler friendly, and ships
JavaScript plus TypeScript declarations. CI tests the current Node.js LTS and
latest release lines.

## Install

```shell
npm install personalitygen
```

## Generate a character

```typescript
import {
  AdaptiveBifurcatedProfile,
  BigFivePersonality,
  LifeStage,
} from "personalitygen";

const personality = BigFivePersonality.random(LifeStage.Adult);
const adaptive = AdaptiveBifurcatedProfile.fromBigFive(
  personality.traits,
);

console.log(personality.conflictResolution.style);
console.log(adaptive.vector);
console.log(adaptive.dominantPoles(0.25));
```

Random factories use `Math.random` by default. Supply any structural random
source when a game or simulation owns randomness:

```typescript
const rng = {
  uniform(minimum: number, maximum: number): number {
    return minimum + (maximum - minimum) * mySeededRandom();
  },
};

const personality = BigFivePersonality.random(LifeStage.Adult, { rng });
```

`uniform(minimum, maximum)` may return either inclusive endpoint and must
return a finite value inside the requested range; malformed results raise
`RangeError`. A caller-owned seeded source makes a fixed sequence reproducible
within one package version. Outputs from the same integer seed are not
guaranteed to remain identical after a later model or algorithm change.

## Models

- `BigFiveTraits` groups the five OCEAN trait value objects.
- `BigFiveConflictResolution` records a style and its derived concern
  priorities.
- `BigFivePersonality` combines `BigFiveTraits` with
  `BigFiveConflictResolution`.
- `AdaptiveBifurcatedProfile` represents the signed ABBF vector in canonical
  order: order, chaos, cooperation, conflict, competition.
- All exported model instances and enum-like value objects are frozen.
- Numeric constructors throw `RangeError` for invalid ranges; categorical
  constructors throw `TypeError` for unsupported values.

See the [repository documentation](https://github.com/btfranklin/personalitygen)
for the behavioral specification, Python peer, examples, and contribution
guidance.
