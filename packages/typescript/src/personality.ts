import {
  BigFiveConflictResolutionStyle,
  CONFLICT_RESOLUTION_STYLES,
  type BigFiveConflictResolutionStyle as ConflictStyle,
  type LifeStage,
  type PriorityLevel as Priority,
  PriorityLevel,
} from "./enums.js";
import {
  DEFAULT_RANDOM_SOURCE,
  type RandomOptions,
  type RandomSource,
} from "./randomness.js";
import {
  BigFiveAgreeableness,
  BigFiveConscientiousness,
  BigFiveExtraversion,
  BigFiveNeuroticism,
  BigFiveOpenness,
} from "./traits.js";

type TraitName =
  | "openness"
  | "conscientiousness"
  | "extraversion"
  | "agreeableness"
  | "neuroticism";
type TraitScores = Readonly<Record<TraitName, number>>;
type StyleCoefficients = Readonly<Record<TraitName, number>>;
const TRAIT_NAMES: readonly TraitName[] = Object.freeze([
  "openness",
  "conscientiousness",
  "extraversion",
  "agreeableness",
  "neuroticism",
]);

// Exported for compiled conformance tests; the package export map keeps this module internal.
export const CONFLICT_STYLE_COEFFICIENTS: Readonly<
  Record<ConflictStyle, StyleCoefficients>
> = Object.freeze({
  [BigFiveConflictResolutionStyle.Avoiding]: Object.freeze({
    openness: -0.1,
    conscientiousness: -0.2,
    extraversion: 0,
    agreeableness: 0.2,
    neuroticism: 0.7,
  }),
  [BigFiveConflictResolutionStyle.Obliging]: Object.freeze({
    openness: -0.1,
    conscientiousness: 0,
    extraversion: -0.2,
    agreeableness: 0.3,
    neuroticism: 0.2,
  }),
  [BigFiveConflictResolutionStyle.Integrating]: Object.freeze({
    openness: 0.1,
    conscientiousness: 0.1,
    extraversion: 0,
    agreeableness: 0.2,
    neuroticism: 0,
  }),
  [BigFiveConflictResolutionStyle.Dominating]: Object.freeze({
    openness: -0.2,
    conscientiousness: 0.2,
    extraversion: 0.2,
    agreeableness: -0.4,
    neuroticism: -0.2,
  }),
  [BigFiveConflictResolutionStyle.Compromising]: Object.freeze({
    openness: 0,
    conscientiousness: -0.2,
    extraversion: 0.1,
    agreeableness: 0,
    neuroticism: 0.1,
  }),
});

export const CONFLICT_MINIMUM_WEIGHT = 0.1;

export const CONFLICT_CONCERN_MAPPINGS: Readonly<
  Record<ConflictStyle, readonly [Priority, Priority]>
> = Object.freeze({
  [BigFiveConflictResolutionStyle.Avoiding]: Object.freeze([
    PriorityLevel.Low,
    PriorityLevel.Low,
  ] as const),
  [BigFiveConflictResolutionStyle.Obliging]: Object.freeze([
    PriorityLevel.Low,
    PriorityLevel.High,
  ] as const),
  [BigFiveConflictResolutionStyle.Integrating]: Object.freeze([
    PriorityLevel.High,
    PriorityLevel.High,
  ] as const),
  [BigFiveConflictResolutionStyle.Dominating]: Object.freeze([
    PriorityLevel.High,
    PriorityLevel.Low,
  ] as const),
  [BigFiveConflictResolutionStyle.Compromising]: Object.freeze([
    PriorityLevel.Moderate,
    PriorityLevel.Moderate,
  ] as const),
});

function traitScores(traits: BigFiveTraits): TraitScores {
  return {
    openness: traits.openness.score,
    conscientiousness: traits.conscientiousness.score,
    extraversion: traits.extraversion.score,
    agreeableness: traits.agreeableness.score,
    neuroticism: traits.neuroticism.score,
  };
}

export function conflictStyleWeights(
  traits: BigFiveTraits,
): Readonly<Record<ConflictStyle, number>> {
  const scores = traitScores(traits);
  const weights = {} as Record<ConflictStyle, number>;
  for (const style of CONFLICT_RESOLUTION_STYLES) {
    const coefficients = CONFLICT_STYLE_COEFFICIENTS[style];
    const level = TRAIT_NAMES.reduce(
      (total, traitName) => total + scores[traitName] * coefficients[traitName],
      0,
    );
    weights[style] = Math.max(level, CONFLICT_MINIMUM_WEIGHT);
  }
  return Object.freeze(weights);
}

function weightedChoice(
  weights: Readonly<Record<ConflictStyle, number>>,
  rng: RandomSource,
): ConflictStyle {
  const total = CONFLICT_RESOLUTION_STYLES.reduce(
    (sum, style) => sum + weights[style],
    0,
  );
  const draw = rng.uniform(0, total);
  if (!Number.isFinite(draw) || draw < 0 || draw > total) {
    throw new RangeError("RandomSource.uniform returned a value out of range.");
  }
  let threshold = draw;
  for (const style of CONFLICT_RESOLUTION_STYLES) {
    threshold -= weights[style];
    if (threshold <= 0) {
      return style;
    }
  }
  return CONFLICT_RESOLUTION_STYLES[
    CONFLICT_RESOLUTION_STYLES.length - 1
  ] as ConflictStyle;
}

export interface BigFiveTraitsOptions {
  readonly openness: BigFiveOpenness;
  readonly conscientiousness: BigFiveConscientiousness;
  readonly extraversion: BigFiveExtraversion;
  readonly agreeableness: BigFiveAgreeableness;
  readonly neuroticism: BigFiveNeuroticism;
}

export class BigFiveTraits {
  readonly openness: BigFiveOpenness;
  readonly conscientiousness: BigFiveConscientiousness;
  readonly extraversion: BigFiveExtraversion;
  readonly agreeableness: BigFiveAgreeableness;
  readonly neuroticism: BigFiveNeuroticism;

  constructor(options: BigFiveTraitsOptions) {
    if (
      !(options.openness instanceof BigFiveOpenness) ||
      !(options.conscientiousness instanceof BigFiveConscientiousness) ||
      !(options.extraversion instanceof BigFiveExtraversion) ||
      !(options.agreeableness instanceof BigFiveAgreeableness) ||
      !(options.neuroticism instanceof BigFiveNeuroticism)
    ) {
      throw new TypeError("Big Five traits must be Big Five trait objects.");
    }
    this.openness = options.openness;
    this.conscientiousness = options.conscientiousness;
    this.extraversion = options.extraversion;
    this.agreeableness = options.agreeableness;
    this.neuroticism = options.neuroticism;
    Object.freeze(this);
  }

  static random(lifeStage: LifeStage, options: RandomOptions = {}): BigFiveTraits {
    return new BigFiveTraits({
      openness: BigFiveOpenness.random(lifeStage, options),
      conscientiousness: BigFiveConscientiousness.random(lifeStage, options),
      extraversion: BigFiveExtraversion.random(lifeStage, options),
      agreeableness: BigFiveAgreeableness.random(lifeStage, options),
      neuroticism: BigFiveNeuroticism.random(lifeStage, options),
    });
  }
}

export interface BigFiveConflictResolutionOptions {
  readonly style: ConflictStyle;
}

export class BigFiveConflictResolution {
  readonly style: ConflictStyle;
  readonly concernForSelf: Priority;
  readonly concernForOthers: Priority;

  constructor(options: BigFiveConflictResolutionOptions) {
    if (!CONFLICT_RESOLUTION_STYLES.includes(options.style)) {
      throw new TypeError(`Unsupported conflict style: ${String(options.style)}`);
    }
    const [concernForSelf, concernForOthers] = CONFLICT_CONCERN_MAPPINGS[options.style];
    this.style = options.style;
    this.concernForSelf = concernForSelf;
    this.concernForOthers = concernForOthers;
    Object.freeze(this);
  }

  static random(
    traits: BigFiveTraits,
    options: RandomOptions = {},
  ): BigFiveConflictResolution {
    const style = weightedChoice(
      conflictStyleWeights(traits),
      options.rng ?? DEFAULT_RANDOM_SOURCE,
    );
    return new BigFiveConflictResolution({
      style,
    });
  }
}

export interface BigFivePersonalityOptions {
  readonly traits: BigFiveTraits;
  readonly conflictResolution: BigFiveConflictResolution;
}

export class BigFivePersonality {
  readonly traits: BigFiveTraits;
  readonly conflictResolution: BigFiveConflictResolution;

  constructor(options: BigFivePersonalityOptions) {
    if (!(options.traits instanceof BigFiveTraits)) {
      throw new TypeError("traits must be BigFiveTraits.");
    }
    if (!(options.conflictResolution instanceof BigFiveConflictResolution)) {
      throw new TypeError("conflictResolution must be a BigFiveConflictResolution.");
    }
    this.traits = options.traits;
    this.conflictResolution = options.conflictResolution;
    Object.freeze(this);
  }

  static random(lifeStage: LifeStage, options: RandomOptions = {}): BigFivePersonality {
    const traits = BigFiveTraits.random(lifeStage, options);
    return new BigFivePersonality({
      traits,
      conflictResolution: BigFiveConflictResolution.random(traits, options),
    });
  }
}
