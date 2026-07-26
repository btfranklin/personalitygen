import {
  BigFiveConflictResolutionStyle,
  CONFLICT_RESOLUTION_STYLES,
  type BigFiveConflictResolutionStyle as ConflictStyle,
  type LifeStage,
  PRIORITY_LEVELS,
  type PriorityLevel as Priority,
  PriorityLevel,
} from "./enums.js";
import type { RandomOptions, RandomSource } from "./randomness.js";
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

function traitScores(configuration: BigFiveTraitConfiguration): TraitScores {
  return {
    openness: configuration.openness.score,
    conscientiousness: configuration.conscientiousness.score,
    extraversion: configuration.extraversion.score,
    agreeableness: configuration.agreeableness.score,
    neuroticism: configuration.neuroticism.score,
  };
}

export function conflictStyleWeights(
  configuration: BigFiveTraitConfiguration,
): Readonly<Record<ConflictStyle, number>> {
  const scores = traitScores(configuration);
  const avoiding = CONFLICT_STYLE_COEFFICIENTS[BigFiveConflictResolutionStyle.Avoiding];
  const obliging = CONFLICT_STYLE_COEFFICIENTS[BigFiveConflictResolutionStyle.Obliging];
  const integrating =
    CONFLICT_STYLE_COEFFICIENTS[BigFiveConflictResolutionStyle.Integrating];
  const dominating =
    CONFLICT_STYLE_COEFFICIENTS[BigFiveConflictResolutionStyle.Dominating];
  const compromising =
    CONFLICT_STYLE_COEFFICIENTS[BigFiveConflictResolutionStyle.Compromising];

  return Object.freeze({
    [BigFiveConflictResolutionStyle.Avoiding]: Math.max(
      scores.neuroticism * avoiding.neuroticism +
        scores.openness * avoiding.openness +
        scores.agreeableness * avoiding.agreeableness +
        scores.conscientiousness * avoiding.conscientiousness,
      CONFLICT_MINIMUM_WEIGHT,
    ),
    [BigFiveConflictResolutionStyle.Obliging]: Math.max(
      scores.neuroticism * obliging.neuroticism +
        scores.extraversion * obliging.extraversion +
        scores.openness * obliging.openness +
        scores.agreeableness * obliging.agreeableness,
      CONFLICT_MINIMUM_WEIGHT,
    ),
    [BigFiveConflictResolutionStyle.Integrating]: Math.max(
      scores.openness * integrating.openness +
        scores.agreeableness * integrating.agreeableness +
        scores.conscientiousness * integrating.conscientiousness,
      CONFLICT_MINIMUM_WEIGHT,
    ),
    [BigFiveConflictResolutionStyle.Dominating]: Math.max(
      scores.neuroticism * dominating.neuroticism +
        scores.extraversion * dominating.extraversion +
        scores.openness * dominating.openness +
        scores.agreeableness * dominating.agreeableness +
        scores.conscientiousness * dominating.conscientiousness,
      CONFLICT_MINIMUM_WEIGHT,
    ),
    [BigFiveConflictResolutionStyle.Compromising]: Math.max(
      scores.neuroticism * compromising.neuroticism +
        scores.extraversion * compromising.extraversion +
        scores.conscientiousness * compromising.conscientiousness,
      CONFLICT_MINIMUM_WEIGHT,
    ),
  });
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
  return CONFLICT_RESOLUTION_STYLES[0] as ConflictStyle;
}

const DEFAULT_RANDOM_SOURCE: RandomSource = Object.freeze({
  uniform(minimum: number, maximum: number): number {
    return minimum + (maximum - minimum) * Math.random();
  },
});

export interface BigFiveTraitConfigurationOptions {
  readonly openness: BigFiveOpenness;
  readonly conscientiousness: BigFiveConscientiousness;
  readonly extraversion: BigFiveExtraversion;
  readonly agreeableness: BigFiveAgreeableness;
  readonly neuroticism: BigFiveNeuroticism;
}

export class BigFiveTraitConfiguration {
  readonly openness: BigFiveOpenness;
  readonly conscientiousness: BigFiveConscientiousness;
  readonly extraversion: BigFiveExtraversion;
  readonly agreeableness: BigFiveAgreeableness;
  readonly neuroticism: BigFiveNeuroticism;

  constructor(options: BigFiveTraitConfigurationOptions) {
    if (
      !(options.openness instanceof BigFiveOpenness) ||
      !(options.conscientiousness instanceof BigFiveConscientiousness) ||
      !(options.extraversion instanceof BigFiveExtraversion) ||
      !(options.agreeableness instanceof BigFiveAgreeableness) ||
      !(options.neuroticism instanceof BigFiveNeuroticism)
    ) {
      throw new TypeError("Trait configuration values must be Big Five trait objects.");
    }
    this.openness = options.openness;
    this.conscientiousness = options.conscientiousness;
    this.extraversion = options.extraversion;
    this.agreeableness = options.agreeableness;
    this.neuroticism = options.neuroticism;
    Object.freeze(this);
  }

  static random(
    lifeStage: LifeStage,
    options: RandomOptions = {},
  ): BigFiveTraitConfiguration {
    return new BigFiveTraitConfiguration({
      openness: BigFiveOpenness.random(lifeStage, options),
      conscientiousness: BigFiveConscientiousness.random(lifeStage, options),
      extraversion: BigFiveExtraversion.random(lifeStage, options),
      agreeableness: BigFiveAgreeableness.random(lifeStage, options),
      neuroticism: BigFiveNeuroticism.random(lifeStage, options),
    });
  }
}

export interface BigFiveConflictResolutionConfigurationOptions {
  readonly conflictResolutionStyle: ConflictStyle;
  readonly concernForSelf: Priority;
  readonly concernForOthers: Priority;
}

export class BigFiveConflictResolutionConfiguration {
  readonly conflictResolutionStyle: ConflictStyle;
  readonly concernForSelf: Priority;
  readonly concernForOthers: Priority;

  constructor(options: BigFiveConflictResolutionConfigurationOptions) {
    if (!CONFLICT_RESOLUTION_STYLES.includes(options.conflictResolutionStyle)) {
      throw new TypeError(
        `Unsupported conflict style: ${String(options.conflictResolutionStyle)}`,
      );
    }
    if (!PRIORITY_LEVELS.includes(options.concernForSelf)) {
      throw new TypeError(
        `Unsupported priority level: ${String(options.concernForSelf)}`,
      );
    }
    if (!PRIORITY_LEVELS.includes(options.concernForOthers)) {
      throw new TypeError(
        `Unsupported priority level: ${String(options.concernForOthers)}`,
      );
    }
    this.conflictResolutionStyle = options.conflictResolutionStyle;
    this.concernForSelf = options.concernForSelf;
    this.concernForOthers = options.concernForOthers;
    Object.freeze(this);
  }

  static random(
    traitConfiguration: BigFiveTraitConfiguration,
    options: RandomOptions = {},
  ): BigFiveConflictResolutionConfiguration {
    const style = weightedChoice(
      conflictStyleWeights(traitConfiguration),
      options.rng ?? DEFAULT_RANDOM_SOURCE,
    );
    const [concernForSelf, concernForOthers] = CONFLICT_CONCERN_MAPPINGS[style];
    return new BigFiveConflictResolutionConfiguration({
      conflictResolutionStyle: style,
      concernForSelf,
      concernForOthers,
    });
  }
}

export interface BigFivePersonalityOptions {
  readonly traitConfiguration: BigFiveTraitConfiguration;
  readonly conflictResolutionConfiguration: BigFiveConflictResolutionConfiguration;
}

export class BigFivePersonality {
  readonly traitConfiguration: BigFiveTraitConfiguration;
  readonly conflictResolutionConfiguration: BigFiveConflictResolutionConfiguration;

  constructor(options: BigFivePersonalityOptions) {
    if (!(options.traitConfiguration instanceof BigFiveTraitConfiguration)) {
      throw new TypeError("traitConfiguration must be a BigFiveTraitConfiguration.");
    }
    if (
      !(
        options.conflictResolutionConfiguration instanceof
        BigFiveConflictResolutionConfiguration
      )
    ) {
      throw new TypeError(
        "conflictResolutionConfiguration must be a BigFiveConflictResolutionConfiguration.",
      );
    }
    this.traitConfiguration = options.traitConfiguration;
    this.conflictResolutionConfiguration = options.conflictResolutionConfiguration;
    Object.freeze(this);
  }

  static random(lifeStage: LifeStage, options: RandomOptions = {}): BigFivePersonality {
    const traitConfiguration = BigFiveTraitConfiguration.random(lifeStage, options);
    return new BigFivePersonality({
      traitConfiguration,
      conflictResolutionConfiguration: BigFiveConflictResolutionConfiguration.random(
        traitConfiguration,
        options,
      ),
    });
  }
}
