import { TRAIT_SAMPLE_MIN, UNIT_RANGE_MAX } from "./constants.js";
import { assertLifeStage, LifeStage } from "./enums.js";
import { type RandomOptions, type RandomSource, randomGaussian } from "./randomness.js";
import { componentAverageScore } from "./scoring.js";

type TraitMeans = readonly [number, number, number];

interface TraitSamplingConfiguration {
  readonly standardDeviation: number;
  readonly means: Readonly<Record<LifeStage, TraitMeans>>;
}

function traitMeans(first: number, second: number, third: number): TraitMeans {
  return Object.freeze([first, second, third]);
}

function sampleTrait(
  lifeStage: LifeStage,
  configuration: TraitSamplingConfiguration,
  rng?: RandomSource,
): TraitMeans {
  assertLifeStage(lifeStage);
  const means = configuration.means[lifeStage];
  return [
    randomGaussian({
      mean: means[0],
      standardDeviation: configuration.standardDeviation,
      minimum: TRAIT_SAMPLE_MIN,
      maximum: UNIT_RANGE_MAX,
      ...(rng === undefined ? {} : { rng }),
    }),
    randomGaussian({
      mean: means[1],
      standardDeviation: configuration.standardDeviation,
      minimum: TRAIT_SAMPLE_MIN,
      maximum: UNIT_RANGE_MAX,
      ...(rng === undefined ? {} : { rng }),
    }),
    randomGaussian({
      mean: means[2],
      standardDeviation: configuration.standardDeviation,
      minimum: TRAIT_SAMPLE_MIN,
      maximum: UNIT_RANGE_MAX,
      ...(rng === undefined ? {} : { rng }),
    }),
  ];
}

// Exported for compiled conformance tests; the package export map keeps this module internal.
export const TRAIT_SAMPLING_CONFIGURATIONS = Object.freeze({
  openness: Object.freeze({
    standardDeviation: 0.16,
    means: Object.freeze({
      [LifeStage.Child]: traitMeans(0.8, 0.85, 0.85),
      [LifeStage.YoungAdult]: traitMeans(0.7, 0.75, 0.75),
      [LifeStage.Adult]: traitMeans(0.6, 0.65, 0.65),
    }),
  }),
  conscientiousness: Object.freeze({
    standardDeviation: 0.22,
    means: Object.freeze({
      [LifeStage.Child]: traitMeans(0.5, 0.55, 0.5),
      [LifeStage.YoungAdult]: traitMeans(0.6, 0.65, 0.6),
      [LifeStage.Adult]: traitMeans(0.7, 0.75, 0.7),
    }),
  }),
  extraversion: Object.freeze({
    standardDeviation: 0.27,
    means: Object.freeze({
      [LifeStage.Child]: traitMeans(0.72, 0.7, 0.72),
      [LifeStage.YoungAdult]: traitMeans(0.62, 0.6, 0.62),
      [LifeStage.Adult]: traitMeans(0.52, 0.5, 0.52),
    }),
  }),
  agreeableness: Object.freeze({
    standardDeviation: 0.18,
    means: Object.freeze({
      [LifeStage.Child]: traitMeans(0.55, 0.55, 0.4),
      [LifeStage.YoungAdult]: traitMeans(0.65, 0.65, 0.5),
      [LifeStage.Adult]: traitMeans(0.75, 0.75, 0.6),
    }),
  }),
  neuroticism: Object.freeze({
    standardDeviation: 0.32,
    means: Object.freeze({
      [LifeStage.Child]: traitMeans(0.7, 0.6, 0.55),
      [LifeStage.YoungAdult]: traitMeans(0.6, 0.5, 0.45),
      [LifeStage.Adult]: traitMeans(0.5, 0.4, 0.35),
    }),
  }),
} satisfies Readonly<Record<string, TraitSamplingConfiguration>>);

export interface BigFiveOpennessScores {
  readonly aestheticSensitivityScore: number;
  readonly creativeImaginationScore: number;
  readonly intellectualCuriosityScore: number;
}

export class BigFiveOpenness {
  readonly aestheticSensitivityScore: number;
  readonly creativeImaginationScore: number;
  readonly intellectualCuriosityScore: number;
  readonly score: number;

  constructor(scores: BigFiveOpennessScores) {
    this.aestheticSensitivityScore = scores.aestheticSensitivityScore;
    this.creativeImaginationScore = scores.creativeImaginationScore;
    this.intellectualCuriosityScore = scores.intellectualCuriosityScore;
    this.score = componentAverageScore(
      this.aestheticSensitivityScore,
      this.creativeImaginationScore,
      this.intellectualCuriosityScore,
    );
    Object.freeze(this);
  }

  static random(lifeStage: LifeStage, options: RandomOptions = {}): BigFiveOpenness {
    const [aestheticSensitivity, creativeImagination, intellectualCuriosity] =
      sampleTrait(lifeStage, TRAIT_SAMPLING_CONFIGURATIONS.openness, options.rng);
    return new BigFiveOpenness({
      aestheticSensitivityScore: aestheticSensitivity,
      creativeImaginationScore: creativeImagination,
      intellectualCuriosityScore: intellectualCuriosity,
    });
  }
}

export interface BigFiveConscientiousnessScores {
  readonly organizationScore: number;
  readonly responsibilityScore: number;
  readonly productivityScore: number;
}

export class BigFiveConscientiousness {
  readonly organizationScore: number;
  readonly responsibilityScore: number;
  readonly productivityScore: number;
  readonly score: number;

  constructor(scores: BigFiveConscientiousnessScores) {
    this.organizationScore = scores.organizationScore;
    this.responsibilityScore = scores.responsibilityScore;
    this.productivityScore = scores.productivityScore;
    this.score = componentAverageScore(
      this.organizationScore,
      this.responsibilityScore,
      this.productivityScore,
    );
    Object.freeze(this);
  }

  static random(
    lifeStage: LifeStage,
    options: RandomOptions = {},
  ): BigFiveConscientiousness {
    const [organization, responsibility, productivity] = sampleTrait(
      lifeStage,
      TRAIT_SAMPLING_CONFIGURATIONS.conscientiousness,
      options.rng,
    );
    return new BigFiveConscientiousness({
      organizationScore: organization,
      responsibilityScore: responsibility,
      productivityScore: productivity,
    });
  }
}

export interface BigFiveExtraversionScores {
  readonly assertivenessScore: number;
  readonly sociabilityScore: number;
  readonly energyLevelScore: number;
}

export class BigFiveExtraversion {
  readonly assertivenessScore: number;
  readonly sociabilityScore: number;
  readonly energyLevelScore: number;
  readonly score: number;

  constructor(scores: BigFiveExtraversionScores) {
    this.assertivenessScore = scores.assertivenessScore;
    this.sociabilityScore = scores.sociabilityScore;
    this.energyLevelScore = scores.energyLevelScore;
    this.score = componentAverageScore(
      this.assertivenessScore,
      this.sociabilityScore,
      this.energyLevelScore,
    );
    Object.freeze(this);
  }

  static random(
    lifeStage: LifeStage,
    options: RandomOptions = {},
  ): BigFiveExtraversion {
    const [assertiveness, sociability, energyLevel] = sampleTrait(
      lifeStage,
      TRAIT_SAMPLING_CONFIGURATIONS.extraversion,
      options.rng,
    );
    return new BigFiveExtraversion({
      assertivenessScore: assertiveness,
      sociabilityScore: sociability,
      energyLevelScore: energyLevel,
    });
  }
}

export interface BigFiveAgreeablenessScores {
  readonly compassionScore: number;
  readonly respectfulnessScore: number;
  readonly trustScore: number;
}

export class BigFiveAgreeableness {
  readonly compassionScore: number;
  readonly respectfulnessScore: number;
  readonly trustScore: number;
  readonly score: number;

  constructor(scores: BigFiveAgreeablenessScores) {
    this.compassionScore = scores.compassionScore;
    this.respectfulnessScore = scores.respectfulnessScore;
    this.trustScore = scores.trustScore;
    this.score = componentAverageScore(
      this.compassionScore,
      this.respectfulnessScore,
      this.trustScore,
    );
    Object.freeze(this);
  }

  static random(
    lifeStage: LifeStage,
    options: RandomOptions = {},
  ): BigFiveAgreeableness {
    const [compassion, respectfulness, trust] = sampleTrait(
      lifeStage,
      TRAIT_SAMPLING_CONFIGURATIONS.agreeableness,
      options.rng,
    );
    return new BigFiveAgreeableness({
      compassionScore: compassion,
      respectfulnessScore: respectfulness,
      trustScore: trust,
    });
  }
}

export interface BigFiveNeuroticismScores {
  readonly anxietyScore: number;
  readonly emotionalVolatilityScore: number;
  readonly depressionScore: number;
}

export class BigFiveNeuroticism {
  readonly anxietyScore: number;
  readonly emotionalVolatilityScore: number;
  readonly depressionScore: number;
  readonly score: number;

  constructor(scores: BigFiveNeuroticismScores) {
    this.anxietyScore = scores.anxietyScore;
    this.emotionalVolatilityScore = scores.emotionalVolatilityScore;
    this.depressionScore = scores.depressionScore;
    this.score = componentAverageScore(
      this.anxietyScore,
      this.emotionalVolatilityScore,
      this.depressionScore,
    );
    Object.freeze(this);
  }

  static random(lifeStage: LifeStage, options: RandomOptions = {}): BigFiveNeuroticism {
    const [anxiety, emotionalVolatility, depression] = sampleTrait(
      lifeStage,
      TRAIT_SAMPLING_CONFIGURATIONS.neuroticism,
      options.rng,
    );
    return new BigFiveNeuroticism({
      anxietyScore: anxiety,
      emotionalVolatilityScore: emotionalVolatility,
      depressionScore: depression,
    });
  }
}
