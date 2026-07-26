import { ABBF_STANDARD_DEVIATION } from "./constants.js";
import {
  ADAPTIVE_BIFURCATED_DOMAINS,
  ADAPTIVE_BIFURCATED_POLES,
  AdaptiveBifurcatedDomain,
  AdaptiveBifurcatedPole,
  type AdaptiveBifurcatedDomain as Domain,
  type AdaptiveBifurcatedPole as Pole,
} from "./enums.js";
import type { BigFiveTraitConfiguration } from "./personality.js";
import { type RandomOptions, randomGaussian } from "./randomness.js";
import {
  cosineSimilarity,
  dotProduct,
  unitToSigned,
  validateSignedRange,
  weightedSignedAverage,
} from "./scoring.js";

export interface AdaptiveBifurcatedAxisOptions {
  readonly domain: Domain;
  readonly positivePole: Pole;
  readonly negativePole: Pole;
  readonly score: number;
}

export class AdaptiveBifurcatedAxis {
  readonly domain: Domain;
  readonly positivePole: Pole;
  readonly negativePole: Pole;
  readonly score: number;

  constructor(options: AdaptiveBifurcatedAxisOptions) {
    if (!ADAPTIVE_BIFURCATED_DOMAINS.includes(options.domain)) {
      throw new TypeError(`Unsupported ABBF domain: ${String(options.domain)}`);
    }
    if (!ADAPTIVE_BIFURCATED_POLES.includes(options.positivePole)) {
      throw new TypeError(`Unsupported ABBF pole: ${String(options.positivePole)}`);
    }
    if (!ADAPTIVE_BIFURCATED_POLES.includes(options.negativePole)) {
      throw new TypeError(`Unsupported ABBF pole: ${String(options.negativePole)}`);
    }
    validateSignedRange(options.score);
    this.domain = options.domain;
    this.positivePole = options.positivePole;
    this.negativePole = options.negativePole;
    this.score = options.score;
    Object.freeze(this);
  }

  get dominantPole(): Pole | undefined {
    if (this.score > 0) {
      return this.positivePole;
    }
    if (this.score < 0) {
      return this.negativePole;
    }
    return undefined;
  }
}

type AxisDefinition = Readonly<{
  readonly field: keyof AdaptiveBifurcatedProfileOptions;
  readonly domain: Domain;
  readonly positivePole: Pole;
  readonly negativePole: Pole;
}>;

export const AXIS_DEFINITIONS: readonly AxisDefinition[] = Object.freeze([
  Object.freeze({
    field: "orderScore",
    domain: AdaptiveBifurcatedDomain.Order,
    positivePole: AdaptiveBifurcatedPole.Strategizing,
    negativePole: AdaptiveBifurcatedPole.Improvisation,
  }),
  Object.freeze({
    field: "chaosScore",
    domain: AdaptiveBifurcatedDomain.Chaos,
    positivePole: AdaptiveBifurcatedPole.Ethicality,
    negativePole: AdaptiveBifurcatedPole.Instinctivity,
  }),
  Object.freeze({
    field: "cooperationScore",
    domain: AdaptiveBifurcatedDomain.Cooperation,
    positivePole: AdaptiveBifurcatedPole.Collaboration,
    negativePole: AdaptiveBifurcatedPole.Independence,
  }),
  Object.freeze({
    field: "conflictScore",
    domain: AdaptiveBifurcatedDomain.Conflict,
    positivePole: AdaptiveBifurcatedPole.Harmonizing,
    negativePole: AdaptiveBifurcatedPole.Utilitarianism,
  }),
  Object.freeze({
    field: "competitionScore",
    domain: AdaptiveBifurcatedDomain.Competition,
    positivePole: AdaptiveBifurcatedPole.Proficiency,
    negativePole: AdaptiveBifurcatedPole.Dominancy,
  }),
]);

export const ABBF_PROJECTION_COEFFICIENTS = Object.freeze({
  order: Object.freeze({ openness: 1 }),
  chaos: Object.freeze({
    agreeableness: 0.45,
    conscientiousness: 0.35,
    neuroticism: -0.2,
  }),
  cooperation: Object.freeze({
    extraversion: 0.55,
    agreeableness: 0.45,
  }),
  conflict: Object.freeze({
    agreeableness: 0.4,
    neuroticism: 0.3,
    openness: 0.2,
    conscientiousness: -0.1,
  }),
  competition: Object.freeze({
    conscientiousness: 0.5,
    agreeableness: 0.3,
    extraversion: -0.2,
  }),
});

export interface AdaptiveBifurcatedProfileOptions {
  readonly orderScore: number;
  readonly chaosScore: number;
  readonly cooperationScore: number;
  readonly conflictScore: number;
  readonly competitionScore: number;
}

function sampleSignedScore(options: RandomOptions): number {
  return randomGaussian({
    mean: 0,
    standardDeviation: ABBF_STANDARD_DEVIATION,
    minimum: -1,
    maximum: 1,
    ...options,
  });
}

export class AdaptiveBifurcatedProfile {
  readonly orderScore: number;
  readonly chaosScore: number;
  readonly cooperationScore: number;
  readonly conflictScore: number;
  readonly competitionScore: number;

  constructor(options: AdaptiveBifurcatedProfileOptions) {
    validateSignedRange(
      options.orderScore,
      options.chaosScore,
      options.cooperationScore,
      options.conflictScore,
      options.competitionScore,
    );
    this.orderScore = options.orderScore;
    this.chaosScore = options.chaosScore;
    this.cooperationScore = options.cooperationScore;
    this.conflictScore = options.conflictScore;
    this.competitionScore = options.competitionScore;
    Object.freeze(this);
  }

  static random(options: RandomOptions = {}): AdaptiveBifurcatedProfile {
    return new AdaptiveBifurcatedProfile({
      orderScore: sampleSignedScore(options),
      chaosScore: sampleSignedScore(options),
      cooperationScore: sampleSignedScore(options),
      conflictScore: sampleSignedScore(options),
      competitionScore: sampleSignedScore(options),
    });
  }

  static fromBigFive(traits: BigFiveTraitConfiguration): AdaptiveBifurcatedProfile {
    const openness = unitToSigned(traits.openness.score);
    const conscientiousness = unitToSigned(traits.conscientiousness.score);
    const extraversion = unitToSigned(traits.extraversion.score);
    const agreeableness = unitToSigned(traits.agreeableness.score);
    const neuroticism = unitToSigned(traits.neuroticism.score);

    return new AdaptiveBifurcatedProfile({
      orderScore: weightedSignedAverage([
        openness,
        ABBF_PROJECTION_COEFFICIENTS.order.openness,
      ]),
      chaosScore: weightedSignedAverage(
        [agreeableness, ABBF_PROJECTION_COEFFICIENTS.chaos.agreeableness],
        [conscientiousness, ABBF_PROJECTION_COEFFICIENTS.chaos.conscientiousness],
        [neuroticism, ABBF_PROJECTION_COEFFICIENTS.chaos.neuroticism],
      ),
      cooperationScore: weightedSignedAverage(
        [extraversion, ABBF_PROJECTION_COEFFICIENTS.cooperation.extraversion],
        [agreeableness, ABBF_PROJECTION_COEFFICIENTS.cooperation.agreeableness],
      ),
      conflictScore: weightedSignedAverage(
        [agreeableness, ABBF_PROJECTION_COEFFICIENTS.conflict.agreeableness],
        [neuroticism, ABBF_PROJECTION_COEFFICIENTS.conflict.neuroticism],
        [openness, ABBF_PROJECTION_COEFFICIENTS.conflict.openness],
        [conscientiousness, ABBF_PROJECTION_COEFFICIENTS.conflict.conscientiousness],
      ),
      competitionScore: weightedSignedAverage(
        [conscientiousness, ABBF_PROJECTION_COEFFICIENTS.competition.conscientiousness],
        [agreeableness, ABBF_PROJECTION_COEFFICIENTS.competition.agreeableness],
        [extraversion, ABBF_PROJECTION_COEFFICIENTS.competition.extraversion],
      ),
    });
  }

  get vector(): readonly [number, number, number, number, number] {
    return Object.freeze([
      this.orderScore,
      this.chaosScore,
      this.cooperationScore,
      this.conflictScore,
      this.competitionScore,
    ]);
  }

  get axes(): readonly AdaptiveBifurcatedAxis[] {
    return Object.freeze(
      AXIS_DEFINITIONS.map(
        ({ field, domain, positivePole, negativePole }) =>
          new AdaptiveBifurcatedAxis({
            domain,
            positivePole,
            negativePole,
            score: this[field],
          }),
      ),
    );
  }

  dominantPoles(threshold = 0): Readonly<Partial<Record<Domain, Pole>>> {
    if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
      throw new RangeError("threshold must be in the range 0.0...1.0");
    }
    const poles: Partial<Record<Domain, Pole>> = {};
    for (const axis of this.axes) {
      if (axis.score > threshold) {
        poles[axis.domain] = axis.positivePole;
      } else if (axis.score < -threshold) {
        poles[axis.domain] = axis.negativePole;
      }
    }
    return Object.freeze(poles);
  }

  dotProduct(other: AdaptiveBifurcatedProfile): number {
    return dotProduct(this.vector, other.vector);
  }

  cosineSimilarity(other: AdaptiveBifurcatedProfile): number {
    return cosineSimilarity(this.vector, other.vector);
  }
}
