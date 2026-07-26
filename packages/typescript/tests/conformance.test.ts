import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { test } from "node:test";
import { ABBF_PROJECTION_COEFFICIENTS, AXIS_DEFINITIONS } from "../dist/adaptive.js";
import {
  ABBF_STANDARD_DEVIATION,
  CDF_EPSILON,
  SIGNED_RANGE_MAX,
  SIGNED_RANGE_MIN,
  TRAIT_SAMPLE_MIN,
  UNIT_RANGE_MAX,
  UNIT_RANGE_MIN,
} from "../dist/constants.js";
import {
  CONFLICT_RESOLUTION_STYLES,
  LIFE_STAGES,
  PRIORITY_LEVELS,
} from "../dist/enums.js";
import {
  AdaptiveBifurcatedProfile,
  BigFiveAgreeableness,
  BigFiveConflictResolution,
  BigFiveConscientiousness,
  BigFiveExtraversion,
  BigFiveNeuroticism,
  BigFiveOpenness,
  BigFiveTraits,
  type LifeStage,
  type RandomSource,
} from "../dist/index.js";
import {
  CONFLICT_CONCERN_MAPPINGS,
  CONFLICT_MINIMUM_WEIGHT,
  CONFLICT_STYLE_COEFFICIENTS,
  conflictStyleWeights,
} from "../dist/personality.js";
import { randomGaussian } from "../dist/randomness.js";
import { componentAverageScore } from "../dist/scoring.js";
import { TRAIT_SAMPLING_CONFIGURATIONS } from "../dist/traits.js";

interface Model {
  readonly schemaVersion: number;
  readonly tolerances: {
    readonly arithmetic: number;
    readonly gaussian: number;
  };
  readonly ranges: {
    readonly unit: readonly number[];
    readonly traitSample: readonly number[];
    readonly signed: readonly number[];
  };
  readonly randomness: {
    readonly cdfEpsilon: number;
    readonly abbfStandardDeviation: number;
  };
  readonly lifeStages: readonly string[];
  readonly priorityLevels: readonly string[];
  readonly traits: Readonly<
    Record<
      string,
      {
        readonly components: readonly string[];
        readonly standardDeviation: number;
        readonly means: Readonly<Record<string, readonly number[]>>;
      }
    >
  >;
  readonly conflictResolution: {
    readonly minimumWeight: number;
    readonly styles: Readonly<
      Record<
        string,
        {
          readonly coefficients: Readonly<Record<string, number>>;
          readonly concerns: readonly string[];
        }
      >
    >;
  };
  readonly adaptive: {
    readonly axes: readonly {
      readonly domain: string;
      readonly positivePole: string;
      readonly negativePole: string;
    }[];
    readonly projection: Readonly<Record<string, Readonly<Record<string, number>>>>;
  };
}

const specRoot = new URL("../../../spec/", import.meta.url);

function loadJson<T>(path: string): T {
  return JSON.parse(readFileSync(new URL(path, specRoot), "utf8")) as T;
}

const model = loadJson<Model>("model.json");
const arithmeticTolerance = model.tolerances.arithmetic;
const gaussianTolerance = model.tolerances.gaussian;
const CONFORMANCE_FIXTURES = Object.freeze({
  adaptive: "adaptive.json",
  aggregateScoring: "aggregate-scoring.json",
  conflictResolution: "conflict-resolution.json",
  gaussianSampling: "gaussian-sampling.json",
  lifeStageSampling: "life-stage-sampling.json",
  validation: "validation.json",
});
const TRAIT_COMPONENT_PROPERTIES = Object.freeze({
  openness: Object.freeze({
    aesthetic_sensitivity: "aestheticSensitivityScore",
    creative_imagination: "creativeImaginationScore",
    intellectual_curiosity: "intellectualCuriosityScore",
  }),
  conscientiousness: Object.freeze({
    organization: "organizationScore",
    responsibility: "responsibilityScore",
    productivity: "productivityScore",
  }),
  extraversion: Object.freeze({
    assertiveness: "assertivenessScore",
    sociability: "sociabilityScore",
    energy_level: "energyLevelScore",
  }),
  agreeableness: Object.freeze({
    compassion: "compassionScore",
    respectfulness: "respectfulnessScore",
    trust: "trustScore",
  }),
  neuroticism: Object.freeze({
    anxiety: "anxietyScore",
    emotional_volatility: "emotionalVolatilityScore",
    depression: "depressionScore",
  }),
});

function assertClose(actual: number, expected: number, tolerance: number): void {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `expected ${actual} to be within ${tolerance} of ${expected}`,
  );
}

function fractionSource(fractions: readonly number[]): RandomSource {
  let index = 0;
  return {
    uniform(minimum, maximum) {
      const fraction = fractions[index];
      assert.notEqual(fraction, undefined, "random fixture was exhausted");
      index += 1;
      return minimum + (maximum - minimum) * (fraction as number);
    },
  };
}

function bigFiveTraits(
  scores: Readonly<{
    openness: number;
    conscientiousness: number;
    extraversion: number;
    agreeableness: number;
    neuroticism: number;
  }>,
): BigFiveTraits {
  return new BigFiveTraits({
    openness: new BigFiveOpenness({
      aestheticSensitivityScore: scores.openness,
      creativeImaginationScore: scores.openness,
      intellectualCuriosityScore: scores.openness,
    }),
    conscientiousness: new BigFiveConscientiousness({
      organizationScore: scores.conscientiousness,
      responsibilityScore: scores.conscientiousness,
      productivityScore: scores.conscientiousness,
    }),
    extraversion: new BigFiveExtraversion({
      assertivenessScore: scores.extraversion,
      sociabilityScore: scores.extraversion,
      energyLevelScore: scores.extraversion,
    }),
    agreeableness: new BigFiveAgreeableness({
      compassionScore: scores.agreeableness,
      respectfulnessScore: scores.agreeableness,
      trustScore: scores.agreeableness,
    }),
    neuroticism: new BigFiveNeuroticism({
      anxietyScore: scores.neuroticism,
      emotionalVolatilityScore: scores.neuroticism,
      depressionScore: scores.neuroticism,
    }),
  });
}

function profile(vector: readonly number[]): AdaptiveBifurcatedProfile {
  return new AdaptiveBifurcatedProfile({
    orderScore: vector[0] as number,
    chaosScore: vector[1] as number,
    cooperationScore: vector[2] as number,
    conflictScore: vector[3] as number,
    competitionScore: vector[4] as number,
  });
}

test("implementation constants match the language-neutral model", () => {
  assert.equal(model.schemaVersion, 1);
  assert.deepEqual([UNIT_RANGE_MIN, UNIT_RANGE_MAX], model.ranges.unit);
  assert.deepEqual([TRAIT_SAMPLE_MIN, UNIT_RANGE_MAX], model.ranges.traitSample);
  assert.deepEqual([SIGNED_RANGE_MIN, SIGNED_RANGE_MAX], model.ranges.signed);
  assert.equal(CDF_EPSILON, model.randomness.cdfEpsilon);
  assert.equal(ABBF_STANDARD_DEVIATION, model.randomness.abbfStandardDeviation);
  assert.deepEqual(LIFE_STAGES, model.lifeStages);
  assert.deepEqual(PRIORITY_LEVELS, model.priorityLevels);
  assert.deepEqual(
    CONFLICT_RESOLUTION_STYLES,
    Object.keys(model.conflictResolution.styles),
  );
  assert.deepEqual(Object.keys(TRAIT_SAMPLING_CONFIGURATIONS), [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
  ]);
  for (const [name, expected] of Object.entries(model.traits)) {
    const actual =
      TRAIT_SAMPLING_CONFIGURATIONS[name as keyof typeof TRAIT_SAMPLING_CONFIGURATIONS];
    assert.equal(actual.standardDeviation, expected.standardDeviation);
    assert.deepEqual(actual.means, expected.means);
    assert.deepEqual(
      Object.keys(
        TRAIT_COMPONENT_PROPERTIES[name as keyof typeof TRAIT_COMPONENT_PROPERTIES],
      ),
      expected.components,
    );
  }
  assert.equal(CONFLICT_MINIMUM_WEIGHT, model.conflictResolution.minimumWeight);
  for (const [style, expected] of Object.entries(model.conflictResolution.styles)) {
    assert.deepEqual(
      CONFLICT_STYLE_COEFFICIENTS[style as keyof typeof CONFLICT_STYLE_COEFFICIENTS],
      expected.coefficients,
    );
    assert.deepEqual(
      CONFLICT_CONCERN_MAPPINGS[style as keyof typeof CONFLICT_CONCERN_MAPPINGS],
      expected.concerns,
    );
  }
  assert.deepEqual(
    AXIS_DEFINITIONS.map(({ domain, positivePole, negativePole }) => ({
      domain,
      positivePole,
      negativePole,
    })),
    model.adaptive.axes,
  );
  assert.deepEqual(ABBF_PROJECTION_COEFFICIENTS, model.adaptive.projection);
});

test("conformance fixture set is explicit", () => {
  const actual = readdirSync(new URL("conformance/", specRoot))
    .filter((path) => path.endsWith(".json"))
    .sort();
  assert.deepEqual(actual, Object.values(CONFORMANCE_FIXTURES).sort());
});

test("aggregate scoring fixtures", () => {
  const fixtures = loadJson<{
    readonly cases: readonly {
      readonly components: readonly number[];
      readonly expected: number;
    }[];
  }>(`conformance/${CONFORMANCE_FIXTURES.aggregateScoring}`);
  for (const fixture of fixtures.cases) {
    assertClose(
      componentAverageScore(...fixture.components),
      fixture.expected,
      arithmeticTolerance,
    );
  }
});

test("validation fixtures", () => {
  const fixtures = loadJson<{
    readonly unitScores: {
      readonly valid: readonly number[];
      readonly invalid: readonly number[];
    };
    readonly signedScores: {
      readonly valid: readonly number[];
      readonly invalid: readonly number[];
    };
    readonly dominantPoleThresholds: {
      readonly valid: readonly number[];
      readonly invalid: readonly number[];
    };
    readonly gaussianParameters: {
      readonly invalid: readonly {
        readonly mean: number;
        readonly standardDeviation: number;
        readonly minimum: number;
        readonly maximum: number;
      }[];
    };
  }>(`conformance/${CONFORMANCE_FIXTURES.validation}`);
  for (const score of fixtures.unitScores.valid) {
    assert.doesNotThrow(
      () =>
        new BigFiveOpenness({
          aestheticSensitivityScore: score,
          creativeImaginationScore: score,
          intellectualCuriosityScore: score,
        }),
    );
  }
  for (const score of fixtures.unitScores.invalid) {
    assert.throws(
      () =>
        new BigFiveOpenness({
          aestheticSensitivityScore: score,
          creativeImaginationScore: 0.5,
          intellectualCuriosityScore: 0.5,
        }),
      RangeError,
    );
  }
  for (const score of fixtures.signedScores.valid) {
    assert.doesNotThrow(() => profile([score, 0, 0, 0, 0]));
  }
  for (const score of fixtures.signedScores.invalid) {
    assert.throws(() => profile([score, 0, 0, 0, 0]), RangeError);
  }
  const neutral = profile([0, 0, 0, 0, 0]);
  for (const threshold of fixtures.dominantPoleThresholds.valid) {
    assert.doesNotThrow(() => neutral.dominantPoles(threshold));
  }
  for (const threshold of fixtures.dominantPoleThresholds.invalid) {
    assert.throws(() => neutral.dominantPoles(threshold), RangeError);
  }
  for (const parameters of fixtures.gaussianParameters.invalid) {
    assert.throws(
      () =>
        randomGaussian({
          ...parameters,
          rng: fractionSource([0.5]),
        }),
      RangeError,
    );
  }
});

test("truncated Gaussian fixtures", () => {
  const fixtures = loadJson<{
    readonly cases: readonly {
      readonly mean: number;
      readonly standardDeviation: number;
      readonly minimum: number;
      readonly maximum: number;
      readonly uniformFraction: number;
      readonly expected: number;
    }[];
  }>(`conformance/${CONFORMANCE_FIXTURES.gaussianSampling}`);
  for (const fixture of fixtures.cases) {
    assertClose(
      randomGaussian({
        mean: fixture.mean,
        standardDeviation: fixture.standardDeviation,
        minimum: fixture.minimum,
        maximum: fixture.maximum,
        rng: fractionSource([fixture.uniformFraction]),
      }),
      fixture.expected,
      gaussianTolerance,
    );
  }
});

test("life-stage generation fixtures", () => {
  const fixtures = loadJson<{
    readonly cases: readonly {
      readonly lifeStage: LifeStage;
      readonly uniformFractions: readonly number[];
      readonly expected: Readonly<Record<string, readonly number[]>>;
    }[];
  }>(`conformance/${CONFORMANCE_FIXTURES.lifeStageSampling}`);
  for (const fixture of fixtures.cases) {
    const traits = BigFiveTraits.random(fixture.lifeStage, {
      rng: fractionSource(fixture.uniformFractions),
    });
    const actual = Object.fromEntries(
      Object.entries(TRAIT_COMPONENT_PROPERTIES).map(
        ([traitName, componentProperties]) => {
          const trait = traits[
            traitName as keyof typeof TRAIT_COMPONENT_PROPERTIES
          ] as unknown as Readonly<Record<string, number>>;
          return [
            traitName,
            Object.values(componentProperties).map((propertyName) => {
              const value = trait[propertyName];
              assert.equal(
                typeof value,
                "number",
                `${traitName}.${propertyName} is not a numeric public property`,
              );
              return value as number;
            }),
          ];
        },
      ),
    );
    for (const [name, values] of Object.entries(actual)) {
      const expected = fixture.expected[name] as readonly number[];
      values.forEach((value, index) => {
        assertClose(value, expected[index] as number, gaussianTolerance);
      });
    }
  }
});

test("weighted conflict selection fixtures", () => {
  const fixtures = loadJson<{
    readonly cases: readonly {
      readonly traits: Readonly<{
        openness: number;
        conscientiousness: number;
        extraversion: number;
        agreeableness: number;
        neuroticism: number;
      }>;
      readonly weights: Readonly<Record<string, number>>;
      readonly selections: readonly {
        readonly uniformFraction: number;
        readonly expected: string;
      }[];
    }[];
  }>(`conformance/${CONFORMANCE_FIXTURES.conflictResolution}`);
  for (const fixture of fixtures.cases) {
    const traits = bigFiveTraits(fixture.traits);
    const weights = conflictStyleWeights(traits);
    for (const [style, expected] of Object.entries(fixture.weights)) {
      assertClose(
        weights[style as keyof typeof weights],
        expected,
        arithmeticTolerance,
      );
    }
    for (const selection of fixture.selections) {
      assert.equal(
        BigFiveConflictResolution.random(traits, {
          rng: fractionSource([selection.uniformFraction]),
        }).style,
        selection.expected,
      );
    }
  }
});

test("ABBF projection, poles, dot products, and cosine fixtures", () => {
  const fixtures = loadJson<{
    readonly projectionCases: readonly {
      readonly bigFive: Readonly<{
        openness: number;
        conscientiousness: number;
        extraversion: number;
        agreeableness: number;
        neuroticism: number;
      }>;
      readonly expected: readonly number[];
    }[];
    readonly dominantPoleCases: readonly {
      readonly vector: readonly number[];
      readonly threshold: number;
      readonly expected: Readonly<Record<string, string>>;
    }[];
    readonly vectorCases: readonly {
      readonly left: readonly number[];
      readonly right: readonly number[];
      readonly dotProduct: number;
      readonly cosineSimilarity: number;
    }[];
  }>(`conformance/${CONFORMANCE_FIXTURES.adaptive}`);
  for (const fixture of fixtures.projectionCases) {
    const actual = AdaptiveBifurcatedProfile.fromBigFive(
      bigFiveTraits(fixture.bigFive),
    ).vector;
    actual.forEach((value, index) => {
      assertClose(value, fixture.expected[index] as number, arithmeticTolerance);
    });
  }
  for (const fixture of fixtures.dominantPoleCases) {
    assert.deepEqual(
      profile(fixture.vector).dominantPoles(fixture.threshold),
      fixture.expected,
    );
  }
  for (const fixture of fixtures.vectorCases) {
    const left = profile(fixture.left);
    const right = profile(fixture.right);
    assertClose(left.dotProduct(right), fixture.dotProduct, arithmeticTolerance);
    assertClose(
      left.cosineSimilarity(right),
      fixture.cosineSimilarity,
      arithmeticTolerance,
    );
  }
});
