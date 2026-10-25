import assert from "node:assert/strict";
import { readdirSync } from "node:fs";
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
  BigFivePersonality,
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
import { Fixture, loadFixture } from "./fixture-data.js";

const specRoot = new URL("../../../spec/", import.meta.url);
const model = loadFixture(new URL("model.json", specRoot));
const arithmeticTolerance = model.mapping("tolerances").number("arithmetic");
const gaussianTolerance = model.mapping("tolerances").number("gaussian");
const CONFORMANCE_FIXTURES = Object.freeze({
  adaptive: "adaptive.json",
  aggregateScoring: "aggregate-scoring.json",
  conflictResolution: "conflict-resolution.json",
  gaussianSampling: "gaussian-sampling.json",
  lifeStageSampling: "life-stage-sampling.json",
  validation: "validation.json",
});

function traitComponents(traits: BigFiveTraits) {
  return {
    openness: {
      aesthetic_sensitivity: traits.openness.aestheticSensitivityScore,
      creative_imagination: traits.openness.creativeImaginationScore,
      intellectual_curiosity: traits.openness.intellectualCuriosityScore,
    },
    conscientiousness: {
      organization: traits.conscientiousness.organizationScore,
      responsibility: traits.conscientiousness.responsibilityScore,
      productivity: traits.conscientiousness.productivityScore,
    },
    extraversion: {
      assertiveness: traits.extraversion.assertivenessScore,
      sociability: traits.extraversion.sociabilityScore,
      energy_level: traits.extraversion.energyLevelScore,
    },
    agreeableness: {
      compassion: traits.agreeableness.compassionScore,
      respectfulness: traits.agreeableness.respectfulnessScore,
      trust: traits.agreeableness.trustScore,
    },
    neuroticism: {
      anxiety: traits.neuroticism.anxietyScore,
      emotional_volatility: traits.neuroticism.emotionalVolatilityScore,
      depression: traits.neuroticism.depressionScore,
    },
  };
}

function lifeStage(fixture: Fixture): LifeStage {
  const stage = fixture.string("lifeStage");
  for (const allowed of LIFE_STAGES) {
    if (stage === allowed) {
      return allowed;
    }
  }
  throw new TypeError(`Unknown life stage: ${stage}`);
}

function fixtureFile(name: string): Fixture {
  return loadFixture(new URL(`conformance/${name}`, specRoot));
}

function assertClose(actual: number, expected: number, tolerance: number): void {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `expected ${actual} to be within ${tolerance} of ${expected}`,
  );
}

function fractionSource(
  fractions: readonly number[],
): RandomSource & { assertConsumed(): void } {
  let index = 0;
  return {
    uniform(minimum, maximum) {
      const fraction = fractions[index];
      assert.ok(fraction !== undefined, "random fixture was exhausted");
      index += 1;
      return minimum + (maximum - minimum) * fraction;
    },
    assertConsumed() {
      assert.equal(index, fractions.length, "random fixture has unused draws");
    },
  };
}

function bigFiveTraits(scores: Fixture): BigFiveTraits {
  return new BigFiveTraits({
    openness: new BigFiveOpenness({
      aestheticSensitivityScore: scores.number("openness"),
      creativeImaginationScore: scores.number("openness"),
      intellectualCuriosityScore: scores.number("openness"),
    }),
    conscientiousness: new BigFiveConscientiousness({
      organizationScore: scores.number("conscientiousness"),
      responsibilityScore: scores.number("conscientiousness"),
      productivityScore: scores.number("conscientiousness"),
    }),
    extraversion: new BigFiveExtraversion({
      assertivenessScore: scores.number("extraversion"),
      sociabilityScore: scores.number("extraversion"),
      energyLevelScore: scores.number("extraversion"),
    }),
    agreeableness: new BigFiveAgreeableness({
      compassionScore: scores.number("agreeableness"),
      respectfulnessScore: scores.number("agreeableness"),
      trustScore: scores.number("agreeableness"),
    }),
    neuroticism: new BigFiveNeuroticism({
      anxietyScore: scores.number("neuroticism"),
      emotionalVolatilityScore: scores.number("neuroticism"),
      depressionScore: scores.number("neuroticism"),
    }),
  });
}

function profile(vector: readonly number[]): AdaptiveBifurcatedProfile {
  assert.equal(vector.length, 5);
  const [orderScore, chaosScore, cooperationScore, conflictScore, competitionScore] =
    vector;
  assert.ok(orderScore !== undefined);
  assert.ok(chaosScore !== undefined);
  assert.ok(cooperationScore !== undefined);
  assert.ok(conflictScore !== undefined);
  assert.ok(competitionScore !== undefined);
  return new AdaptiveBifurcatedProfile({
    orderScore,
    chaosScore,
    cooperationScore,
    conflictScore,
    competitionScore,
  });
}

function assertTraitComponents(traits: BigFiveTraits, expected: Fixture): void {
  const components = traitComponents(traits);
  assert.deepEqual(expected.keys(), Object.keys(components));
  for (const [traitName, properties] of Object.entries(components)) {
    const values = expected.numbers(traitName);
    assert.equal(values.length, Object.keys(properties).length);
    Object.entries(properties).forEach(([propertyName, actual], index) => {
      assert.equal(
        typeof actual,
        "number",
        `${traitName}.${propertyName} is not a numeric public property`,
      );
      const value = values[index];
      assert.ok(value !== undefined);
      assertClose(actual, value, gaussianTolerance);
    });
  }
}

test("implementation constants match the language-neutral model", () => {
  assert.equal(model.number("schemaVersion"), 1);
  const ranges = model.mapping("ranges");
  assert.deepEqual([UNIT_RANGE_MIN, UNIT_RANGE_MAX], ranges.numbers("unit"));
  assert.deepEqual([TRAIT_SAMPLE_MIN, UNIT_RANGE_MAX], ranges.numbers("traitSample"));
  assert.deepEqual([SIGNED_RANGE_MIN, SIGNED_RANGE_MAX], ranges.numbers("signed"));
  const randomness = model.mapping("randomness");
  assert.equal(CDF_EPSILON, randomness.number("cdfEpsilon"));
  assert.equal(ABBF_STANDARD_DEVIATION, randomness.number("abbfStandardDeviation"));
  assert.deepEqual(LIFE_STAGES, model.strings("lifeStages"));
  assert.deepEqual(PRIORITY_LEVELS, model.strings("priorityLevels"));
  const conflict = model.mapping("conflictResolution");
  const styles = conflict.mapping("styles");
  assert.deepEqual(CONFLICT_RESOLUTION_STYLES, styles.keys());
  assert.deepEqual(Object.keys(TRAIT_SAMPLING_CONFIGURATIONS), [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
  ]);
  const modelTraits = model.mapping("traits");
  assert.deepEqual(modelTraits.keys(), Object.keys(TRAIT_SAMPLING_CONFIGURATIONS));
  const components = traitComponents(
    bigFiveTraits(
      new Fixture({
        openness: 0.5,
        conscientiousness: 0.5,
        extraversion: 0.5,
        agreeableness: 0.5,
        neuroticism: 0.5,
      }),
    ),
  );
  for (const [name, actual] of Object.entries(TRAIT_SAMPLING_CONFIGURATIONS)) {
    const expected = modelTraits.mapping(name);
    assert.equal(actual.standardDeviation, expected.number("standardDeviation"));
    const means = expected.mapping("means");
    assert.deepEqual(Object.keys(actual.means), means.keys());
    for (const [stage, actualMeans] of Object.entries(actual.means)) {
      assert.deepEqual(actualMeans, means.numbers(stage));
    }
  }
  for (const [name, properties] of Object.entries(components)) {
    assert.deepEqual(
      Object.keys(properties),
      modelTraits.mapping(name).strings("components"),
    );
  }
  assert.equal(CONFLICT_MINIMUM_WEIGHT, conflict.number("minimumWeight"));
  for (const [style, coefficients] of Object.entries(CONFLICT_STYLE_COEFFICIENTS)) {
    assert.deepEqual(
      coefficients,
      styles.mapping(style).mapping("coefficients").numberValues(),
    );
  }
  assert.deepEqual(Object.keys(CONFLICT_STYLE_COEFFICIENTS), styles.keys());
  for (const [style, concerns] of Object.entries(CONFLICT_CONCERN_MAPPINGS)) {
    assert.deepEqual(concerns, styles.mapping(style).strings("concerns"));
  }
  assert.deepEqual(Object.keys(CONFLICT_CONCERN_MAPPINGS), styles.keys());
  const adaptive = model.mapping("adaptive");
  assert.deepEqual(
    AXIS_DEFINITIONS.map(({ domain, positivePole, negativePole }) => ({
      domain,
      positivePole,
      negativePole,
    })),
    adaptive.objects("axes").map((axis) => {
      assert.deepEqual(axis.keys(), ["domain", "positivePole", "negativePole"]);
      return {
        domain: axis.string("domain"),
        positivePole: axis.string("positivePole"),
        negativePole: axis.string("negativePole"),
      };
    }),
  );
  const projection = adaptive.mapping("projection");
  assert.deepEqual(Object.keys(ABBF_PROJECTION_COEFFICIENTS), projection.keys());
  for (const [domain, coefficients] of Object.entries(ABBF_PROJECTION_COEFFICIENTS)) {
    assert.deepEqual(coefficients, projection.mapping(domain).numberValues());
  }
});

test("conformance fixture set is explicit", () => {
  const actual = readdirSync(new URL("conformance/", specRoot))
    .filter((path) => path.endsWith(".json"))
    .sort();
  assert.deepEqual(actual, Object.values(CONFORMANCE_FIXTURES).sort());
});

test("aggregate scoring fixtures", () => {
  const fixtures = fixtureFile(CONFORMANCE_FIXTURES.aggregateScoring);
  for (const fixture of fixtures.objects("cases")) {
    assertClose(
      componentAverageScore(...fixture.numbers("components")),
      fixture.number("expected"),
      arithmeticTolerance,
    );
  }
});

test("validation fixtures", () => {
  const fixtures = fixtureFile(CONFORMANCE_FIXTURES.validation);
  for (const score of fixtures.mapping("unitScores").numbers("valid")) {
    assert.doesNotThrow(
      () =>
        new BigFiveOpenness({
          aestheticSensitivityScore: score,
          creativeImaginationScore: score,
          intellectualCuriosityScore: score,
        }),
    );
  }
  for (const score of fixtures.mapping("unitScores").numbers("invalid")) {
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
  for (const score of fixtures.mapping("signedScores").numbers("valid")) {
    assert.doesNotThrow(() => profile([score, 0, 0, 0, 0]));
  }
  for (const score of fixtures.mapping("signedScores").numbers("invalid")) {
    assert.throws(() => profile([score, 0, 0, 0, 0]), RangeError);
  }
  const neutral = profile([0, 0, 0, 0, 0]);
  for (const threshold of fixtures.mapping("dominantPoleThresholds").numbers("valid")) {
    assert.doesNotThrow(() => neutral.dominantPoles(threshold));
  }
  for (const threshold of fixtures
    .mapping("dominantPoleThresholds")
    .numbers("invalid")) {
    assert.throws(() => neutral.dominantPoles(threshold), RangeError);
  }
  for (const parameters of fixtures.mapping("gaussianParameters").objects("invalid")) {
    assert.throws(
      () =>
        randomGaussian({
          mean: parameters.number("mean"),
          standardDeviation: parameters.number("standardDeviation"),
          minimum: parameters.number("minimum"),
          maximum: parameters.number("maximum"),
          rng: fractionSource([0.5]),
        }),
      RangeError,
    );
  }
});

test("truncated Gaussian fixtures", () => {
  const fixtures = fixtureFile(CONFORMANCE_FIXTURES.gaussianSampling);
  for (const fixture of fixtures.objects("cases")) {
    assertClose(
      randomGaussian({
        mean: fixture.number("mean"),
        standardDeviation: fixture.number("standardDeviation"),
        minimum: fixture.number("minimum"),
        maximum: fixture.number("maximum"),
        rng: fractionSource([fixture.number("uniformFraction")]),
      }),
      fixture.number("expected"),
      gaussianTolerance,
    );
  }
});

test("life-stage generation fixtures", () => {
  const fixtures = fixtureFile(CONFORMANCE_FIXTURES.lifeStageSampling);
  for (const fixture of fixtures.objects("cases")) {
    const rng = fractionSource(fixture.numbers("uniformFractions"));
    const traits = BigFiveTraits.random(lifeStage(fixture), { rng });
    rng.assertConsumed();
    assertTraitComponents(traits, fixture.mapping("expected"));
  }
});

test("full personality generation consumes the shared draw sequence", () => {
  const fixtures = fixtureFile(CONFORMANCE_FIXTURES.lifeStageSampling);
  for (const fixture of fixtures.objects("personalityCases")) {
    const rng = fractionSource(fixture.numbers("uniformFractions"));
    const personality = BigFivePersonality.random(lifeStage(fixture), { rng });
    rng.assertConsumed();
    assertTraitComponents(personality.traits, fixture.mapping("expected"));
    assert.deepEqual(
      {
        style: personality.conflictResolution.style,
        concernForSelf: personality.conflictResolution.concernForSelf,
        concernForOthers: personality.conflictResolution.concernForOthers,
      },
      fixture.mapping("expectedConflictResolution").stringValues(),
    );
  }
});

test("weighted conflict selection fixtures", () => {
  const fixtures = fixtureFile(CONFORMANCE_FIXTURES.conflictResolution);
  for (const fixture of fixtures.objects("cases")) {
    const traits = bigFiveTraits(fixture.mapping("traits"));
    const weights = conflictStyleWeights(traits);
    const expectedWeights = fixture.mapping("weights");
    assert.deepEqual(Object.keys(weights), expectedWeights.keys());
    for (const [style, weight] of Object.entries(weights)) {
      assertClose(weight, expectedWeights.number(style), arithmeticTolerance);
    }
    for (const selection of fixture.objects("selections")) {
      assert.equal(
        BigFiveConflictResolution.random(traits, {
          rng: fractionSource([selection.number("uniformFraction")]),
        }).style,
        selection.string("expected"),
      );
    }
  }
});

test("ABBF projection, poles, dot products, and cosine fixtures", () => {
  const fixtures = fixtureFile(CONFORMANCE_FIXTURES.adaptive);
  for (const fixture of fixtures.objects("projectionCases")) {
    const actual = AdaptiveBifurcatedProfile.fromBigFive(
      bigFiveTraits(fixture.mapping("bigFive")),
    ).vector;
    const expected = fixture.numbers("expected");
    assert.equal(actual.length, expected.length);
    actual.forEach((value, index) => {
      const expectedValue = expected[index];
      assert.ok(expectedValue !== undefined);
      assertClose(value, expectedValue, arithmeticTolerance);
    });
  }
  for (const fixture of fixtures.objects("dominantPoleCases")) {
    assert.deepEqual(
      profile(fixture.numbers("vector")).dominantPoles(fixture.number("threshold")),
      fixture.mapping("expected").stringValues(),
    );
  }
  for (const fixture of fixtures.objects("vectorCases")) {
    const left = profile(fixture.numbers("left"));
    const right = profile(fixture.numbers("right"));
    assertClose(
      left.dotProduct(right),
      fixture.number("dotProduct"),
      arithmeticTolerance,
    );
    assertClose(
      left.cosineSimilarity(right),
      fixture.number("cosineSimilarity"),
      arithmeticTolerance,
    );
  }
});
