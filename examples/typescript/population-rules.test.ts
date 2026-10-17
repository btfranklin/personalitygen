import assert from "node:assert/strict";
import test from "node:test";
import {
  BigFiveAgreeableness,
  BigFiveConflictResolution,
  BigFiveConflictResolutionStyle,
  BigFiveConscientiousness,
  BigFiveExtraversion,
  BigFiveNeuroticism,
  BigFiveOpenness,
  BigFivePersonality,
  BigFiveTraits,
} from "../../packages/typescript/dist/index.js";
import {
  generatePopulation,
  highAnxietyProfiles,
  lowSociabilityProfiles,
  summarizePopulation,
} from "./population-rules.js";

test("population repeats in full and summaries count all profiles and styles", () => {
  const population = generatePopulation();
  assert.deepEqual(population, generatePopulation());
  assert.notDeepEqual(population, generatePopulation(43));
  assert.equal(population.length, 90);
  assert.deepEqual(
    ["child", "young_adult", "adult"].map(
      (stage) => population.filter((member) => member.stage === stage).length,
    ),
    [30, 30, 30],
  );

  const summary = summarizePopulation(population);
  assert.equal(Object.keys(summary.means).length, 5);
  assert.equal(Object.keys(summary.styleCounts).length, 5);
  assert.equal(
    Object.values(summary.styleCounts).reduce((sum, count) => sum + count),
    90,
  );
  assert.ok(Object.values(summary.means).every(Number.isFinite));
});

test("population size and empty summaries have clear boundaries", () => {
  for (const size of [0, -1, 1.5, Number.NaN]) {
    assert.throws(() => generatePopulation(42, size), /positive integer/);
  }
  assert.throws(() => summarizePopulation([]), /empty population/);
});

test("filters use anxiety and sociability component scores", () => {
  const makeMember = (
    id: string,
    anxiety: number,
    otherNeuroticism: number,
    sociability: number,
    otherExtraversion: number,
  ) => {
    const personality = new BigFivePersonality({
      traits: new BigFiveTraits({
        openness: new BigFiveOpenness({
          aestheticSensitivityScore: 0.5,
          creativeImaginationScore: 0.5,
          intellectualCuriosityScore: 0.5,
        }),
        conscientiousness: new BigFiveConscientiousness({
          organizationScore: 0.5,
          responsibilityScore: 0.5,
          productivityScore: 0.5,
        }),
        extraversion: new BigFiveExtraversion({
          assertivenessScore: otherExtraversion,
          sociabilityScore: sociability,
          energyLevelScore: otherExtraversion,
        }),
        agreeableness: new BigFiveAgreeableness({
          compassionScore: 0.5,
          respectfulnessScore: 0.5,
          trustScore: 0.5,
        }),
        neuroticism: new BigFiveNeuroticism({
          anxietyScore: anxiety,
          emotionalVolatilityScore: otherNeuroticism,
          depressionScore: otherNeuroticism,
        }),
      }),
      conflictResolution: new BigFiveConflictResolution({
        style: BigFiveConflictResolutionStyle.Avoiding,
      }),
    });
    return { id, stage: "adult" as const, personality };
  };
  const componentMatchesAggregateMiss = makeMember(
    "component-match",
    0.9,
    0.1,
    0.1,
    0.9,
  );
  const aggregateMatchesComponentMiss = makeMember(
    "aggregate-match",
    0.6,
    0.9,
    0.4,
    0.1,
  );
  const population = [componentMatchesAggregateMiss, aggregateMatchesComponentMiss];

  assert.ok(
    Math.abs(
      componentMatchesAggregateMiss.personality.traits.neuroticism.score - 1.1 / 3,
    ) < 1e-12,
  );
  assert.ok(
    Math.abs(
      aggregateMatchesComponentMiss.personality.traits.neuroticism.score - 2.4 / 3,
    ) < 1e-12,
  );
  assert.deepEqual(
    highAnxietyProfiles(population).map(({ id }) => id),
    ["component-match"],
  );
  assert.deepEqual(
    lowSociabilityProfiles(population).map(({ id }) => id),
    ["component-match"],
  );
});
