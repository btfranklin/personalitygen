import assert from "node:assert/strict";
import { test } from "node:test";
import {
  AdaptiveBifurcatedAxis,
  AdaptiveBifurcatedDomain,
  AdaptiveBifurcatedPole,
  BigFiveOpenness,
  BigFivePersonality,
  LifeStage,
} from "../dist/index.js";

test("public value objects are immutable", () => {
  const openness = new BigFiveOpenness({
    aestheticSensitivityScore: 0.3,
    creativeImaginationScore: 0.6,
    intellectualCuriosityScore: 0.9,
  });
  assert.equal(openness.score, 0.6);
  assert.ok(Object.isFrozen(openness));
});

test("random personality generation uses the supplied random source", () => {
  const personality = BigFivePersonality.random(LifeStage.Adult, {
    rng: {
      uniform(minimum, maximum) {
        return minimum + (maximum - minimum) * 0.5;
      },
    },
  });
  assert.ok(Object.isFrozen(personality));
  assert.ok(Object.isFrozen(personality.traitConfiguration));
  assert.ok(Object.isFrozen(personality.conflictResolutionConfiguration));
});

test("malformed categorical and numeric inputs use idiomatic errors", () => {
  assert.throws(() => BigFivePersonality.random("elder" as never), TypeError);
  assert.throws(
    () =>
      new BigFiveOpenness({
        aestheticSensitivityScore: 2,
        creativeImaginationScore: 0.5,
        intellectualCuriosityScore: 0.5,
      }),
    RangeError,
  );
  assert.throws(
    () =>
      new AdaptiveBifurcatedAxis({
        domain: AdaptiveBifurcatedDomain.Order,
        positivePole: AdaptiveBifurcatedPole.Strategizing,
        negativePole: "unsupported" as never,
        score: 0.5,
      }),
    TypeError,
  );
  assert.throws(
    () =>
      BigFivePersonality.random(LifeStage.Adult, {
        rng: {
          uniform() {
            return Number.NaN;
          },
        },
      }),
    RangeError,
  );
});
