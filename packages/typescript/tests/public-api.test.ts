import assert from "node:assert/strict";
import { test } from "node:test";
import {
  AdaptiveBifurcatedAxis,
  AdaptiveBifurcatedDomain,
  AdaptiveBifurcatedPole,
  BigFiveConflictResolution,
  BigFiveConflictResolutionStyle,
  BigFiveOpenness,
  BigFivePersonality,
  LifeStage,
  PriorityLevel,
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
  assert.ok(Object.isFrozen(personality.traits));
  assert.ok(Object.isFrozen(personality.conflictResolution));
});

test("conflict concerns are derived from the authored style", () => {
  const conflict = new BigFiveConflictResolution({
    style: BigFiveConflictResolutionStyle.Avoiding,
  });

  assert.equal(conflict.style, BigFiveConflictResolutionStyle.Avoiding);
  assert.equal(conflict.concernForSelf, PriorityLevel.Low);
  assert.equal(conflict.concernForOthers, PriorityLevel.Low);
  assert.ok(Object.isFrozen(conflict));
});

test("random generation accepts an inclusive upper endpoint", () => {
  const openness = BigFiveOpenness.random(LifeStage.Child, {
    rng: {
      uniform(_minimum, maximum) {
        return maximum;
      },
    },
  });

  assert.equal(openness.aestheticSensitivityScore, 1);
  assert.equal(openness.creativeImaginationScore, 1);
  assert.equal(openness.intellectualCuriosityScore, 1);
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
  const midpointPersonality = BigFivePersonality.random(LifeStage.Adult, {
    rng: {
      uniform(minimum, maximum) {
        return minimum + (maximum - minimum) * 0.5;
      },
    },
  });
  assert.throws(
    () =>
      BigFiveConflictResolution.random(midpointPersonality.traits, {
        rng: {
          uniform() {
            return Number.POSITIVE_INFINITY;
          },
        },
      }),
    RangeError,
  );
});
