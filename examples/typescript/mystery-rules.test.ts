import assert from "node:assert/strict";
import test from "node:test";
import {
  AdaptiveBifurcatedProfile,
  type AdaptiveBifurcatedProfileOptions,
} from "../../packages/typescript/dist/index.js";
import { castMystery, type MysteryCandidate } from "./mystery-rules.js";

const candidate = (
  id: string,
  scores: AdaptiveBifurcatedProfileOptions,
): MysteryCandidate => ({
  id,
  profile: new AdaptiveBifurcatedProfile(scores),
});

test("casting picks the highest eligible protagonist and unique optimal allies", () => {
  const candidates = [
    candidate("protagonist-low", {
      orderScore: 0,
      chaosScore: 0,
      cooperationScore: 0,
      conflictScore: 0,
      competitionScore: 0.2,
    }),
    candidate("protagonist", {
      orderScore: 1,
      chaosScore: 0,
      cooperationScore: 0,
      conflictScore: 0,
      competitionScore: 0.8,
    }),
    candidate("ally", {
      orderScore: 1,
      chaosScore: 0.1,
      cooperationScore: 0,
      conflictScore: 0,
      competitionScore: 0.7,
    }),
    candidate("middle", {
      orderScore: 0,
      chaosScore: 0,
      cooperationScore: 1,
      conflictScore: 0,
      competitionScore: -0.1,
    }),
    candidate("rival", {
      orderScore: -1,
      chaosScore: 0,
      cooperationScore: 0,
      conflictScore: 0,
      competitionScore: -0.8,
    }),
  ];
  const cast = castMystery(candidates);

  assert.deepEqual(
    [cast.protagonist.id, cast.ally.id, cast.rival.id],
    ["protagonist", "ally", "rival"],
  );
  assert.equal(
    cast.similarities.protagonistAlly,
    cast.protagonist.profile.cosineSimilarity(cast.ally.profile),
  );
  assert.equal(
    cast.similarities.protagonistRival,
    cast.protagonist.profile.cosineSimilarity(cast.rival.profile),
  );
  assert.equal(
    cast.similarities.allyRival,
    cast.ally.profile.cosineSimilarity(cast.rival.profile),
  );
});

test("casting rejects short pools and pools with no eligible protagonist", () => {
  const ineligible = (id: string) =>
    candidate(id, {
      orderScore: 0,
      chaosScore: 0,
      cooperationScore: 0,
      conflictScore: 0,
      competitionScore: 0.1,
    });
  assert.throws(
    () => castMystery([ineligible("short-1"), ineligible("short-2")]),
    /at least three/,
  );
  assert.throws(
    () => castMystery([ineligible("one"), ineligible("two"), ineligible("three")]),
    /dominant proficiency above 0\.1/,
  );
  const repeatedId = candidate("duplicate", {
    orderScore: 0,
    chaosScore: 0,
    cooperationScore: 0,
    conflictScore: 0,
    competitionScore: 0.8,
  });
  assert.throws(
    () => castMystery([repeatedId, { ...repeatedId }, ineligible("third")]),
    /unique candidate IDs/,
  );
});

test("casting keeps the first candidate when protagonist and ally scores tie", () => {
  const candidates = [
    candidate("protagonist-first", {
      orderScore: 1,
      chaosScore: 0,
      cooperationScore: 0,
      conflictScore: 0,
      competitionScore: 0.8,
    }),
    candidate("protagonist-tie", {
      orderScore: -1,
      chaosScore: 0,
      cooperationScore: 0,
      conflictScore: 0,
      competitionScore: 0.8,
    }),
    candidate("ally-first", {
      orderScore: 1,
      chaosScore: 0,
      cooperationScore: 0,
      conflictScore: 0,
      competitionScore: 0.7,
    }),
    candidate("ally-tie", {
      orderScore: 1,
      chaosScore: 0,
      cooperationScore: 0,
      conflictScore: 0,
      competitionScore: 0.7,
    }),
    candidate("rival", {
      orderScore: -1,
      chaosScore: 0,
      cooperationScore: 0,
      conflictScore: 0,
      competitionScore: -0.8,
    }),
  ];
  const cast = castMystery(candidates);

  assert.equal(cast.protagonist.id, "protagonist-first");
  assert.equal(cast.ally.id, "ally-first");
});
