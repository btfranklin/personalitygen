import assert from "node:assert/strict";
import test from "node:test";
import {
  BigFiveConflictResolution,
  BigFivePersonality,
  BigFiveTraits,
  LifeStage,
} from "../../packages/typescript/dist/index.js";
import {
  type PartyMember,
  preferredPlan,
  responseTo,
  settleDisagreement,
  trustChange,
} from "./rpg-rules.js";

const traits = BigFiveTraits.random(LifeStage.Adult, {
  rng: { uniform: (minimum, maximum) => minimum + (maximum - minimum) / 2 },
});

test("opposition uses the saved style; agreement does not cause a dispute", () => {
  const responses = {
    avoiding: "stand aside",
    obliging: "yield",
    integrating: "seek a joint plan",
    dominating: "refuse",
    compromising: "offer a limited visit",
  } as const;
  const styles = [
    "avoiding",
    "obliging",
    "integrating",
    "dominating",
    "compromising",
  ] as const;
  for (const style of styles) {
    const member: PartyMember = {
      name: "Bram",
      personality: new BigFivePersonality({
        traits,
        conflictResolution: new BigFiveConflictResolution({ style }),
      }),
      goal: "protect the party",
      wounded: true,
    };
    assert.equal(responseTo(member, "enter the ruins"), responses[style]);
    assert.equal(responseTo(member, "return to camp"), "accept");
    assert.equal(trustChange(member, "scout the entrance"), 1);
    assert.equal(trustChange(member, "enter the ruins"), -1);
    assert.equal(trustChange(member, "return to camp"), 0);
    assert.equal(
      preferredPlan({ ...member, goal: "find the relic", wounded: false }),
      "enter the ruins",
    );
  }
});

test("a refusal stops the visit; a joint plan limits it", () => {
  assert.equal(
    settleDisagreement(["accept", "yield"], "enter the ruins"),
    "enter the ruins",
  );
  assert.equal(
    settleDisagreement(["accept", "seek a joint plan"], "enter the ruins"),
    "scout the entrance",
  );
  assert.equal(
    settleDisagreement(["refuse", "offer a limited visit"], "enter the ruins"),
    "return to camp",
  );
});
