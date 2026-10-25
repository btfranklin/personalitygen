import { BigFivePersonality, LifeStage } from "../../packages/typescript/dist/index.js";
import {
  type PartyMember,
  preferredPlan,
  responseTo,
  settleDisagreement,
  trustChange,
} from "./rpg-rules.js";

// This example owns its random source. Python uses a different generator.
let seed = 42;
const rng = {
  uniform(minimum: number, maximum: number): number {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return minimum + (maximum - minimum) * (seed / 4294967296);
  },
};
const party = [
  {
    name: "Asha",
    personality: BigFivePersonality.random(LifeStage.Adult, { rng }),
    goal: "find the relic",
    wounded: false,
  },
  {
    name: "Bram",
    personality: BigFivePersonality.random(LifeStage.Adult, { rng }),
    goal: "protect the party",
    wounded: true,
  },
  {
    name: "Cato",
    personality: BigFivePersonality.random(LifeStage.Adult, { rng }),
    goal: "find the relic",
    wounded: false,
  },
] as const satisfies readonly PartyMember[];
const leader = party.at(0);
if (leader === undefined) {
  throw new Error("The party requires a leader.");
}
const proposal = preferredPlan(leader);
console.log("Scene: The party reaches ruins before nightfall. Bram is wounded.");
console.log(`Leader's proposal: ${proposal}.`);
const responses = party.map((member) => {
  const response = responseTo(member, proposal);
  const traits = member.personality.traits;
  console.log(
    `${member.name}: goal=${member.goal}; curiosity=${traits.openness.intellectualCuriosityScore.toFixed(2)}; anxiety=${traits.neuroticism.anxietyScore.toFixed(2)}; style=${member.personality.conflictResolution.style}`,
  );
  console.log(`  Preference: ${preferredPlan(member)}. Response: ${response}.`);
  return response;
});
const decision = settleDisagreement(responses, proposal);
console.log(`Party decision: ${decision}.`);
for (const member of party.slice(1)) {
  const before = 2;
  console.log(
    `${member.name}: trust in leader ${before} -> ${before + trustChange(member, decision)}`,
  );
}
console.log("The game owns goals, injuries, plans, and trust. Profiles stay fixed.");
