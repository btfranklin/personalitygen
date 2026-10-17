import {
  AdaptiveBifurcatedProfile,
  BigFivePersonality,
  LifeStage,
} from "../../packages/typescript/dist/index.js";
import { castMystery, type MysteryCandidate } from "./mystery-rules.js";
import { createSeededRandom } from "./seeded-random.js";

const rng = createSeededRandom(42);
const candidates: MysteryCandidate[] = Array.from({ length: 40 }, (_, index) => {
  const personality = BigFivePersonality.random(LifeStage.Adult, { rng });
  return {
    id: `Candidate-${String(index + 1).padStart(2, "0")}`,
    profile: AdaptiveBifurcatedProfile.fromBigFive(personality.traits),
  };
});
const cast = castMystery(candidates);

console.log("Mystery cast from 40 simulated adult profiles:");
console.log("Roles are story rules. Similarity is not conflict probability.");
console.log("The rival role does not mean the character is a villain.");
for (const [role, candidate] of Object.entries({
  protagonist: cast.protagonist,
  ally: cast.ally,
  rival: cast.rival,
})) {
  console.log(
    `${role}: ${candidate.id}; dominant poles=${JSON.stringify(candidate.profile.dominantPoles(0.1))}`,
  );
}
console.log("Selection reasons:");
console.log(
  `protagonist: highest competition score among candidates with dominant proficiency (${cast.protagonist.profile.competitionScore.toFixed(2)})`,
);
console.log(
  `ally: most cosine-similar remaining candidate (${cast.similarities.protagonistAlly.toFixed(2)})`,
);
console.log(
  `rival: least cosine-similar candidate remaining after ally selection (${cast.similarities.protagonistRival.toFixed(2)})`,
);
console.log("Pair similarities:");
console.log(`protagonist / ally: ${cast.similarities.protagonistAlly.toFixed(2)}`);
console.log(`protagonist / rival: ${cast.similarities.protagonistRival.toFixed(2)}`);
console.log(`ally / rival: ${cast.similarities.allyRival.toFixed(2)}`);
