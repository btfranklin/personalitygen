import {
  AdaptiveBifurcatedProfile,
  BigFivePersonality,
  LifeStage,
} from "../dist/index.js";

const personality = BigFivePersonality.random(LifeStage.Adult);
const adaptive = AdaptiveBifurcatedProfile.fromBigFive(personality.traits);
const conflict = personality.conflictResolution;

console.log("NPC: Quartermaster Ilya");
console.log(`Conflict style: ${conflict.style}`);
console.log(
  `Concern priorities: self=${conflict.concernForSelf}, others=${conflict.concernForOthers}`,
);
console.log(`ABBF vector: ${adaptive.vector.map((value) => value.toFixed(2))}`);
console.log("Dominant ABBF poles:", adaptive.dominantPoles(0.25));
