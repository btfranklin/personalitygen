import assert from "node:assert/strict";
import {
  generatePopulation,
  highAnxietyProfiles,
  lowSociabilityProfiles,
  summarizePopulation,
} from "./population-rules.js";

const population = generatePopulation(42, 30);
assert.deepEqual(population, generatePopulation(42, 30));
console.log("Repeat check passed: full profiles match for seed 42.");
const summary = summarizePopulation(population);
const highAnxiety = highAnxietyProfiles(population);
const lowSociability = lowSociabilityProfiles(population);

console.log("Simulated population: 30 profiles per life stage; seed 42.");
console.log("This sample reflects this generator, not a demographic forecast.");
console.log("Save selected profile data with its seed and thresholds for tests.");
for (const stage of ["child", "young_adult", "adult"] as const) {
  const members = population.filter((member) => member.stage === stage);
  const stageSummary = summarizePopulation(members);
  console.log(`${stage}: ${members.length} profiles; Big Five means:`);
  for (const [trait, mean] of Object.entries(stageSummary.means)) {
    console.log(`  ${trait}: ${mean.toFixed(2)}`);
  }
}
console.log("Aggregate Big Five means:");
for (const [trait, mean] of Object.entries(summary.means)) {
  console.log(`${trait}: ${mean.toFixed(2)}`);
}
console.log("Conflict style counts:");
for (const [style, count] of Object.entries(summary.styleCounts)) {
  console.log(`${style}: ${count}`);
}
console.log(`High anxiety (component score >= 0.7): ${highAnxiety.length}`);
for (const member of highAnxiety.slice(0, 3)) {
  const traits = member.personality.traits;
  console.log(
    `  ${member.id}: anxiety=${traits.neuroticism.anxietyScore.toFixed(2)}, neuroticism=${traits.neuroticism.score.toFixed(2)}`,
  );
}
console.log(`Low sociability (component score <= 0.3): ${lowSociability.length}`);
for (const member of lowSociability.slice(0, 3)) {
  const traits = member.personality.traits;
  console.log(
    `  ${member.id}: sociability=${traits.extraversion.sociabilityScore.toFixed(2)}, extraversion=${traits.extraversion.score.toFixed(2)}`,
  );
}
