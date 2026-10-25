import { AdaptiveBifurcatedProfile } from "../../packages/typescript/dist/index.js";

const cast = {
  diplomat: new AdaptiveBifurcatedProfile({
    orderScore: 0.4,
    chaosScore: 0.6,
    cooperationScore: 0.9,
    conflictScore: 0.8,
    competitionScore: 0.2,
  }),
  scout: new AdaptiveBifurcatedProfile({
    orderScore: -0.2,
    chaosScore: -0.1,
    cooperationScore: -0.4,
    conflictScore: 0.1,
    competitionScore: 0.5,
  }),
  warlord: new AdaptiveBifurcatedProfile({
    orderScore: 0.2,
    chaosScore: -0.7,
    cooperationScore: -0.8,
    conflictScore: -0.6,
    competitionScore: -0.9,
  }),
};

const characters = [
  ["diplomat", cast.diplomat],
  ["scout", cast.scout],
  ["warlord", cast.warlord],
] as const;
for (let leftIndex = 0; leftIndex < characters.length; leftIndex += 1) {
  const leftEntry = characters.at(leftIndex);
  if (leftEntry === undefined) continue;
  for (
    let rightIndex = leftIndex + 1;
    rightIndex < characters.length;
    rightIndex += 1
  ) {
    const rightEntry = characters.at(rightIndex);
    if (rightEntry === undefined) continue;
    const [leftName, left] = leftEntry;
    const [rightName, right] = rightEntry;
    console.log(
      `${leftName} / ${rightName}: dot=${left.dotProduct(right).toFixed(2)}, cosine=${left.cosineSimilarity(right).toFixed(2)}`,
    );
  }
}
