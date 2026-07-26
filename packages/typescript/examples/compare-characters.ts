import { AdaptiveBifurcatedProfile } from "../dist/index.js";

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

const names = Object.keys(cast) as (keyof typeof cast)[];
for (let leftIndex = 0; leftIndex < names.length; leftIndex += 1) {
  for (let rightIndex = leftIndex + 1; rightIndex < names.length; rightIndex += 1) {
    const leftName = names[leftIndex] as keyof typeof cast;
    const rightName = names[rightIndex] as keyof typeof cast;
    const left = cast[leftName];
    const right = cast[rightName];
    console.log(
      `${leftName} / ${rightName}: dot=${left.dotProduct(right).toFixed(2)}, cosine=${left.cosineSimilarity(right).toFixed(2)}`,
    );
  }
}
