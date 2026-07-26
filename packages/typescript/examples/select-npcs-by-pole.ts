import {
  AdaptiveBifurcatedDomain,
  type AdaptiveBifurcatedDomainValue,
  AdaptiveBifurcatedPole,
  type AdaptiveBifurcatedPoleValue,
  AdaptiveBifurcatedProfile,
} from "../dist/index.js";

const profile = (
  vector: readonly [number, number, number, number, number],
): AdaptiveBifurcatedProfile =>
  new AdaptiveBifurcatedProfile({
    orderScore: vector[0],
    chaosScore: vector[1],
    cooperationScore: vector[2],
    conflictScore: vector[3],
    competitionScore: vector[4],
  });

const cast = {
  Asha: profile([0.7, 0.5, 0.8, 0.4, 0.3]),
  Bram: profile([-0.4, -0.7, -0.6, -0.5, -0.8]),
  Cato: profile([0.2, 0.1, -0.2, -0.7, 0.8]),
  Dima: profile([-0.6, 0.6, 0.5, 0.7, 0.1]),
};

function selectByPole(
  domain: AdaptiveBifurcatedDomainValue,
  pole: AdaptiveBifurcatedPoleValue,
  threshold = 0.4,
): string[] {
  return Object.entries(cast)
    .filter(([, candidate]) => candidate.dominantPoles(threshold)[domain] === pole)
    .map(([name]) => name);
}

console.log(
  "Collaborative NPCs:",
  selectByPole(
    AdaptiveBifurcatedDomain.Cooperation,
    AdaptiveBifurcatedPole.Collaboration,
  ),
);
console.log(
  "Dominancy-leaning NPCs:",
  selectByPole(AdaptiveBifurcatedDomain.Competition, AdaptiveBifurcatedPole.Dominancy),
);
