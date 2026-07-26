import {
  AdaptiveBifurcatedProfile,
  BigFiveAgreeableness,
  BigFiveConscientiousness,
  BigFiveExtraversion,
  BigFiveNeuroticism,
  BigFiveOpenness,
  BigFiveTraits,
} from "../dist/index.js";

const traits = new BigFiveTraits({
  openness: new BigFiveOpenness({
    aestheticSensitivityScore: 0.75,
    creativeImaginationScore: 0.7,
    intellectualCuriosityScore: 0.8,
  }),
  conscientiousness: new BigFiveConscientiousness({
    organizationScore: 0.65,
    responsibilityScore: 0.7,
    productivityScore: 0.75,
  }),
  extraversion: new BigFiveExtraversion({
    assertivenessScore: 0.4,
    sociabilityScore: 0.45,
    energyLevelScore: 0.5,
  }),
  agreeableness: new BigFiveAgreeableness({
    compassionScore: 0.8,
    respectfulnessScore: 0.75,
    trustScore: 0.7,
  }),
  neuroticism: new BigFiveNeuroticism({
    anxietyScore: 0.25,
    emotionalVolatilityScore: 0.3,
    depressionScore: 0.2,
  }),
});

const profile = AdaptiveBifurcatedProfile.fromBigFive(traits);
for (const axis of profile.axes) {
  console.log(
    `${axis.domain}: ${axis.score.toFixed(2)} (${axis.dominantPole ?? "balanced"})`,
  );
}
