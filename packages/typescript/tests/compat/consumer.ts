import {
  AdaptiveBifurcatedDomain,
  AdaptiveBifurcatedPole,
  AdaptiveBifurcatedProfile,
  BigFiveConflictResolutionStyle,
  BigFivePersonality,
  LifeStage,
  PriorityLevel,
  type RandomSource,
} from "personalitygen";

const rng: RandomSource = {
  uniform(minimum, maximum) {
    return minimum + (maximum - minimum) * 0.5;
  },
};

const lifeStage: LifeStage = LifeStage.Adult;
const domain: AdaptiveBifurcatedDomain = AdaptiveBifurcatedDomain.Order;
const pole: AdaptiveBifurcatedPole = AdaptiveBifurcatedPole.Strategizing;
const style: BigFiveConflictResolutionStyle =
  BigFiveConflictResolutionStyle.Integrating;
const priority: PriorityLevel = PriorityLevel.High;

const personality = BigFivePersonality.random(lifeStage, { rng });
const adaptive = AdaptiveBifurcatedProfile.fromBigFive(personality.traits);

adaptive.dominantPoles(0.25);
if (adaptive.vector.length !== 5) {
  throw new Error("Unexpected vector length.");
}
void [domain, pole, style, priority];
