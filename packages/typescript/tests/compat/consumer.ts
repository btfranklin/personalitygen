import {
  AdaptiveBifurcatedProfile,
  BigFivePersonality,
  LifeStage,
  type RandomSource,
} from "../../dist/index.js";

const rng: RandomSource = {
  uniform(minimum, maximum) {
    return minimum + (maximum - minimum) * 0.5;
  },
};

const personality = BigFivePersonality.random(LifeStage.Adult, { rng });
const adaptive = AdaptiveBifurcatedProfile.fromBigFive(personality.traitConfiguration);

adaptive.dominantPoles(0.25);
