import {
  BigFiveConflictResolutionStyle,
  BigFivePersonality,
  type BigFiveConflictResolutionStyle as ConflictStyle,
  LifeStage,
  type LifeStage as Stage,
} from "../../packages/typescript/dist/index.js";
import { createSeededRandom } from "./seeded-random.js";

export interface PopulationMember {
  readonly id: string;
  readonly stage: Stage;
  readonly personality: BigFivePersonality;
}

export interface PopulationSummary {
  readonly means: Readonly<
    Record<
      | "openness"
      | "conscientiousness"
      | "extraversion"
      | "agreeableness"
      | "neuroticism",
      number
    >
  >;
  readonly styleCounts: Readonly<Record<ConflictStyle, number>>;
}

export function generatePopulation(
  seed = 42,
  profilesPerStage = 30,
): PopulationMember[] {
  if (!Number.isInteger(profilesPerStage) || profilesPerStage < 1) {
    throw new RangeError("profilesPerStage must be a positive integer.");
  }
  const rng = createSeededRandom(seed);
  const stages = [LifeStage.Child, LifeStage.YoungAdult, LifeStage.Adult] as const;
  return stages.flatMap((stage) =>
    Array.from({ length: profilesPerStage }, (_, index) => ({
      id: `${stage}-${String(index + 1).padStart(2, "0")}`,
      stage,
      personality: BigFivePersonality.random(stage, { rng }),
    })),
  );
}

export function summarizePopulation(
  population: readonly PopulationMember[],
): PopulationSummary {
  if (population.length === 0) {
    throw new RangeError("Cannot summarize an empty population.");
  }
  const means = {
    openness:
      population.reduce(
        (sum, member) => sum + member.personality.traits.openness.score,
        0,
      ) / population.length,
    conscientiousness:
      population.reduce(
        (sum, member) => sum + member.personality.traits.conscientiousness.score,
        0,
      ) / population.length,
    extraversion:
      population.reduce(
        (sum, member) => sum + member.personality.traits.extraversion.score,
        0,
      ) / population.length,
    agreeableness:
      population.reduce(
        (sum, member) => sum + member.personality.traits.agreeableness.score,
        0,
      ) / population.length,
    neuroticism:
      population.reduce(
        (sum, member) => sum + member.personality.traits.neuroticism.score,
        0,
      ) / population.length,
  };
  const styleCounts: Record<ConflictStyle, number> = {
    [BigFiveConflictResolutionStyle.Avoiding]: 0,
    [BigFiveConflictResolutionStyle.Obliging]: 0,
    [BigFiveConflictResolutionStyle.Integrating]: 0,
    [BigFiveConflictResolutionStyle.Dominating]: 0,
    [BigFiveConflictResolutionStyle.Compromising]: 0,
  };
  for (const member of population) {
    styleCounts[member.personality.conflictResolution.style] += 1;
  }
  return { means, styleCounts };
}

export function highAnxietyProfiles(
  population: readonly PopulationMember[],
): PopulationMember[] {
  return population.filter(
    (member) => member.personality.traits.neuroticism.anxietyScore >= 0.7,
  );
}

export function lowSociabilityProfiles(
  population: readonly PopulationMember[],
): PopulationMember[] {
  return population.filter(
    (member) => member.personality.traits.extraversion.sociabilityScore <= 0.3,
  );
}
