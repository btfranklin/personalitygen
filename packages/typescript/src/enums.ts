function stringValues<const T extends Readonly<Record<string, string>>>(
  values: T,
): readonly T[keyof T][] {
  return Object.values(values) as T[keyof T][];
}

function isStringValue<const T extends Readonly<Record<string, string>>>(
  values: T,
  value: unknown,
): value is T[keyof T] {
  return (
    typeof value === "string" && stringValues(values).includes(value as T[keyof T])
  );
}

export const LifeStage = Object.freeze({
  Child: "child",
  YoungAdult: "young_adult",
  Adult: "adult",
} as const);
export type LifeStage = (typeof LifeStage)[keyof typeof LifeStage];

export const PriorityLevel = Object.freeze({
  Low: "low",
  Moderate: "moderate",
  High: "high",
} as const);
export type PriorityLevel = (typeof PriorityLevel)[keyof typeof PriorityLevel];

export const BigFiveConflictResolutionStyle = Object.freeze({
  Avoiding: "avoiding",
  Obliging: "obliging",
  Integrating: "integrating",
  Dominating: "dominating",
  Compromising: "compromising",
} as const);
export type BigFiveConflictResolutionStyle =
  (typeof BigFiveConflictResolutionStyle)[keyof typeof BigFiveConflictResolutionStyle];

export const AdaptiveBifurcatedDomain = Object.freeze({
  Order: "order",
  Chaos: "chaos",
  Cooperation: "cooperation",
  Conflict: "conflict",
  Competition: "competition",
} as const);
export type AdaptiveBifurcatedDomain =
  (typeof AdaptiveBifurcatedDomain)[keyof typeof AdaptiveBifurcatedDomain];

export const AdaptiveBifurcatedPole = Object.freeze({
  Strategizing: "strategizing",
  Improvisation: "improvisation",
  Ethicality: "ethicality",
  Instinctivity: "instinctivity",
  Collaboration: "collaboration",
  Independence: "independence",
  Harmonizing: "harmonizing",
  Utilitarianism: "utilitarianism",
  Proficiency: "proficiency",
  Dominancy: "dominancy",
} as const);
export type AdaptiveBifurcatedPole =
  (typeof AdaptiveBifurcatedPole)[keyof typeof AdaptiveBifurcatedPole];

export const LIFE_STAGES = Object.freeze(stringValues(LifeStage));
export const PRIORITY_LEVELS = Object.freeze(stringValues(PriorityLevel));
export const CONFLICT_RESOLUTION_STYLES = Object.freeze(
  stringValues(BigFiveConflictResolutionStyle),
);
export const ADAPTIVE_BIFURCATED_DOMAINS = Object.freeze(
  stringValues(AdaptiveBifurcatedDomain),
);
export const ADAPTIVE_BIFURCATED_POLES = Object.freeze(
  stringValues(AdaptiveBifurcatedPole),
);

export function assertLifeStage(value: unknown): asserts value is LifeStage {
  if (!isStringValue(LifeStage, value)) {
    throw new TypeError(`Unsupported life stage: ${String(value)}`);
  }
}
