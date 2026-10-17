import {
  AdaptiveBifurcatedPole,
  type AdaptiveBifurcatedProfile,
} from "../../packages/typescript/dist/index.js";

export interface MysteryCandidate {
  readonly id: string;
  readonly profile: AdaptiveBifurcatedProfile;
}

export interface MysteryCast {
  readonly protagonist: MysteryCandidate;
  readonly ally: MysteryCandidate;
  readonly rival: MysteryCandidate;
  readonly similarities: Readonly<{
    protagonistAlly: number;
    protagonistRival: number;
    allyRival: number;
  }>;
}

export function castMystery(candidates: readonly MysteryCandidate[]): MysteryCast {
  if (candidates.length < 3) {
    throw new RangeError("Mystery casting requires at least three candidates.");
  }
  if (new Set(candidates.map((candidate) => candidate.id)).size !== candidates.length) {
    throw new RangeError("Mystery casting requires unique candidate IDs.");
  }

  const eligible = candidates.filter(
    (candidate) =>
      candidate.profile.dominantPoles(0.1).competition ===
      AdaptiveBifurcatedPole.Proficiency,
  );
  const protagonist = eligible.reduce<MysteryCandidate | undefined>(
    (best, candidate) =>
      best === undefined ||
      candidate.profile.competitionScore > best.profile.competitionScore
        ? candidate
        : best,
    undefined,
  );
  if (protagonist === undefined) {
    throw new RangeError(
      "Mystery casting requires a candidate with dominant proficiency above 0.1.",
    );
  }

  const remaining = candidates.filter((candidate) => candidate !== protagonist);
  const ally = remaining.reduce((best, candidate) =>
    candidate.profile.cosineSimilarity(protagonist.profile) >
    best.profile.cosineSimilarity(protagonist.profile)
      ? candidate
      : best,
  );
  const afterAlly = remaining.filter((candidate) => candidate !== ally);
  const rival = afterAlly.reduce((best, candidate) =>
    candidate.profile.cosineSimilarity(protagonist.profile) <
    best.profile.cosineSimilarity(protagonist.profile)
      ? candidate
      : best,
  );

  return {
    protagonist,
    ally,
    rival,
    similarities: {
      protagonistAlly: protagonist.profile.cosineSimilarity(ally.profile),
      protagonistRival: protagonist.profile.cosineSimilarity(rival.profile),
      allyRival: ally.profile.cosineSimilarity(rival.profile),
    },
  };
}
