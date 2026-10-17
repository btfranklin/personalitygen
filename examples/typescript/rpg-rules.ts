// These game rules are separate from package behavior.
import type { BigFivePersonality } from "../../packages/typescript/dist/index.js";

export type Plan = "enter the ruins" | "return to camp" | "scout the entrance";

export interface PartyMember {
  readonly name: string;
  readonly personality: BigFivePersonality;
  readonly goal: "find the relic" | "protect the party";
  readonly wounded: boolean;
}

export function preferredPlan(member: PartyMember): Plan {
  const traits = member.personality.traits;
  const goalBias = member.goal === "find the relic" ? 0.3 : -0.3;
  const interest =
    traits.openness.intellectualCuriosityScore -
    traits.neuroticism.anxietyScore +
    goalBias -
    (member.wounded ? 0.4 : 0);
  return interest > 0 ? "enter the ruins" : "return to camp";
}

export function responseTo(member: PartyMember, proposal: Plan): string {
  if (preferredPlan(member) === proposal) return "accept";
  return {
    avoiding: "stand aside",
    obliging: "yield",
    integrating: "seek a joint plan",
    dominating: "refuse",
    compromising: "offer a limited visit",
  }[member.personality.conflictResolution.style];
}

export function settleDisagreement(responses: readonly string[], proposal: Plan): Plan {
  if (responses.includes("refuse")) return "return to camp";
  if (
    responses.some((response) =>
      ["seek a joint plan", "offer a limited visit"].includes(response),
    )
  ) {
    return "scout the entrance";
  }
  return proposal;
}

export function trustChange(member: PartyMember, decision: Plan): number {
  if (decision === "scout the entrance") return 1;
  if (decision === "enter the ruins" && preferredPlan(member) === "return to camp") {
    return -1;
  }
  return 0;
}
