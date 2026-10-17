import type { RandomSource } from "../../packages/typescript/dist/index.js";

export function createSeededRandom(seed: number): RandomSource {
  let state = seed >>> 0;
  return {
    uniform(minimum: number, maximum: number): number {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return minimum + (maximum - minimum) * (state / 4294967296);
    },
  };
}
