import {
  SIGNED_RANGE_MAX,
  SIGNED_RANGE_MIN,
  UNIT_RANGE_MAX,
  UNIT_RANGE_MIN,
} from "./constants.js";

function validateFinite(value: number): void {
  if (!Number.isFinite(value)) {
    throw new RangeError("Scores must be finite numbers.");
  }
}

export function validateUnitRange(...values: readonly number[]): void {
  for (const value of values) {
    validateFinite(value);
    if (value < UNIT_RANGE_MIN || value > UNIT_RANGE_MAX) {
      throw new RangeError("All trait components must be in the range 0.0...1.0");
    }
  }
}

export function validateSignedRange(...values: readonly number[]): void {
  for (const value of values) {
    validateFinite(value);
    if (value < SIGNED_RANGE_MIN || value > SIGNED_RANGE_MAX) {
      throw new RangeError("All signed scores must be in the range -1.0...1.0");
    }
  }
}

export function componentAverageScore(...values: readonly number[]): number {
  if (values.length === 0) {
    throw new RangeError("At least one component score is required.");
  }
  validateUnitRange(...values);
  return values.reduce((total, value) => total + value, 0) / values.length;
}

export function unitToSigned(value: number): number {
  validateUnitRange(value);
  return value * 2 - 1;
}

export function weightedSignedAverage(
  ...weightedValues: readonly (readonly [number, number])[]
): number {
  if (weightedValues.length === 0) {
    throw new RangeError("At least one weighted value is required.");
  }
  const totalWeight = weightedValues.reduce(
    (total, [, weight]) => total + Math.abs(weight),
    0,
  );
  if (totalWeight <= 0) {
    throw new RangeError("At least one weight must be non-zero.");
  }
  for (const [value] of weightedValues) {
    validateSignedRange(value);
  }
  const score =
    weightedValues.reduce((total, [value, weight]) => total + value * weight, 0) /
    totalWeight;
  return Math.max(SIGNED_RANGE_MIN, Math.min(SIGNED_RANGE_MAX, score));
}

export function dotProduct(left: readonly number[], right: readonly number[]): number {
  if (left.length !== right.length) {
    throw new RangeError("Vectors must have the same length.");
  }
  validateSignedRange(...left);
  validateSignedRange(...right);
  return left.reduce(
    (total, leftValue, index) => total + leftValue * (right[index] ?? 0),
    0,
  );
}

export function cosineSimilarity(
  left: readonly number[],
  right: readonly number[],
): number {
  if (left.length !== right.length) {
    throw new RangeError("Vectors must have the same length.");
  }
  validateSignedRange(...left);
  validateSignedRange(...right);
  const leftNorm = Math.sqrt(left.reduce((total, value) => total + value * value, 0));
  const rightNorm = Math.sqrt(right.reduce((total, value) => total + value * value, 0));
  if (leftNorm === 0 || rightNorm === 0) {
    return 0;
  }
  return dotProduct(left, right) / (leftNorm * rightNorm);
}
