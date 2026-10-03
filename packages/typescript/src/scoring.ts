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
  return left.reduce((total, leftValue, index) => {
    const rightValue = right[index];
    if (rightValue === undefined) {
      throw new RangeError("Vectors must have the same length.");
    }
    return total + leftValue * rightValue;
  }, 0);
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
  const leftScale = left.reduce(
    (maximum, value) => Math.max(maximum, Math.abs(value)),
    0,
  );
  const rightScale = right.reduce(
    (maximum, value) => Math.max(maximum, Math.abs(value)),
    0,
  );
  if (leftScale === 0 || rightScale === 0) {
    return 0;
  }

  const scaledLeft = left.map((value) => value / leftScale);
  const scaledRight = right.map((value) => value / rightScale);
  const leftNorm = Math.sqrt(
    scaledLeft.reduce((total, value) => total + value * value, 0),
  );
  const rightNorm = Math.sqrt(
    scaledRight.reduce((total, value) => total + value * value, 0),
  );
  const dot = scaledLeft.reduce((total, value, index) => {
    const rightValue = scaledRight[index];
    if (rightValue === undefined) {
      throw new RangeError("Vectors must have the same length.");
    }
    return total + value * rightValue;
  }, 0);
  return Math.max(-1, Math.min(1, dot / (leftNorm * rightNorm)));
}
