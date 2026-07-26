import { CDF_EPSILON } from "./constants.js";

export interface RandomSource {
  uniform(minimum: number, maximum: number): number;
}

export interface RandomOptions {
  readonly rng?: RandomSource;
}

const DEFAULT_RANDOM_SOURCE: RandomSource = Object.freeze({
  uniform(minimum: number, maximum: number): number {
    return minimum + (maximum - minimum) * Math.random();
  },
});

function errorFunction(value: number): number {
  const sign = value < 0 ? -1 : 1;
  const magnitude = Math.abs(value);
  const t = 1 / (1 + 0.3275911 * magnitude);
  const polynomial =
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
      0.254829592) *
    t;
  return sign * (1 - polynomial * Math.exp(-(magnitude * magnitude)));
}

function normalCdf(value: number, mean: number, standardDeviation: number): number {
  return 0.5 * (1 + errorFunction((value - mean) / (standardDeviation * Math.SQRT2)));
}

function inverseStandardNormal(probability: number): number {
  if (!(probability > 0 && probability < 1)) {
    throw new RangeError("Probability must be strictly between 0 and 1.");
  }

  const a = [
    -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2,
    -3.066479806614716e1, 2.506628277459239,
  ] as const;
  const b = [
    -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
    6.680131188771972e1, -1.328068155288572e1,
  ] as const;
  const c = [
    -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
    -2.549732539343734, 4.374664141464968, 2.938163982698783,
  ] as const;
  const d = [
    7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416,
  ] as const;
  const lowerBoundary = 0.02425;
  const upperBoundary = 1 - lowerBoundary;

  if (probability < lowerBoundary) {
    const q = Math.sqrt(-2 * Math.log(probability));
    return (
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  }
  if (probability > upperBoundary) {
    const q = Math.sqrt(-2 * Math.log(1 - probability));
    return -(
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  }

  const q = probability - 0.5;
  const r = q * q;
  return (
    ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
  );
}

export function randomGaussian(options: {
  readonly mean: number;
  readonly standardDeviation: number;
  readonly minimum: number;
  readonly maximum: number;
  readonly rng?: RandomSource;
}): number {
  const {
    mean,
    standardDeviation,
    minimum,
    maximum,
    rng = DEFAULT_RANDOM_SOURCE,
  } = options;
  if (
    !Number.isFinite(mean) ||
    !Number.isFinite(minimum) ||
    !Number.isFinite(maximum)
  ) {
    throw new RangeError("Gaussian parameters must be finite.");
  }
  if (!Number.isFinite(standardDeviation) || standardDeviation <= 0) {
    throw new RangeError("standardDeviation must be positive.");
  }
  if (minimum > maximum) {
    throw new RangeError("minimum must be less than or equal to maximum.");
  }

  const lowerCdf = normalCdf(minimum, mean, standardDeviation);
  const upperCdf = normalCdf(maximum, mean, standardDeviation);
  if (lowerCdf >= upperCdf) {
    return Math.max(minimum, Math.min(maximum, mean));
  }

  const lower = Math.max(lowerCdf, CDF_EPSILON);
  const upper = Math.min(upperCdf, 1 - CDF_EPSILON);
  if (lower >= upper) {
    return Math.max(minimum, Math.min(maximum, mean));
  }

  const probability = rng.uniform(lower, upper);
  if (!Number.isFinite(probability) || probability < lower || probability > upper) {
    throw new RangeError("RandomSource.uniform returned a value out of range.");
  }
  return mean + standardDeviation * inverseStandardNormal(probability);
}
