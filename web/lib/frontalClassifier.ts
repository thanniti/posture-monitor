// web/lib/frontalClassifier.ts
import {
  FRONT_LATERAL_TILT_TOLERANCE_DEG,
  FRONT_NECK_SHRINK_TOLERANCE,
  FRONT_SHOULDER_PROTRACTION_TOLERANCE,
} from "./config";
import type { FrontalFeatures } from "./features";

const BASELINE_FIELDS = [
  "nose_to_shoulder_vertical_px",
  "ear_to_shoulder_vertical_px",
  "shoulder_width_px",
  "shoulder_tilt_deg",
  "interocular_distance_px",
] as const;

export type FrontalBaseline = Record<(typeof BASELINE_FIELDS)[number], number>;

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

export function computeFrontalBaseline(rows: FrontalFeatures[]): FrontalBaseline {
  const baseline = {} as FrontalBaseline;
  for (const field of BASELINE_FIELDS) {
    baseline[field] = median(rows.map((row) => row[field]));
  }
  return baseline;
}

function severityToConfidence(ratio: number): number {
  if (ratio >= 1.0) {
    return Math.min(99.0, Math.round((50 + 50 * (ratio - 1.0)) * 10) / 10);
  }
  return Math.min(100.0, Math.round((1.0 - ratio) * 100 * 10) / 10);
}

export function classifyFrontal(raw: FrontalFeatures, baseline: FrontalBaseline): [string, number] {
  const liveInterocular = raw.interocular_distance_px;
  const baselineInterocular = baseline.interocular_distance_px;

  const liveNoseRatio = raw.nose_to_shoulder_vertical_px / liveInterocular;
  const baselineNoseRatio = baseline.nose_to_shoulder_vertical_px / baselineInterocular;
  const noseShrinkFraction = (baselineNoseRatio - liveNoseRatio) / baselineNoseRatio;

  const liveEarRatio = raw.ear_to_shoulder_vertical_px / liveInterocular;
  const baselineEarRatio = baseline.ear_to_shoulder_vertical_px / baselineInterocular;
  const earShrinkFraction = (baselineEarRatio - liveEarRatio) / baselineEarRatio;

  const liveShoulderRatio = raw.shoulder_width_px / liveInterocular;
  const baselineShoulderRatio = baseline.shoulder_width_px / baselineInterocular;
  const shoulderShrinkFraction = (baselineShoulderRatio - liveShoulderRatio) / baselineShoulderRatio;

  const shoulderTiltDeviation = Math.abs(raw.shoulder_tilt_deg - baseline.shoulder_tilt_deg);

  const neckCompressionRatio = Math.max(noseShrinkFraction, earShrinkFraction) / FRONT_NECK_SHRINK_TOLERANCE;
  const protractionRatio = shoulderShrinkFraction / FRONT_SHOULDER_PROTRACTION_TOLERANCE;
  const tiltRatio = shoulderTiltDeviation / FRONT_LATERAL_TILT_TOLERANCE_DEG;

  // Tracked by category (not a bare Math.max of all three) so the Slouching-vs-
  // LateralLean decision never depends on floating-point equality between
  // ratios computed via different formulas -- same reasoning as the Python original.
  const categoryRatios: Array<[string, number]> = [
    ["Slouching", Math.max(neckCompressionRatio, protractionRatio)],
    ["LateralLean", tiltRatio],
  ];
  let label = categoryRatios[0][0];
  let severity = categoryRatios[0][1];
  for (const [category, ratio] of categoryRatios) {
    if (ratio > severity) {
      severity = ratio;
      label = category;
    }
  }
  const confidence = severityToConfidence(severity);

  if (severity < 1.0) {
    return ["CorrectSitting", confidence];
  }
  return [label, confidence];
}
