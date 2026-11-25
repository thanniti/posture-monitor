// web/lib/calibration.ts
import { NORMALIZED_FEATURE_ORDER } from "./config";
import type { SideFeatures } from "./features";

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

export function computeScaleReference(rows: SideFeatures[]): number {
  const lengths = rows.map((r) => r.nose_to_ear_px).filter((v) => v > 0);
  if (lengths.length === 0) {
    throw new Error("computeScaleReference: no rows with positive nose_to_ear_px");
  }
  return median(lengths);
}

export interface NormalizedFeatures {
  neck_angle_deg: number;
  nose_shoulder_angle_deg: number;
  head_forward_offset_norm: number;
  nose_forward_offset_norm: number;
  mean_visibility: number;
}

export function normalizeFeatures(raw: SideFeatures, scaleReference: number): NormalizedFeatures {
  if (scaleReference <= 0) {
    throw new Error(`normalizeFeatures: scaleReference must be positive, got ${scaleReference}`);
  }
  return {
    neck_angle_deg: raw.neck_angle_deg,
    nose_shoulder_angle_deg: raw.nose_shoulder_angle_deg,
    head_forward_offset_norm: raw.head_forward_offset_px / scaleReference,
    nose_forward_offset_norm: raw.nose_forward_offset_px / scaleReference,
    mean_visibility: raw.mean_visibility,
  };
}

export function featureVector(normalized: NormalizedFeatures): number[] {
  return NORMALIZED_FEATURE_ORDER.map((name) => normalized[name]);
}
