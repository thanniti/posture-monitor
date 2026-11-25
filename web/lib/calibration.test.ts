// web/lib/calibration.test.ts
import { describe, expect, it } from "vitest";
import { computeScaleReference, featureVector, normalizeFeatures } from "./calibration";
import type { SideFeatures } from "./features";

describe("computeScaleReference", () => {
  it("returns the median nose_to_ear_px across rows", () => {
    const rows = [18.0, 20.0, 22.0].map((v) => ({ nose_to_ear_px: v }) as SideFeatures);
    expect(computeScaleReference(rows)).toBeCloseTo(20.0, 5);
  });

  it("throws when no rows have a positive nose_to_ear_px", () => {
    const rows = [{ nose_to_ear_px: 0 } as SideFeatures, { nose_to_ear_px: -1 } as SideFeatures];
    expect(() => computeScaleReference(rows)).toThrow();
  });
});

describe("normalizeFeatures", () => {
  function makeRaw(overrides: Partial<SideFeatures> = {}): SideFeatures {
    return {
      neck_angle_deg: 12.0, nose_shoulder_angle_deg: -8.0,
      head_forward_offset_px: 5.0, nose_forward_offset_px: 10.0,
      nose_to_ear_px: 20.0, mean_visibility: 0.9,
      ...overrides,
    };
  }

  it("divides offset features by the scale reference and leaves angles/visibility untouched", () => {
    const normalized = normalizeFeatures(makeRaw(), 10.0);
    expect(normalized.head_forward_offset_norm).toBeCloseTo(0.5, 5);
    expect(normalized.nose_forward_offset_norm).toBeCloseTo(1.0, 5);
    expect(normalized.neck_angle_deg).toBe(12.0);
  });

  it("throws when scaleReference is not positive", () => {
    expect(() => normalizeFeatures(makeRaw(), 0)).toThrow();
  });
});

describe("featureVector", () => {
  it("extracts values in NORMALIZED_FEATURE_ORDER", () => {
    const normalized = {
      neck_angle_deg: 1, nose_shoulder_angle_deg: 2,
      head_forward_offset_norm: 3, nose_forward_offset_norm: 4, mean_visibility: 5,
    };
    expect(featureVector(normalized)).toEqual([1, 2, 3, 4, 5]);
  });
});
