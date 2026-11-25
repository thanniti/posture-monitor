// web/lib/frontalClassifier.test.ts
import { describe, expect, it } from "vitest";
import { classifyFrontal, computeFrontalBaseline } from "./frontalClassifier";
import type { FrontalFeatures } from "./features";

function makeFrontalRow(overrides: Partial<FrontalFeatures> = {}): FrontalFeatures {
  return {
    nose_to_shoulder_vertical_px: 20.0,
    ear_to_shoulder_vertical_px: 15.0,
    shoulder_width_px: 100.0,
    shoulder_tilt_deg: 0.0,
    ear_tilt_deg: 0.0,
    interocular_distance_px: 40.0,
    mean_visibility: 1.0,
    ...overrides,
  };
}

describe("computeFrontalBaseline", () => {
  it("computes the median per field", () => {
    const rows = [18.0, 20.0, 22.0].map((v) => makeFrontalRow({ nose_to_shoulder_vertical_px: v }));
    expect(computeFrontalBaseline(rows).nose_to_shoulder_vertical_px).toBeCloseTo(20.0, 5);
  });
});

describe("classifyFrontal", () => {
  it("matches the baseline -> CorrectSitting at 100%", () => {
    const baseline = computeFrontalBaseline([makeFrontalRow()]);
    const [label, confidence] = classifyFrontal(makeFrontalRow(), baseline);
    expect(label).toBe("CorrectSitting");
    expect(confidence).toBeCloseTo(100.0, 5);
  });

  it("flags Slouching when the neck compresses", () => {
    const baseline = computeFrontalBaseline([makeFrontalRow({ nose_to_shoulder_vertical_px: 20.0, ear_to_shoulder_vertical_px: 15.0, interocular_distance_px: 40.0 })]);
    const live = makeFrontalRow({ nose_to_shoulder_vertical_px: 10.0, ear_to_shoulder_vertical_px: 8.0, interocular_distance_px: 40.0 });
    const [label, confidence] = classifyFrontal(live, baseline);
    expect(label).toBe("Slouching");
    expect(confidence).toBeGreaterThan(50.0);
  });

  it("stays CorrectSitting for a small neck change", () => {
    const baseline = computeFrontalBaseline([makeFrontalRow({ nose_to_shoulder_vertical_px: 20.0, interocular_distance_px: 40.0 })]);
    const live = makeFrontalRow({ nose_to_shoulder_vertical_px: 19.0, interocular_distance_px: 40.0 });
    expect(classifyFrontal(live, baseline)[0]).toBe("CorrectSitting");
  });

  it("flags Slouching when shoulders narrow", () => {
    const baseline = computeFrontalBaseline([makeFrontalRow({ shoulder_width_px: 100.0, interocular_distance_px: 40.0 })]);
    const live = makeFrontalRow({ shoulder_width_px: 70.0, interocular_distance_px: 40.0 });
    expect(classifyFrontal(live, baseline)[0]).toBe("Slouching");
  });

  it("flags LateralLean on shoulder tilt", () => {
    const baseline = computeFrontalBaseline([makeFrontalRow({ shoulder_tilt_deg: 0.0 })]);
    const live = makeFrontalRow({ shoulder_tilt_deg: 15.0 });
    expect(classifyFrontal(live, baseline)[0]).toBe("LateralLean");
  });

  it("gives higher confidence for more severe deviation", () => {
    const baseline = computeFrontalBaseline([makeFrontalRow({ nose_to_shoulder_vertical_px: 20.0, interocular_distance_px: 40.0 })]);
    const mild = makeFrontalRow({ nose_to_shoulder_vertical_px: 16.0, interocular_distance_px: 40.0 });
    const severe = makeFrontalRow({ nose_to_shoulder_vertical_px: 5.0, interocular_distance_px: 40.0 });
    expect(classifyFrontal(severe, baseline)[1]).toBeGreaterThan(classifyFrontal(mild, baseline)[1]);
  });

  it("caps confidence at 100 for a stretched neck", () => {
    const baseline = computeFrontalBaseline([makeFrontalRow({ nose_to_shoulder_vertical_px: 20.0, interocular_distance_px: 40.0 })]);
    const live = makeFrontalRow({ nose_to_shoulder_vertical_px: 30.0, interocular_distance_px: 40.0 });
    const [label, confidence] = classifyFrontal(live, baseline);
    expect(label).toBe("CorrectSitting");
    expect(confidence).toBeCloseTo(100.0, 5);
  });
});
