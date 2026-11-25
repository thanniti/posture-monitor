// web/lib/features.test.ts
import { describe, expect, it } from "vitest";
import { extractFeatures, extractFrontalFeatures, type Keypoints } from "./features";

describe("extractFeatures", () => {
  it("computes a zero neck angle when the ear is directly above the shoulder", () => {
    const keypoints = {
      nose: [50, 0, 1.0],
      left_ear: [40, 20, 1.0], right_ear: [40, 20, 0.9],
      left_shoulder: [40, 60, 1.0], right_shoulder: [60, 60, 1.0],
    } as unknown as Keypoints;
    const features = extractFeatures(keypoints);
    expect(features.neck_angle_deg).toBeCloseTo(0, 5);
  });

  it("picks the higher-visibility side", () => {
    const keypoints = {
      nose: [50, 0, 1.0],
      left_ear: [40, 20, 0.2], right_ear: [45, 25, 0.9],
      left_shoulder: [40, 60, 0.2], right_shoulder: [45, 65, 0.9],
    } as unknown as Keypoints;
    const features = extractFeatures(keypoints);
    expect(features.nose_to_ear_px).toBeCloseTo(Math.hypot(50 - 45, 0 - 25), 5);
  });

  it("averages visibility across the picked nose/ear/shoulder", () => {
    const keypoints = {
      nose: [50, 0, 1.0],
      left_ear: [40, 20, 0.5], right_ear: [40, 20, 0.4],
      left_shoulder: [40, 60, 0.6], right_shoulder: [40, 60, 0.5],
    } as unknown as Keypoints;
    const features = extractFeatures(keypoints);
    expect(features.mean_visibility).toBeCloseTo((1.0 + 0.5 + 0.6) / 3, 5);
  });
});

describe("extractFrontalFeatures", () => {
  function makeSymmetricKeypoints(): Keypoints {
    return {
      nose: [60, 30, 1.0],
      left_eye: [55, 20, 1.0], right_eye: [65, 20, 1.0],
      left_ear: [45, 25, 1.0], right_ear: [75, 25, 1.0],
      left_shoulder: [40, 80, 1.0], right_shoulder: [80, 80, 1.0],
    } as unknown as Keypoints;
  }

  it("computes shoulder width and interocular distance as euclidean distances", () => {
    const features = extractFrontalFeatures(makeSymmetricKeypoints());
    expect(features.shoulder_width_px).toBeCloseTo(40, 5);
    expect(features.interocular_distance_px).toBeCloseTo(10, 5);
  });

  it("computes zero shoulder tilt when both shoulders are level", () => {
    const features = extractFrontalFeatures(makeSymmetricKeypoints());
    expect(features.shoulder_tilt_deg).toBeCloseTo(0, 5);
  });
});
