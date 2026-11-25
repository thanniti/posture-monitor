// web/lib/session.test.ts
import { describe, expect, it } from "vitest";
import {
  applyAngleTolerance,
  classifySideFrame,
  PostureSession,
  swapForwardBackward,
} from "./session";
import type { ForestModel } from "./classifier";
import type { Keypoints } from "./features";
import type { SideFeatures } from "./features";

describe("applyAngleTolerance", () => {
  it("overrides a lean label to CorrectSitting when both angles are within tolerance", () => {
    const raw = { neck_angle_deg: 5.0, nose_shoulder_angle_deg: -3.0 } as SideFeatures;
    expect(applyAngleTolerance(raw, "LeanForward")).toBe("CorrectSitting");
  });

  it("leaves a large lean unchanged", () => {
    const raw = { neck_angle_deg: 25.0, nose_shoulder_angle_deg: 10.0 } as SideFeatures;
    expect(applyAngleTolerance(raw, "LeanForward")).toBe("LeanForward");
  });

  it("does not affect PoseUncertain", () => {
    const raw = { neck_angle_deg: 0.0, nose_shoulder_angle_deg: 0.0 } as SideFeatures;
    expect(applyAngleTolerance(raw, "PoseUncertain")).toBe("PoseUncertain");
  });
});

describe("swapForwardBackward", () => {
  it("swaps LeanForward and LeanBackward, leaves everything else alone", () => {
    expect(swapForwardBackward("LeanForward")).toBe("LeanBackward");
    expect(swapForwardBackward("LeanBackward")).toBe("LeanForward");
    expect(swapForwardBackward("CorrectSitting")).toBe("CorrectSitting");
  });
});

function makeStumpModel(): ForestModel {
  return {
    classes: ["CorrectSitting", "LeanForward"],
    featureOrder: ["neck_angle_deg", "nose_shoulder_angle_deg", "head_forward_offset_norm", "nose_forward_offset_norm", "mean_visibility"],
    scaleReference: 1.0,
    trees: [
      {
        feature: [0, -2, -2],
        threshold: [0.0, -2.0, -2.0],
        childrenLeft: [1, -1, -1],
        childrenRight: [2, -1, -1],
        value: [[0, 0], [10, 0], [0, 10]],
      },
    ],
  };
}

describe("classifySideFrame", () => {
  it("returns PoseUncertain with null confidence below the visibility threshold", () => {
    const raw = {
      neck_angle_deg: 0, nose_shoulder_angle_deg: 0,
      head_forward_offset_px: 1, nose_forward_offset_px: 1,
      nose_to_ear_px: 10, mean_visibility: 0.1,
    } as SideFeatures;
    const [label, confidence] = classifySideFrame(raw, makeStumpModel(), 10.0);
    expect(label).toBe("PoseUncertain");
    expect(confidence).toBeNull();
  });

  it("returns the model's predicted label and confidence outside tolerance", () => {
    const raw = {
      neck_angle_deg: 30, nose_shoulder_angle_deg: 30,
      head_forward_offset_px: 3, nose_forward_offset_px: 3,
      nose_to_ear_px: 10, mean_visibility: 0.95,
    } as SideFeatures;
    const [label, confidence] = classifySideFrame(raw, makeStumpModel(), 1.0);
    expect(label).toBe("LeanForward");
    expect(confidence).toBeCloseTo(100.0, 1);
  });
});

function makeSideKeypoints(): Keypoints {
  return {
    nose: [60, 10, 1.0],
    left_eye: [58, 8, 1.0], right_eye: [58, 8, 1.0],
    left_ear: [50, 20, 1.0], right_ear: [50, 20, 1.0],
    left_shoulder: [50, 60, 1.0], right_shoulder: [50, 60, 1.0],
    left_hip: [50, 100, 1.0], right_hip: [50, 100, 1.0],
    left_knee: [50, 140, 1.0], right_knee: [50, 140, 1.0],
    left_ankle: [50, 180, 1.0], right_ankle: [50, 180, 1.0],
  } as unknown as Keypoints;
}

function makeFrontKeypoints(): Keypoints {
  return {
    nose: [60, 30, 1.0],
    left_eye: [55, 20, 1.0], right_eye: [65, 20, 1.0],
    left_ear: [45, 25, 1.0], right_ear: [75, 25, 1.0],
    left_shoulder: [40, 80, 1.0], right_shoulder: [80, 80, 1.0],
    left_hip: [40, 120, 1.0], right_hip: [80, 120, 1.0],
    left_knee: [40, 160, 1.0], right_knee: [80, 160, 1.0],
    left_ankle: [40, 200, 1.0], right_ankle: [80, 200, 1.0],
  } as unknown as Keypoints;
}

describe("PostureSession", () => {
  it("reports no-pose and resets timing when keypoints are null", () => {
    const session = new PostureSession(makeStumpModel());
    expect(session.processFrame(null, 0)).toEqual({ kind: "no-pose" });
  });

  it("runs side calibration, then front calibration, then live side classification", () => {
    const session = new PostureSession(makeStumpModel());
    let status;

    // Side calibration: 17 frames (t=0..16) of side-shaped keypoints. The
    // phase completes internally on the t=15 frame (phaseElapsed reaches 15s
    // via 1s deltas), but that frame still reports phase "side" (matching
    // run()'s Python behavior where completion is detected before the
    // return); the NEXT frame (t=16) is the first to report phase "front".
    for (let t = 0; t <= 16; t++) {
      status = session.processFrame(makeSideKeypoints(), t);
    }
    expect(status).toMatchObject({ kind: "calibrating", phase: "front", matches: false });

    // Front calibration: enough consecutive front-shaped frames (1s deltas)
    // to accumulate 15s of phaseElapsed.
    for (let t = 17; t <= 32; t++) {
      status = session.processFrame(makeFrontKeypoints(), t);
    }

    // Back to side view: both calibrations are done, so this now classifies live.
    for (let t = 33; t <= 35; t++) {
      status = session.processFrame(makeSideKeypoints(), t);
    }
    expect(status).toMatchObject({ kind: "live", mode: "side" });
  });
});
