// web/lib/session.ts
import type { ForestModel } from "./classifier";
import { predict } from "./classifier";
import { computeFrontalBaseline, classifyFrontal, type FrontalBaseline } from "./frontalClassifier";
import { computeScaleReference, featureVector, normalizeFeatures } from "./calibration";
import {
  extractFeatures,
  extractFrontalFeatures,
  type FrontalFeatures,
  type Keypoints,
  type SideFeatures,
} from "./features";
import { OrientationState } from "./orientation";
import { PostureMonitorState } from "./smoothing";
import {
  CALIBRATION_SECONDS,
  CORRECT_SITTING_ANGLE_TOLERANCE_DEG,
  DEBOUNCE_SECONDS,
  FRONT_VIEW_RATIO_THRESHOLD,
  MIN_INTEROCULAR_DISTANCE_PX,
  MIN_MEAN_VISIBILITY,
  ORIENTATION_SMOOTHING_WINDOW_SECONDS,
  SIDE_VIEW_RATIO_THRESHOLD,
  SMOOTHING_WINDOW_SECONDS,
  SWAP_LEAN_FORWARD_BACKWARD,
} from "./config";

export function swapForwardBackward(label: string): string {
  if (label === "LeanForward") return "LeanBackward";
  if (label === "LeanBackward") return "LeanForward";
  return label;
}

export function applyAngleTolerance(raw: SideFeatures, predictedLabel: string): string {
  if (predictedLabel !== "LeanForward" && predictedLabel !== "LeanBackward") {
    return predictedLabel;
  }
  const withinTolerance =
    Math.abs(raw.neck_angle_deg) <= CORRECT_SITTING_ANGLE_TOLERANCE_DEG &&
    Math.abs(raw.nose_shoulder_angle_deg) <= CORRECT_SITTING_ANGLE_TOLERANCE_DEG;
  return withinTolerance ? "CorrectSitting" : predictedLabel;
}

export function classifySideFrame(
  raw: SideFeatures,
  model: ForestModel,
  scaleReference: number
): [string, number | null] {
  if (raw.mean_visibility < MIN_MEAN_VISIBILITY) {
    return ["PoseUncertain", null];
  }
  const normalized = normalizeFeatures(raw, scaleReference);
  const { label: rawLabel, confidence } = predict(model, featureVector(normalized));
  const swapped = SWAP_LEAN_FORWARD_BACKWARD ? swapForwardBackward(rawLabel) : rawLabel;
  const label = applyAngleTolerance(raw, swapped);
  return label !== swapped ? [label, null] : [label, confidence];
}

export function isFrontalDegenerate(raw: FrontalFeatures): boolean {
  return raw.mean_visibility < MIN_MEAN_VISIBILITY || raw.interocular_distance_px < MIN_INTEROCULAR_DISTANCE_PX;
}

export function classifyFrontFrame(raw: FrontalFeatures, baseline: FrontalBaseline): [string, number | null] {
  if (isFrontalDegenerate(raw)) {
    return ["PoseUncertain", null];
  }
  return classifyFrontal(raw, baseline);
}

export type SessionStatus =
  | { kind: "no-pose" }
  | { kind: "calibrating"; phase: "side" | "front"; matches: boolean; remainingSeconds: number }
  | { kind: "live"; mode: "side" | "front"; label: string; confidence: number | null; flagged: boolean };

export class PostureSession {
  private scaleReference: number | null = null;
  private baseline: FrontalBaseline | null = null;
  private orientationState = new OrientationState(ORIENTATION_SMOOTHING_WINDOW_SECONDS, "side");
  private smoothingState = new PostureMonitorState(SMOOTHING_WINDOW_SECONDS, DEBOUNCE_SECONDS);

  private sideCalibrationRows: SideFeatures[] = [];
  private frontCalibrationRows: FrontalFeatures[] = [];
  private phaseElapsed = 0;
  private lastTimestamp: number | null = null;

  constructor(private model: ForestModel) {}

  processFrame(keypoints: Keypoints | null, timestamp: number): SessionStatus {
    if (keypoints === null) {
      this.lastTimestamp = null;
      return { kind: "no-pose" };
    }

    const rawSide = extractFeatures(keypoints);
    const rawFront = extractFrontalFeatures(keypoints);
    const ratio = rawSide.nose_to_ear_px > 0 ? rawFront.shoulder_width_px / rawSide.nose_to_ear_px : 0.0;

    const delta = this.lastTimestamp !== null ? Math.max(0, timestamp - this.lastTimestamp) : 0;
    this.lastTimestamp = timestamp;

    if (this.scaleReference === null) {
      const matches = ratio < SIDE_VIEW_RATIO_THRESHOLD;
      if (matches) {
        this.phaseElapsed += delta;
        if (rawSide.mean_visibility >= MIN_MEAN_VISIBILITY) {
          this.sideCalibrationRows.push(rawSide);
        }
      }
      const remaining = Math.max(0, CALIBRATION_SECONDS - this.phaseElapsed);
      if (this.phaseElapsed >= CALIBRATION_SECONDS && this.sideCalibrationRows.length > 0) {
        this.scaleReference = computeScaleReference(this.sideCalibrationRows);
        this.phaseElapsed = 0;
      }
      return { kind: "calibrating", phase: "side", matches, remainingSeconds: remaining };
    }

    if (this.baseline === null) {
      const matches = ratio > FRONT_VIEW_RATIO_THRESHOLD;
      if (matches) {
        this.phaseElapsed += delta;
        if (!isFrontalDegenerate(rawFront)) {
          this.frontCalibrationRows.push(rawFront);
        }
      }
      const remaining = Math.max(0, CALIBRATION_SECONDS - this.phaseElapsed);
      if (this.phaseElapsed >= CALIBRATION_SECONDS && this.frontCalibrationRows.length > 0) {
        this.baseline = computeFrontalBaseline(this.frontCalibrationRows);
        this.phaseElapsed = 0;
      }
      return { kind: "calibrating", phase: "front", matches, remainingSeconds: remaining };
    }

    const mode = this.orientationState.update(timestamp, ratio);
    const [label, confidence] =
      mode === "side"
        ? classifySideFrame(rawSide, this.model, this.scaleReference)
        : classifyFrontFrame(rawFront, this.baseline);

    const [smoothed, flagged] = this.smoothingState.update(timestamp, label);
    const displayConfidence = label === smoothed ? confidence : null;

    return { kind: "live", mode, label: smoothed, confidence: displayConfidence, flagged };
  }
}
