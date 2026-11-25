// web/lib/config.ts
export const CALIBRATION_SECONDS = 15;
export const SMOOTHING_WINDOW_SECONDS = 2.5;
export const DEBOUNCE_SECONDS = 5.0;
export const CORRECT_SITTING_ANGLE_TOLERANCE_DEG = 8.0;
export const MIN_MEAN_VISIBILITY = 0.5;
export const SIDE_VIEW_RATIO_THRESHOLD = 1.3;
export const FRONT_VIEW_RATIO_THRESHOLD = 1.6;
export const ORIENTATION_SMOOTHING_WINDOW_SECONDS = 1.5;
export const MIN_INTEROCULAR_DISTANCE_PX = 8.0;
export const FRONT_NECK_SHRINK_TOLERANCE = 0.12;
export const FRONT_SHOULDER_PROTRACTION_TOLERANCE = 0.12;
export const FRONT_LATERAL_TILT_TOLERANCE_DEG = 6.0;
// The Python side currently has this off (see config.py's comment: the blanket
// swap wasn't a sound fix for camera mirroring and is disabled pending a real
// per-session sign-correction approach). Keep it in sync with config.py.
export const SWAP_LEAN_FORWARD_BACKWARD = false;

export const NORMALIZED_FEATURE_ORDER = [
  "neck_angle_deg",
  "nose_shoulder_angle_deg",
  "head_forward_offset_norm",
  "nose_forward_offset_norm",
  "mean_visibility",
] as const;
