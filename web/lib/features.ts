// web/lib/features.ts
export type Keypoint = [number, number, number]; // x_px, y_px, visibility
export type Keypoints = Record<string, Keypoint>;

function euclidean(p1: Keypoint, p2: Keypoint): number {
  return Math.hypot(p1[0] - p2[0], p1[1] - p2[1]);
}

function angleFromVertical(bottomPoint: Keypoint, topPoint: Keypoint): number {
  const dx = topPoint[0] - bottomPoint[0];
  const dy = topPoint[1] - bottomPoint[1];
  return (Math.atan2(dx, -dy) * 180) / Math.PI;
}

function angleFromHorizontal(leftPoint: Keypoint, rightPoint: Keypoint): number {
  const dx = rightPoint[0] - leftPoint[0];
  const dy = rightPoint[1] - leftPoint[1];
  return (Math.atan2(dy, dx) * 180) / Math.PI;
}

function pickSide(keypoints: Keypoints, partName: string): Keypoint {
  const left = keypoints[`left_${partName}`];
  const right = keypoints[`right_${partName}`];
  return left[2] >= right[2] ? left : right;
}

function midpoint(p1: Keypoint, p2: Keypoint): Keypoint {
  return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2];
}

export interface SideFeatures {
  neck_angle_deg: number;
  nose_shoulder_angle_deg: number;
  head_forward_offset_px: number;
  nose_forward_offset_px: number;
  nose_to_ear_px: number;
  mean_visibility: number;
}

export function extractFeatures(keypoints: Keypoints): SideFeatures {
  const nose = keypoints["nose"];
  const ear = pickSide(keypoints, "ear");
  const shoulder = pickSide(keypoints, "shoulder");
  const visibilities = [nose[2], ear[2], shoulder[2]];

  return {
    neck_angle_deg: angleFromVertical(shoulder, ear),
    nose_shoulder_angle_deg: angleFromVertical(shoulder, nose),
    head_forward_offset_px: Math.abs(ear[0] - shoulder[0]),
    nose_forward_offset_px: Math.abs(nose[0] - shoulder[0]),
    nose_to_ear_px: euclidean(nose, ear),
    mean_visibility: visibilities.reduce((a, b) => a + b, 0) / visibilities.length,
  };
}

export interface FrontalFeatures {
  nose_to_shoulder_vertical_px: number;
  ear_to_shoulder_vertical_px: number;
  shoulder_width_px: number;
  shoulder_tilt_deg: number;
  ear_tilt_deg: number;
  interocular_distance_px: number;
  mean_visibility: number;
}

export function extractFrontalFeatures(keypoints: Keypoints): FrontalFeatures {
  const nose = keypoints["nose"];
  const leftEye = keypoints["left_eye"];
  const rightEye = keypoints["right_eye"];
  const leftEar = keypoints["left_ear"];
  const rightEar = keypoints["right_ear"];
  const leftShoulder = keypoints["left_shoulder"];
  const rightShoulder = keypoints["right_shoulder"];

  const earMid = midpoint(leftEar, rightEar);
  const shoulderMid = midpoint(leftShoulder, rightShoulder);

  const visibilities = [
    nose[2], leftEye[2], rightEye[2], leftEar[2], rightEar[2],
    leftShoulder[2], rightShoulder[2],
  ];

  return {
    nose_to_shoulder_vertical_px: Math.abs(nose[1] - shoulderMid[1]),
    ear_to_shoulder_vertical_px: Math.abs(earMid[1] - shoulderMid[1]),
    shoulder_width_px: euclidean(leftShoulder, rightShoulder),
    shoulder_tilt_deg: angleFromHorizontal(leftShoulder, rightShoulder),
    ear_tilt_deg: angleFromHorizontal(leftEar, rightEar),
    interocular_distance_px: euclidean(leftEye, rightEye),
    mean_visibility: visibilities.reduce((a, b) => a + b, 0) / visibilities.length,
  };
}
