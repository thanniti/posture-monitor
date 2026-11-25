// web/lib/pose.ts
import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";
import type { Keypoints } from "./features";

const LANDMARK_INDICES: Record<string, number> = {
  nose: 0,
  left_eye: 2, right_eye: 5,
  left_ear: 7, right_ear: 8,
  left_shoulder: 11, right_shoulder: 12,
  left_hip: 23, right_hip: 24,
  left_knee: 25, right_knee: 26,
  left_ankle: 27, right_ankle: 28,
};

export async function createLandmarker(modelAssetPath: string): Promise<PoseLandmarker> {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  return PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath },
    runningMode: "VIDEO",
  });
}

export function extractKeypoints(
  landmarker: PoseLandmarker,
  video: HTMLVideoElement,
  timestampMs: number
): Keypoints | null {
  const result = landmarker.detectForVideo(video, timestampMs);
  if (!result.landmarks || result.landmarks.length === 0) {
    return null;
  }
  const landmarks = result.landmarks[0];
  const width = video.videoWidth;
  const height = video.videoHeight;
  const keypoints: Keypoints = {};
  for (const [name, idx] of Object.entries(LANDMARK_INDICES)) {
    const lm = landmarks[idx];
    keypoints[name] = [lm.x * width, lm.y * height, lm.visibility ?? 0];
  }
  return keypoints;
}

const SKELETON_CONNECTIONS: Array<[string, string]> = [
  ["left_ear", "left_shoulder"], ["right_ear", "right_shoulder"],
  ["left_shoulder", "right_shoulder"],
  ["left_shoulder", "left_hip"], ["right_shoulder", "right_hip"],
  ["left_hip", "right_hip"],
  ["left_hip", "left_knee"], ["right_hip", "right_knee"],
  ["left_knee", "left_ankle"], ["right_knee", "right_ankle"],
];

export function drawSkeleton(ctx: CanvasRenderingContext2D, keypoints: Keypoints): void {
  ctx.strokeStyle = "rgba(255, 214, 10, 0.9)";
  ctx.lineWidth = 2;
  for (const [a, b] of SKELETON_CONNECTIONS) {
    const [xa, ya] = keypoints[a];
    const [xb, yb] = keypoints[b];
    ctx.beginPath();
    ctx.moveTo(xa, ya);
    ctx.lineTo(xb, yb);
    ctx.stroke();
  }
  ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
  for (const [x, y] of Object.values(keypoints)) {
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}
