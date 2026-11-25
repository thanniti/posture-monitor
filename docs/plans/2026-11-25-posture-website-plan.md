# Posture Monitor Website Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing Python posture-monitoring project into a public website: an Apple-style landing page (benefits of good posture, how the tool works) plus a fully in-browser live version of the posture-detection tool, deployed on Vercel with no backend and no retraining required by visitors.

**Architecture:** A Next.js (App Router) + TypeScript + Tailwind app in a new `web/` directory. The trained RandomForest is exported once to a small JSON tree structure and walked by a hand-written TypeScript function (no ML runtime dependency). MediaPipe's official JS build (`@mediapipe/tasks-vision`) reuses the existing `.task` model file unchanged. All of the Python feature-extraction, calibration, smoothing, and orientation-detection logic is ported 1:1 into small, independently testable `lib/` modules.

**Tech Stack:** Next.js, TypeScript, Tailwind CSS, `@mediapipe/tasks-vision`, Vitest, deployed on Vercel.

**Design spec:** `docs/specs/2026-11-25-posture-website-design.md` — read it for the full rationale; this plan implements it.

---

## Task 0: Merge the ML work into `master` and branch for the website

**Context:** All the real Python ML code currently lives on the `posture-ml-model` branch; `master` only has docs. `scripts/export_model.py` (Task 1) needs the Python code, so it must land on `master` first.

**Files:** none (git operations only), run from `/Users/tun/Desktop/posture-monitor` (the main checkout, not a worktree).

- [ ] **Step 1: Merge `posture-ml-model` into `master`**

```bash
cd /Users/tun/Desktop/posture-monitor
git checkout master
git merge posture-ml-model --no-edit
```

Expected: a clean fast-forward or merge commit, no conflicts (the two branches haven't diverged on the same files).

- [ ] **Step 2: Push master**

```bash
git push origin master
```

- [ ] **Step 3: Set up an isolated worktree for the website branch**

Use the `superpowers:using-git-worktrees` skill to create a worktree for a new `posture-website` branch off the now-updated `master`. All remaining tasks in this plan happen inside that worktree.

---

## Task 1: Export the trained model to JSON

**Context:** The browser can't run scikit-learn. We export the already-trained `RandomForestClassifier`'s tree structure (not retrain, not approximate) into a JSON file the TypeScript classifier (Task 10) walks directly.

**Files:**
- Create: `scripts/export_model.py`
- Create (generated, committed): `web/public/models/classifier.json` — wait, created in Task 2's `web/` directory; run this step again after Task 2 if `web/` doesn't exist yet. For now, output to a temp path and move it once `web/` exists (see Step 3).

- [ ] **Step 1: Write the export script**

```python
# scripts/export_model.py
import json
import sys

import joblib

from posture_monitor.calibration.calibrate import NORMALIZED_FEATURE_ORDER


def _export_tree(tree):
    return {
        "feature": tree.feature.tolist(),
        "threshold": tree.threshold.tolist(),
        "childrenLeft": tree.children_left.tolist(),
        "childrenRight": tree.children_right.tolist(),
        "value": [node_value[0].tolist() for node_value in tree.value],
    }


def export_model(model_bundle_path, output_path):
    bundle = joblib.load(model_bundle_path)
    model = bundle["model"]
    if type(model).__name__ != "RandomForestClassifier":
        raise ValueError(
            f"export_model only supports RandomForestClassifier, got {type(model).__name__}. "
            "If train.py selected a different model type, this export script needs updating "
            "to match (e.g. a different tree-walk, or an SVM-specific export)."
        )
    payload = {
        "classes": list(model.classes_),
        "featureOrder": NORMALIZED_FEATURE_ORDER,
        "scaleReference": bundle["scale_reference"],
        "trees": [_export_tree(estimator.tree_) for estimator in model.estimators_],
    }
    with open(output_path, "w") as f:
        json.dump(payload, f)
    print(f"Exported {len(payload['trees'])} trees, {len(payload['classes'])} classes to {output_path}")


if __name__ == "__main__":
    model_bundle_path = sys.argv[1] if len(sys.argv) > 1 else "posture_monitor/models/classifier.joblib"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "web/public/models/classifier.json"
    export_model(model_bundle_path, output_path)
```

- [ ] **Step 2: Run it (after Task 2 creates `web/`) and verify the output**

```bash
.venv/bin/python3 scripts/export_model.py
python3 -c "import json; d = json.load(open('web/public/models/classifier.json')); print(d['classes'], len(d['trees']), d['scaleReference'])"
```

Expected: prints the class list (e.g. `['CorrectSitting', 'LeanBackward', 'LeanForward']`), `200` trees, and the scale reference float.

- [ ] **Step 3: Commit**

```bash
git add scripts/export_model.py web/public/models/classifier.json
git commit -m "Export trained RandomForest to a browser-walkable JSON tree structure"
```

---

## Task 2: Scaffold the Next.js app

**Files:** everything under `web/` created by the scaffolding tool, plus `web/vitest.config.ts` and `web/package.json` edits.

- [ ] **Step 1: Scaffold with create-next-app**

```bash
cd /Users/tun/Desktop/posture-monitor
npx create-next-app@latest web --typescript --tailwind --eslint --app --no-src-dir --import-alias "@/*" --use-npm
```

- [ ] **Step 2: Install the extra dependencies**

```bash
cd web
npm install @mediapipe/tasks-vision
npm install -D vitest
```

- [ ] **Step 3: Add the Vitest config**

```typescript
// web/vitest.config.ts
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    include: ["lib/**/*.test.ts"],
  },
});
```

- [ ] **Step 4: Add the test script**

Edit `web/package.json`'s `"scripts"` block to add:

```json
"test": "vitest run"
```

- [ ] **Step 5: Verify the scaffold builds and the (empty) test suite runs**

```bash
npm run test
npm run build
```

Expected: `npm run test` reports "No test files found" (not an error — there are no `lib/` files yet) or passes with 0 tests; `npm run build` succeeds.

- [ ] **Step 6: Commit**

```bash
cd /Users/tun/Desktop/posture-monitor
git add web/
git commit -m "Scaffold the Next.js website app"
```

---

## Task 3: Add the pose landmarker model as a build-time download

**Context:** `pose_landmarker_lite.task` is a 5.5MB third-party asset, already gitignored on the Python side and fetched via `scripts/download_model.sh`. Rather than committing the binary into `web/`, mirror that pattern so Vercel fetches it fresh at build time from the same URL already used by the Python project.

**Files:**
- Create: `web/scripts/download-model.sh`
- Modify: `web/package.json` (add `prebuild` script)
- Modify: `web/.gitignore` (ignore `public/models/pose_landmarker_lite.task`, but NOT `public/models/classifier.json`, which stays committed from Task 1)

- [ ] **Step 1: Write the download script**

```bash
#!/usr/bin/env bash
# web/scripts/download-model.sh
set -euo pipefail
mkdir -p "$(dirname "$0")/../public/models"
curl -sL -o "$(dirname "$0")/../public/models/pose_landmarker_lite.task" \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
echo "Downloaded public/models/pose_landmarker_lite.task"
```

```bash
chmod +x web/scripts/download-model.sh
```

- [ ] **Step 2: Wire it into the build**

Edit `web/package.json`'s `"scripts"` block to add:

```json
"prebuild": "bash scripts/download-model.sh"
```

(`npm run build` automatically runs `prebuild` first — this is how Vercel will fetch the asset on every deploy.)

- [ ] **Step 3: Ignore the downloaded binary**

Add to `web/.gitignore`:

```
public/models/pose_landmarker_lite.task
```

- [ ] **Step 4: Run it locally and verify**

```bash
cd web
npm run prebuild
ls -la public/models/pose_landmarker_lite.task
```

Expected: the file exists and is a few MB (not an HTML error page — if `curl` got a 404/error page the file will be tiny; check with `file public/models/pose_landmarker_lite.task` if the size looks wrong).

- [ ] **Step 5: Commit**

```bash
git add web/scripts/download-model.sh web/package.json web/.gitignore
git commit -m "Fetch the pose landmarker model at build time instead of committing it"
```

---

## Task 4: `lib/config.ts`

**Context:** Ported constants from `posture_monitor/config.py`, using the values currently active on `master` (15s calibration, swap disabled, orientation-smoothing fix included). If the Python config changes later, update this file to match — it is not auto-synced.

**Files:** Create: `web/lib/config.ts`

- [ ] **Step 1: Write the file**

```typescript
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
```

- [ ] **Step 2: Commit**

```bash
git add lib/config.ts
git commit -m "Add ported config constants"
```

---

## Task 5: `lib/smoothing.ts`

**Context:** Port of `posture_monitor/inference/smoothing.py`'s `PostureMonitorState`.

**Files:** Create: `web/lib/smoothing.ts`, `web/lib/smoothing.test.ts`

- [ ] **Step 1: Write the failing tests**

```typescript
// web/lib/smoothing.test.ts
import { describe, expect, it } from "vitest";
import { PostureMonitorState } from "./smoothing";

describe("PostureMonitorState", () => {
  it("returns the single label as smoothed when history has one entry", () => {
    const state = new PostureMonitorState(2.5, 5.0);
    const [smoothed, flagged] = state.update(0.0, "CorrectSitting");
    expect(smoothed).toBe("CorrectSitting");
    expect(flagged).toBe(false);
  });

  it("smooths via majority vote within the window", () => {
    const state = new PostureMonitorState(2.5, 5.0);
    state.update(0.0, "LeanForward");
    state.update(1.0, "LeanForward");
    const [smoothed] = state.update(2.0, "CorrectSitting");
    expect(smoothed).toBe("LeanForward");
  });

  it("drops entries older than the window", () => {
    const state = new PostureMonitorState(1.0, 5.0);
    state.update(0.0, "LeanForward");
    const [smoothed] = state.update(5.0, "CorrectSitting");
    expect(smoothed).toBe("CorrectSitting");
  });

  it("does not flag until debounceSeconds of sustained bad posture has passed", () => {
    const state = new PostureMonitorState(0.1, 5.0);
    state.update(0.0, "LeanForward");
    const [, flaggedEarly] = state.update(1.0, "LeanForward");
    expect(flaggedEarly).toBe(false);
    const [, flaggedLate] = state.update(6.0, "LeanForward");
    expect(flaggedLate).toBe(true);
  });

  it("resets and never flags once CorrectSitting reappears", () => {
    const state = new PostureMonitorState(0.1, 5.0);
    state.update(0.0, "LeanForward");
    state.update(6.0, "CorrectSitting");
    const [, flagged] = state.update(6.1, "LeanForward");
    expect(flagged).toBe(false);
  });
});
```

- [ ] **Step 2: Run and confirm it fails**

```bash
npm run test -- smoothing
```

Expected: fails with "Cannot find module './smoothing'" or similar.

- [ ] **Step 3: Write the implementation**

```typescript
// web/lib/smoothing.ts
export class PostureMonitorState {
  private history: Array<[number, string]> = [];
  private badPostureSince: number | null = null;

  constructor(
    private windowSeconds: number,
    private debounceSeconds: number
  ) {}

  update(timestamp: number, label: string): [string, boolean] {
    this.history.push([timestamp, label]);
    const cutoff = timestamp - this.windowSeconds;
    while (this.history.length > 0 && this.history[0][0] < cutoff) {
      this.history.shift();
    }

    const counts = new Map<string, number>();
    for (const [, l] of this.history) {
      counts.set(l, (counts.get(l) ?? 0) + 1);
    }
    let smoothed = this.history[0][1];
    let bestCount = -1;
    for (const [label_, count] of counts) {
      if (count > bestCount) {
        bestCount = count;
        smoothed = label_;
      }
    }

    if (smoothed === "CorrectSitting") {
      this.badPostureSince = null;
      return [smoothed, false];
    }

    if (this.badPostureSince === null) {
      this.badPostureSince = timestamp;
    }
    const flagged = timestamp - this.badPostureSince >= this.debounceSeconds;
    return [smoothed, flagged];
  }
}
```

- [ ] **Step 4: Run and confirm it passes**

```bash
npm run test -- smoothing
```

- [ ] **Step 5: Commit**

```bash
git add lib/smoothing.ts lib/smoothing.test.ts
git commit -m "Port PostureMonitorState to TypeScript"
```

---

## Task 6: `lib/orientation.ts`

**Context:** Port of `OrientationState` and `_raw_orientation_reading` from `posture_monitor/inference/realtime.py` (the majority-vote fix for the LateralLean-in-side-view bug).

**Files:** Create: `web/lib/orientation.ts`, `web/lib/orientation.test.ts`

- [ ] **Step 1: Write the failing tests**

```typescript
// web/lib/orientation.test.ts
import { describe, expect, it } from "vitest";
import { OrientationState, rawOrientationReading } from "./orientation";

describe("rawOrientationReading", () => {
  it("reads side below the side threshold", () => {
    expect(rawOrientationReading(0.5)).toBe("side");
  });

  it("reads front above the front threshold", () => {
    expect(rawOrientationReading(2.0)).toBe("front");
  });

  it("is null (ambiguous) between the thresholds", () => {
    expect(rawOrientationReading(1.45)).toBeNull();
  });
});

describe("OrientationState", () => {
  it("starts at the initial mode", () => {
    const state = new OrientationState(1.5, "side");
    expect(state.confirmedMode).toBe("side");
  });

  it("does not flip on a single extreme-lean frame", () => {
    const state = new OrientationState(1.5, "side");
    state.update(0.0, 0.9); // side
    const mode = state.update(1.0, 2.0); // one front-reading frame, outvoted by the prior side reading
    expect(mode).toBe("side");
  });

  it("flips once a second consecutive reading agrees", () => {
    const state = new OrientationState(1.5, "side");
    state.update(0.0, 0.9); // side
    state.update(1.0, 2.0); // front (outvoted, stays side)
    const mode = state.update(2.0, 2.0); // second consecutive front reading wins
    expect(mode).toBe("front");
  });

  it("does not let ambiguous readings affect the mode", () => {
    const state = new OrientationState(1.5, "side");
    const mode = state.update(0.0, 1.45); // ambiguous, no vote
    expect(mode).toBe("side");
  });

  it("ages old readings out of the window", () => {
    const state = new OrientationState(1.5, "side");
    state.update(0.0, 0.9); // side
    const mode = state.update(10.0, 2.0); // far enough that the old side reading is gone
    expect(mode).toBe("front");
  });
});
```

- [ ] **Step 2: Run and confirm it fails**

```bash
npm run test -- orientation
```

- [ ] **Step 3: Write the implementation**

```typescript
// web/lib/orientation.ts
import { FRONT_VIEW_RATIO_THRESHOLD, SIDE_VIEW_RATIO_THRESHOLD } from "./config";

export type Orientation = "side" | "front";

export function rawOrientationReading(ratio: number): Orientation | null {
  if (ratio < SIDE_VIEW_RATIO_THRESHOLD) return "side";
  if (ratio > FRONT_VIEW_RATIO_THRESHOLD) return "front";
  return null;
}

export class OrientationState {
  confirmedMode: Orientation;
  private history: Array<[number, Orientation | null]> = [];

  constructor(
    private windowSeconds: number,
    initialMode: Orientation = "side"
  ) {
    this.confirmedMode = initialMode;
  }

  update(timestamp: number, ratio: number): Orientation {
    const reading = rawOrientationReading(ratio);
    this.history.push([timestamp, reading]);
    const cutoff = timestamp - this.windowSeconds;
    while (this.history.length > 0 && this.history[0][0] < cutoff) {
      this.history.shift();
    }

    let sideVotes = 0;
    let frontVotes = 0;
    for (const [, r] of this.history) {
      if (r === "side") sideVotes++;
      else if (r === "front") frontVotes++;
    }
    if (sideVotes > frontVotes) this.confirmedMode = "side";
    else if (frontVotes > sideVotes) this.confirmedMode = "front";
    return this.confirmedMode;
  }
}
```

- [ ] **Step 4: Run and confirm it passes**

```bash
npm run test -- orientation
```

- [ ] **Step 5: Commit**

```bash
git add lib/orientation.ts lib/orientation.test.ts
git commit -m "Port OrientationState to TypeScript"
```

---

## Task 7: `lib/features.ts`

**Context:** Port of `posture_monitor/features/extraction.py` — `extract_features` and `extract_frontal_features`, plus their geometry helpers.

**Files:** Create: `web/lib/features.ts`, `web/lib/features.test.ts`

- [ ] **Step 1: Write the failing tests**

```typescript
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
```

- [ ] **Step 2: Run and confirm it fails**

```bash
npm run test -- features
```

- [ ] **Step 3: Write the implementation**

```typescript
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
```

- [ ] **Step 4: Run and confirm it passes**

```bash
npm run test -- features
```

- [ ] **Step 5: Commit**

```bash
git add lib/features.ts lib/features.test.ts
git commit -m "Port feature extraction to TypeScript"
```

---

## Task 8: `lib/calibration.ts`

**Context:** Port of `posture_monitor/calibration/calibrate.py`.

**Files:** Create: `web/lib/calibration.ts`, `web/lib/calibration.test.ts`

- [ ] **Step 1: Write the failing tests**

```typescript
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
```

- [ ] **Step 2: Run and confirm it fails**

```bash
npm run test -- calibration
```

- [ ] **Step 3: Write the implementation**

```typescript
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
  return NORMALIZED_FEATURE_ORDER.map((name) => (normalized as unknown as Record<string, number>)[name]);
}
```

- [ ] **Step 4: Run and confirm it passes**

```bash
npm run test -- calibration
```

- [ ] **Step 5: Commit**

```bash
git add lib/calibration.ts lib/calibration.test.ts
git commit -m "Port calibration (scale reference, normalization) to TypeScript"
```

---

## Task 9: `lib/frontalClassifier.ts`

**Context:** Port of `posture_monitor/inference/frontal_classifier.py` — the rule-based front-view classifier. Test cases mirror `tests/test_frontal_classifier.py` 1:1.

**Files:** Create: `web/lib/frontalClassifier.ts`, `web/lib/frontalClassifier.test.ts`

- [ ] **Step 1: Write the failing tests**

```typescript
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
```

- [ ] **Step 2: Run and confirm it fails**

```bash
npm run test -- frontalClassifier
```

- [ ] **Step 3: Write the implementation**

```typescript
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
```

- [ ] **Step 4: Run and confirm it passes**

```bash
npm run test -- frontalClassifier
```

- [ ] **Step 5: Commit**

```bash
git add lib/frontalClassifier.ts lib/frontalClassifier.test.ts
git commit -m "Port the rule-based front-view classifier to TypeScript"
```

---

## Task 10: `lib/classifier.ts`

**Context:** Walks the JSON tree structure exported in Task 1 — this IS the trained RandomForest, re-expressed as data. Tests use a tiny hand-built fixture tree, not the real (200-tree) exported model, so they stay fast and deterministic.

**Files:** Create: `web/lib/classifier.ts`, `web/lib/classifier.test.ts`

- [ ] **Step 1: Write the failing tests**

```typescript
// web/lib/classifier.test.ts
import { describe, expect, it } from "vitest";
import { predict, predictProba, type ForestModel } from "./classifier";

function makeStumpModel(): ForestModel {
  // A single tree with one split on feature 0 at threshold 0.0: left leaf favors
  // class 0 (10 votes), right leaf favors class 1 (10 votes). childrenLeft/Right
  // use -1 for leaves, feature/threshold use -2 for leaves (matches sklearn's
  // tree_ convention).
  const tree = {
    feature: [0, -2, -2],
    threshold: [0.0, -2.0, -2.0],
    childrenLeft: [1, -1, -1],
    childrenRight: [2, -1, -1],
    value: [[0, 0], [10, 0], [0, 10]],
  };
  return {
    classes: ["CorrectSitting", "LeanForward"],
    featureOrder: ["x"],
    scaleReference: 1.0,
    trees: [tree],
  };
}

describe("classifier tree-walk", () => {
  it("predicts the left leaf's class when the feature is below threshold", () => {
    const result = predict(makeStumpModel(), [-1.0]);
    expect(result.label).toBe("CorrectSitting");
    expect(result.confidence).toBeCloseTo(100.0, 1);
  });

  it("predicts the right leaf's class when the feature is above threshold", () => {
    const result = predict(makeStumpModel(), [1.0]);
    expect(result.label).toBe("LeanForward");
  });

  it("averages per-class probabilities across multiple trees", () => {
    const model = makeStumpModel();
    model.trees.push(model.trees[0]);
    const proba = predictProba(model, [-1.0]);
    expect(proba[0]).toBeCloseTo(1.0, 5);
    expect(proba[1]).toBeCloseTo(0.0, 5);
  });
});
```

- [ ] **Step 2: Run and confirm it fails**

```bash
npm run test -- classifier
```

- [ ] **Step 3: Write the implementation**

```typescript
// web/lib/classifier.ts
export interface ForestTree {
  feature: number[];
  threshold: number[];
  childrenLeft: number[];
  childrenRight: number[];
  value: number[][];
}

export interface ForestModel {
  classes: string[];
  featureOrder: string[];
  scaleReference: number;
  trees: ForestTree[];
}

function predictTreeProba(tree: ForestTree, features: number[]): number[] {
  let node = 0;
  while (tree.childrenLeft[node] !== -1) {
    const goLeft = features[tree.feature[node]] <= tree.threshold[node];
    node = goLeft ? tree.childrenLeft[node] : tree.childrenRight[node];
  }
  const counts = tree.value[node];
  const total = counts.reduce((a, b) => a + b, 0);
  return counts.map((c) => c / total);
}

export function predictProba(model: ForestModel, features: number[]): number[] {
  const nClasses = model.classes.length;
  const sums = new Array(nClasses).fill(0);
  for (const tree of model.trees) {
    const proba = predictTreeProba(tree, features);
    for (let i = 0; i < nClasses; i++) sums[i] += proba[i];
  }
  return sums.map((s) => s / model.trees.length);
}

export function predict(model: ForestModel, features: number[]): { label: string; confidence: number } {
  const proba = predictProba(model, features);
  let bestIndex = 0;
  for (let i = 1; i < proba.length; i++) {
    if (proba[i] > proba[bestIndex]) bestIndex = i;
  }
  return { label: model.classes[bestIndex], confidence: Math.round(proba[bestIndex] * 1000) / 10 };
}

export async function loadModel(url: string): Promise<ForestModel> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load classifier model from ${url}: ${response.status}`);
  }
  return response.json();
}
```

- [ ] **Step 4: Run and confirm it passes**

```bash
npm run test -- classifier
```

- [ ] **Step 5: Commit**

```bash
git add lib/classifier.ts lib/classifier.test.ts
git commit -m "Add RandomForest tree-walk inference for the exported model"
```

---

## Task 11: `lib/session.ts`

**Context:** Port of `run()`'s per-frame logic from `posture_monitor/inference/realtime.py`, restructured as a class with a `processFrame` method called once per animation frame (instead of iterating a Python generator). This is the orchestration layer: guided two-phase calibration, orientation detection, dispatch to side/front classification, smoothing.

**Files:** Create: `web/lib/session.ts`, `web/lib/session.test.ts`

- [ ] **Step 1: Write the failing tests**

```typescript
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
```

- [ ] **Step 2: Run and confirm it fails**

```bash
npm run test -- session
```

- [ ] **Step 3: Write the implementation**

```typescript
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
```

- [ ] **Step 4: Run and confirm it passes**

```bash
npm run test -- session
```

If the `PostureSession` integration test doesn't pass exactly as written (e.g. an off-by-one in which frame reports which phase), treat the test's intent as authoritative — it's checking "side calibration finishes, then front calibration finishes, then live side-view classification happens" — and adjust the exact frame-index assertions to match what the real deltas/thresholds produce, rather than changing `processFrame`'s logic to force the original assertion to pass.

- [ ] **Step 5: Commit**

```bash
git add lib/session.ts lib/session.test.ts
git commit -m "Port the run() orchestration loop to a per-frame PostureSession class"
```

---

## Task 12: `lib/statusText.ts`

**Context:** Port of `_status_text_and_color` and `_guided_calibration_text` from `realtime.py`, adapted to take a `SessionStatus` (Task 11) and return display info for the overlay.

**Files:** Create: `web/lib/statusText.ts`, `web/lib/statusText.test.ts`

- [ ] **Step 1: Write the failing tests**

```typescript
// web/lib/statusText.test.ts
import { describe, expect, it } from "vitest";
import { statusDisplay } from "./statusText";

describe("statusDisplay", () => {
  it("shows No pose detected in red", () => {
    const { text, colorClass } = statusDisplay({ kind: "no-pose" });
    expect(text).toBe("No pose detected");
    expect(colorClass).toBe("status-red");
  });

  it("prompts to turn to the expected view when calibration isn't matching", () => {
    const { text, colorClass } = statusDisplay({ kind: "calibrating", phase: "side", matches: false, remainingSeconds: 10 });
    expect(text).toBe("Please turn to SIDE view to calibrate");
    expect(colorClass).toBe("status-yellow");
  });

  it("shows a countdown while calibration is matching", () => {
    const { text } = statusDisplay({ kind: "calibrating", phase: "front", matches: true, remainingSeconds: 7.0 });
    expect(text).toBe("Calibrating Front view... 7s");
  });

  it("shows mode, label, and confidence for a live CorrectSitting frame in green", () => {
    const { text, colorClass } = statusDisplay({ kind: "live", mode: "side", label: "CorrectSitting", confidence: 92.3, flagged: false });
    expect(text).toBe("[Side] CorrectSitting (92%)");
    expect(colorClass).toBe("status-green");
  });

  it("omits the confidence suffix when confidence is null, and is red for non-CorrectSitting", () => {
    const { text, colorClass } = statusDisplay({ kind: "live", mode: "front", label: "Slouching", confidence: null, flagged: true });
    expect(text).toBe("[Front] Slouching");
    expect(colorClass).toBe("status-red");
  });
});
```

- [ ] **Step 2: Run and confirm it fails**

```bash
npm run test -- statusText
```

- [ ] **Step 3: Write the implementation**

```typescript
// web/lib/statusText.ts
import type { SessionStatus } from "./session";

export interface StatusDisplay {
  text: string;
  colorClass: "status-green" | "status-red" | "status-yellow";
}

function capitalize(word: string): string {
  return word[0].toUpperCase() + word.slice(1);
}

export function statusDisplay(status: SessionStatus): StatusDisplay {
  if (status.kind === "no-pose") {
    return { text: "No pose detected", colorClass: "status-red" };
  }
  if (status.kind === "calibrating") {
    if (!status.matches) {
      return {
        text: `Please turn to ${status.phase.toUpperCase()} view to calibrate`,
        colorClass: "status-yellow",
      };
    }
    return {
      text: `Calibrating ${capitalize(status.phase)} view... ${Math.round(status.remainingSeconds)}s`,
      colorClass: "status-yellow",
    };
  }
  const confidenceSuffix = status.confidence !== null ? ` (${Math.round(status.confidence)}%)` : "";
  return {
    text: `[${capitalize(status.mode)}] ${status.label}${confidenceSuffix}`,
    colorClass: status.label === "CorrectSitting" ? "status-green" : "status-red",
  };
}
```

- [ ] **Step 4: Run and confirm it passes**

```bash
npm run test -- statusText
```

- [ ] **Step 5: Commit**

```bash
git add lib/statusText.ts lib/statusText.test.ts
git commit -m "Port status overlay text formatting to TypeScript"
```

---

## Task 13: `lib/pose.ts`

**Context:** Thin wrapper around `@mediapipe/tasks-vision`, matching the keypoint shape Python's `extract_keypoints` produces, plus the skeleton-drawing helper (port of `_draw_skeleton`). Not unit tested — it needs a real browser/WASM/canvas environment; verify manually in Task 14's dev-server check.

**Files:** Create: `web/lib/pose.ts`

- [ ] **Step 1: Write the file**

```typescript
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
```

- [ ] **Step 2: Commit**

```bash
git add lib/pose.ts
git commit -m "Add MediaPipe pose detection wrapper and skeleton drawing"
```

---

## Task 14: `components/TryItTool.tsx`

**Context:** The live tool UI: webcam capture, canvas skeleton overlay, status text overlay, calibration prompts, and error states. Visual/interactive, verified manually with the dev server (not unit tested, consistent with the design spec's testing scope).

**Files:** Create: `web/components/TryItTool.tsx`

- [ ] **Step 1: Write the component**

```tsx
// web/components/TryItTool.tsx
"use client";

import { useEffect, useRef, useState } from "react";
import type { PoseLandmarker } from "@mediapipe/tasks-vision";
import { createLandmarker, drawSkeleton, extractKeypoints } from "@/lib/pose";
import { loadModel, type ForestModel } from "@/lib/classifier";
import { PostureSession, type SessionStatus } from "@/lib/session";
import { statusDisplay } from "@/lib/statusText";

type ToolState = "idle" | "loading" | "running" | "camera-error" | "load-error";

export default function TryItTool() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<PoseLandmarker | null>(null);
  const sessionRef = useRef<PostureSession | null>(null);
  const rafRef = useRef<number | null>(null);

  const [toolState, setToolState] = useState<ToolState>("idle");
  const [status, setStatus] = useState<SessionStatus>({ kind: "no-pose" });

  function loop() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const landmarker = landmarkerRef.current;
    const session = sessionRef.current;
    if (!video || !canvas || !landmarker || !session) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");

    const keypoints = extractKeypoints(landmarker, video, performance.now());
    const newStatus = session.processFrame(keypoints, performance.now() / 1000);
    setStatus(newStatus);

    if (ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (keypoints) drawSkeleton(ctx, keypoints);
    }

    rafRef.current = requestAnimationFrame(loop);
  }

  async function start() {
    setToolState("loading");
    try {
      const [landmarker, model] = await Promise.all([
        createLandmarker("/models/pose_landmarker_lite.task"),
        loadModel("/models/classifier.json") as Promise<ForestModel>,
      ]);
      landmarkerRef.current = landmarker;
      sessionRef.current = new PostureSession(model);
    } catch {
      setToolState("load-error");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    } catch {
      setToolState("camera-error");
      return;
    }

    setToolState("running");
    rafRef.current = requestAnimationFrame(loop);
  }

  useEffect(() => {
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      landmarkerRef.current?.close();
      const stream = videoRef.current?.srcObject as MediaStream | null;
      stream?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  const { text, colorClass } = statusDisplay(status);

  return (
    <div className="relative mx-auto aspect-video w-full max-w-3xl overflow-hidden rounded-2xl bg-black shadow-xl">
      <video ref={videoRef} className="h-full w-full object-cover" muted playsInline />
      <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />

      {toolState === "running" && (
        <div className={`absolute left-4 top-4 text-lg font-semibold drop-shadow ${colorClass}`}>{text}</div>
      )}

      {toolState === "idle" && (
        <div className="absolute inset-0 flex items-center justify-center bg-white/90">
          <button
            onClick={start}
            className="rounded-full bg-accent px-6 py-3 font-medium text-white shadow-md transition hover:opacity-90"
          >
            Start camera
          </button>
        </div>
      )}

      {toolState === "loading" && (
        <div className="absolute inset-0 flex items-center justify-center bg-white/90 text-ink">
          Loading posture detector…
        </div>
      )}

      {toolState === "camera-error" && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-white/90 px-6 text-center text-ink">
          <p>Camera access is needed to use this tool.</p>
          <button onClick={start} className="rounded-full bg-accent px-6 py-2 font-medium text-white">
            Try again
          </button>
        </div>
      )}

      {toolState === "load-error" && (
        <div className="absolute inset-0 flex items-center justify-center bg-white/90 px-6 text-center text-ink">
          Couldn&apos;t load the posture detector — please refresh.
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add components/TryItTool.tsx
git commit -m "Add the live TryItTool component"
```

---

## Task 15: Marketing section components

**Context:** The static content sections from the design spec's content outline, Light & Monochrome styling. Presentational only, verified visually (Step 2).

**Files:**
- Create: `web/components/Nav.tsx`
- Create: `web/components/Hero.tsx`
- Create: `web/components/Benefits.tsx`
- Create: `web/components/HowItWorks.tsx`
- Create: `web/components/HowToUse.tsx`
- Create: `web/components/Footer.tsx`

- [ ] **Step 1: Write `components/Nav.tsx`**

```tsx
// web/components/Nav.tsx
export default function Nav() {
  return (
    <header className="sticky top-0 z-10 flex items-center justify-between border-b border-hairline bg-white/80 px-6 py-4 backdrop-blur">
      <span className="text-lg font-semibold text-ink">PostureIQ</span>
      <nav className="flex gap-6 text-sm text-ink">
        <a href="#benefits" className="hover:text-accent">Benefits</a>
        <a href="#how-it-works" className="hover:text-accent">How it works</a>
        <a
          href="#try-it"
          className="rounded-full bg-accent px-4 py-1.5 font-medium text-white hover:opacity-90"
        >
          Try it now
        </a>
      </nav>
    </header>
  );
}
```

- [ ] **Step 2: Write `components/Hero.tsx`**

```tsx
// web/components/Hero.tsx
export default function Hero() {
  return (
    <section className="flex flex-col items-center gap-4 px-6 py-24 text-center">
      <p className="text-xs font-semibold uppercase tracking-widest text-muted">PostureIQ</p>
      <h1 className="max-w-2xl text-5xl font-semibold tracking-tight text-ink sm:text-6xl">
        Sit better. Feel better.
      </h1>
      <p className="max-w-xl text-lg text-muted">
        Real-time posture coaching that runs entirely in your browser.
      </p>
      <a
        href="#try-it"
        className="mt-4 rounded-full bg-accent px-8 py-3 font-medium text-white shadow-md transition hover:opacity-90"
      >
        Try it now
      </a>
      <p className="text-sm text-muted">No sign-up. Nothing leaves your device.</p>
    </section>
  );
}
```

- [ ] **Step 3: Write `components/Benefits.tsx`**

```tsx
// web/components/Benefits.tsx
const BENEFITS = [
  { title: "Less strain", body: "Reduces back and neck pain caused by hours of slouching." },
  { title: "Better breathing", body: "An upright spine improves breathing and circulation." },
  { title: "Sharper focus", body: "Good posture is linked to more energy and concentration." },
  { title: "Long-term health", body: "Prevents the spinal issues that build up from years of bad habits." },
];

export default function Benefits() {
  return (
    <section id="benefits" className="mx-auto max-w-5xl px-6 py-20">
      <h2 className="text-center text-3xl font-semibold tracking-tight text-ink">
        Why good posture matters
      </h2>
      <div className="mt-12 grid gap-8 sm:grid-cols-2 lg:grid-cols-4">
        {BENEFITS.map((benefit) => (
          <div key={benefit.title} className="rounded-2xl border border-hairline p-6">
            <h3 className="font-semibold text-ink">{benefit.title}</h3>
            <p className="mt-2 text-sm text-muted">{benefit.body}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
```

- [ ] **Step 4: Write `components/HowItWorks.tsx`**

```tsx
// web/components/HowItWorks.tsx
const STEPS = [
  {
    title: "We watch your pose",
    body: "MediaPipe tracks your head, neck, and shoulders only. Nothing is recorded or sent anywhere.",
  },
  {
    title: "15-second calibration",
    body: "It learns your own proportions first — it doesn't assume you're sitting correctly yet.",
  },
  {
    title: "Live feedback",
    body: "Flags slouching or leaning in real time. Side view uses a trained ML model; facing the camera uses geometric rules against your own calibrated baseline.",
  },
];

export default function HowItWorks() {
  return (
    <section id="how-it-works" className="mx-auto max-w-5xl px-6 py-20">
      <h2 className="text-center text-3xl font-semibold tracking-tight text-ink">How it works</h2>
      <div className="mt-12 grid gap-8 sm:grid-cols-3">
        {STEPS.map((step, i) => (
          <div key={step.title}>
            <div className="text-sm font-semibold text-accent">{String(i + 1).padStart(2, "0")}</div>
            <h3 className="mt-2 font-semibold text-ink">{step.title}</h3>
            <p className="mt-2 text-sm text-muted">{step.body}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
```

- [ ] **Step 5: Write `components/HowToUse.tsx`**

```tsx
// web/components/HowToUse.tsx
const TIPS = [
  "Use good lighting so your face and shoulders are clearly visible.",
  "Both sitting side-on and facing the camera work.",
  "Sit naturally during the 15-second calibration — don't force \"perfect\" posture.",
];

export default function HowToUse() {
  return (
    <section className="mx-auto max-w-3xl px-6 py-20">
      <h2 className="text-center text-3xl font-semibold tracking-tight text-ink">How to use it</h2>
      <ul className="mt-8 space-y-3">
        {TIPS.map((tip) => (
          <li key={tip} className="flex gap-3 text-muted">
            <span className="text-accent">•</span>
            <span>{tip}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
```

- [ ] **Step 6: Write `components/Footer.tsx`**

```tsx
// web/components/Footer.tsx
export default function Footer() {
  return (
    <footer className="border-t border-hairline px-6 py-10 text-center text-sm text-muted">
      <p>Built as a class project.</p>
      <a
        href="https://github.com/thanniti/posture-monitor"
        className="mt-2 inline-block text-accent hover:underline"
      >
        View the code on GitHub
      </a>
    </footer>
  );
}
```

- [ ] **Step 7: Commit**

```bash
git add components/Nav.tsx components/Hero.tsx components/Benefits.tsx components/HowItWorks.tsx components/HowToUse.tsx components/Footer.tsx
git commit -m "Add marketing section components"
```

---

## Task 16: Assemble the page, theme, and layout

**Context:** Wires all the section components together into `app/page.tsx`, adds the Light & Monochrome color tokens to Tailwind, and the `status-*` classes used by `TryItTool`.

**Files:**
- Modify: `web/tailwind.config.ts`
- Modify: `web/app/globals.css`
- Modify: `web/app/layout.tsx`
- Modify: `web/app/page.tsx`

- [ ] **Step 1: Add theme colors to `tailwind.config.ts`**

Edit the generated `tailwind.config.ts`'s `theme.extend` to add:

```typescript
colors: {
  ink: "#1d1d1f",
  accent: "#0071e3",
  muted: "#86868b",
  hairline: "#e8e8ed",
},
```

- [ ] **Step 2: Add overlay color classes to `app/globals.css`**

Append to the generated `app/globals.css` (after the `@tailwind` directives):

```css
body {
  color: #1d1d1f;
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Helvetica, Arial, sans-serif;
}

.status-green {
  color: #30d158;
}
.status-red {
  color: #ff3b30;
}
.status-yellow {
  color: #ff9f0a;
}
```

- [ ] **Step 3: Write `app/layout.tsx`**

```tsx
// web/app/layout.tsx
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PostureIQ — Real-time posture coaching",
  description: "A browser-based posture monitor that watches your sitting posture and gives real-time feedback, powered by an on-device ML model.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-white">{children}</body>
    </html>
  );
}
```

- [ ] **Step 4: Write `app/page.tsx`**

```tsx
// web/app/page.tsx
import Nav from "@/components/Nav";
import Hero from "@/components/Hero";
import Benefits from "@/components/Benefits";
import HowItWorks from "@/components/HowItWorks";
import HowToUse from "@/components/HowToUse";
import TryItTool from "@/components/TryItTool";
import Footer from "@/components/Footer";

export default function Home() {
  return (
    <>
      <Nav />
      <main>
        <Hero />
        <Benefits />
        <HowItWorks />
        <HowToUse />
        <section id="try-it" className="px-6 py-20">
          <h2 className="mb-12 text-center text-3xl font-semibold tracking-tight text-ink">Try it</h2>
          <TryItTool />
        </section>
      </main>
      <Footer />
    </>
  );
}
```

- [ ] **Step 5: Run the dev server and visually verify**

```bash
npm run dev
```

Open `http://localhost:3000`. Verify: the nav, hero, benefits, how-it-works, how-to-use, and try-it sections all render in the Light & Monochrome style; clicking "Start camera" prompts for webcam permission and, once granted, shows the video feed with a skeleton overlay and status text that progresses through the guided calibration prompts.

- [ ] **Step 6: Run the full test suite one more time**

```bash
npm run test
npm run build
```

Expected: all `lib/` tests pass, production build succeeds.

- [ ] **Step 7: Commit**

```bash
git add tailwind.config.ts app/globals.css app/layout.tsx app/page.tsx
git commit -m "Assemble the landing page with the Light & Monochrome theme"
```

---

## Task 17: Deploy to Vercel

**Files:** none (deployment configuration only).

- [ ] **Step 1: Push the branch**

```bash
git push -u origin posture-website
```

- [ ] **Step 2: Create the Vercel project**

In the Vercel dashboard (or `vercel` CLI), import the `posture-monitor` GitHub repo, and set the project's **Root Directory** to `web`. Leave build/output settings at Next.js defaults (Vercel auto-detects `npm run build`, which runs `prebuild` first per Task 3).

- [ ] **Step 3: Deploy and verify**

Trigger a deploy (push to the branch, or "Deploy" in the dashboard). Once live, open the deployed URL on a device with a camera and confirm: the page loads, the theme renders correctly, and "Start camera" successfully starts the live tool end-to-end (calibration → live classification), matching the local dev-server check from Task 16.

- [ ] **Step 4: Merge to master**

Once verified, open a PR from `posture-website` into `master` and merge it (or merge directly if working solo) so the deployed site tracks `master` going forward.
