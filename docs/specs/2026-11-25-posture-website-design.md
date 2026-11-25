# Posture Monitor Website — Design Spec

## Goal

Turn the existing Python posture-monitoring ML project into a public website: a marketing landing page explaining the benefits of good posture and how the tool works, plus a live in-browser version of the posture-detection tool itself. Deployed on Vercel so anyone with the link can use it without installing anything or retraining any model.

## Non-goals

- No user accounts, no data persistence, no analytics/tracking of posture data.
- No backend server. The site is pure client-side static/Next.js — webcam video and all processing stay in the visitor's browser.
- Not re-solving the existing model-quality issues (e.g. LeanForward/LeanBackward confusion) — this project ports the current Python behavior as-is, bugs and all. Model-quality fixes happen in the Python project and get re-exported.

## Visual direction

**Light & Monochrome**, Apple-product-page inspired: white background, near-black (`#1d1d1f`) text, a single blue accent (`#0071e3`), generous whitespace, large confident type (`-apple-system`/SF Pro stack), rounded pill buttons, minimal sticky nav.

## Site structure

Single scrolling page (`web/app/page.tsx`) with a sticky nav (`PostureIQ | Benefits | How it works | Try it`) that anchor-links to sections on the same page — no separate `/tool` route.

Sections, top to bottom:

1. **Hero** — Headline "Sit better. Feel better.", subhead "Real-time posture coaching that runs entirely in your browser.", a "Try it now" button that smooth-scrolls to the Try It section, and a small reassurance line ("no sign-up, nothing leaves your device").
2. **Benefits** — 4 short cards: reduces back/neck strain; better breathing & circulation; sharper focus & energy; prevents long-term spinal issues. One icon + 1-2 sentences each.
3. **How it works** — 3 numbered steps, written to honestly match the real pipeline: (1) "We watch your pose" — MediaPipe tracks head/neck/shoulders only, nothing recorded or sent anywhere; (2) "15-second calibration" — learns the visitor's own proportions, explicitly does not assume they're sitting correctly during this window; (3) "Live feedback" — flags slouching/leaning in real time; side view uses a trained ML classifier, front view (facing the camera) uses geometric rules against the visitor's own calibrated baseline.
4. **How to use** — practical tips: use good lighting; both side-view and facing-the-camera work; sit naturally (not deliberately "perfect") during the 15s calibration so the baseline reflects normal posture.
5. **Try it** — the live tool itself (see "Try It section" below).
6. **Footer** — short "built as a class project" note and a link to the GitHub repo.

## Try It section layout

Full-width video element with the pose skeleton drawn on top via canvas overlay, matching the current desktop app's look:
- Status text overlaid directly on the video (top-left): `[Mode] Label (confidence%)` using the same color rules as today (green for CorrectSitting, red otherwise, yellow while calibrating, red "No pose detected" if tracking is lost).
- A thin calibration progress bar overlaid near the bottom of the video during each calibration phase.
- A "Please turn to SIDE/FRONT view to calibrate" prompt overlay when the guided calibration phase's expected orientation isn't currently detected (pausing the countdown), mirroring the Python app's guided calibration.
- Before the webcam starts: a "Start camera" button (so camera access is an explicit visitor action, not automatic on page load) and a one-line camera-permission-denied fallback message with a retry button if `getUserMedia` fails.

## Scope: full feature parity

The website reproduces the current Python app's full feature set in the browser:
- Guided two-phase calibration (side view first, then front view, 15s each, pausing/prompting if the visitor isn't in the expected orientation).
- Automatic per-frame orientation detection with the majority-vote `OrientationState` stabilization (no instant single-frame flips).
- Side-view ML classification (CorrectSitting / LeanForward / LeanBackward) via the actual trained RandomForest model, exported to data (see below) — not retrained, not reimplemented as rules.
- Front-view rule-based classification (CorrectSitting / Slouching / LateralLean) against the visitor's own calibrated baseline.
- Angle-tolerance override, majority-vote label smoothing, and debounce, matching current constants.
- Skeleton overlay drawn from the same keypoints used for classification.

Any config constants (calibration seconds, thresholds, tolerances) are ported with the same values as `posture_monitor/config.py` at the time of export, not re-tuned as part of this project.

## Technical architecture

**Stack:** Next.js (App Router) + TypeScript + Tailwind CSS, in a new `web/` directory in the existing `posture-monitor` repo. Deployed on Vercel, rooted at `web/`, as a pure static/client app with no backend API routes required for the tool itself.

**Directory layout (`web/`):**
- `app/page.tsx` — the landing page, composed from section components.
- `components/Hero.tsx`, `Benefits.tsx`, `HowItWorks.tsx`, `HowToUse.tsx`, `Footer.tsx` — static marketing sections.
- `components/TryItTool.tsx` — owns the `<video>` element, the overlay `<canvas>`, the "Start camera" / permission-denied UI, and drives the per-frame loop via `requestAnimationFrame`, calling into `lib/session.ts`.
- `lib/pose.ts` — thin wrapper around `@mediapipe/tasks-vision`'s `PoseLandmarker`, loading `public/models/pose_landmarker_lite.task` and returning the same named-keypoint shape (`{nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle}`, each `[x, y, visibility]`) as `posture_monitor/features/landmarks.py`.
- `lib/features.ts` — 1:1 TypeScript port of `posture_monitor/features/extraction.py` (`extractFeatures`, `extractFrontalFeatures`, and their shared geometry helpers).
- `lib/calibration.ts` — port of `posture_monitor/calibration/calibrate.py` (`computeScaleReference`, `normalizeFeatures`, `featureVector`).
- `lib/classifier.ts` — loads `public/models/classifier.json` (the exported RandomForest, see below) and implements tree-walking prediction with per-class vote averaging, exposing the same `predict`/`predictProba`-shaped API the Python `_classify_side_frame` relies on.
- `lib/frontalClassifier.ts` — port of `posture_monitor/inference/frontal_classifier.py` (`computeFrontalBaseline`, `classifyFrontal`).
- `lib/orientation.ts` — port of `OrientationState` and `_rawOrientationReading` from `realtime.py`.
- `lib/smoothing.ts` — port of `PostureMonitorState`.
- `lib/session.ts` — orchestration: the port of `run()`'s per-frame logic (guided calibration bookkeeping, orientation update, dispatch to side/front classification, smoothing/debounce), restructured as a plain function callable once per animation frame rather than a Python generator loop.
- `public/models/pose_landmarker_lite.task` — copied from the Python project's `assets/`.
- `public/models/classifier.json` — generated by the export script below.

**Model export:** `scripts/export_model.py`, added to the existing Python project (`posture-ml-model` branch, alongside `scripts/download_model.sh`), run once per retrain with output committed to `web/public/models/classifier.json`. It walks the trained `RandomForestClassifier`'s `estimators_`, and for each tree serializes `feature` (index into `NORMALIZED_FEATURE_ORDER`), `threshold`, `children_left`, `children_right`, and per-leaf class-vote arrays into a compact JSON array of trees, plus the `classes_` list and `scale_reference`/feature order metadata needed by `lib/classifier.ts`. This is the actual trained model re-expressed as data — no behavior change, no retraining, no approximation.

**Data flow (steady state, one `requestAnimationFrame` tick):** webcam frame → MediaPipe `PoseLandmarker.detectForVideo` → keypoints → `lib/features.ts` (both side and front feature sets computed every frame, same as Python) → ratio computed → during calibration: accumulate into the active phase's rows; after both calibrations: `lib/orientation.ts` decides side/front → dispatch to `lib/classifier.ts` or `lib/frontalClassifier.ts` → `lib/smoothing.ts` → draw skeleton + status overlay on the canvas.

## Error handling

- Camera permission denied or no camera available: `TryItTool` shows an inline message ("Camera access is needed to use this tool") with a retry button; no uncaught exception, no broken layout.
- Pose not detected in a frame: same "No pose detected" red status text as the Python app; the session loop continues waiting for the next frame rather than erroring.
- `pose_landmarker_lite.task` or `classifier.json` fails to fetch/parse: a dedicated "Couldn't load the posture detector — please refresh" state, shown instead of a silently frozen camera preview.
- Browser lacking `getUserMedia` or WASM support: a plain-text fallback message in place of the tool, rather than a crash.

## Testing

Vitest unit tests for everything in `lib/`, mirroring the existing Python test names/cases 1:1 wherever the ported logic is behaviorally identical (`lib/orientation.test.ts` mirrors `test_realtime.py`'s `OrientationState`/`_raw_orientation_reading` tests, `lib/smoothing.test.ts` mirrors `PostureMonitorState`'s tests, `lib/features.test.ts` mirrors `test_extraction.py`, `lib/frontalClassifier.test.ts` mirrors `test_frontal_classifier.py`, `lib/calibration.test.ts` mirrors `test_calibrate.py`). This doubles as a correctness check that the TypeScript port actually matches the Python behavior it's derived from. No tests depend on a real webcam or real MediaPipe inference (that part is exercised manually in a browser).

## Branching

The Python ML code currently lives on the `posture-ml-model` branch; `master` only has docs. Before starting this work, merge `posture-ml-model` into `master` (so `scripts/export_model.py` and the trained model it reads are available on the trunk), then branch the website work from `master`.

## Deployment

A Vercel project pointed at the `web/` subdirectory of the existing `posture-monitor` GitHub repo, auto-deploying on push. No environment variables or backend services required. `public/models/*` (the `.task` file and `classifier.json`) ship as static assets in the deployed app, so visitors never need Python, scikit-learn, or a retrain — they get the exact already-trained model.
