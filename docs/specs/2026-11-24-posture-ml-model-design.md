# Posture Monitoring ML Model — Design

Date: 2026-11-24

## Goal

Train a model on a labeled seating-posture image dataset, then use it to monitor a live
side-view webcam feed of a student and flag incorrect sitting posture in real time. A
30-second calibration window at the start of each session measures the student's own body
proportions so the model isn't thrown off by people of different body size/build than the
training subject.

This spec covers the **ML model and pipeline only**. The browser-based web app that will
eventually host this (webcam capture UI, alerts UI) is an explicit follow-on project, out of
scope here. The output contract of the inference pipeline (see "Real-time inference loop")
is designed so it can be wired into that web app later without rework.

Project context: class/hackathon demo. Favor a working, honestly-evaluated prototype over
production concerns (no accounts, no deployment infra, no persistence beyond local files).

## Dataset (as found)

Location: `posture-monitor/data/` (`train/`, `test/` subfolders), 4 classes:
`CorrectSitting`, `LeanBackward`, `LeanForward`, `LegTwisting`.

- Train: CorrectSitting 606, LeanBackward 432, LeanForward 619, LegTwisting 795 (2,452 total)
- Test: CorrectSitting 64, LeanBackward 57, LeanForward 49, LegTwisting 84 (254 total)
- 720×1280 JPEGs, single subject throughout, side-view camera for `CorrectSitting` /
  `LeanForward` / `LeanBackward`; `LegTwisting` is captured with the subject rotated more
  toward the camera, legs pulled onto the chair — not a pure sagittal-plane lean.

**Important finding — data is single-session, not single-example-diverse:** filenames
(`image_0.jpg` … `image_896.jpg`) and image content confirm each class folder is frames
extracted from one continuous video clip per posture (same clothes/background/lighting
throughout), not hundreds of independent sittings. Index ranges overlap between `train/` and
`test/` for the same class, and same-numbered files in `train/` vs `test/` are different but
visually near-identical frames a fraction of a second apart. **The provided train/test split
is a shuffle of one clip's frames, not a hold-out of a separate session** — accuracy measured
on it will be optimistic and does not demonstrate generalization to a new sitting session or
a new student.

## Known limitations (carried forward into methodology, not blocking this scope)

1. **Single subject, single session per class.** Real generalization to other students is
   unproven until tested on new recordings. The 30s calibration (see below) is the mechanism
   meant to help bridge this, but it only rescales body proportions — it does not fix
   appearance-based overfitting, which is why the model is built on geometric features, not
   raw pixels.
2. **`LegTwisting` is a different kind of signal** than the other three classes (leg/hip
   position and body rotation, not trunk lean) and needs its own feature cluster.
3. Recommend capturing at least one additional session (different day/clothing/lighting,
   ideally a second person) before trusting this model's real-world accuracy. Not required
   to complete this phase of the project, but should be called out in any demo/report.

## Architecture

```
Frame (recorded image or live webcam frame)
      │
      ▼
[1] Pose Estimation — MediaPipe Pose (pretrained, not trained on our data)
      │   → 2D keypoints: ear, eye/nose, shoulder, hip, knee, ankle + per-landmark visibility
      ▼
[2] Geometric Feature Extraction (single shared implementation used by training AND inference)
      ▼
[3] Proportion Normalization (distance features scaled by the subject's own
      │   calibrated body-segment lengths)
      ▼
[4] Classifier (scikit-learn, trained on our labeled dataset's extracted+normalized features)
      ▼
Predicted class + confidence (CorrectSitting / LeanForward / LeanBackward / LegTwisting /
PoseUncertain)
```

Rationale: MediaPipe's pose model already generalizes across body types (trained on a large
diverse dataset) and is used frozen/untrained. The only part trained on our small,
single-person dataset is a lightweight classifier over a handful of geometric numbers, not
raw pixels — this is the main defense against overfitting to this one subject's appearance,
clothing, and background.

## Feature set

| Feature | Captures | Primarily separates |
|---|---|---|
| Neck flexion angle (ear–shoulder vs. vertical) | Forward head posture | Correct vs LeanForward |
| Torso lean angle (shoulder–hip vs. vertical) | Trunk tilt | LeanForward vs LeanBackward vs Correct |
| Head-forward offset (horizontal ear–shoulder distance, normalized by torso length) | Head jutting forward | LeanForward |
| Hip–shoulder vertical compression ratio | Reclining/slouch-down | LeanBackward |
| Hip–knee–ankle angle | Leg position | LegTwisting |
| Leg-landmark visibility/occlusion score | Legs pulled up/crossed | LegTwisting |
| Shoulder/hip rotation asymmetry (left/right landmark depth cues) | Body turned out of profile | LegTwisting |
| Mean visibility score across key landmarks | Pose-tracking reliability | Triggers `PoseUncertain` instead of a forced guess |

`LegTwisting` uses its own feature cluster rather than reusing the trunk-angle features,
since it is a different failure mode (legs/rotation vs. sagittal lean), and MediaPipe
confidence commonly drops on this pose (self-occlusion, subject turned toward camera).

## Calibration (first 30 seconds of a live session)

- Run steps [1]+[2] every frame for 30s without classifying/flagging.
- For each distance-based feature, take the **median** raw (un-normalized) measurement
  across all 30s of frames. Median is robust to occasional bad frames/noisy keypoints.
- Skeletal segment lengths (e.g. shoulder-to-hip pixel distance, shoulder width) barely
  change whether the student is slouching or sitting upright, so **no assumption is made
  that the student is sitting correctly during calibration** — this is purely a body-scale
  reference, not a posture baseline.
- Angle-based features don't need distance scaling (already proportion-invariant) but are
  still sanity-checked during this window (e.g. reject frames with low visibility).
- Store the resulting medians as the session's reference scale; subsequent frames' distance
  features are divided by them before classification.
- Training data is normalized the same way for consistency: medians computed once across the
  whole combined training dataset (all 4 classes, since it's one subject), not from any
  single class assumed to be "correct."

## Training methodology

1. **De-duplication.** Compute each frame's raw geometric feature vector, then drop frames
   whose feature vector is near-identical to the preceding frame in the same class/clip
   (e.g. cosine distance below a threshold). This produces an honest count of *effectively
   distinct* poses per class — likely far fewer than the ~2,450 raw training frames — and
   prevents the classifier from simply memorizing a handful of poses repeated hundreds of
   times.
2. **Split.** After de-duplication, split each class's clip by contiguous time position
   (e.g. first ~80% of the timeline → train, last ~20% → held-out test) instead of random
   shuffling. This is still same-session validation (see limitations above) but at least
   avoids testing on frames a fraction of a second from training frames.
3. **Class imbalance.** Use `class_weight='balanced'` in scikit-learn models rather than
   resampling, given the moderate (432–795) imbalance.
4. **Model selection.** Train and compare Random Forest (primary candidate — handles mixed
   feature scales, gives feature importances for debugging) and SVM with RBF kernel
   (baseline comparison). Select by cross-validated per-class F1, not overall accuracy.
5. **Evaluation.** Report per-class precision/recall/F1 and a confusion matrix. Explicitly
   label results as session-level validation accuracy, with a note that cross-session/
   cross-subject accuracy is unverified pending additional recordings.

## Real-time inference loop

```
per frame (post-calibration):
  keypoints = pose_estimate(frame)
  if mean_visibility(keypoints) < threshold:
      current_prediction = "PoseUncertain"
  else:
      features = extract_features(keypoints)
      features = normalize(features, calibration_medians)
      current_prediction = classifier.predict(features)

  smoothed = majority_vote(last N seconds of current_prediction)   # ~2-3s window, de-flickers
  if smoothed != "CorrectSitting" persists for >= debounce_seconds (~5s):
      emit_flag(timestamp, smoothed_class, confidence)
```

The debounce avoids flagging brief transient movements (e.g. reaching for a pen). The output
of `emit_flag` is the integration point for the future web app (e.g. a Python
generator/callback or a local event stream) — not implemented in this phase, but the
interface is designed now to avoid rework later.

## Project structure

```
posture-monitor/
  data/                  # already in place (train/, test/)
  posture_monitor/
    features/            # shared geometric feature extraction (training + live use this)
    calibration/         # 30s proportion-normalization logic
    training/            # dedup, split, train, evaluate scripts
    inference/           # webcam loop, smoothing/debounce, flagging
    models/              # saved trained classifier
  tests/                 # unit tests: feature extraction, calibration, smoothing
  docs/specs/            # this design doc
```

## Testing plan

- Unit tests: feature-extraction determinism (same keypoints → same features), calibration
  median computation, majority-vote smoothing, and debounce logic.
- Model evaluation: cross-validated classification report (precision/recall/F1 per class) +
  confusion matrix on the de-duplicated, time-aware split.
- Qualitative/manual test: run the live pipeline (after calibration) on a team member other
  than the original dataset subject, to get an early read on cross-subject generalization
  even though no formal held-out second-subject dataset exists yet.

## Out of scope for this phase

- The browser web app (webcam capture UI, alert UI, any backend serving).
- Any deployment, accounts, or persistence beyond local model/files.
- Formal cross-subject accuracy guarantees (flagged as a limitation, not solved here).
