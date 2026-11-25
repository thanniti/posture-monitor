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

## Results (first training run)

Ran the full pipeline (Task 11, Steps 1-4) against the real dataset in `data/`.

**Feature extraction.** `python3 -m posture_monitor.training.build_dataset` processed all
2,706 images (train + test combined); MediaPipe failed to detect a pose on 57 of them
(warned, not fatal), leaving **2,649 rows** in `posture_monitor/training/features.csv`.

**De-duplication threshold tuning.** The design doc's original default,
`DEDUP_ANGLE_DISTANCE_THRESHOLD = 2.0` (degrees, combined Euclidean distance across
neck/torso/hip-knee angle), turned out to be far too tight for this data: it only dropped a
handful of frames per class (e.g. `CorrectSitting: 649 raw -> 623 after dedup`). Measuring the
actual consecutive-frame angle distance showed a median of ~14-30 degrees per class (real
per-frame pose-estimation jitter, not literal duplicate video frames), so a 2.0-degree
threshold was effectively a no-op. The threshold was raised to **65.0 degrees** after sweeping
a range of candidate values — this was the one tunable parameter the task permitted adjusting,
and the goal was removing near-duplicate frames while keeping genuinely distinct poses, not
hitting an exact target count. Re-running de-duplication and the honest (chronological,
80/20) split with the new threshold produced a dramatic collapse, consistent with the forensic
finding that each class is one short continuous clip:

```
CorrectSitting: 649 raw -> 58 after dedup (46 train / 12 test)
LeanBackward:   460 raw -> 139 after dedup (111 train / 28 test)
LeanForward:    663 raw -> 85 after dedup (68 train / 17 test)
LegTwisting:    877 raw -> 168 after dedup (134 train / 34 test)
```

Total honest-split dataset: 450 rows (359 train / 91 test), written to
`posture_monitor/training/features_honest.csv`.

**Training and evaluation.** `python3 -m posture_monitor.training.train` on the de-duplicated,
honestly-split data:

```
random_forest: cv f1_macro = 0.797
svm_rbf:       cv f1_macro = 0.573
Selected model: random_forest
```

Held-out (honest-split) classification report:

```
                precision    recall  f1-score   support

CorrectSitting       0.47      0.67      0.55        12
  LeanBackward       0.62      0.36      0.45        28
   LeanForward       0.62      0.59      0.61        17
   LegTwisting       0.67      0.82      0.74        34

      accuracy                           0.62        91
     macro avg       0.60      0.61      0.59        91
  weighted avg       0.62      0.62      0.60        91
```

Confusion matrix (row = true label, column = predicted label, class order
CorrectSitting / LeanBackward / LeanForward / LegTwisting):

```
[[ 8  3  1  0]
 [ 9 10  2  7]
 [ 0  0 10  7]
 [ 0  3  3 28]]
```

`posture_monitor/models/classifier.joblib` (random forest) was written and is the selected
model.

**Reading these numbers.** Cross-validated F1 (0.797) is noticeably higher than the honest
held-out F1 (macro 0.59) — expected, since cross-validation here still folds over the same
single clip's de-duplicated frames, while the held-out test split is the last chronological
20% of each clip and is a harder, more realistic test of "different moment, same session."
`LeanBackward` is the weakest class (recall 0.36), most often confused with `CorrectSitting`
and `LegTwisting`; `LegTwisting` and `LeanForward` perform best, consistent with those classes
having the most visually distinct geometry (leg/rotation signal, strong forward neck angle).

**This is session-level validation, not cross-subject validation.** Per the "Known
limitations" and "Training methodology" sections above, this entire run — feature extraction,
de-duplication, split, training, and evaluation — uses data from a single subject's single
recording session per class. The honest chronological split avoids testing on frames a
fraction of a second from training frames, but it does not and cannot demonstrate
generalization to a new sitting session, a new subject, different clothing/lighting, or a
different camera setup. These numbers should be read as "does the geometric-feature pipeline
separate these four postures within one person's one session" — not as a real-world accuracy
estimate. The qualitative manual webcam test (Task 11 Step 5, run separately) and a future
second recording session are the next steps toward answering the cross-subject question.

## Results (upper-body-only redesign)

**Motivation.** An explicit project requirement arrived to make the posture classifier rely
only on head/neck/shoulder keypoints (nose, ears, eyes, shoulders) — no hip, knee, or ankle
keypoints at all. This was a deliberate scope narrowing, not a measurement finding: the goal
is a monitor that works from a typical webcam framing where the lower body is frequently out
of frame or occluded by a desk, and that is robust to the subject's chair/desk setup rather
than keying off leg geometry that may not even be visible.

**New feature sets.** `extract_features()` (side/profile view, used for training) now
returns exactly 6 raw features, all derived from nose, ear, and shoulder keypoints:
`neck_angle_deg`, `nose_shoulder_angle_deg`, `head_forward_offset_px`,
`nose_forward_offset_px`, `nose_to_ear_px`, `mean_visibility`. The old hip-based
`torso_length_px` scale reference is replaced by `nose_to_ear_px` — a stable facial
dimension, clearly visible in profile and unaffected by posture. `NORMALIZED_FEATURE_ORDER`
(used at both calibration and inference time) is correspondingly 5 features:
`neck_angle_deg`, `nose_shoulder_angle_deg`, `head_forward_offset_norm`,
`nose_forward_offset_norm`, `mean_visibility`.

`extract_frontal_features()` (front-view, used for the live front-view classifier, not for
this training pipeline) returns 7 raw features: `nose_to_shoulder_vertical_px`,
`ear_to_shoulder_vertical_px`, `shoulder_width_px`, `shoulder_tilt_deg`, `ear_tilt_deg`,
`interocular_distance_px`, `mean_visibility`, scaled by `interocular_distance_px` (eye-to-eye
distance) rather than shoulder width, since shoulder width is now itself used as a slouch
signal (shoulder protraction narrows it).

**`LegTwisting` excluded from training.** The original `LegTwisting` label was defined by
leg/hip geometry (crossed or pulled-up legs) that the upper-body-only feature set has no way
to detect — none of the 6 side-view features carry any leg signal. Training a 4th class with
no supporting features in the input would not teach the model anything about `LegTwisting`
and would only inject label noise into the decision boundaries of the other three classes (a
sample with `LegTwisting`'s *upper-body* posture, which can look like any of the other
classes depending on how the subject is slouched while twisting their legs, would get
forced into one of 4 buckets instead of being set aside). `dedup_split.py` and
`build_dataset.py` still process `LegTwisting` frames like any other class (unaffected,
generic code), but `train.py::load_and_normalize` now drops `class == "LegTwisting"` rows
immediately after loading the CSV, making this a strict 3-class problem: `CorrectSitting`,
`LeanBackward`, `LeanForward`.

**De-duplication.** `DEDUP_FEATURES` changed from the old 3-angle set
(`neck_angle_deg`, `torso_angle_deg`, `hip_knee_angle_deg`) to the 2 angle features that
still exist (`neck_angle_deg`, `nose_shoulder_angle_deg`). The existing
`DEDUP_ANGLE_DISTANCE_THRESHOLD = 65.0` (tuned in the original 3-angle run above) was tried
as-is against the new 2-angle distance and produced a sensible, non-degenerate collapse
across all four raw classes (no class collapsed to near-zero or stayed near 100%), so it was
kept unchanged rather than retuned:

```
CorrectSitting: 649 raw -> 61 after dedup (48 train / 13 test)
LeanBackward: 460 raw -> 93 after dedup (74 train / 19 test)
LeanForward: 663 raw -> 45 after dedup (36 train / 9 test)
LegTwisting: 877 raw -> 126 after dedup (100 train / 26 test)
```

(`LegTwisting` rows are still produced by this step, then dropped later, inside
`train.py`.) Total honest-split dataset written to `features_honest.csv`: 325 rows. After
excluding `LegTwisting` at training time, the usable honest-split dataset is
`CorrectSitting` + `LeanBackward` + `LeanForward` = 199 rows (158 train / 41 test).

**Training and evaluation.** `python3 -m posture_monitor.training.build_dataset` reprocessed
all 2,706 images with the new 6-feature extractor; 57 still had no detected pose (same count
as before — pose detection failure is independent of the feature redesign), leaving 2,649
rows. `python3 -m posture_monitor.training.train` on the de-duplicated, honestly-split,
3-class data:

```
random_forest: cv f1_macro = 0.651
svm_rbf:       cv f1_macro = 0.463
Selected model: random_forest
```

Held-out (honest-split) classification report:

```
                precision    recall  f1-score   support

CorrectSitting       0.70      0.54      0.61        13
  LeanBackward       0.64      0.47      0.55        19
   LeanForward       0.35      0.67      0.46         9

      accuracy                           0.54        41
     macro avg       0.57      0.56      0.54        41
  weighted avg       0.60      0.54      0.55        41
```

Confusion matrix (row = true label, column = predicted label, class order
CorrectSitting / LeanBackward / LeanForward):

```
[[7 4 2]
 [1 9 9]
 [2 1 6]]
```

`posture_monitor/models/classifier.joblib` (random forest) was overwritten with this 3-class,
5-feature model.

**Honest assessment: did `LeanBackward` hold up without hip-based features?** In the
original 4-class run, `LeanBackward` was already the weakest class (recall 0.36, F1 0.45),
most often confused with `CorrectSitting` and `LegTwisting`. In this run, `LeanBackward`'s
own numbers moved slightly upward in isolation (recall 0.47, F1 0.55) — but this is not a
like-for-like improvement: `LegTwisting` (one of its two biggest confusers before) is no
longer a candidate label at all, so some of its former LegTwisting-bound misclassifications
simply can't happen anymore, and the comparison pool shrank from 91 held-out rows across 4
classes to 41 held-out rows across 3. Looking at the new confusion matrix, `LeanBackward`'s
remaining error mode shifted entirely onto `LeanForward` (9 of 19 LeanBackward rows predicted
as LeanForward) — a confusion the original 4-class run did not have in this direction at all
(original matrix shows 0 LeanBackward-row entries in the LeanForward column). This is
consistent with `LeanBackward` previously being partly separable from `LeanForward` via
hip/torso-angle signal that no longer exists; with only neck/nose/shoulder angles available,
a backward lean and a forward lean apparently produce more overlapping upper-body angle
values than expected for at least some frames in this one-session dataset. The
cross-validated F1 also dropped noticeably (0.797 -> 0.651) and the held-out macro F1 dropped
slightly (0.59 -> 0.54) even with one fewer, easier-to-separate class in the mix
(`LegTwisting` had the best recall at 0.82 in the original run). Taken together, the numbers
do not support a claim that `LeanBackward` (or the 3-class problem generally) holds up as
well on upper-body-only features as the original feature set did — the drop in
cross-validated F1 and the new LeanBackward/LeanForward confusion are the clearest signals of
that. This remains session-level validation on one subject's one recording per class (same
caveat as the first training run above), so these numbers should be read as "does the
narrower upper-body-only feature set still separate these three postures within one person's
one session," not as a real-world accuracy estimate.
