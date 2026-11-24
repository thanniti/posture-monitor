# Posture Monitoring ML Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and train a posture classifier (CorrectSitting / LeanForward / LeanBackward / LegTwisting) from the labeled side-view image dataset, and a real-time inference pipeline that calibrates to an individual student for 30s then flags sustained bad posture from a live frame source.

**Architecture:** MediaPipe Pose (pretrained, frozen) extracts body keypoints from each frame. A small shared feature-extraction module computes 8 interpretable geometric features (angles, occlusion/visibility, normalized distances) from those keypoints. A 30-second calibration window (or, for training, the whole training set) establishes a per-subject scale reference used to normalize distance-based features. A scikit-learn classifier (Random Forest, compared against SVM) trained on these normalized features predicts the posture class; a smoothing + debounce layer turns per-frame predictions into a stable real-time flag.

**Tech Stack:** Python 3.14 (arm64 macOS, confirmed working), MediaPipe 1.2.3 (Tasks API — `mediapipe.tasks.python.vision.PoseLandmarker`, NOT the old `mp.solutions.pose` API, which no longer exists in this version), OpenCV, scikit-learn, pandas, joblib, pytest.

---

## Environment facts established before writing this plan (do not re-derive)

- Confirmed locally: Python 3.14.3, arm64, only Python available via Homebrew on this machine.
- MediaPipe 1.2.3 ships a `py3-none` wheel (works on 3.14) but **only exposes the new Tasks API**. `import mediapipe; mediapipe.solutions` does not exist. You must use `mediapipe.tasks.python.vision.PoseLandmarker` with a downloaded `.task` model bundle.
- Model bundle used and already downloaded to `assets/pose_landmarker_lite.task` (5.7MB) from:
  `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task`
- Verified end-to-end on a real dataset image (`data/train/CorrectSitting/image_0.jpg`): pose detected, 33 landmarks returned with `.x`, `.y` (normalized 0-1) and `.visibility` fields. Landmark indices confirmed by inspection: `0`=nose, `7`/`8`=left/right ear, `11`/`12`=left/right shoulder, `23`/`24`=left/right hip, `25`/`26`=left/right knee, `27`/`28`=left/right ankle.
- Dataset location: `data/train/<ClassName>/image_<N>.jpg` and `data/test/<ClassName>/image_<N>.jpg`, classes: `CorrectSitting`, `LeanBackward`, `LeanForward`, `LegTwisting`. Confirmed in brainstorming phase: these are frames from one continuous clip per class (not independent sittings) — this plan's Task 8 implements the de-dup + honest re-split required to get a trustworthy accuracy number.
- A venv already exists at `.venv/` with all required packages installed (versions pinned in Task 1's `requirements.txt`).

---

## Design decisions locked in (read before implementing — avoids re-deriving math)

**Why calibration normalizes by a fixed median, not per-frame self-normalization:** distance features (e.g. head-forward offset) could in principle be divided by that *same frame's* torso length to get a scale-invariant ratio instantly, without any 30s calibration step. We deliberately do **not** do this: a slouched/reclined torso's projected 2D length is itself compressed by the posture we're trying to detect, so dividing by the current frame's own torso length would partially cancel out the very signal we want to measure (circular normalization). Instead we fix the scale once, from a median over many frames of varied motion (the calibration window, or the whole training set), and apply that fixed scale to every subsequent frame.

**Keypoint dict contract** (output of `features/landmarks.py`, input to `features/extraction.py`), pixel coordinates (not normalized 0-1 — multiplied by actual frame width/height so distances are comparable across different camera resolutions/aspect ratios):

```python
{
  "nose": (x_px, y_px, visibility),
  "left_ear": (x_px, y_px, visibility), "right_ear": (x_px, y_px, visibility),
  "left_shoulder": (...), "right_shoulder": (...),
  "left_hip": (...), "right_hip": (...),
  "left_knee": (...), "right_knee": (...),
  "left_ankle": (...), "right_ankle": (...),
}
```

**Raw feature set** (output of `features/extraction.py::extract_features`, pre-normalization):

| Key | Meaning | Normalization needed? |
|---|---|---|
| `neck_angle_deg` | angle of (shoulder→ear) vector from vertical | No (already scale-invariant) |
| `torso_angle_deg` | angle of (hip→shoulder) vector from vertical | No |
| `head_forward_offset_px` | horizontal distance, ear to shoulder (chosen side) | Yes — divide by scale reference |
| `hip_shoulder_vertical_px` | vertical distance, hip to shoulder (chosen side) | Yes |
| `hip_knee_angle_deg` | joint angle at knee, hip–knee–ankle (chosen side) | No |
| `leg_visibility` | mean visibility of chosen-side knee+ankle | No (already 0-1) |
| `shoulder_span_px` | euclidean distance, left_shoulder to right_shoulder | Yes |
| `mean_visibility` | mean visibility across ear/shoulder/hip/knee/ankle (chosen side) + both shoulders | No |
| `torso_length_px` | euclidean distance, hip to shoulder (chosen side) — **not a classifier feature itself, this IS the scale reference measurement** | n/a |

"Chosen side" = whichever of left/right has higher visibility for that landmark pair, picked independently per pair.

**Normalized feature vector fed to the classifier** (order matters, defined once in `calibration/calibrate.py::NORMALIZED_FEATURE_ORDER`, reused everywhere): `neck_angle_deg, torso_angle_deg, head_forward_offset_norm, hip_shoulder_vertical_norm, hip_knee_angle_deg, leg_visibility, shoulder_span_norm, mean_visibility`.

---

## File Structure

```
posture-monitor/
  assets/
    pose_landmarker_lite.task     # already downloaded, gitignored (binary, reproducible via script)
  scripts/
    download_model.sh             # Task 1
  posture_monitor/
    __init__.py
    config.py                     # Task 2
    features/
      __init__.py
      landmarks.py                # Task 3
      extraction.py                # Task 4
    calibration/
      __init__.py
      calibrate.py                 # Task 5
    inference/
      __init__.py
      smoothing.py                 # Task 6
      realtime.py                  # Task 10
    training/
      __init__.py
      build_dataset.py              # Task 7
      dedup_split.py                 # Task 8
      train.py                       # Task 9
    models/
      .gitkeep                      # trained classifier.joblib saved here, gitignored
  tests/
    __init__.py
    test_landmarks.py               # Task 3
    test_extraction.py              # Task 4
    test_calibrate.py               # Task 5
    test_smoothing.py               # Task 6
    test_build_dataset.py           # Task 7
    test_dedup_split.py             # Task 8
    test_train.py                   # Task 9
    test_realtime.py                # Task 10
  data/                              # already in place, gitignored
  requirements.txt                  # Task 1
  docs/specs/2026-11-24-posture-ml-model-design.md   # already committed
  docs/plans/2026-11-24-posture-ml-model-plan.md      # this file
```

---

### Task 1: Project scaffolding, dependencies, model asset

**Files:**
- Create: `requirements.txt`
- Create: `scripts/download_model.sh`
- Create: `posture_monitor/__init__.py`
- Create: `posture_monitor/features/__init__.py`
- Create: `posture_monitor/calibration/__init__.py`
- Create: `posture_monitor/inference/__init__.py`
- Create: `posture_monitor/training/__init__.py`
- Create: `posture_monitor/models/.gitkeep`
- Create: `tests/__init__.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write `requirements.txt`**

```
mediapipe==1.2.3
opencv-python==5.1.0.95
scikit-learn==1.9.1
numpy==2.5.5
pandas==3.1.1
joblib==1.6.0
pytest==9.2.0
```

- [ ] **Step 2: Write `scripts/download_model.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$(dirname "$0")/../assets"
curl -sL -o "$(dirname "$0")/../assets/pose_landmarker_lite.task" \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
echo "Downloaded assets/pose_landmarker_lite.task"
```

- [ ] **Step 3: Make it executable and create package directories**

Run:
```bash
chmod +x scripts/download_model.sh
mkdir -p posture_monitor/features posture_monitor/calibration posture_monitor/inference posture_monitor/training posture_monitor/models tests
touch posture_monitor/__init__.py posture_monitor/features/__init__.py posture_monitor/calibration/__init__.py posture_monitor/inference/__init__.py posture_monitor/training/__init__.py posture_monitor/models/.gitkeep tests/__init__.py
```

- [ ] **Step 4: Update `.gitignore`** (append, don't duplicate existing `data/` entry)

```
assets/
posture_monitor/models/*.joblib
.pytest_cache/
```

- [ ] **Step 5: Verify the existing venv has everything `requirements.txt` declares**

Run: `source .venv/bin/activate && pip install -r requirements.txt && python3 -c "import mediapipe, cv2, sklearn, pandas, joblib, pytest; print('all imports OK')"`
Expected: `all imports OK` (packages already installed during brainstorming, this just confirms versions match).

- [ ] **Step 6: Commit**

```bash
git add requirements.txt scripts/.gitignore posture_monitor tests .gitignore
git commit -m "Scaffold posture_monitor package structure and dependencies"
```

---

### Task 2: Config module

**Files:**
- Create: `posture_monitor/config.py`

- [ ] **Step 1: Write the config file (no test needed — pure constants)**

```python
POSE_MODEL_PATH = "assets/pose_landmarker_lite.task"

# Below this mean landmark visibility, report "PoseUncertain" instead of a class guess.
MIN_MEAN_VISIBILITY = 0.5

# Calibration window length at the start of a live session.
CALIBRATION_SECONDS = 30

# Rolling majority-vote window for smoothing per-frame predictions.
SMOOTHING_WINDOW_SECONDS = 2.5

# A non-CorrectSitting (or PoseUncertain) smoothed label must persist this long before flagging.
DEBOUNCE_SECONDS = 5.0

# Euclidean distance (in degrees, across neck/torso/knee angles) below which two
# consecutive frames in the same class are treated as near-duplicates and the later
# one is dropped during training-set de-duplication.
DEDUP_ANGLE_DISTANCE_THRESHOLD = 2.0

# Fraction of each class's chronologically-sorted frames used for the honest
# re-split's training portion (remainder becomes the held-out test portion).
HONEST_SPLIT_TRAIN_FRACTION = 0.8
```

- [ ] **Step 2: Verify it imports**

Run: `source .venv/bin/activate && python3 -c "from posture_monitor import config; print(config.CALIBRATION_SECONDS)"`
Expected: `30`

- [ ] **Step 3: Commit**

```bash
git add posture_monitor/config.py
git commit -m "Add posture_monitor config constants"
```

---

### Task 3: Landmark extraction wrapper

**Files:**
- Create: `posture_monitor/features/landmarks.py`
- Test: `tests/test_landmarks.py`

- [ ] **Step 1: Write the failing test (real image, real model, no mocks)**

```python
# tests/test_landmarks.py
from posture_monitor.config import POSE_MODEL_PATH
from posture_monitor.features.landmarks import create_landmarker, extract_keypoints_from_file


def test_extract_keypoints_from_real_sample_image():
    landmarker = create_landmarker(POSE_MODEL_PATH)
    keypoints = extract_keypoints_from_file(
        landmarker, "data/train/CorrectSitting/image_0.jpg"
    )

    assert keypoints is not None
    expected_names = {
        "nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
        "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    }
    assert set(keypoints.keys()) == expected_names

    # Sanity check using known pose in this specific sample image (profile view,
    # sitting upright-ish, writing at a desk): ear should be above (smaller y,
    # since image y increases downward) the shoulder, and visibility should be high
    # for the clearly-visible side of the body.
    ear_y = keypoints["left_ear"][1]
    shoulder_y = keypoints["left_shoulder"][1]
    assert ear_y < shoulder_y

    shoulder_visibility = max(keypoints["left_shoulder"][2], keypoints["right_shoulder"][2])
    assert shoulder_visibility > 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_landmarks.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'posture_monitor.features.landmarks'`

- [ ] **Step 3: Write the implementation**

```python
# posture_monitor/features/landmarks.py
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

LANDMARK_INDICES = {
    "nose": 0,
    "left_ear": 7, "right_ear": 8,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}


def create_landmarker(model_path):
    base_options = BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.PoseLandmarker.create_from_options(options)


def extract_keypoints(landmarker, mp_image):
    """mp_image: a mediapipe.Image. Returns a dict of named keypoints in pixel
    coordinates (x_px, y_px, visibility), or None if no pose was detected."""
    width = mp_image.width
    height = mp_image.height
    result = landmarker.detect(mp_image)
    if not result.pose_landmarks:
        return None
    landmarks = result.pose_landmarks[0]
    return {
        name: (landmarks[idx].x * width, landmarks[idx].y * height, landmarks[idx].visibility)
        for name, idx in LANDMARK_INDICES.items()
    }


def extract_keypoints_from_file(landmarker, image_path):
    mp_image = mp.Image.create_from_file(image_path)
    return extract_keypoints(landmarker, mp_image)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_landmarks.py -v`
Expected: PASS (MediaPipe will print `INFO`/`WARNING` log lines to stderr — that's expected noise, not a failure)

- [ ] **Step 5: Commit**

```bash
git add posture_monitor/features/landmarks.py tests/test_landmarks.py
git commit -m "Add MediaPipe pose landmark extraction wrapper"
```

---

### Task 4: Geometric feature extraction

**Files:**
- Create: `posture_monitor/features/extraction.py`
- Test: `tests/test_extraction.py`

- [ ] **Step 1: Write the failing tests (synthetic keypoints, no camera/model needed)**

```python
# tests/test_extraction.py
import math
import pytest

from posture_monitor.features.extraction import extract_features


def make_keypoints(**overrides):
    """A simple upright reference pose: hip at origin, shoulder directly above,
    ear directly above shoulder, knee/ankle directly below hip (straight leg).
    Individual coordinates can be overridden per test."""
    base = {
        "nose": (0, -220, 1.0),
        "left_ear": (0, -200, 1.0), "right_ear": (0, -200, 0.3),
        "left_shoulder": (0, -100, 1.0), "right_shoulder": (0, -100, 1.0),
        "left_hip": (0, 0, 1.0), "right_hip": (0, 0, 0.9),
        "left_knee": (0, 100, 1.0), "right_knee": (0, 100, 0.9),
        "left_ankle": (0, 200, 1.0), "right_ankle": (0, 200, 0.9),
    }
    base.update(overrides)
    return base


def test_upright_pose_has_near_zero_angles():
    features = extract_features(make_keypoints())
    assert features["neck_angle_deg"] == pytest.approx(0.0, abs=0.01)
    assert features["torso_angle_deg"] == pytest.approx(0.0, abs=0.01)
    assert features["hip_knee_angle_deg"] == pytest.approx(180.0, abs=0.01)


def test_forward_lean_produces_positive_torso_angle():
    leaned = make_keypoints(left_shoulder=(50, -100, 1.0), right_shoulder=(50, -100, 1.0))
    features = extract_features(leaned)
    assert features["torso_angle_deg"] > 10


def test_bent_knee_reduces_hip_knee_angle():
    bent = make_keypoints(left_knee=(80, 60, 1.0), right_knee=(80, 60, 0.9))
    features = extract_features(bent)
    assert features["hip_knee_angle_deg"] < 150


def test_chooses_higher_visibility_side_for_ear():
    # right_ear has lower visibility (0.3) than left_ear (1.0) in the base pose,
    # so left_ear's offset should be used for head_forward_offset_px.
    features_base = extract_features(make_keypoints())
    shifted = make_keypoints(left_ear=(40, -200, 1.0))
    features_shifted = extract_features(shifted)
    assert features_shifted["head_forward_offset_px"] != features_base["head_forward_offset_px"]


def test_shoulder_span_and_torso_length_are_positive():
    features = extract_features(make_keypoints())
    assert features["shoulder_span_px"] >= 0
    assert features["torso_length_px"] == pytest.approx(100.0, abs=0.01)


def test_mean_visibility_in_zero_one_range():
    features = extract_features(make_keypoints())
    assert 0.0 <= features["mean_visibility"] <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_extraction.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'posture_monitor.features.extraction'`

- [ ] **Step 3: Write the implementation**

```python
# posture_monitor/features/extraction.py
import math


def _euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _angle_from_vertical(bottom_point, top_point):
    """Degrees between the vector bottom->top and straight-up vertical.
    0 = perfectly upright. Positive/negative = leaning one way or the other."""
    dx = top_point[0] - bottom_point[0]
    dy = top_point[1] - bottom_point[1]
    return math.degrees(math.atan2(dx, -dy))


def _joint_angle(vertex, point_a, point_b):
    """Degrees at `vertex` between rays to point_a and point_b. 180 = straight line."""
    v1 = (point_a[0] - vertex[0], point_a[1] - vertex[1])
    v2 = (point_b[0] - vertex[0], point_b[1] - vertex[1])
    mag1, mag2 = math.hypot(*v1), math.hypot(*v2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (mag1 * mag2)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def _pick_side(keypoints, part_name):
    left = keypoints[f"left_{part_name}"]
    right = keypoints[f"right_{part_name}"]
    return left if left[2] >= right[2] else right


def extract_features(keypoints):
    ear = _pick_side(keypoints, "ear")
    shoulder = _pick_side(keypoints, "shoulder")
    hip = _pick_side(keypoints, "hip")
    knee = _pick_side(keypoints, "knee")
    ankle = _pick_side(keypoints, "ankle")
    left_shoulder = keypoints["left_shoulder"]
    right_shoulder = keypoints["right_shoulder"]

    visibilities = [ear[2], shoulder[2], hip[2], knee[2], ankle[2],
                     left_shoulder[2], right_shoulder[2]]

    return {
        "neck_angle_deg": _angle_from_vertical(shoulder, ear),
        "torso_angle_deg": _angle_from_vertical(hip, shoulder),
        "head_forward_offset_px": abs(ear[0] - shoulder[0]),
        "hip_shoulder_vertical_px": abs(hip[1] - shoulder[1]),
        "hip_knee_angle_deg": _joint_angle(knee, hip, ankle),
        "leg_visibility": (knee[2] + ankle[2]) / 2,
        "shoulder_span_px": _euclidean(left_shoulder, right_shoulder),
        "mean_visibility": sum(visibilities) / len(visibilities),
        "torso_length_px": _euclidean(hip, shoulder),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_extraction.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add posture_monitor/features/extraction.py tests/test_extraction.py
git commit -m "Add geometric feature extraction from pose keypoints"
```

---

### Task 5: Calibration & shared normalization

**Files:**
- Create: `posture_monitor/calibration/calibrate.py`
- Test: `tests/test_calibrate.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_calibrate.py
import pytest

from posture_monitor.calibration.calibrate import (
    NORMALIZED_FEATURE_ORDER,
    compute_scale_reference,
    feature_vector,
    normalize_features,
)


def make_raw(torso_length_px=100.0, **overrides):
    base = {
        "neck_angle_deg": 5.0,
        "torso_angle_deg": 10.0,
        "head_forward_offset_px": 20.0,
        "hip_shoulder_vertical_px": 90.0,
        "hip_knee_angle_deg": 170.0,
        "leg_visibility": 0.95,
        "shoulder_span_px": 40.0,
        "mean_visibility": 0.97,
        "torso_length_px": torso_length_px,
    }
    base.update(overrides)
    return base


def test_scale_reference_is_median_of_torso_length():
    rows = [make_raw(torso_length_px=v) for v in [90.0, 100.0, 110.0, 1000.0]]
    # median of [90, 100, 110, 1000] is (100+110)/2 = 105
    assert compute_scale_reference(rows) == pytest.approx(105.0)


def test_scale_reference_ignores_zero_length_rows():
    rows = [make_raw(torso_length_px=0.0), make_raw(torso_length_px=100.0)]
    assert compute_scale_reference(rows) == pytest.approx(100.0)


def test_normalize_features_divides_distance_features_by_scale_reference():
    raw = make_raw(head_forward_offset_px=20.0, hip_shoulder_vertical_px=90.0, shoulder_span_px=40.0)
    normalized = normalize_features(raw, scale_reference=100.0)

    assert normalized["head_forward_offset_norm"] == pytest.approx(0.2)
    assert normalized["hip_shoulder_vertical_norm"] == pytest.approx(0.9)
    assert normalized["shoulder_span_norm"] == pytest.approx(0.4)


def test_normalize_features_leaves_angle_and_visibility_features_unchanged():
    raw = make_raw()
    normalized = normalize_features(raw, scale_reference=100.0)

    assert normalized["neck_angle_deg"] == raw["neck_angle_deg"]
    assert normalized["torso_angle_deg"] == raw["torso_angle_deg"]
    assert normalized["hip_knee_angle_deg"] == raw["hip_knee_angle_deg"]
    assert normalized["leg_visibility"] == raw["leg_visibility"]
    assert normalized["mean_visibility"] == raw["mean_visibility"]


def test_feature_vector_follows_declared_order():
    raw = make_raw()
    normalized = normalize_features(raw, scale_reference=100.0)
    vector = feature_vector(normalized)

    assert len(vector) == len(NORMALIZED_FEATURE_ORDER)
    assert vector == [normalized[name] for name in NORMALIZED_FEATURE_ORDER]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_calibrate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'posture_monitor.calibration.calibrate'`

- [ ] **Step 3: Write the implementation**

```python
# posture_monitor/calibration/calibrate.py
import statistics

NORMALIZED_FEATURE_ORDER = [
    "neck_angle_deg",
    "torso_angle_deg",
    "head_forward_offset_norm",
    "hip_shoulder_vertical_norm",
    "hip_knee_angle_deg",
    "leg_visibility",
    "shoulder_span_norm",
    "mean_visibility",
]


def compute_scale_reference(raw_feature_rows):
    """Median torso_length_px across rows. Used both for the live 30s calibration
    window and for the training-set-wide reference (see docs/specs design doc for
    why we use a fixed median rather than per-frame self-normalization)."""
    lengths = [row["torso_length_px"] for row in raw_feature_rows if row["torso_length_px"] > 0]
    return statistics.median(lengths)


def normalize_features(raw, scale_reference):
    return {
        "neck_angle_deg": raw["neck_angle_deg"],
        "torso_angle_deg": raw["torso_angle_deg"],
        "head_forward_offset_norm": raw["head_forward_offset_px"] / scale_reference,
        "hip_shoulder_vertical_norm": raw["hip_shoulder_vertical_px"] / scale_reference,
        "hip_knee_angle_deg": raw["hip_knee_angle_deg"],
        "leg_visibility": raw["leg_visibility"],
        "shoulder_span_norm": raw["shoulder_span_px"] / scale_reference,
        "mean_visibility": raw["mean_visibility"],
    }


def feature_vector(normalized):
    return [normalized[name] for name in NORMALIZED_FEATURE_ORDER]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_calibrate.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add posture_monitor/calibration/calibrate.py tests/test_calibrate.py
git commit -m "Add calibration scale reference and shared feature normalization"
```

---

### Task 6: Smoothing and debounce state machine

**Files:**
- Create: `posture_monitor/inference/smoothing.py`
- Test: `tests/test_smoothing.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_smoothing.py
from posture_monitor.inference.smoothing import PostureMonitorState


def test_majority_vote_within_window():
    state = PostureMonitorState(window_seconds=3.0, debounce_seconds=100.0)
    labels = ["CorrectSitting", "LeanForward", "LeanForward"]
    smoothed = None
    for i, label in enumerate(labels):
        smoothed, _ = state.update(timestamp=float(i), label=label)
    assert smoothed == "LeanForward"


def test_old_entries_fall_out_of_window():
    state = PostureMonitorState(window_seconds=2.0, debounce_seconds=100.0)
    state.update(timestamp=0.0, label="LeanForward")
    state.update(timestamp=1.0, label="LeanForward")
    # Jump far ahead: both earlier entries should be evicted, only new one counts.
    smoothed, _ = state.update(timestamp=10.0, label="CorrectSitting")
    assert smoothed == "CorrectSitting"


def test_no_flag_before_debounce_elapses():
    state = PostureMonitorState(window_seconds=1.0, debounce_seconds=5.0)
    _, flagged_early = state.update(timestamp=0.0, label="LeanForward")
    _, flagged_mid = state.update(timestamp=2.0, label="LeanForward")
    assert flagged_early is False
    assert flagged_mid is False


def test_flags_after_debounce_elapses():
    state = PostureMonitorState(window_seconds=1.0, debounce_seconds=5.0)
    state.update(timestamp=0.0, label="LeanForward")
    _, flagged = state.update(timestamp=6.0, label="LeanForward")
    assert flagged is True


def test_returning_to_correct_sitting_resets_debounce_timer():
    state = PostureMonitorState(window_seconds=1.0, debounce_seconds=5.0)
    state.update(timestamp=0.0, label="LeanForward")
    state.update(timestamp=6.0, label="CorrectSitting")
    _, flagged = state.update(timestamp=7.0, label="LeanForward")
    assert flagged is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_smoothing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'posture_monitor.inference.smoothing'`

- [ ] **Step 3: Write the implementation**

```python
# posture_monitor/inference/smoothing.py
from collections import deque


class PostureMonitorState:
    """Turns a stream of per-frame (timestamp, label) predictions into a smoothed
    label plus a debounced bad-posture flag."""

    def __init__(self, window_seconds, debounce_seconds):
        self.window_seconds = window_seconds
        self.debounce_seconds = debounce_seconds
        self.history = deque()  # (timestamp, label)
        self.bad_posture_since = None

    def update(self, timestamp, label):
        self.history.append((timestamp, label))
        cutoff = timestamp - self.window_seconds
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

        labels_in_window = [l for _, l in self.history]
        smoothed = max(set(labels_in_window), key=labels_in_window.count)

        if smoothed == "CorrectSitting":
            self.bad_posture_since = None
            return smoothed, False

        if self.bad_posture_since is None:
            self.bad_posture_since = timestamp

        flagged = (timestamp - self.bad_posture_since) >= self.debounce_seconds
        return smoothed, flagged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_smoothing.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add posture_monitor/inference/smoothing.py tests/test_smoothing.py
git commit -m "Add majority-vote smoothing and debounce logic for posture flagging"
```

---

### Task 7: Build feature dataset from labeled images

**Files:**
- Create: `posture_monitor/training/build_dataset.py`
- Test: `tests/test_build_dataset.py`

- [ ] **Step 1: Write the failing test (real images, real model — uses a tiny temp copy of a few real files so the test stays fast)**

```python
# tests/test_build_dataset.py
import shutil
from pathlib import Path

from posture_monitor.training.build_dataset import CLASSES, build_feature_table


def test_build_feature_table_on_small_sample(tmp_path):
    data_dir = tmp_path / "data"
    for split in ["train", "test"]:
        for cls in CLASSES:
            dest = data_dir / split / cls
            dest.mkdir(parents=True)
            src_dir = Path("data") / split / cls
            for src_file in sorted(src_dir.glob("image_*.jpg"))[:2]:
                shutil.copy(src_file, dest / src_file.name)

    output_csv = tmp_path / "features.csv"
    rows = build_feature_table(str(data_dir), str(output_csv))

    assert output_csv.exists()
    assert len(rows) == 2 * 2 * len(CLASSES)  # 2 splits * 2 files * N classes
    first_row = rows[0]
    for key in ("split", "class", "filename", "frame_index", "torso_length_px", "neck_angle_deg"):
        assert key in first_row
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_build_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'posture_monitor.training.build_dataset'`

- [ ] **Step 3: Write the implementation**

```python
# posture_monitor/training/build_dataset.py
import csv
import re
from pathlib import Path

from posture_monitor.config import POSE_MODEL_PATH
from posture_monitor.features.extraction import extract_features
from posture_monitor.features.landmarks import create_landmarker, extract_keypoints_from_file

CLASSES = ["CorrectSitting", "LeanBackward", "LeanForward", "LegTwisting"]
FRAME_INDEX_RE = re.compile(r"image_(\d+)\.jpg")


def build_feature_table(data_dir, output_csv):
    landmarker = create_landmarker(POSE_MODEL_PATH)
    rows = []

    for split in ["train", "test"]:
        for cls in CLASSES:
            folder = Path(data_dir) / split / cls
            if not folder.exists():
                continue
            for img_path in sorted(folder.glob("image_*.jpg")):
                match = FRAME_INDEX_RE.match(img_path.name)
                if not match:
                    continue
                keypoints = extract_keypoints_from_file(landmarker, str(img_path))
                if keypoints is None:
                    continue
                features = extract_features(keypoints)
                rows.append({
                    "split": split,
                    "class": cls,
                    "filename": img_path.name,
                    "frame_index": int(match.group(1)),
                    **features,
                })

    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return rows


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "posture_monitor/training/features.csv"
    rows = build_feature_table(data_dir, output_csv)
    print(f"Wrote {len(rows)} rows to {output_csv}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_build_dataset.py -v`
Expected: PASS (1 test; will take a few seconds since it runs real pose estimation on 16 images)

- [ ] **Step 5: Commit**

```bash
git add posture_monitor/training/build_dataset.py tests/test_build_dataset.py
git commit -m "Add feature-table builder that runs pose estimation over labeled images"
```

---

### Task 8: De-duplication and honest time-aware split

**Files:**
- Create: `posture_monitor/training/dedup_split.py`
- Test: `tests/test_dedup_split.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dedup_split.py
from posture_monitor.training.dedup_split import deduplicate, honest_split, process_class_rows


def make_row(frame_index, neck=0.0, torso=0.0, knee=180.0, **overrides):
    row = {
        "frame_index": frame_index,
        "neck_angle_deg": neck,
        "torso_angle_deg": torso,
        "hip_knee_angle_deg": knee,
    }
    row.update(overrides)
    return row


def test_deduplicate_drops_near_identical_consecutive_frames():
    rows = [
        make_row(0, neck=0.0),
        make_row(1, neck=0.1),   # near-duplicate of previous, should be dropped
        make_row(2, neck=0.2),   # near-duplicate of kept[-1]=row0 (distance 0.2), still dropped at threshold 2.0
        make_row(3, neck=10.0),  # clearly different, kept
    ]
    kept = deduplicate(rows, threshold=2.0)
    assert [r["frame_index"] for r in kept] == [0, 3]


def test_honest_split_is_chronological_not_random():
    rows = [make_row(i) for i in range(10)]
    train_rows, test_rows = honest_split(rows, train_fraction=0.8)
    assert [r["frame_index"] for r in train_rows] == list(range(8))
    assert [r["frame_index"] for r in test_rows] == [8, 9]


def test_process_class_rows_pools_then_dedups_then_splits():
    # Simulates original train/test folders both drawn from one clip, frame indices
    # interleaved across the two folders (the leakage pattern found in the real data).
    original_train = [make_row(0, neck=0.0), make_row(2, neck=0.2), make_row(4, neck=20.0)]
    original_test = [make_row(1, neck=0.1), make_row(3, neck=0.3)]

    train_rows, test_rows = process_class_rows(
        original_train + original_test, dedup_threshold=2.0, train_fraction=0.5
    )

    all_kept_indices = [r["frame_index"] for r in train_rows + test_rows]
    assert all_kept_indices == sorted(all_kept_indices)  # chronological order preserved
    assert 4 in all_kept_indices  # the clearly-different frame must survive dedup
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_dedup_split.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'posture_monitor.training.dedup_split'`

- [ ] **Step 3: Write the implementation**

```python
# posture_monitor/training/dedup_split.py
import math

DEDUP_FEATURES = ["neck_angle_deg", "torso_angle_deg", "hip_knee_angle_deg"]


def _angle_distance(row_a, row_b):
    return math.sqrt(sum((row_a[f] - row_b[f]) ** 2 for f in DEDUP_FEATURES))


def deduplicate(rows_sorted_by_frame_index, threshold):
    """Drops frames whose angle features are within `threshold` (degrees, combined
    Euclidean distance) of the most recently KEPT frame. Input must already be
    sorted by frame_index (chronological order)."""
    kept = []
    for row in rows_sorted_by_frame_index:
        if not kept or _angle_distance(kept[-1], row) >= threshold:
            kept.append(row)
    return kept


def honest_split(rows_sorted_by_frame_index, train_fraction):
    cutoff = int(len(rows_sorted_by_frame_index) * train_fraction)
    return rows_sorted_by_frame_index[:cutoff], rows_sorted_by_frame_index[cutoff:]


def process_class_rows(all_rows_for_class, dedup_threshold, train_fraction):
    """Pools rows regardless of their original train/test folder (the original split
    is a leaky shuffle of one clip, see design doc), re-sorts chronologically,
    de-duplicates, then re-splits by time position."""
    pooled = sorted(all_rows_for_class, key=lambda r: r["frame_index"])
    deduped = deduplicate(pooled, dedup_threshold)
    return honest_split(deduped, train_fraction)


if __name__ == "__main__":
    import sys
    from collections import defaultdict

    import pandas as pd

    from posture_monitor.config import DEDUP_ANGLE_DISTANCE_THRESHOLD, HONEST_SPLIT_TRAIN_FRACTION

    input_csv = sys.argv[1] if len(sys.argv) > 1 else "posture_monitor/training/features.csv"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "posture_monitor/training/features_honest.csv"

    df = pd.read_csv(input_csv)
    by_class = defaultdict(list)
    for row in df.to_dict("records"):
        by_class[row["class"]].append(row)

    all_train, all_test = [], []
    for cls, rows in by_class.items():
        train_rows, test_rows = process_class_rows(
            rows, DEDUP_ANGLE_DISTANCE_THRESHOLD, HONEST_SPLIT_TRAIN_FRACTION
        )
        print(f"{cls}: {len(rows)} raw -> {len(train_rows) + len(test_rows)} after dedup "
              f"({len(train_rows)} train / {len(test_rows)} test)")
        all_train.extend(train_rows)
        all_test.extend(test_rows)

    for row in all_train:
        row["honest_split"] = "train"
    for row in all_test:
        row["honest_split"] = "test"

    pd.DataFrame(all_train + all_test).to_csv(output_csv, index=False)
    print(f"Wrote {len(all_train) + len(all_test)} rows to {output_csv}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_dedup_split.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add posture_monitor/training/dedup_split.py tests/test_dedup_split.py
git commit -m "Add de-duplication and honest time-aware train/test split"
```

---

### Task 9: Train and evaluate the classifier

**Files:**
- Create: `posture_monitor/training/train.py`
- Test: `tests/test_train.py`

- [ ] **Step 1: Write the failing test (synthetic, separable data — keeps the test fast; the real dataset run happens in Task 11)**

```python
# tests/test_train.py
import random

import joblib
import pandas as pd

from posture_monitor.training.train import train_and_evaluate


def make_synthetic_csv(path):
    random.seed(0)
    rows = []
    # Two easily-separable "classes" driven by neck_angle_deg, repeated across an
    # honest train/test split, with enough rows for 5-fold CV on the train portion.
    for honest_split, count in [("train", 40), ("test", 10)]:
        for i in range(count):
            cls = "CorrectSitting" if i % 2 == 0 else "LeanForward"
            neck = random.uniform(-2, 2) if cls == "CorrectSitting" else random.uniform(28, 32)
            rows.append({
                "class": cls,
                "honest_split": honest_split,
                "neck_angle_deg": neck,
                "torso_angle_deg": random.uniform(-2, 2),
                "head_forward_offset_px": random.uniform(5, 10),
                "hip_shoulder_vertical_px": random.uniform(85, 95),
                "hip_knee_angle_deg": random.uniform(170, 180),
                "leg_visibility": random.uniform(0.9, 1.0),
                "shoulder_span_px": random.uniform(35, 45),
                "mean_visibility": random.uniform(0.9, 1.0),
                "torso_length_px": random.uniform(95, 105),
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def test_train_and_evaluate_produces_a_usable_model(tmp_path):
    csv_path = tmp_path / "synthetic_features.csv"
    model_path = tmp_path / "classifier.joblib"
    make_synthetic_csv(csv_path)

    model = train_and_evaluate(str(csv_path), str(model_path))

    assert model_path.exists()
    bundle = joblib.load(model_path)
    assert "model" in bundle and "scale_reference" in bundle and "feature_order" in bundle
    # The two classes are trivially separable by neck_angle_deg, so accuracy should be high.
    predictions = bundle["model"].predict([[0.0, 0.0, 0.08, 0.9, 175.0, 0.95, 0.4, 0.95]])
    assert predictions[0] == "CorrectSitting"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'posture_monitor.training.train'`

- [ ] **Step 3: Write the implementation**

```python
# posture_monitor/training/train.py
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from posture_monitor.calibration.calibrate import NORMALIZED_FEATURE_ORDER, feature_vector, normalize_features


def _to_matrix(sub_df, scale_reference):
    X = [feature_vector(normalize_features(row, scale_reference)) for row in sub_df.to_dict("records")]
    y = sub_df["class"].tolist()
    return X, y


def load_and_normalize(csv_path):
    df = pd.read_csv(csv_path)
    train_df = df[df.honest_split == "train"]
    test_df = df[df.honest_split == "test"]
    scale_reference = train_df["torso_length_px"].median()

    X_train, y_train = _to_matrix(train_df, scale_reference)
    X_test, y_test = _to_matrix(test_df, scale_reference)
    return X_train, y_train, X_test, y_test, scale_reference


def train_and_evaluate(csv_path, model_out_path):
    X_train, y_train, X_test, y_test, scale_reference = load_and_normalize(csv_path)

    candidates = {
        "random_forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
        "svm_rbf": SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42),
    }

    best_name, best_model, best_score = None, None, -1.0
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro")
        mean_score = scores.mean()
        print(f"{name}: cv f1_macro = {mean_score:.3f}")
        if mean_score > best_score:
            best_name, best_model, best_score = name, model, mean_score

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    print(f"Selected model: {best_name}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred, labels=sorted(set(y_train))))

    joblib.dump(
        {"model": best_model, "scale_reference": scale_reference, "feature_order": NORMALIZED_FEATURE_ORDER},
        model_out_path,
    )
    return best_model


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "posture_monitor/training/features_honest.csv"
    model_out_path = sys.argv[2] if len(sys.argv) > 2 else "posture_monitor/models/classifier.joblib"
    train_and_evaluate(csv_path, model_out_path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_train.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add posture_monitor/training/train.py tests/test_train.py
git commit -m "Add classifier training, model selection, and evaluation"
```

---

### Task 10: Real-time inference loop

**Files:**
- Create: `posture_monitor/inference/realtime.py`
- Test: `tests/test_realtime.py`

- [ ] **Step 1: Write the failing test (real images from the dataset fed through a fake frame source with synthetic timestamps, real model trained on synthetic data from Task 9's test helper — no real camera, no wall-clock waiting)**

```python
# tests/test_realtime.py
import cv2
import joblib
from sklearn.ensemble import RandomForestClassifier

from posture_monitor.inference.realtime import run


def _train_tiny_always_correct_model(tmp_path):
    # A model that always predicts CorrectSitting, so this test can assert the
    # pipeline runs end-to-end (calibration -> classification -> smoothing ->
    # debounce) without needing a real trained classifier.
    model = RandomForestClassifier(n_estimators=5, random_state=0)
    X = [[0, 0, 0.1, 0.9, 175, 0.95, 0.4, 0.95]] * 4
    y = ["CorrectSitting"] * 4
    model.fit(X, y)
    model_path = tmp_path / "model.joblib"
    joblib.dump({"model": model, "scale_reference": 100.0, "feature_order": None}, model_path)
    return str(model_path)


def _fake_frame_source(num_frames, start_timestamp=0.0, step_seconds=1.0):
    frame = cv2.imread("data/train/CorrectSitting/image_0.jpg")
    for i in range(num_frames):
        yield start_timestamp + i * step_seconds, frame


def test_run_completes_calibration_and_emits_no_flags_for_correct_sitting(tmp_path):
    model_path = _train_tiny_always_correct_model(tmp_path)
    flags = []

    # 35 frames at 1s/frame covers the 30s calibration window plus 5s of live inference.
    run(_fake_frame_source(35), model_path, on_flag=lambda ts, label: flags.append((ts, label)))

    assert flags == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_realtime.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'posture_monitor.inference.realtime'`

- [ ] **Step 3: Write the implementation**

```python
# posture_monitor/inference/realtime.py
import time

import cv2
import joblib
import mediapipe as mp

from posture_monitor.calibration.calibrate import compute_scale_reference, feature_vector, normalize_features
from posture_monitor.config import (
    CALIBRATION_SECONDS,
    DEBOUNCE_SECONDS,
    MIN_MEAN_VISIBILITY,
    POSE_MODEL_PATH,
    SMOOTHING_WINDOW_SECONDS,
)
from posture_monitor.features.extraction import extract_features
from posture_monitor.features.landmarks import create_landmarker, extract_keypoints
from posture_monitor.inference.smoothing import PostureMonitorState


def run(frame_source, model_bundle_path, on_flag=print):
    bundle = joblib.load(model_bundle_path)
    model = bundle["model"]
    landmarker = create_landmarker(POSE_MODEL_PATH)
    state = PostureMonitorState(SMOOTHING_WINDOW_SECONDS, DEBOUNCE_SECONDS)

    calibration_rows = []
    calibration_start = None
    scale_reference = None

    for timestamp, frame_bgr in frame_source:
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        keypoints = extract_keypoints(landmarker, mp_image)
        if keypoints is None:
            continue
        raw = extract_features(keypoints)

        if scale_reference is None:
            if calibration_start is None:
                calibration_start = timestamp
            calibration_rows.append(raw)
            if timestamp - calibration_start >= CALIBRATION_SECONDS:
                scale_reference = compute_scale_reference(calibration_rows)
            continue

        if raw["mean_visibility"] < MIN_MEAN_VISIBILITY:
            label = "PoseUncertain"
        else:
            normalized = normalize_features(raw, scale_reference)
            label = model.predict([feature_vector(normalized)])[0]

        smoothed, flagged = state.update(timestamp, label)
        if flagged:
            on_flag(timestamp, smoothed)


def webcam_frame_source(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield time.time(), frame
    finally:
        cap.release()


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "posture_monitor/models/classifier.joblib"
    run(webcam_frame_source(), model_path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_realtime.py -v`
Expected: PASS (1 test; will take a little while since it runs real pose estimation on 35 frames)

- [ ] **Step 5: Commit**

```bash
git add posture_monitor/inference/realtime.py tests/test_realtime.py
git commit -m "Add real-time calibration, classification, smoothing, and flagging loop"
```

---

### Task 11: Run the full pipeline on the real dataset and record results

This is the step that produces real numbers — everything before it was building and unit-testing the machinery.

**Files:**
- Modify: `docs/specs/2026-11-24-posture-ml-model-design.md` (append a "Results" section)

- [ ] **Step 1: Build the feature table from the real dataset**

Run: `source .venv/bin/activate && python3 -m posture_monitor.training.build_dataset`
Expected: Prints `Wrote <N> rows to posture_monitor/training/features.csv` where N is close to 2,706 (total image count across both folders) minus any frames where pose detection failed.

- [ ] **Step 2: Run de-duplication and honest split, observe how much the clips collapse**

Run: `source .venv/bin/activate && python3 -m posture_monitor.training.dedup_split`
Expected: Per-class lines like `CorrectSitting: 670 raw -> <M> after dedup (<train> train / <test> test)`. Given the forensic finding that each class is one short continuous clip, expect `<M>` to be dramatically smaller than the raw count — if it isn't (e.g. barely any frames dropped), the threshold in `config.DEDUP_ANGLE_DISTANCE_THRESHOLD` is too tight and should be raised and re-run before continuing.

- [ ] **Step 3: Train and evaluate on the honestly-split, de-duplicated data**

Run: `source .venv/bin/activate && python3 -m posture_monitor.training.train`
Expected: cross-validated F1 scores for both candidate models, a selected model name, a classification report, and a confusion matrix printed to the terminal. `posture_monitor/models/classifier.joblib` is created.

- [ ] **Step 4: Record the results in the design doc**

Append a `## Results (first training run)` section to `docs/specs/2026-11-24-posture-ml-model-design.md` with: the per-class dedup counts from Step 2, the chosen model, and the classification report/confusion matrix from Step 3. Note explicitly that this is session-level validation per the spec's documented limitation, not cross-subject validation.

- [ ] **Step 5: Try the live pipeline manually (qualitative check from the spec's testing plan)**

Run: `source .venv/bin/activate && python3 -m posture_monitor.inference.realtime`
Sit in front of the webcam side-on for 30s (calibration, any posture is fine), then deliberately slouch/lean and confirm a flag prints to the terminal after ~5s, then sit correctly and confirm flags stop. Ideally have a teammate other than the original dataset subject try this too, per the spec's cross-subject qualitative check.

- [ ] **Step 6: Commit**

```bash
git add docs/specs/2026-11-24-posture-ml-model-design.md posture_monitor/training/features.csv posture_monitor/training/features_honest.csv
git commit -m "Record first training run results against the real dataset"
```

(Note: `posture_monitor/models/classifier.joblib` stays untracked per `.gitignore` — it's a build artifact, regenerate it by re-running Steps 1-3.)

---

## Self-review notes

- **Spec coverage:** Architecture (Task 3-5), feature set table (Task 4), calibration (Task 5), training methodology incl. dedup/honest-split/class-imbalance/model-selection (Task 7-9), real-time loop incl. smoothing/debounce (Task 6, 10), project structure (all tasks), testing plan incl. qualitative cross-subject check (Task 11 Step 5), out-of-scope web app correctly excluded. `LegTwisting`'s distinct feature cluster (hip-knee-angle, leg visibility, shoulder span/rotation) is covered in Task 4.
- **Type/name consistency checked:** `NORMALIZED_FEATURE_ORDER` defined once in `calibration/calibrate.py` (Task 5) and imported by `training/train.py` (Task 9) and `inference/realtime.py` (Task 10) — not redefined. `extract_features` keys (Task 4) match exactly what `normalize_features` (Task 5) and `build_dataset.py` (Task 7) expect. `honest_split`/`class` column names match between `dedup_split.py` (Task 8) output and `train.py` (Task 9) input.
- **No placeholders:** every step has complete, runnable code; no "TBD"/"add error handling" style steps.
