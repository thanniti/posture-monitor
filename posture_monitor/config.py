POSE_MODEL_PATH = "assets/pose_landmarker_lite.task"

# Below this mean landmark visibility, report "PoseUncertain" instead of a class guess.
MIN_MEAN_VISIBILITY = 0.5

# Calibration window length at the start of a live session.
CALIBRATION_SECONDS = 15

# Rolling majority-vote window for smoothing per-frame predictions.
SMOOTHING_WINDOW_SECONDS = 2.5

# A non-CorrectSitting (or PoseUncertain) smoothed label must persist this long before flagging.
DEBOUNCE_SECONDS = 5.0

# Euclidean distance (in degrees, across neck/torso/knee angles) below which two
# consecutive frames in the same class are treated as near-duplicates and the later
# one is dropped during training-set de-duplication.
#
# Tuned against the real dataset (Task 11): the original default of 2.0 was far too
# tight. Real per-frame pose-estimation output has substantial frame-to-frame jitter
# (median consecutive-frame angle distance was ~14-30 degrees depending on class), so
# a 2.0-degree threshold dropped almost nothing (e.g. CorrectSitting only went from
# 649 -> 623 raw frames). Raised to 65.0, which collapses each class's near-duplicate
# frames by roughly 80-91% while still leaving enough distinct rows per class for a
# meaningful honest train/test split.
DEDUP_ANGLE_DISTANCE_THRESHOLD = 65.0

# Fraction of each class's chronologically-sorted frames used for the honest
# re-split's training portion (remainder becomes the held-out test portion).
HONEST_SPLIT_TRAIN_FRACTION = 0.8

# If the model predicts LeanForward/LeanBackward but both torso_angle_deg and
# neck_angle_deg are within this many degrees of vertical, override to
# CorrectSitting instead. Widens the "sitting correctly" margin since the trained
# model's decision boundary (learned from a small CorrectSitting sample) can be
# oversensitive to minor natural sway. Increase this value for a wider margin,
# decrease it to make the classifier stricter.
CORRECT_SITTING_ANGLE_TOLERANCE_DEG = 8.0

# Front-view (facing-camera) detection tolerances — rule-based, not ML, since there
# is no labeled front-view training data. Each compares a live frame's frontal
# feature (normalized by interocular_distance_px, a stable facial dimension that
# doesn't change with posture) against a personal baseline recorded during that
# session's calibration.

# Fraction the live nose-to-shoulder OR ear-to-shoulder vertical distance (each
# normalized by interocular distance) can shrink below its calibrated baseline
# ratio before it's flagged as slouching. Lowered from an initial 0.25 (required
# near-horizontal leaning to trigger) to make moderate slouching detectable.
FRONT_NECK_SHRINK_TOLERANCE = 0.12

# Fraction the live shoulder-width-to-interocular-distance ratio can shrink below
# its calibrated baseline before it's flagged as slouching (shoulders rounding
# forward/inward narrows their apparent width in a frontal view).
FRONT_SHOULDER_PROTRACTION_TOLERANCE = 0.12

# Degrees the live shoulder-tilt angle can deviate from its calibrated baseline
# before it's flagged as LateralLean (leaning/tilting to one side), rather than
# Slouching (which covers neck compression and shoulder protraction).
FRONT_LATERAL_TILT_TOLERANCE_DEG = 6.0

# Below this eye-to-eye pixel distance, treat front-view detection as unreliable
# (PoseUncertain) rather than running classify_frontal — guards against
# division-by-near-zero in its normalization math when MediaPipe detects the two
# eyes implausibly close together.
MIN_INTEROCULAR_DISTANCE_PX = 8.0

# Orientation auto-detection: ratio of shoulder_span_px / torso_length_px (from the
# side-view feature extraction, but meaningful regardless of which pipeline ends up
# classifying the frame). Calibrated from the real labeled dataset: true side-view
# poses (CorrectSitting/LeanForward/LeanBackward) measured ~0.13-0.19 median ratio.
# There is no genuine front-facing capture in the dataset to calibrate the upper
# threshold against directly, so FRONT_VIEW_RATIO_THRESHOLD is an extrapolated lower
# bound based on the LegTwisting clip (a side-view-labeled clip where the subject was
# rotated partway toward the camera, not a true frontal pose), which measured ~0.42
# median -- a genuinely frontal pose should read well above this. Below
# SIDE_VIEW_RATIO_THRESHOLD -> side view; above FRONT_VIEW_RATIO_THRESHOLD -> front
# view; the gap in between is a deliberate hysteresis band -- ambiguous/borderline
# frames keep whatever mode was already active, instead of flickering between modes
# on noisy single-frame readings.
SIDE_VIEW_RATIO_THRESHOLD = 0.25
FRONT_VIEW_RATIO_THRESHOLD = 0.40
