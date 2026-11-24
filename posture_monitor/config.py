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
# feature against a personal baseline recorded during that session's calibration.

# Fraction the live neck-length/shoulder-width ratio can shrink below the
# calibrated baseline ratio before it's flagged as slouching (facing the camera,
# ducking your head down/forward shortens this ratio; dividing by shoulder width
# cancels out the effect of being closer to or farther from the camera).
# Lowered from an initial 0.25 (required near-horizontal leaning to trigger) to
# make moderate slouching detectable. Decrease further for more sensitivity,
# increase if normal posture shifts start triggering false positives.
FRONT_NECK_SHRINK_TOLERANCE = 0.12

# Degrees the live head-shoulder angle can deviate from the calibrated baseline
# before it's flagged as slouching. Lowered from an initial 10.0 for the same
# sensitivity reason as FRONT_NECK_SHRINK_TOLERANCE above.
FRONT_HEAD_ANGLE_TOLERANCE_DEG = 6.0

# Degrees the live hip-knee-ankle angle can drop below the calibrated baseline
# before it's flagged as leg twisting (a bent/crossed leg reduces this angle).
FRONT_HIP_KNEE_ANGLE_TOLERANCE_DEG = 20.0

# How much the live leg-landmark visibility can drop below the calibrated
# baseline before it's flagged as leg twisting (legs pulled up/occluded reduce
# MediaPipe's visibility confidence for knee/ankle).
FRONT_LEG_VISIBILITY_DROP_TOLERANCE = 0.3

# Below this shoulder-width pixel distance, treat front-view detection as
# unreliable (PoseUncertain) rather than running classify_frontal — guards
# against division-by-near-zero in its neck-length/shoulder-width ratio when
# MediaPipe detects the two shoulders implausibly close together.
MIN_SHOULDER_WIDTH_PX = 20.0

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
