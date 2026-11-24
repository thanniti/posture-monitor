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
