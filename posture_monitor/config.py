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
