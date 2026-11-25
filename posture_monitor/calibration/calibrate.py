import statistics

NORMALIZED_FEATURE_ORDER = [
    "neck_angle_deg",
    "nose_shoulder_angle_deg",
    "head_forward_offset_norm",
    "nose_forward_offset_norm",
    "mean_visibility",
]


def compute_scale_reference(raw_feature_rows):
    """Median nose_to_ear_px across rows. Used both for the live calibration
    window and for the training-set-wide reference (see docs/specs design doc
    for why we use a fixed median rather than per-frame self-normalization).
    nose_to_ear_px is a stable facial dimension, clearly visible in profile,
    unaffected by posture -- unlike the old hip-based torso_length_px, which
    this replaces (upper-body-only requirement)."""
    lengths = [row["nose_to_ear_px"] for row in raw_feature_rows if row["nose_to_ear_px"] > 0]
    if not lengths:
        raise ValueError("compute_scale_reference: no rows with positive nose_to_ear_px")
    return statistics.median(lengths)


def normalize_features(raw, scale_reference):
    """Normalize raw features by dividing distance features by scale_reference."""
    if scale_reference <= 0:
        raise ValueError(f"normalize_features: scale_reference must be positive, got {scale_reference}")
    return {
        "neck_angle_deg": raw["neck_angle_deg"],
        "nose_shoulder_angle_deg": raw["nose_shoulder_angle_deg"],
        "head_forward_offset_norm": raw["head_forward_offset_px"] / scale_reference,
        "nose_forward_offset_norm": raw["nose_forward_offset_px"] / scale_reference,
        "mean_visibility": raw["mean_visibility"],
    }


def feature_vector(normalized):
    """Extract features in the declared order."""
    return [normalized[name] for name in NORMALIZED_FEATURE_ORDER]
