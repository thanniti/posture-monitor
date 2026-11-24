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
    if not lengths:
        raise ValueError("compute_scale_reference: no rows with positive torso_length_px")
    return statistics.median(lengths)


def normalize_features(raw, scale_reference):
    """Normalize raw features by dividing distance features by scale_reference."""
    if scale_reference <= 0:
        raise ValueError(f"normalize_features: scale_reference must be positive, got {scale_reference}")
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
    """Extract features in the declared order."""
    return [normalized[name] for name in NORMALIZED_FEATURE_ORDER]
