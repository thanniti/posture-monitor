import statistics

from posture_monitor.config import (
    FRONT_HEAD_ANGLE_TOLERANCE_DEG,
    FRONT_HIP_KNEE_ANGLE_TOLERANCE_DEG,
    FRONT_LEG_VISIBILITY_DROP_TOLERANCE,
    FRONT_NECK_SHRINK_TOLERANCE,
)

BASELINE_FIELDS = [
    "neck_length_px",
    "shoulder_width_px",
    "head_shoulder_angle_deg",
    "hip_knee_angle_deg",
    "leg_visibility",
]


def compute_frontal_baseline(raw_feature_rows):
    """Median of each relevant field across a session's calibration-window rows."""
    return {
        field: statistics.median([row[field] for row in raw_feature_rows])
        for field in BASELINE_FIELDS
    }


def classify_frontal(raw, baseline):
    """Rule-based front-view posture classification: compares live frontal
    features against a personal baseline (from compute_frontal_baseline) with
    fixed tolerance margins. Does not handle PoseUncertain — callers should check
    mean_visibility themselves before calling this, same as the side-view path."""
    baseline_neck_ratio = baseline["neck_length_px"] / baseline["shoulder_width_px"]
    live_neck_ratio = raw["neck_length_px"] / raw["shoulder_width_px"]
    neck_shrink_fraction = (baseline_neck_ratio - live_neck_ratio) / baseline_neck_ratio
    head_angle_deviation = abs(raw["head_shoulder_angle_deg"] - baseline["head_shoulder_angle_deg"])

    if neck_shrink_fraction > FRONT_NECK_SHRINK_TOLERANCE or head_angle_deviation > FRONT_HEAD_ANGLE_TOLERANCE_DEG:
        return "Slouching"

    hip_knee_drop = baseline["hip_knee_angle_deg"] - raw["hip_knee_angle_deg"]
    leg_visibility_drop = baseline["leg_visibility"] - raw["leg_visibility"]
    if hip_knee_drop > FRONT_HIP_KNEE_ANGLE_TOLERANCE_DEG or leg_visibility_drop > FRONT_LEG_VISIBILITY_DROP_TOLERANCE:
        return "LegTwisting"

    return "CorrectSitting"
