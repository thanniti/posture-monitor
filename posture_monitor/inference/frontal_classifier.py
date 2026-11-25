import statistics

from posture_monitor.config import (
    FRONT_LATERAL_TILT_TOLERANCE_DEG,
    FRONT_NECK_SHRINK_TOLERANCE,
    FRONT_SHOULDER_PROTRACTION_TOLERANCE,
)

BASELINE_FIELDS = [
    "nose_to_shoulder_vertical_px",
    "ear_to_shoulder_vertical_px",
    "shoulder_width_px",
    "shoulder_tilt_deg",
    "interocular_distance_px",
]


def compute_frontal_baseline(raw_feature_rows):
    """Median of each relevant field across a session's calibration-window rows."""
    return {
        field: statistics.median([row[field] for row in raw_feature_rows])
        for field in BASELINE_FIELDS
    }


def _severity_to_confidence(ratio):
    """Converts a deviation-to-tolerance ratio into a 0-100 display confidence.
    Not a true statistical confidence (this is rule-based, not probabilistic) --
    it's how far past (or below) the triggering threshold the strongest signal
    is, scaled so 'right at the threshold' reads as 50%."""
    if ratio >= 1.0:
        return min(99.0, round(50 + 50 * (ratio - 1.0), 1))
    return min(100.0, round((1.0 - ratio) * 100, 1))


def classify_frontal(raw, baseline):
    """Rule-based front-view posture classification: compares live frontal
    features against a personal baseline (from compute_frontal_baseline).
    Normalizes the compression/protraction signals by interocular_distance_px
    (a stable facial dimension, immune to shoulder rounding or camera
    distance) instead of shoulder width, since shoulder width is now itself a
    slouch signal (protraction), not a safe reference. Returns (label,
    confidence) -- confidence is NOT a probability, see _severity_to_confidence.
    Does not handle PoseUncertain -- callers should check mean_visibility and
    interocular_distance_px themselves before calling this."""
    live_interocular = raw["interocular_distance_px"]
    baseline_interocular = baseline["interocular_distance_px"]

    live_nose_ratio = raw["nose_to_shoulder_vertical_px"] / live_interocular
    baseline_nose_ratio = baseline["nose_to_shoulder_vertical_px"] / baseline_interocular
    nose_shrink_fraction = (baseline_nose_ratio - live_nose_ratio) / baseline_nose_ratio

    live_ear_ratio = raw["ear_to_shoulder_vertical_px"] / live_interocular
    baseline_ear_ratio = baseline["ear_to_shoulder_vertical_px"] / baseline_interocular
    ear_shrink_fraction = (baseline_ear_ratio - live_ear_ratio) / baseline_ear_ratio

    live_shoulder_ratio = raw["shoulder_width_px"] / live_interocular
    baseline_shoulder_ratio = baseline["shoulder_width_px"] / baseline_interocular
    shoulder_shrink_fraction = (baseline_shoulder_ratio - live_shoulder_ratio) / baseline_shoulder_ratio

    shoulder_tilt_deviation = abs(raw["shoulder_tilt_deg"] - baseline["shoulder_tilt_deg"])

    neck_compression_ratio = max(nose_shrink_fraction, ear_shrink_fraction) / FRONT_NECK_SHRINK_TOLERANCE
    protraction_ratio = shoulder_shrink_fraction / FRONT_SHOULDER_PROTRACTION_TOLERANCE
    tilt_ratio = shoulder_tilt_deviation / FRONT_LATERAL_TILT_TOLERANCE_DEG

    # Tracked by category (not a bare max() of all three) so the Slouching-vs-
    # LateralLean decision never depends on floating-point equality between
    # ratios computed via different formulas.
    category_ratios = {
        "Slouching": max(neck_compression_ratio, protraction_ratio),
        "LateralLean": tilt_ratio,
    }
    label = max(category_ratios, key=category_ratios.get)
    severity = category_ratios[label]
    confidence = _severity_to_confidence(severity)

    if severity < 1.0:
        return "CorrectSitting", confidence
    return label, confidence
