import pytest

from posture_monitor.calibration.calibrate import (
    NORMALIZED_FEATURE_ORDER,
    compute_scale_reference,
    feature_vector,
    normalize_features,
)


def make_raw(nose_to_ear_px=100.0, **overrides):
    base = {
        "neck_angle_deg": 5.0,
        "nose_shoulder_angle_deg": 8.0,
        "head_forward_offset_px": 20.0,
        "nose_forward_offset_px": 25.0,
        "mean_visibility": 0.97,
        "nose_to_ear_px": nose_to_ear_px,
    }
    base.update(overrides)
    return base


def test_scale_reference_is_median_of_nose_to_ear():
    rows = [make_raw(nose_to_ear_px=v) for v in [90.0, 100.0, 110.0, 1000.0]]
    assert compute_scale_reference(rows) == pytest.approx(105.0)


def test_scale_reference_ignores_zero_length_rows():
    rows = [make_raw(nose_to_ear_px=0.0), make_raw(nose_to_ear_px=100.0)]
    assert compute_scale_reference(rows) == pytest.approx(100.0)


def test_scale_reference_raises_on_all_non_positive_rows():
    rows = [make_raw(nose_to_ear_px=0.0), make_raw(nose_to_ear_px=-5.0)]
    with pytest.raises(ValueError, match="no rows with positive nose_to_ear_px"):
        compute_scale_reference(rows)


def test_normalize_features_divides_distance_features_by_scale_reference():
    raw = make_raw(head_forward_offset_px=20.0, nose_forward_offset_px=25.0)
    normalized = normalize_features(raw, scale_reference=100.0)
    assert normalized["head_forward_offset_norm"] == pytest.approx(0.2)
    assert normalized["nose_forward_offset_norm"] == pytest.approx(0.25)


def test_normalize_features_leaves_angle_and_visibility_features_unchanged():
    raw = make_raw()
    normalized = normalize_features(raw, scale_reference=100.0)
    assert normalized["neck_angle_deg"] == raw["neck_angle_deg"]
    assert normalized["nose_shoulder_angle_deg"] == raw["nose_shoulder_angle_deg"]
    assert normalized["mean_visibility"] == raw["mean_visibility"]


def test_normalize_features_raises_on_non_positive_scale_reference():
    raw = make_raw()
    with pytest.raises(ValueError, match="scale_reference must be positive"):
        normalize_features(raw, scale_reference=0)
    with pytest.raises(ValueError, match="scale_reference must be positive"):
        normalize_features(raw, scale_reference=-5.0)


def test_feature_vector_follows_declared_order():
    raw = make_raw()
    normalized = normalize_features(raw, scale_reference=100.0)
    vector = feature_vector(normalized)
    assert len(vector) == len(NORMALIZED_FEATURE_ORDER)
    assert vector == [normalized[name] for name in NORMALIZED_FEATURE_ORDER]
