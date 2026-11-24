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
