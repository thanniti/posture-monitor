import math
import pytest

from posture_monitor.features.extraction import extract_features


def make_keypoints(**overrides):
    """A simple upright reference pose: hip at origin, shoulder directly above,
    ear directly above shoulder, knee/ankle directly below hip (straight leg).
    Individual coordinates can be overridden per test."""
    base = {
        "nose": (0, -220, 1.0),
        "left_ear": (0, -200, 1.0), "right_ear": (0, -200, 0.3),
        "left_shoulder": (0, -100, 1.0), "right_shoulder": (0, -100, 1.0),
        "left_hip": (0, 0, 1.0), "right_hip": (0, 0, 0.9),
        "left_knee": (0, 100, 1.0), "right_knee": (0, 100, 0.9),
        "left_ankle": (0, 200, 1.0), "right_ankle": (0, 200, 0.9),
    }
    base.update(overrides)
    return base


def test_upright_pose_has_near_zero_angles():
    features = extract_features(make_keypoints())
    assert features["neck_angle_deg"] == pytest.approx(0.0, abs=0.01)
    assert features["torso_angle_deg"] == pytest.approx(0.0, abs=0.01)
    assert features["hip_knee_angle_deg"] == pytest.approx(180.0, abs=0.01)


def test_forward_lean_produces_positive_torso_angle():
    leaned = make_keypoints(left_shoulder=(50, -100, 1.0), right_shoulder=(50, -100, 1.0))
    features = extract_features(leaned)
    assert features["torso_angle_deg"] > 10


def test_bent_knee_reduces_hip_knee_angle():
    bent = make_keypoints(left_knee=(80, 60, 1.0), right_knee=(80, 60, 0.9))
    features = extract_features(bent)
    assert features["hip_knee_angle_deg"] < 150


def test_chooses_higher_visibility_side_for_ear():
    # right_ear has lower visibility (0.3) than left_ear (1.0) in the base pose,
    # so left_ear's offset should be used for head_forward_offset_px.
    features_base = extract_features(make_keypoints())
    shifted = make_keypoints(left_ear=(40, -200, 1.0))
    features_shifted = extract_features(shifted)
    assert features_shifted["head_forward_offset_px"] != features_base["head_forward_offset_px"]


def test_shoulder_span_and_torso_length_are_positive():
    features = extract_features(make_keypoints())
    assert features["shoulder_span_px"] >= 0
    assert features["torso_length_px"] == pytest.approx(100.0, abs=0.01)


def test_mean_visibility_in_zero_one_range():
    features = extract_features(make_keypoints())
    assert 0.0 <= features["mean_visibility"] <= 1.0
