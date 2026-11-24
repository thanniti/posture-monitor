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


from posture_monitor.features.extraction import extract_frontal_features


def test_level_ears_and_shoulders_have_zero_tilt():
    features = extract_frontal_features(make_keypoints())
    assert features["shoulder_tilt_deg"] == pytest.approx(0.0, abs=0.01)
    assert features["ear_tilt_deg"] == pytest.approx(0.0, abs=0.01)


def test_tilted_shoulders_produce_nonzero_shoulder_tilt():
    tilted = make_keypoints(right_shoulder=(0, -80, 1.0))  # right shoulder raised
    features = extract_frontal_features(tilted)
    assert features["shoulder_tilt_deg"] != pytest.approx(0.0, abs=0.01)


def test_head_directly_above_shoulders_has_near_zero_head_shoulder_angle():
    features = extract_frontal_features(make_keypoints())
    assert features["head_shoulder_angle_deg"] == pytest.approx(0.0, abs=0.01)


def test_neck_length_and_shoulder_width_are_positive():
    # The shared base fixture stacks left/right shoulder at the identical point
    # (it's a side-view fixture with no left/right offset modeled -- the existing
    # side-view test even asserts shoulder_span_px >= 0, not > 0, for this reason).
    # Give shoulders a realistic left/right offset here so shoulder_width_px, which
    # is meaningful for a frontal pose, is actually exercised.
    frontal = make_keypoints(left_shoulder=(-50, -100, 1.0), right_shoulder=(50, -100, 1.0))
    features = extract_frontal_features(frontal)
    assert features["neck_length_px"] > 0
    assert features["shoulder_width_px"] > 0


def test_shortened_neck_reduces_neck_length():
    normal = extract_frontal_features(make_keypoints())
    # Move ears down closer to shoulders, simulating a ducked/slouched head.
    hunched = make_keypoints(
        left_ear=(0, -120, 1.0), right_ear=(0, -120, 0.3),
    )
    hunched_features = extract_frontal_features(hunched)
    assert hunched_features["neck_length_px"] < normal["neck_length_px"]


def test_frontal_hip_knee_angle_matches_side_view_computation():
    # Sanity check: leg features should be identical between the two functions
    # since they're not view-dependent and use the same underlying computation.
    from posture_monitor.features.extraction import extract_features
    keypoints = make_keypoints()
    assert extract_frontal_features(keypoints)["hip_knee_angle_deg"] == pytest.approx(
        extract_features(keypoints)["hip_knee_angle_deg"]
    )
