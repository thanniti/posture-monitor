import math

import pytest

from posture_monitor.features.extraction import extract_features, extract_frontal_features


def make_keypoints(**overrides):
    """A simple upright reference pose: shoulder at origin-ish, ear directly
    above shoulder, nose slightly above and level with ear. Individual
    coordinates can be overridden per test. Eyes are placed level with each
    other and with the nose, a fixed inter-eye gap apart (frontal reference)."""
    # Head points are centered above the left shoulder (x=-50) rather than at
    # x=0, since the picked side for both shoulder and ear ties to "left"
    # (_pick_side favors left on equal visibility) -- aligning them keeps the
    # upright reference pose actually upright (near-zero angles) for the
    # side-view features, which only ever look at one picked side.
    base = {
        "nose": (-50, -220, 1.0),
        "left_eye": (-65, -215, 1.0), "right_eye": (-35, -215, 1.0),
        "left_ear": (-50, -200, 1.0), "right_ear": (-50, -200, 0.3),
        "left_shoulder": (-50, -100, 1.0), "right_shoulder": (50, -100, 1.0),
        "left_hip": (-40, 0, 1.0), "right_hip": (40, 0, 0.9),
        "left_knee": (-40, 100, 1.0), "right_knee": (40, 100, 0.9),
        "left_ankle": (-40, 200, 1.0), "right_ankle": (40, 200, 0.9),
    }
    base.update(overrides)
    return base


# --- extract_features (side view) ---

def test_upright_side_pose_has_near_zero_angles():
    features = extract_features(make_keypoints())
    assert features["neck_angle_deg"] == pytest.approx(0.0, abs=0.01)
    assert features["nose_shoulder_angle_deg"] == pytest.approx(0.0, abs=0.01)


def test_forward_head_produces_nonzero_offsets_and_angles():
    jutting = make_keypoints(
        left_ear=(40, -200, 1.0), right_ear=(40, -200, 0.3),
        nose=(50, -220, 1.0),
    )
    features = extract_features(jutting)
    assert features["neck_angle_deg"] != pytest.approx(0.0, abs=0.01)
    assert features["nose_shoulder_angle_deg"] != pytest.approx(0.0, abs=0.01)
    assert features["head_forward_offset_px"] > 0
    assert features["nose_forward_offset_px"] > 0


def test_side_view_chooses_higher_visibility_ear():
    # right_ear has lower visibility (0.3) than left_ear (1.0) in the base pose.
    base = extract_features(make_keypoints())
    shifted = extract_features(make_keypoints(left_ear=(40, -200, 1.0)))
    assert shifted["head_forward_offset_px"] != base["head_forward_offset_px"]


def test_nose_to_ear_is_positive_and_does_not_depend_on_shoulder():
    features = extract_features(make_keypoints())
    assert features["nose_to_ear_px"] > 0
    moved_shoulder = extract_features(make_keypoints(left_shoulder=(-200, -50, 1.0), right_shoulder=(200, -50, 1.0)))
    assert moved_shoulder["nose_to_ear_px"] == pytest.approx(features["nose_to_ear_px"])


def test_side_view_has_no_hip_or_knee_dependent_keys():
    features = extract_features(make_keypoints())
    assert "torso_angle_deg" not in features
    assert "hip_knee_angle_deg" not in features
    assert "leg_visibility" not in features
    assert "torso_length_px" not in features


def test_side_view_mean_visibility_in_zero_one_range():
    features = extract_features(make_keypoints())
    assert 0.0 <= features["mean_visibility"] <= 1.0


# --- extract_frontal_features (front view) ---

def test_level_ears_and_shoulders_have_zero_tilt():
    features = extract_frontal_features(make_keypoints())
    assert features["shoulder_tilt_deg"] == pytest.approx(0.0, abs=0.01)
    assert features["ear_tilt_deg"] == pytest.approx(0.0, abs=0.01)


def test_tilted_shoulders_produce_nonzero_shoulder_tilt():
    tilted = make_keypoints(right_shoulder=(50, -120, 1.0))
    features = extract_frontal_features(tilted)
    assert features["shoulder_tilt_deg"] != pytest.approx(0.0, abs=0.01)


def test_shortened_neck_reduces_ear_to_shoulder_vertical():
    normal = extract_frontal_features(make_keypoints())
    hunched = make_keypoints(left_ear=(0, -120, 1.0), right_ear=(0, -120, 0.3))
    hunched_features = extract_frontal_features(hunched)
    assert hunched_features["ear_to_shoulder_vertical_px"] < normal["ear_to_shoulder_vertical_px"]


def test_narrower_shoulders_reduce_shoulder_width():
    normal = extract_frontal_features(make_keypoints())
    narrow = make_keypoints(left_shoulder=(-20, -100, 1.0), right_shoulder=(20, -100, 1.0))
    narrow_features = extract_frontal_features(narrow)
    assert narrow_features["shoulder_width_px"] < normal["shoulder_width_px"]


def test_interocular_distance_is_positive_and_stable_regardless_of_shoulders():
    features = extract_frontal_features(make_keypoints())
    assert features["interocular_distance_px"] > 0
    narrowed_shoulders = extract_frontal_features(
        make_keypoints(left_shoulder=(-20, -100, 1.0), right_shoulder=(20, -100, 1.0))
    )
    assert narrowed_shoulders["interocular_distance_px"] == pytest.approx(features["interocular_distance_px"])


def test_frontal_view_has_no_hip_or_knee_dependent_keys():
    features = extract_frontal_features(make_keypoints())
    assert "hip_knee_angle_deg" not in features
    assert "leg_visibility" not in features


def test_frontal_mean_visibility_in_zero_one_range():
    features = extract_frontal_features(make_keypoints())
    assert 0.0 <= features["mean_visibility"] <= 1.0
