import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from posture_monitor.inference.realtime import run_front_view, run_side_view
from posture_monitor.inference.realtime import _status_text_and_color
from posture_monitor.inference.realtime import _draw_skeleton
from posture_monitor.inference.realtime import _apply_angle_tolerance
from posture_monitor.inference.realtime import _suppress_leg_twisting
from posture_monitor.inference.realtime import _infer_orientation, run


def _train_tiny_always_correct_model(tmp_path):
    # A model that always predicts CorrectSitting, so this test can assert the
    # pipeline runs end-to-end (calibration -> classification -> smoothing ->
    # debounce) without needing a real trained classifier.
    model = RandomForestClassifier(n_estimators=5, random_state=0)
    X = [[0, 0, 0.1, 0.9, 175, 0.95, 0.4, 0.95]] * 4
    y = ["CorrectSitting"] * 4
    model.fit(X, y)
    model_path = tmp_path / "model.joblib"
    joblib.dump({"model": model, "scale_reference": 100.0, "feature_order": None}, model_path)
    return str(model_path)


def _fake_frame_source(num_frames, start_timestamp=0.0, step_seconds=1.0):
    frame = cv2.imread("data/train/CorrectSitting/image_0.jpg")
    for i in range(num_frames):
        yield start_timestamp + i * step_seconds, frame


def test_run_completes_calibration_and_emits_no_flags_for_correct_sitting(tmp_path):
    model_path = _train_tiny_always_correct_model(tmp_path)
    flags = []

    # 35 frames at 1s/frame covers the 30s calibration window plus 5s of live inference.
    run_side_view(_fake_frame_source(35), model_path, on_flag=lambda ts, label: flags.append((ts, label)))

    assert flags == []


def test_status_text_no_pose_detected():
    text, color = _status_text_and_color(
        keypoints_detected=False, calibrating=False, remaining_calibration_seconds=0, label=None
    )
    assert text == "No pose detected"
    assert color == (0, 0, 255)


def test_status_text_calibrating_shows_countdown():
    text, color = _status_text_and_color(
        keypoints_detected=True, calibrating=True, remaining_calibration_seconds=12.4, label=None
    )
    assert text == "Calibrating... 12s"
    assert color == (255, 255, 0)


def test_status_text_correct_sitting_is_green():
    text, color = _status_text_and_color(
        keypoints_detected=True, calibrating=False, remaining_calibration_seconds=0, label="CorrectSitting"
    )
    assert text == "CorrectSitting"
    assert color == (0, 255, 0)


def test_status_text_bad_posture_is_red():
    text, color = _status_text_and_color(
        keypoints_detected=True, calibrating=False, remaining_calibration_seconds=0, label="LeanForward"
    )
    assert text == "LeanForward"
    assert color == (0, 0, 255)


def _make_test_keypoints():
    return {
        "nose": (50, 10, 1.0),
        "left_ear": (40, 20, 1.0), "right_ear": (60, 20, 1.0),
        "left_shoulder": (30, 50, 1.0), "right_shoulder": (70, 50, 1.0),
        "left_hip": (35, 100, 1.0), "right_hip": (65, 100, 1.0),
        "left_knee": (35, 150, 1.0), "right_knee": (65, 150, 1.0),
        "left_ankle": (35, 200, 1.0), "right_ankle": (65, 200, 1.0),
    }


def test_draw_skeleton_draws_something_on_blank_frame():
    frame = np.zeros((220, 100, 3), dtype=np.uint8)
    _draw_skeleton(frame, _make_test_keypoints())
    assert frame.any()


def test_draw_skeleton_draws_a_dot_at_each_keypoint_location():
    frame = np.zeros((220, 100, 3), dtype=np.uint8)
    keypoints = _make_test_keypoints()
    _draw_skeleton(frame, keypoints)
    x, y, _ = keypoints["nose"]
    assert frame[int(y), int(x)].any()


def _make_raw(torso=0.0, neck=0.0):
    return {"torso_angle_deg": torso, "neck_angle_deg": neck}


def test_angle_tolerance_overrides_lean_forward_within_tolerance():
    raw = _make_raw(torso=5.0, neck=-3.0)
    assert _apply_angle_tolerance(raw, "LeanForward") == "CorrectSitting"


def test_angle_tolerance_overrides_lean_backward_within_tolerance():
    raw = _make_raw(torso=-6.0, neck=4.0)
    assert _apply_angle_tolerance(raw, "LeanBackward") == "CorrectSitting"


def test_angle_tolerance_does_not_override_large_lean():
    raw = _make_raw(torso=25.0, neck=10.0)
    assert _apply_angle_tolerance(raw, "LeanForward") == "LeanForward"


def test_angle_tolerance_does_not_affect_leg_twisting_or_pose_uncertain():
    raw = _make_raw(torso=0.0, neck=0.0)
    assert _apply_angle_tolerance(raw, "LegTwisting") == "LegTwisting"
    assert _apply_angle_tolerance(raw, "PoseUncertain") == "PoseUncertain"


def test_run_front_view_completes_calibration_and_emits_no_flags_for_self_consistent_frame():
    # Feeding the SAME real frame throughout both calibration and live phases
    # means the computed baseline exactly matches that frame's own live features
    # (zero deviation), so classify_frontal must always return CorrectSitting —
    # this verifies the full real pose-estimation + calibration + rule-based
    # classification + smoothing wiring end-to-end without needing a labeled
    # frontal dataset (none exists yet).
    flags = []

    run_front_view(_fake_frame_source(35), on_flag=lambda ts, label: flags.append((ts, label)))

    assert flags == []


def test_suppress_leg_twisting_remaps_to_correct_sitting():
    assert _suppress_leg_twisting("LegTwisting") == "CorrectSitting"


def test_suppress_leg_twisting_leaves_other_labels_unchanged():
    assert _suppress_leg_twisting("CorrectSitting") == "CorrectSitting"
    assert _suppress_leg_twisting("LeanForward") == "LeanForward"
    assert _suppress_leg_twisting("LeanBackward") == "LeanBackward"
    assert _suppress_leg_twisting("Slouching") == "Slouching"
    assert _suppress_leg_twisting("PoseUncertain") == "PoseUncertain"


def test_run_side_view_never_emits_leg_twisting_label(tmp_path):
    # Train a tiny model that always predicts LegTwisting, feed it through
    # run_side_view, and confirm the suppression means it's never flagged as
    # LegTwisting at the point flags are emitted (it gets treated as
    # CorrectSitting instead, which never triggers a flag).
    model = RandomForestClassifier(n_estimators=5, random_state=0)
    X = [[0, 0, 0.1, 0.9, 175, 0.95, 0.4, 0.95]] * 4
    y = ["LegTwisting"] * 4
    model.fit(X, y)
    model_path = tmp_path / "model.joblib"
    joblib.dump({"model": model, "scale_reference": 100.0, "feature_order": None}, model_path)

    flags = []
    run_side_view(_fake_frame_source(35), str(model_path), on_flag=lambda ts, label: flags.append((ts, label)))

    assert flags == []


def test_infer_orientation_below_threshold_is_side():
    assert _infer_orientation(0.1, current_mode="front") == "side"


def test_infer_orientation_above_threshold_is_front():
    assert _infer_orientation(0.6, current_mode="side") == "front"


def test_infer_orientation_ambiguous_zone_keeps_current_mode():
    assert _infer_orientation(0.3, current_mode="side") == "side"
    assert _infer_orientation(0.3, current_mode="front") == "front"


def test_run_unified_locks_to_side_mode_and_emits_no_flags_for_correct_sitting(tmp_path):
    # data/train/CorrectSitting/image_0.jpg measures shoulder_span/torso_length
    # ratio ~0.3375 -- inside the ambiguous hysteresis band, not clearly below
    # SIDE_VIEW_RATIO_THRESHOLD. So this test verifies the hysteresis default
    # (current_mode starts as "side" and an ambiguous ratio keeps it there),
    # not that the image is unambiguously classified as side view on its own.
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=5, random_state=0)
    X = [[0, 0, 0.1, 0.9, 175, 0.95, 0.4, 0.95]] * 4
    y = ["CorrectSitting"] * 4
    model.fit(X, y)
    model_path = tmp_path / "model.joblib"
    joblib.dump({"model": model, "scale_reference": 100.0, "feature_order": None}, model_path)

    flags = []
    run(_fake_frame_source(35), str(model_path), on_flag=lambda ts, label: flags.append((ts, label)))

    assert flags == []
