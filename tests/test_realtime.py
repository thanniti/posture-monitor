import cv2
import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from posture_monitor.inference.realtime import (
    _apply_angle_tolerance,
    _classify_front_frame,
    _classify_side_frame,
    _draw_skeleton,
    _guided_calibration_text,
    _infer_orientation,
    _status_text_and_color,
    run,
    run_front_view,
    run_side_view,
)


# --- _draw_skeleton ---

def _make_test_keypoints():
    return {
        "nose": (50, 10, 1.0),
        "left_eye": (45, 5, 1.0), "right_eye": (55, 5, 1.0),
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


# --- _apply_angle_tolerance ---

def _make_raw(neck=0.0, nose_shoulder=0.0):
    return {"neck_angle_deg": neck, "nose_shoulder_angle_deg": nose_shoulder}


def test_angle_tolerance_overrides_lean_forward_within_tolerance():
    raw = _make_raw(neck=5.0, nose_shoulder=-3.0)
    assert _apply_angle_tolerance(raw, "LeanForward") == "CorrectSitting"


def test_angle_tolerance_overrides_lean_backward_within_tolerance():
    raw = _make_raw(neck=-6.0, nose_shoulder=4.0)
    assert _apply_angle_tolerance(raw, "LeanBackward") == "CorrectSitting"


def test_angle_tolerance_does_not_override_large_lean():
    raw = _make_raw(neck=25.0, nose_shoulder=10.0)
    assert _apply_angle_tolerance(raw, "LeanForward") == "LeanForward"


def test_angle_tolerance_does_not_affect_pose_uncertain():
    raw = _make_raw(neck=0.0, nose_shoulder=0.0)
    assert _apply_angle_tolerance(raw, "PoseUncertain") == "PoseUncertain"


# --- _status_text_and_color ---

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


def test_status_text_includes_mode_prefix():
    text, _ = _status_text_and_color(
        keypoints_detected=True, calibrating=False, remaining_calibration_seconds=0,
        label="CorrectSitting", mode="side",
    )
    assert text == "[Side] CorrectSitting"


def test_status_text_includes_confidence_percentage():
    text, _ = _status_text_and_color(
        keypoints_detected=True, calibrating=False, remaining_calibration_seconds=0,
        label="CorrectSitting", confidence=87.3,
    )
    assert text == "CorrectSitting (87%)"


def test_status_text_includes_both_mode_and_confidence():
    text, _ = _status_text_and_color(
        keypoints_detected=True, calibrating=False, remaining_calibration_seconds=0,
        label="Slouching", mode="front", confidence=62.0,
    )
    assert text == "[Front] Slouching (62%)"


# --- _guided_calibration_text ---

def test_guided_calibration_text_prompts_when_not_matching():
    text, color = _guided_calibration_text("side", matches=False, remaining_seconds=10.0)
    assert text == "Please turn to SIDE view to calibrate"
    assert color == (0, 165, 255)


def test_guided_calibration_text_shows_countdown_when_matching():
    text, color = _guided_calibration_text("front", matches=True, remaining_seconds=7.0)
    assert text == "Calibrating Front view... 7s"
    assert color == (255, 255, 0)


# --- _infer_orientation ---

def test_infer_orientation_below_threshold_is_side():
    assert _infer_orientation(0.5, current_mode="front") == "side"


def test_infer_orientation_above_threshold_is_front():
    assert _infer_orientation(2.0, current_mode="side") == "front"


def test_infer_orientation_ambiguous_zone_keeps_current_mode():
    assert _infer_orientation(1.45, current_mode="side") == "side"
    assert _infer_orientation(1.45, current_mode="front") == "front"


# --- _classify_side_frame / _classify_front_frame (confidence propagation) ---

def _train_tiny_model(labels):
    model = RandomForestClassifier(n_estimators=5, random_state=0)
    X = [[0.0, 0.0, 0.1, 0.1, 0.95]] * len(labels)
    model.fit(X, labels)
    return model


def test_classify_side_frame_returns_label_and_confidence():
    model = _train_tiny_model(["CorrectSitting", "CorrectSitting", "CorrectSitting", "CorrectSitting"])
    raw = {
        "neck_angle_deg": 0.0, "nose_shoulder_angle_deg": 0.0,
        "head_forward_offset_px": 1.0, "nose_forward_offset_px": 1.0,
        "mean_visibility": 0.95,
    }
    label, confidence = _classify_side_frame(raw, model, scale_reference=10.0)
    assert label == "CorrectSitting"
    assert 0.0 <= confidence <= 100.0


def test_classify_side_frame_pose_uncertain_has_zero_confidence():
    model = _train_tiny_model(["CorrectSitting"] * 4)
    raw = {
        "neck_angle_deg": 0.0, "nose_shoulder_angle_deg": 0.0,
        "head_forward_offset_px": 1.0, "nose_forward_offset_px": 1.0,
        "mean_visibility": 0.1,
    }
    label, confidence = _classify_side_frame(raw, model, scale_reference=10.0)
    assert label == "PoseUncertain"
    assert confidence == 0.0


def test_classify_front_frame_pose_uncertain_on_low_interocular_distance():
    raw = {
        "nose_to_shoulder_vertical_px": 20.0, "ear_to_shoulder_vertical_px": 15.0,
        "shoulder_width_px": 100.0, "shoulder_tilt_deg": 0.0,
        "interocular_distance_px": 2.0, "mean_visibility": 0.95,
    }
    baseline = {
        "nose_to_shoulder_vertical_px": 20.0, "ear_to_shoulder_vertical_px": 15.0,
        "shoulder_width_px": 100.0, "shoulder_tilt_deg": 0.0,
        "interocular_distance_px": 40.0,
    }
    label, confidence = _classify_front_frame(raw, baseline)
    assert label == "PoseUncertain"
    assert confidence == 0.0


def test_classify_front_frame_returns_classify_frontal_result_when_not_degenerate():
    raw = {
        "nose_to_shoulder_vertical_px": 20.0, "ear_to_shoulder_vertical_px": 15.0,
        "shoulder_width_px": 100.0, "shoulder_tilt_deg": 0.0,
        "interocular_distance_px": 40.0, "mean_visibility": 0.95,
    }
    baseline = {
        "nose_to_shoulder_vertical_px": 20.0, "ear_to_shoulder_vertical_px": 15.0,
        "shoulder_width_px": 100.0, "shoulder_tilt_deg": 0.0,
        "interocular_distance_px": 40.0,
    }
    label, confidence = _classify_front_frame(raw, baseline)
    assert label == "CorrectSitting"
    assert confidence == pytest.approx(100.0)


# --- End-to-end integration tests (real images, real MediaPipe, real smoothing) ---

def _fake_frame_source(num_frames, start_timestamp=0.0, step_seconds=1.0):
    frame = cv2.imread("data/train/CorrectSitting/image_0.jpg")
    for i in range(num_frames):
        yield start_timestamp + i * step_seconds, frame


def _train_tiny_side_model(label, n_classes_labels):
    # n_classes_labels: list of label strings the model is trained to always predict
    # (repeat the same label n times so RandomForestClassifier has >1 sample).
    model = RandomForestClassifier(n_estimators=5, random_state=0)
    X = [[0.0, 0.0, 0.1, 0.1, 0.95]] * len(n_classes_labels)
    model.fit(X, n_classes_labels)
    return model


def test_run_side_view_completes_calibration_and_emits_no_flags_for_correct_sitting(tmp_path):
    model = _train_tiny_side_model(None, ["CorrectSitting"] * 4)
    model_path = tmp_path / "model.joblib"
    joblib.dump({"model": model, "scale_reference": 100.0, "feature_order": None}, model_path)

    flags = []
    run_side_view(_fake_frame_source(20), str(model_path), on_flag=lambda ts, label: flags.append((ts, label)))

    assert flags == []


def test_run_front_view_completes_calibration_and_emits_no_flags_for_self_consistent_frame():
    # Feeding the SAME real frame throughout both calibration and live phases
    # means the computed baseline exactly matches that frame's own live
    # features (zero deviation), so classify_frontal must always return
    # CorrectSitting -- this verifies the full real pose-estimation +
    # calibration + rule-based classification + smoothing wiring end-to-end.
    flags = []
    run_front_view(_fake_frame_source(20), on_flag=lambda ts, label: flags.append((ts, label)))
    assert flags == []


def _alternating_fake_frame_source(side_frames, front_frames, extra_frames, step_seconds=1.0):
    # data/train/CorrectSitting/image_9.jpg measures shoulder_width/nose_to_ear
    # ratio ~1.20 -- clearly SIDE (below SIDE_VIEW_RATIO_THRESHOLD=1.3).
    # data/train/LegTwisting/image_72.jpg measures ratio ~4.56 -- clearly FRONT
    # (above FRONT_VIEW_RATIO_THRESHOLD=1.6; this is a side-view-labeled
    # training image where the subject was rotated toward the camera, not a
    # genuine frontal capture, but it reliably triggers the "front" detection
    # threshold, which is all this test needs).
    side_image = cv2.imread("data/train/CorrectSitting/image_9.jpg")
    front_image = cv2.imread("data/train/LegTwisting/image_72.jpg")
    timestamp = 0.0
    for _ in range(side_frames):
        yield timestamp, side_image
        timestamp += step_seconds
    for _ in range(front_frames):
        yield timestamp, front_image
        timestamp += step_seconds
    for _ in range(extra_frames):
        yield timestamp, side_image
        timestamp += step_seconds


def test_run_guided_calibration_completes_both_phases_then_classifies(tmp_path):
    model = RandomForestClassifier(n_estimators=5, random_state=0)
    X = [[30.0, 30.0, 0.3, 0.3, 0.95]] * 4
    y = ["LeanForward"] * 4
    model.fit(X, y)
    model_path = tmp_path / "model.joblib"
    joblib.dump({"model": model, "scale_reference": 100.0, "feature_order": None}, model_path)

    flags = []
    run(
        _alternating_fake_frame_source(side_frames=17, front_frames=17, extra_frames=10),
        str(model_path),
        on_flag=lambda ts, label: flags.append((ts, label)),
    )
    assert len(flags) > 0


def test_run_guided_calibration_does_not_overcredit_elapsed_time_across_a_stall(tmp_path):
    model = RandomForestClassifier(n_estimators=5, random_state=0)
    X = [[30.0, 30.0, 0.3, 0.3, 0.95]] * 4
    y = ["LeanForward"] * 4
    model.fit(X, y)
    model_path = tmp_path / "model.joblib"
    joblib.dump({"model": model, "scale_reference": 100.0, "feature_order": None}, model_path)

    side_image = cv2.imread("data/train/CorrectSitting/image_9.jpg")
    front_image = cv2.imread("data/train/LegTwisting/image_72.jpg")
    blank_frame = np.zeros_like(side_image)

    def frame_source_with_stall():
        yield 0.0, side_image
        yield 0.5, side_image
        yield 1.0, blank_frame
        yield 21.0, side_image
        timestamp = 21.5
        for _ in range(20):
            yield timestamp, side_image
            timestamp += 1.0
        for _ in range(17):
            yield timestamp, front_image
            timestamp += 1.0
        for _ in range(10):
            yield timestamp, side_image
            timestamp += 1.0

    flags = []
    run(frame_source_with_stall(), str(model_path), on_flag=lambda ts, label: flags.append((ts, label)))
    assert len(flags) > 0
