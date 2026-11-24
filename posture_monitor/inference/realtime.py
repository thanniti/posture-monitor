import time

import cv2
import joblib
import mediapipe as mp

from posture_monitor.calibration.calibrate import compute_scale_reference, feature_vector, normalize_features
from posture_monitor.config import (
    CALIBRATION_SECONDS,
    CORRECT_SITTING_ANGLE_TOLERANCE_DEG,
    DEBOUNCE_SECONDS,
    FRONT_VIEW_RATIO_THRESHOLD,
    MIN_MEAN_VISIBILITY,
    MIN_SHOULDER_WIDTH_PX,
    POSE_MODEL_PATH,
    SIDE_VIEW_RATIO_THRESHOLD,
    SMOOTHING_WINDOW_SECONDS,
)
from posture_monitor.features.extraction import extract_features, extract_frontal_features
from posture_monitor.features.landmarks import create_landmarker, extract_keypoints
from posture_monitor.inference.frontal_classifier import classify_frontal, compute_frontal_baseline
from posture_monitor.inference.smoothing import PostureMonitorState

PREVIEW_WINDOW_NAME = "Posture Monitor"

SKELETON_CONNECTIONS = [
    ("left_ear", "left_shoulder"), ("right_ear", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"), ("right_knee", "right_ankle"),
]


def _draw_skeleton(frame_bgr, keypoints):
    """Draws lines between anatomically connected keypoints, then a dot at every
    tracked keypoint (including nose, which has no line connection). Operates on
    a plain numpy array via cv2 drawing calls only — no GUI/display needed, so
    this is directly unit-testable."""
    for name_a, name_b in SKELETON_CONNECTIONS:
        xa, ya, _ = keypoints[name_a]
        xb, yb, _ = keypoints[name_b]
        cv2.line(frame_bgr, (int(xa), int(ya)), (int(xb), int(yb)), (0, 255, 255), 2)
    for x, y, _ in keypoints.values():
        cv2.circle(frame_bgr, (int(x), int(y)), 5, (255, 255, 255), -1)


def _apply_angle_tolerance(raw, predicted_label):
    """If the model predicted a lean (forward or backward) but both torso and
    neck angles are within CORRECT_SITTING_ANGLE_TOLERANCE_DEG of vertical,
    override to CorrectSitting. Does not affect LegTwisting or PoseUncertain —
    this tolerance is specifically about trunk-lean sensitivity, not leg
    position or pose-tracking confidence."""
    if predicted_label not in ("LeanForward", "LeanBackward"):
        return predicted_label
    within_tolerance = (
        abs(raw["torso_angle_deg"]) <= CORRECT_SITTING_ANGLE_TOLERANCE_DEG
        and abs(raw["neck_angle_deg"]) <= CORRECT_SITTING_ANGLE_TOLERANCE_DEG
    )
    return "CorrectSitting" if within_tolerance else predicted_label


def _suppress_leg_twisting(label):
    """LegTwisting detection was found unreliable (both the trained side-view
    classifier and the rule-based front-view classifier can still internally
    predict it) and is deliberately disabled without retraining — any
    LegTwisting prediction is remapped to CorrectSitting so it's never shown
    or flagged."""
    return "CorrectSitting" if label == "LegTwisting" else label


def _status_text_and_color(keypoints_detected, calibrating, remaining_calibration_seconds, label, mode=None):
    """Pure decision logic for what the preview overlay should say, decoupled from
    any actual drawing/GUI calls so it can be unit tested without a display."""
    if not keypoints_detected:
        return "No pose detected", (0, 0, 255)
    if calibrating:
        return f"Calibrating... {remaining_calibration_seconds:.0f}s", (255, 255, 0)
    text = label if mode is None else f"[{mode.capitalize()}] {label}"
    color = (0, 255, 0) if label == "CorrectSitting" else (0, 0, 255)
    return text, color


def _infer_orientation(ratio, current_mode):
    """Hysteresis-based orientation detection from the shoulder-span/torso-length
    ratio: below SIDE_VIEW_RATIO_THRESHOLD means profile/side view, above
    FRONT_VIEW_RATIO_THRESHOLD means facing the camera; in between is ambiguous
    and keeps whatever mode was already active, to avoid flickering between
    modes on borderline or noisy frames."""
    if ratio < SIDE_VIEW_RATIO_THRESHOLD:
        return "side"
    if ratio > FRONT_VIEW_RATIO_THRESHOLD:
        return "front"
    return current_mode


def _render_preview(frame_bgr, text, color, keypoints=None):
    """Draws the skeleton overlay (if keypoints given) and the status text, then
    shows it in a window. Returns True if the user pressed 'q' or Escape to
    request quitting."""
    annotated = frame_bgr.copy()
    if keypoints is not None:
        _draw_skeleton(annotated, keypoints)
    cv2.putText(annotated, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow(PREVIEW_WINDOW_NAME, annotated)
    key = cv2.waitKey(1) & 0xFF
    return key in (ord("q"), 27)


def run_side_view(frame_source, model_bundle_path, on_flag=print, show_preview=False):
    bundle = joblib.load(model_bundle_path)
    model = bundle["model"]
    landmarker = create_landmarker(POSE_MODEL_PATH)
    state = PostureMonitorState(SMOOTHING_WINDOW_SECONDS, DEBOUNCE_SECONDS)

    calibration_rows = []
    calibration_start = None
    scale_reference = None

    try:
        for timestamp, frame_bgr in frame_source:
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            keypoints = extract_keypoints(landmarker, mp_image)

            if keypoints is None:
                if show_preview:
                    text, color = _status_text_and_color(False, False, 0, None)
                    if _render_preview(frame_bgr, text, color):
                        break
                continue

            raw = extract_features(keypoints)

            if scale_reference is None:
                if calibration_start is None:
                    calibration_start = timestamp
                calibration_rows.append(raw)
                remaining = max(0.0, CALIBRATION_SECONDS - (timestamp - calibration_start))
                if show_preview:
                    text, color = _status_text_and_color(True, True, remaining, None)
                    if _render_preview(frame_bgr, text, color, keypoints):
                        break
                if timestamp - calibration_start >= CALIBRATION_SECONDS:
                    scale_reference = compute_scale_reference(calibration_rows)
                continue

            if raw["mean_visibility"] < MIN_MEAN_VISIBILITY:
                label = "PoseUncertain"
            else:
                normalized = normalize_features(raw, scale_reference)
                label = model.predict([feature_vector(normalized)])[0]
                label = _apply_angle_tolerance(raw, label)
                label = _suppress_leg_twisting(label)

            smoothed, flagged = state.update(timestamp, label)

            if show_preview:
                text, color = _status_text_and_color(True, False, 0, smoothed)
                if _render_preview(frame_bgr, text, color, keypoints):
                    break

            if flagged:
                on_flag(timestamp, smoothed)
    finally:
        if show_preview:
            cv2.destroyAllWindows()


def run_front_view(frame_source, on_flag=print, show_preview=False):
    landmarker = create_landmarker(POSE_MODEL_PATH)
    state = PostureMonitorState(SMOOTHING_WINDOW_SECONDS, DEBOUNCE_SECONDS)

    calibration_rows = []
    calibration_start = None
    baseline = None

    try:
        for timestamp, frame_bgr in frame_source:
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            keypoints = extract_keypoints(landmarker, mp_image)

            if keypoints is None:
                if show_preview:
                    text, color = _status_text_and_color(False, False, 0, None)
                    if _render_preview(frame_bgr, text, color):
                        break
                continue

            raw = extract_frontal_features(keypoints)
            degenerate = (
                raw["mean_visibility"] < MIN_MEAN_VISIBILITY
                or raw["shoulder_width_px"] < MIN_SHOULDER_WIDTH_PX
            )

            if baseline is None:
                if calibration_start is None:
                    calibration_start = timestamp
                if not degenerate:
                    calibration_rows.append(raw)
                remaining = max(0.0, CALIBRATION_SECONDS - (timestamp - calibration_start))
                if show_preview:
                    text, color = _status_text_and_color(True, True, remaining, None)
                    if _render_preview(frame_bgr, text, color, keypoints):
                        break
                if timestamp - calibration_start >= CALIBRATION_SECONDS and calibration_rows:
                    baseline = compute_frontal_baseline(calibration_rows)
                continue

            if degenerate:
                label = "PoseUncertain"
            else:
                label = classify_frontal(raw, baseline)
                label = _suppress_leg_twisting(label)

            smoothed, flagged = state.update(timestamp, label)

            if show_preview:
                text, color = _status_text_and_color(True, False, 0, smoothed)
                if _render_preview(frame_bgr, text, color, keypoints):
                    break

            if flagged:
                on_flag(timestamp, smoothed)
    finally:
        if show_preview:
            cv2.destroyAllWindows()


def run(frame_source, model_bundle_path, on_flag=print, show_preview=False):
    """Automatically detects per-frame whether the subject is in side view or
    front view (see _infer_orientation) and routes to the appropriate
    classifier, with no manual mode selection needed. Both calibrations run
    simultaneously during the same 30s window: side-view rows feed
    compute_scale_reference, front-view rows feed compute_frontal_baseline,
    each only from frames detected as that orientation at the time."""
    bundle = joblib.load(model_bundle_path)
    model = bundle["model"]
    landmarker = create_landmarker(POSE_MODEL_PATH)
    state = PostureMonitorState(SMOOTHING_WINDOW_SECONDS, DEBOUNCE_SECONDS)

    side_calibration_rows = []
    front_calibration_rows = []
    calibration_start = None
    scale_reference = None
    baseline = None
    current_mode = "side"

    try:
        for timestamp, frame_bgr in frame_source:
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            keypoints = extract_keypoints(landmarker, mp_image)

            if keypoints is None:
                if show_preview:
                    text, color = _status_text_and_color(False, False, 0, None)
                    if _render_preview(frame_bgr, text, color):
                        break
                continue

            raw_side = extract_features(keypoints)
            raw_front = extract_frontal_features(keypoints)

            orientation_ratio = (
                raw_side["shoulder_span_px"] / raw_side["torso_length_px"]
                if raw_side["torso_length_px"] > 0 else 0.0
            )
            current_mode = _infer_orientation(orientation_ratio, current_mode)

            if scale_reference is None or baseline is None:
                if calibration_start is None:
                    calibration_start = timestamp
                if current_mode == "side" and raw_side["mean_visibility"] >= MIN_MEAN_VISIBILITY:
                    side_calibration_rows.append(raw_side)
                elif current_mode == "front" and (
                    raw_front["mean_visibility"] >= MIN_MEAN_VISIBILITY
                    and raw_front["shoulder_width_px"] >= MIN_SHOULDER_WIDTH_PX
                ):
                    front_calibration_rows.append(raw_front)
                remaining = max(0.0, CALIBRATION_SECONDS - (timestamp - calibration_start))
                if show_preview:
                    text, color = _status_text_and_color(True, True, remaining, None)
                    if _render_preview(frame_bgr, text, color, keypoints):
                        break
                if timestamp - calibration_start >= CALIBRATION_SECONDS:
                    if scale_reference is None and side_calibration_rows:
                        scale_reference = compute_scale_reference(side_calibration_rows)
                    if baseline is None and front_calibration_rows:
                        baseline = compute_frontal_baseline(front_calibration_rows)
                continue

            if current_mode == "side":
                if raw_side["mean_visibility"] < MIN_MEAN_VISIBILITY:
                    label = "PoseUncertain"
                else:
                    normalized = normalize_features(raw_side, scale_reference)
                    label = model.predict([feature_vector(normalized)])[0]
                    label = _apply_angle_tolerance(raw_side, label)
                    label = _suppress_leg_twisting(label)
            else:
                degenerate = (
                    raw_front["mean_visibility"] < MIN_MEAN_VISIBILITY
                    or raw_front["shoulder_width_px"] < MIN_SHOULDER_WIDTH_PX
                )
                if degenerate:
                    label = "PoseUncertain"
                else:
                    label = classify_frontal(raw_front, baseline)
                    label = _suppress_leg_twisting(label)

            smoothed, flagged = state.update(timestamp, label)

            if show_preview:
                text, color = _status_text_and_color(True, False, 0, smoothed, mode=current_mode)
                if _render_preview(frame_bgr, text, color, keypoints):
                    break

            if flagged:
                on_flag(timestamp, smoothed)
    finally:
        if show_preview:
            cv2.destroyAllWindows()


def webcam_frame_source(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield time.time(), frame
    finally:
        cap.release()


def main(model_bundle_path):
    run(webcam_frame_source(), model_bundle_path, show_preview=True)


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "posture_monitor/models/classifier.joblib"
    try:
        main(model_path)
    except KeyboardInterrupt:
        print("\nStopped by user")
