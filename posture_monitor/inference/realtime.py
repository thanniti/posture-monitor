import time

import cv2
import joblib
import mediapipe as mp

from posture_monitor.calibration.calibrate import compute_scale_reference, feature_vector, normalize_features
from posture_monitor.config import (
    CALIBRATION_SECONDS,
    DEBOUNCE_SECONDS,
    MIN_MEAN_VISIBILITY,
    POSE_MODEL_PATH,
    SMOOTHING_WINDOW_SECONDS,
)
from posture_monitor.features.extraction import extract_features
from posture_monitor.features.landmarks import create_landmarker, extract_keypoints
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


def _status_text_and_color(keypoints_detected, calibrating, remaining_calibration_seconds, label):
    """Pure decision logic for what the preview overlay should say, decoupled from
    any actual drawing/GUI calls so it can be unit tested without a display."""
    if not keypoints_detected:
        return "No pose detected", (0, 0, 255)
    if calibrating:
        return f"Calibrating... {remaining_calibration_seconds:.0f}s", (255, 255, 0)
    color = (0, 255, 0) if label == "CorrectSitting" else (0, 0, 255)
    return label, color


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


def run(frame_source, model_bundle_path, on_flag=print, show_preview=False):
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


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "posture_monitor/models/classifier.joblib"
    try:
        run(webcam_frame_source(), model_path, show_preview=True)
    except KeyboardInterrupt:
        print("\nStopped by user")
