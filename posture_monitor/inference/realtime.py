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


def run(frame_source, model_bundle_path, on_flag=print):
    bundle = joblib.load(model_bundle_path)
    model = bundle["model"]
    landmarker = create_landmarker(POSE_MODEL_PATH)
    state = PostureMonitorState(SMOOTHING_WINDOW_SECONDS, DEBOUNCE_SECONDS)

    calibration_rows = []
    calibration_start = None
    scale_reference = None

    for timestamp, frame_bgr in frame_source:
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        keypoints = extract_keypoints(landmarker, mp_image)
        if keypoints is None:
            continue
        raw = extract_features(keypoints)

        if scale_reference is None:
            if calibration_start is None:
                calibration_start = timestamp
            calibration_rows.append(raw)
            if timestamp - calibration_start >= CALIBRATION_SECONDS:
                scale_reference = compute_scale_reference(calibration_rows)
            continue

        if raw["mean_visibility"] < MIN_MEAN_VISIBILITY:
            label = "PoseUncertain"
        else:
            normalized = normalize_features(raw, scale_reference)
            label = model.predict([feature_vector(normalized)])[0]

        smoothed, flagged = state.update(timestamp, label)
        if flagged:
            on_flag(timestamp, smoothed)


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
        run(webcam_frame_source(), model_path)
    except KeyboardInterrupt:
        print("\nStopped by user")
