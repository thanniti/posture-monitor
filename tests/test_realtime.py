import cv2
import joblib
from sklearn.ensemble import RandomForestClassifier

from posture_monitor.inference.realtime import run


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
    run(_fake_frame_source(35), model_path, on_flag=lambda ts, label: flags.append((ts, label)))

    assert flags == []
