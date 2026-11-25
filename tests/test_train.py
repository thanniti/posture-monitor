import random

import joblib
import pandas as pd

from posture_monitor.training.train import train_and_evaluate


def make_synthetic_csv(path):
    random.seed(0)
    rows = []
    # Two easily-separable "classes" driven by neck_angle_deg, repeated across an
    # honest train/test split, with enough rows for 5-fold CV on the train portion.
    for honest_split, count in [("train", 40), ("test", 10)]:
        for i in range(count):
            cls = "CorrectSitting" if i % 2 == 0 else "LeanForward"
            neck = random.uniform(-2, 2) if cls == "CorrectSitting" else random.uniform(28, 32)
            nose_shoulder = random.uniform(-2, 2) if cls == "CorrectSitting" else random.uniform(28, 32)
            rows.append({
                "class": cls,
                "honest_split": honest_split,
                "neck_angle_deg": neck,
                "nose_shoulder_angle_deg": nose_shoulder,
                "head_forward_offset_px": random.uniform(5, 10),
                "nose_forward_offset_px": random.uniform(5, 10),
                "nose_to_ear_px": random.uniform(95, 105),
                "mean_visibility": random.uniform(0.9, 1.0),
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def test_train_and_evaluate_produces_a_usable_model(tmp_path):
    csv_path = tmp_path / "synthetic_features.csv"
    model_path = tmp_path / "classifier.joblib"
    make_synthetic_csv(csv_path)

    model = train_and_evaluate(str(csv_path), str(model_path))

    assert model_path.exists()
    bundle = joblib.load(model_path)
    assert "model" in bundle and "scale_reference" in bundle and "feature_order" in bundle
    # The two classes are trivially separable by neck_angle_deg / nose_shoulder_angle_deg,
    # so accuracy should be high. Vector order matches NORMALIZED_FEATURE_ORDER:
    # neck_angle_deg, nose_shoulder_angle_deg, head_forward_offset_norm,
    # nose_forward_offset_norm, mean_visibility.
    predictions = bundle["model"].predict([[0.0, 0.0, 0.08, 0.08, 0.95]])
    assert predictions[0] == "CorrectSitting"
