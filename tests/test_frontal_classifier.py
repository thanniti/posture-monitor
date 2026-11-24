import pytest

from posture_monitor.inference.frontal_classifier import classify_frontal, compute_frontal_baseline


def make_frontal_row(nose_vert=20.0, ear_vert=15.0, shoulder_width=100.0, shoulder_tilt=0.0, interocular=40.0):
    return {
        "nose_to_shoulder_vertical_px": nose_vert,
        "ear_to_shoulder_vertical_px": ear_vert,
        "shoulder_width_px": shoulder_width,
        "shoulder_tilt_deg": shoulder_tilt,
        "interocular_distance_px": interocular,
    }


def test_compute_frontal_baseline_is_median_per_field():
    rows = [make_frontal_row(nose_vert=v) for v in [18.0, 20.0, 22.0]]
    baseline = compute_frontal_baseline(rows)
    assert baseline["nose_to_shoulder_vertical_px"] == pytest.approx(20.0)


def test_classify_frontal_matches_baseline_is_correct_sitting():
    baseline = compute_frontal_baseline([make_frontal_row()])
    live = make_frontal_row()
    label, confidence = classify_frontal(live, baseline)
    assert label == "CorrectSitting"
    assert confidence == pytest.approx(100.0)


def test_classify_frontal_shrunk_neck_is_slouching():
    baseline = compute_frontal_baseline([make_frontal_row(nose_vert=20.0, ear_vert=15.0, interocular=40.0)])
    live = make_frontal_row(nose_vert=10.0, ear_vert=8.0, interocular=40.0)
    label, confidence = classify_frontal(live, baseline)
    assert label == "Slouching"
    assert confidence > 50.0


def test_classify_frontal_small_neck_change_stays_correct_sitting():
    baseline = compute_frontal_baseline([make_frontal_row(nose_vert=20.0, interocular=40.0)])
    live = make_frontal_row(nose_vert=19.0, interocular=40.0)
    label, confidence = classify_frontal(live, baseline)
    assert label == "CorrectSitting"


def test_classify_frontal_narrowed_shoulders_is_slouching():
    baseline = compute_frontal_baseline([make_frontal_row(shoulder_width=100.0, interocular=40.0)])
    live = make_frontal_row(shoulder_width=70.0, interocular=40.0)
    label, confidence = classify_frontal(live, baseline)
    assert label == "Slouching"


def test_classify_frontal_shoulder_tilt_is_lateral_lean():
    baseline = compute_frontal_baseline([make_frontal_row(shoulder_tilt=0.0)])
    live = make_frontal_row(shoulder_tilt=15.0)
    label, confidence = classify_frontal(live, baseline)
    assert label == "LateralLean"


def test_classify_frontal_confidence_is_higher_for_more_severe_deviation():
    baseline = compute_frontal_baseline([make_frontal_row(nose_vert=20.0, interocular=40.0)])
    mild = make_frontal_row(nose_vert=16.0, interocular=40.0)
    severe = make_frontal_row(nose_vert=5.0, interocular=40.0)
    _, mild_confidence = classify_frontal(mild, baseline)
    _, severe_confidence = classify_frontal(severe, baseline)
    assert severe_confidence > mild_confidence
