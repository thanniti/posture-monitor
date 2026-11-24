import pytest

from posture_monitor.inference.frontal_classifier import compute_frontal_baseline, classify_frontal


def make_frontal_row(neck_length=20.0, shoulder_width=100.0, head_angle=0.0, hip_knee=175.0, leg_vis=0.95):
    return {
        "neck_length_px": neck_length,
        "shoulder_width_px": shoulder_width,
        "head_shoulder_angle_deg": head_angle,
        "hip_knee_angle_deg": hip_knee,
        "leg_visibility": leg_vis,
    }


def test_compute_frontal_baseline_is_median_per_field():
    rows = [
        make_frontal_row(neck_length=18.0),
        make_frontal_row(neck_length=20.0),
        make_frontal_row(neck_length=22.0),
    ]
    baseline = compute_frontal_baseline(rows)
    assert baseline["neck_length_px"] == pytest.approx(20.0)


def test_classify_frontal_matches_baseline_is_correct_sitting():
    baseline = compute_frontal_baseline([make_frontal_row()])
    live = make_frontal_row()
    assert classify_frontal(live, baseline) == "CorrectSitting"


def test_classify_frontal_shrunk_neck_is_slouching():
    baseline = compute_frontal_baseline([make_frontal_row(neck_length=20.0, shoulder_width=100.0)])
    live = make_frontal_row(neck_length=10.0, shoulder_width=100.0)
    assert classify_frontal(live, baseline) == "Slouching"


def test_classify_frontal_small_neck_change_stays_correct_sitting():
    baseline = compute_frontal_baseline([make_frontal_row(neck_length=20.0, shoulder_width=100.0)])
    live = make_frontal_row(neck_length=18.0, shoulder_width=100.0)
    assert classify_frontal(live, baseline) == "CorrectSitting"


def test_classify_frontal_large_head_angle_deviation_is_slouching():
    baseline = compute_frontal_baseline([make_frontal_row(head_angle=0.0)])
    live = make_frontal_row(head_angle=15.0)
    assert classify_frontal(live, baseline) == "Slouching"


def test_classify_frontal_bent_knee_is_leg_twisting():
    baseline = compute_frontal_baseline([make_frontal_row(hip_knee=175.0)])
    live = make_frontal_row(hip_knee=140.0)
    assert classify_frontal(live, baseline) == "LegTwisting"


def test_classify_frontal_reduced_leg_visibility_is_leg_twisting():
    baseline = compute_frontal_baseline([make_frontal_row(leg_vis=0.95)])
    live = make_frontal_row(leg_vis=0.5)
    assert classify_frontal(live, baseline) == "LegTwisting"
