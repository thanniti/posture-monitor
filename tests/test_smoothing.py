from posture_monitor.inference.smoothing import PostureMonitorState


def test_majority_vote_within_window():
    state = PostureMonitorState(window_seconds=3.0, debounce_seconds=100.0)
    labels = ["CorrectSitting", "LeanForward", "LeanForward"]
    smoothed = None
    for i, label in enumerate(labels):
        smoothed, _ = state.update(timestamp=float(i), label=label)
    assert smoothed == "LeanForward"


def test_old_entries_fall_out_of_window():
    state = PostureMonitorState(window_seconds=2.0, debounce_seconds=100.0)
    state.update(timestamp=0.0, label="LeanForward")
    state.update(timestamp=1.0, label="LeanForward")
    # Jump far ahead: both earlier entries should be evicted, only new one counts.
    smoothed, _ = state.update(timestamp=10.0, label="CorrectSitting")
    assert smoothed == "CorrectSitting"


def test_no_flag_before_debounce_elapses():
    state = PostureMonitorState(window_seconds=1.0, debounce_seconds=5.0)
    _, flagged_early = state.update(timestamp=0.0, label="LeanForward")
    _, flagged_mid = state.update(timestamp=2.0, label="LeanForward")
    assert flagged_early is False
    assert flagged_mid is False


def test_flags_after_debounce_elapses():
    state = PostureMonitorState(window_seconds=1.0, debounce_seconds=5.0)
    state.update(timestamp=0.0, label="LeanForward")
    _, flagged = state.update(timestamp=6.0, label="LeanForward")
    assert flagged is True


def test_returning_to_correct_sitting_resets_debounce_timer():
    state = PostureMonitorState(window_seconds=1.0, debounce_seconds=5.0)
    state.update(timestamp=0.0, label="LeanForward")
    state.update(timestamp=6.0, label="CorrectSitting")
    _, flagged = state.update(timestamp=7.0, label="LeanForward")
    assert flagged is False
