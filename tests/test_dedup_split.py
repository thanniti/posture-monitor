from posture_monitor.training.dedup_split import deduplicate, honest_split, process_class_rows


def make_row(frame_index, neck=0.0, nose_shoulder=0.0, **overrides):
    row = {
        "frame_index": frame_index,
        "neck_angle_deg": neck,
        "nose_shoulder_angle_deg": nose_shoulder,
    }
    row.update(overrides)
    return row


def test_deduplicate_drops_near_identical_consecutive_frames():
    rows = [
        make_row(0, neck=0.0),
        make_row(1, neck=0.1),   # near-duplicate of previous, should be dropped
        make_row(2, neck=0.2),   # near-duplicate of kept[-1]=row0 (distance 0.2), still dropped at threshold 2.0
        make_row(3, neck=10.0),  # clearly different, kept
    ]
    kept = deduplicate(rows, threshold=2.0)
    assert [r["frame_index"] for r in kept] == [0, 3]


def test_honest_split_is_chronological_not_random():
    rows = [make_row(i) for i in range(10)]
    train_rows, test_rows = honest_split(rows, train_fraction=0.8)
    assert [r["frame_index"] for r in train_rows] == list(range(8))
    assert [r["frame_index"] for r in test_rows] == [8, 9]


def test_process_class_rows_pools_then_dedups_then_splits():
    # Simulates original train/test folders both drawn from one clip, frame indices
    # interleaved across the two folders (the leakage pattern found in the real data).
    original_train = [make_row(0, neck=0.0), make_row(2, neck=0.2), make_row(4, neck=20.0)]
    original_test = [make_row(1, neck=0.1), make_row(3, neck=0.3)]

    train_rows, test_rows = process_class_rows(
        original_train + original_test, dedup_threshold=2.0, train_fraction=0.5
    )

    all_kept_indices = [r["frame_index"] for r in train_rows + test_rows]
    assert all_kept_indices == sorted(all_kept_indices)  # chronological order preserved
    assert 4 in all_kept_indices  # the clearly-different frame must survive dedup
