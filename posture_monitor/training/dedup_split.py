import math

DEDUP_FEATURES = ["neck_angle_deg", "nose_shoulder_angle_deg"]


def _angle_distance(row_a, row_b):
    return math.sqrt(sum((row_a[f] - row_b[f]) ** 2 for f in DEDUP_FEATURES))


def deduplicate(rows_sorted_by_frame_index, threshold):
    """Drops frames whose angle features are within `threshold` (degrees, combined
    Euclidean distance) of the most recently KEPT frame. Input must already be
    sorted by frame_index (chronological order)."""
    kept = []
    for row in rows_sorted_by_frame_index:
        if not kept or _angle_distance(kept[-1], row) >= threshold:
            kept.append(row)
    return kept


def honest_split(rows_sorted_by_frame_index, train_fraction):
    cutoff = int(len(rows_sorted_by_frame_index) * train_fraction)
    return rows_sorted_by_frame_index[:cutoff], rows_sorted_by_frame_index[cutoff:]


def process_class_rows(all_rows_for_class, dedup_threshold, train_fraction):
    """Pools rows regardless of their original train/test folder (the original split
    is a leaky shuffle of one clip, see design doc), re-sorts chronologically,
    de-duplicates, then re-splits by time position."""
    pooled = sorted(all_rows_for_class, key=lambda r: r["frame_index"])
    deduped = deduplicate(pooled, dedup_threshold)
    return honest_split(deduped, train_fraction)


if __name__ == "__main__":
    import sys
    from collections import defaultdict

    import pandas as pd

    from posture_monitor.config import DEDUP_ANGLE_DISTANCE_THRESHOLD, HONEST_SPLIT_TRAIN_FRACTION

    input_csv = sys.argv[1] if len(sys.argv) > 1 else "posture_monitor/training/features.csv"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "posture_monitor/training/features_honest.csv"

    df = pd.read_csv(input_csv)
    by_class = defaultdict(list)
    for row in df.to_dict("records"):
        by_class[row["class"]].append(row)

    all_train, all_test = [], []
    for cls, rows in by_class.items():
        train_rows, test_rows = process_class_rows(
            rows, DEDUP_ANGLE_DISTANCE_THRESHOLD, HONEST_SPLIT_TRAIN_FRACTION
        )
        print(f"{cls}: {len(rows)} raw -> {len(train_rows) + len(test_rows)} after dedup "
              f"({len(train_rows)} train / {len(test_rows)} test)")
        all_train.extend(train_rows)
        all_test.extend(test_rows)

    for row in all_train:
        row["honest_split"] = "train"
    for row in all_test:
        row["honest_split"] = "test"

    pd.DataFrame(all_train + all_test).to_csv(output_csv, index=False)
    print(f"Wrote {len(all_train) + len(all_test)} rows to {output_csv}")
