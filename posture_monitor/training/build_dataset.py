import csv
import re
from pathlib import Path

from posture_monitor.config import POSE_MODEL_PATH
from posture_monitor.features.extraction import extract_features
from posture_monitor.features.landmarks import create_landmarker, extract_keypoints_from_file

CLASSES = ["CorrectSitting", "LeanBackward", "LeanForward", "LegTwisting"]
FRAME_INDEX_RE = re.compile(r"image_(\d+)\.jpg")


def build_feature_table(data_dir, output_csv):
    landmarker = create_landmarker(POSE_MODEL_PATH)
    rows = []

    for split in ["train", "test"]:
        for cls in CLASSES:
            folder = Path(data_dir) / split / cls
            if not folder.exists():
                continue
            for img_path in sorted(folder.glob("image_*.jpg")):
                match = FRAME_INDEX_RE.match(img_path.name)
                if not match:
                    continue
                keypoints = extract_keypoints_from_file(landmarker, str(img_path))
                if keypoints is None:
                    continue
                features = extract_features(keypoints)
                rows.append({
                    "split": split,
                    "class": cls,
                    "filename": img_path.name,
                    "frame_index": int(match.group(1)),
                    **features,
                })

    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return rows


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "posture_monitor/training/features.csv"
    rows = build_feature_table(data_dir, output_csv)
    print(f"Wrote {len(rows)} rows to {output_csv}")
