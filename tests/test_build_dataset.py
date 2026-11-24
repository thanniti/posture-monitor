import shutil
from pathlib import Path

from posture_monitor.training.build_dataset import CLASSES, build_feature_table


def test_build_feature_table_on_small_sample(tmp_path):
    data_dir = tmp_path / "data"
    for split in ["train", "test"]:
        for cls in CLASSES:
            dest = data_dir / split / cls
            dest.mkdir(parents=True)
            src_dir = Path("data") / split / cls
            for src_file in sorted(src_dir.glob("image_*.jpg"))[:2]:
                shutil.copy(src_file, dest / src_file.name)

    output_csv = tmp_path / "features.csv"
    rows = build_feature_table(str(data_dir), str(output_csv))

    assert output_csv.exists()
    assert len(rows) == 2 * 2 * len(CLASSES)  # 2 splits * 2 files * N classes
    first_row = rows[0]
    for key in ("split", "class", "filename", "frame_index", "torso_length_px", "neck_angle_deg"):
        assert key in first_row
