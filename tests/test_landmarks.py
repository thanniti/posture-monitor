from posture_monitor.config import POSE_MODEL_PATH
from posture_monitor.features.landmarks import create_landmarker, extract_keypoints_from_file


def test_extract_keypoints_from_real_sample_image():
    landmarker = create_landmarker(POSE_MODEL_PATH)
    keypoints = extract_keypoints_from_file(
        landmarker, "data/train/CorrectSitting/image_0.jpg"
    )

    assert keypoints is not None
    expected_names = {
        "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
        "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    }
    assert set(keypoints.keys()) == expected_names

    # Sanity check using known pose in this specific sample image (profile view,
    # sitting upright-ish, writing at a desk): ear should be above (smaller y,
    # since image y increases downward) the shoulder, and visibility should be high
    # for the clearly-visible side of the body.
    ear_y = keypoints["left_ear"][1]
    shoulder_y = keypoints["left_shoulder"][1]
    assert ear_y < shoulder_y

    shoulder_visibility = max(keypoints["left_shoulder"][2], keypoints["right_shoulder"][2])
    assert shoulder_visibility > 0.9
