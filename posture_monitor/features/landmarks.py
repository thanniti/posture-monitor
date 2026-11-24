import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

LANDMARK_INDICES = {
    "nose": 0,
    "left_eye": 2, "right_eye": 5,
    "left_ear": 7, "right_ear": 8,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}


def create_landmarker(model_path):
    base_options = BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.PoseLandmarker.create_from_options(options)


def extract_keypoints(landmarker, mp_image):
    """mp_image: a mediapipe.Image. Returns a dict of named keypoints in pixel
    coordinates (x_px, y_px, visibility), or None if no pose was detected."""
    width = mp_image.width
    height = mp_image.height
    result = landmarker.detect(mp_image)
    if not result.pose_landmarks:
        return None
    landmarks = result.pose_landmarks[0]
    return {
        name: (landmarks[idx].x * width, landmarks[idx].y * height, landmarks[idx].visibility)
        for name, idx in LANDMARK_INDICES.items()
    }


def extract_keypoints_from_file(landmarker, image_path):
    mp_image = mp.Image.create_from_file(image_path)
    return extract_keypoints(landmarker, mp_image)
