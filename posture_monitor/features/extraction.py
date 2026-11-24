import math


def _euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _angle_from_vertical(bottom_point, top_point):
    """Degrees between the vector bottom->top and straight-up vertical.
    0 = perfectly upright. Positive/negative = leaning one way or the other."""
    dx = top_point[0] - bottom_point[0]
    dy = top_point[1] - bottom_point[1]
    return math.degrees(math.atan2(dx, -dy))


def _pick_side(keypoints, part_name):
    left = keypoints[f"left_{part_name}"]
    right = keypoints[f"right_{part_name}"]
    return left if left[2] >= right[2] else right


def _facing_sign(nose, ear):
    """+1.0 if the nose sits in the +x direction from the chosen-side ear,
    -1.0 if -x. The nose is always anatomically in front of the ear, toward
    whichever way the subject is facing -- unlike lean direction, this
    relationship doesn't flip with posture, so its sign reliably canonicalizes
    which way the subject happens to be oriented relative to the camera
    (an arbitrary confound on the lean-angle sign that varies by camera/desk
    setup and doesn't generalize from one recording session to another)."""
    return 1.0 if (nose[0] - ear[0]) >= 0 else -1.0


def _midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2)


def _angle_from_horizontal(left_point, right_point):
    """Degrees between the vector left->right and horizontal. 0 = perfectly level.
    Positive/negative = tilted one way or the other (e.g. right side lower/higher)."""
    dx = right_point[0] - left_point[0]
    dy = right_point[1] - left_point[1]
    return math.degrees(math.atan2(dy, dx))


def extract_features(keypoints):
    """Side-view (profile) geometric features. Deliberately upper-body only
    (nose, ears, shoulders) -- no hip/knee/ankle reliance, per project
    requirements. nose_to_ear_px serves as the scale-normalization reference
    (a stable facial dimension, clearly visible in profile and unaffected by
    posture) in place of the old hip-based torso_length_px. neck_angle_deg
    and nose_shoulder_angle_deg are corrected by _facing_sign so "positive"
    consistently means the same physical lean direction regardless of which
    way the subject happens to face relative to their camera."""
    nose = keypoints["nose"]
    ear = _pick_side(keypoints, "ear")
    shoulder = _pick_side(keypoints, "shoulder")

    facing_sign = _facing_sign(nose, ear)
    visibilities = [nose[2], ear[2], shoulder[2]]

    return {
        "neck_angle_deg": facing_sign * _angle_from_vertical(shoulder, ear),
        "nose_shoulder_angle_deg": facing_sign * _angle_from_vertical(shoulder, nose),
        "head_forward_offset_px": abs(ear[0] - shoulder[0]),
        "nose_forward_offset_px": abs(nose[0] - shoulder[0]),
        "nose_to_ear_px": _euclidean(nose, ear),
        "mean_visibility": sum(visibilities) / len(visibilities),
    }


def extract_frontal_features(keypoints):
    """Front-view (facing-camera) geometric features. Deliberately upper-body
    only (nose, eyes, ears, shoulders) -- no hip/knee/ankle reliance.
    interocular_distance_px (eye-to-eye) serves as the scale-normalization
    reference -- a stable facial dimension, posture-invariant, and (unlike
    shoulder width) not itself a slouch signal, since shoulder_width_px is now
    used AS a signal (shoulder protraction/rounding narrows it)."""
    nose = keypoints["nose"]
    left_eye = keypoints["left_eye"]
    right_eye = keypoints["right_eye"]
    left_ear = keypoints["left_ear"]
    right_ear = keypoints["right_ear"]
    left_shoulder = keypoints["left_shoulder"]
    right_shoulder = keypoints["right_shoulder"]

    ear_mid = _midpoint(left_ear, right_ear)
    shoulder_mid = _midpoint(left_shoulder, right_shoulder)

    visibilities = [
        nose[2], left_eye[2], right_eye[2], left_ear[2], right_ear[2],
        left_shoulder[2], right_shoulder[2],
    ]

    return {
        "nose_to_shoulder_vertical_px": abs(nose[1] - shoulder_mid[1]),
        "ear_to_shoulder_vertical_px": abs(ear_mid[1] - shoulder_mid[1]),
        "shoulder_width_px": _euclidean(left_shoulder, right_shoulder),
        "shoulder_tilt_deg": _angle_from_horizontal(left_shoulder, right_shoulder),
        "ear_tilt_deg": _angle_from_horizontal(left_ear, right_ear),
        "interocular_distance_px": _euclidean(left_eye, right_eye),
        "mean_visibility": sum(visibilities) / len(visibilities),
    }
