import math


def _euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _angle_from_vertical(bottom_point, top_point):
    """Degrees between the vector bottom->top and straight-up vertical.
    0 = perfectly upright. Positive/negative = leaning one way or the other."""
    dx = top_point[0] - bottom_point[0]
    dy = top_point[1] - bottom_point[1]
    return math.degrees(math.atan2(dx, -dy))


def _joint_angle(vertex, point_a, point_b):
    """Degrees at `vertex` between rays to point_a and point_b. 180 = straight line."""
    v1 = (point_a[0] - vertex[0], point_a[1] - vertex[1])
    v2 = (point_b[0] - vertex[0], point_b[1] - vertex[1])
    mag1, mag2 = math.hypot(*v1), math.hypot(*v2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (mag1 * mag2)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def _pick_side(keypoints, part_name):
    left = keypoints[f"left_{part_name}"]
    right = keypoints[f"right_{part_name}"]
    return left if left[2] >= right[2] else right


def _midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2)


def _angle_from_horizontal(left_point, right_point):
    """Degrees between the vector left->right and horizontal. 0 = perfectly level.
    Positive/negative = tilted one way or the other (e.g. right side lower/higher)."""
    dx = right_point[0] - left_point[0]
    dy = right_point[1] - left_point[1]
    return math.degrees(math.atan2(dy, dx))


def extract_features(keypoints):
    ear = _pick_side(keypoints, "ear")
    shoulder = _pick_side(keypoints, "shoulder")
    hip = _pick_side(keypoints, "hip")
    knee = _pick_side(keypoints, "knee")
    ankle = _pick_side(keypoints, "ankle")
    left_shoulder = keypoints["left_shoulder"]
    right_shoulder = keypoints["right_shoulder"]

    visibilities = [ear[2], shoulder[2], hip[2], knee[2], ankle[2],
                     left_shoulder[2], right_shoulder[2]]

    return {
        "neck_angle_deg": _angle_from_vertical(shoulder, ear),
        "torso_angle_deg": _angle_from_vertical(hip, shoulder),
        "head_forward_offset_px": abs(ear[0] - shoulder[0]),
        "hip_shoulder_vertical_px": abs(hip[1] - shoulder[1]),
        "hip_knee_angle_deg": _joint_angle(knee, hip, ankle),
        "leg_visibility": (knee[2] + ankle[2]) / 2,
        "shoulder_span_px": _euclidean(left_shoulder, right_shoulder),
        "mean_visibility": sum(visibilities) / len(visibilities),
        "torso_length_px": _euclidean(hip, shoulder),
    }


def extract_frontal_features(keypoints):
    """Geometric features for a FRONT-VIEW (facing-camera) pose, as opposed to
    extract_features() which assumes a side-view/profile pose. Facing the camera,
    forward/backward lean barely changes 2D x,y angles (that motion is mostly along
    the camera's depth axis), so instead this measures: how level the ear-line and
    shoulder-line are, the angle from shoulder-midpoint up to ear-midpoint, and the
    neck length (ear-midpoint to shoulder-midpoint distance). Leg features are
    reused unchanged since they're not view-dependent."""
    left_ear = keypoints["left_ear"]
    right_ear = keypoints["right_ear"]
    left_shoulder = keypoints["left_shoulder"]
    right_shoulder = keypoints["right_shoulder"]
    knee = _pick_side(keypoints, "knee")
    ankle = _pick_side(keypoints, "ankle")
    hip = _pick_side(keypoints, "hip")

    ear_mid = _midpoint(left_ear, right_ear)
    shoulder_mid = _midpoint(left_shoulder, right_shoulder)

    visibilities = [left_ear[2], right_ear[2], left_shoulder[2], right_shoulder[2]]

    return {
        "shoulder_tilt_deg": _angle_from_horizontal(left_shoulder, right_shoulder),
        "ear_tilt_deg": _angle_from_horizontal(left_ear, right_ear),
        "head_shoulder_angle_deg": _angle_from_vertical(shoulder_mid, ear_mid),
        "neck_length_px": _euclidean(shoulder_mid, ear_mid),
        "shoulder_width_px": _euclidean(left_shoulder, right_shoulder),
        "hip_knee_angle_deg": _joint_angle(knee, hip, ankle),
        "leg_visibility": (knee[2] + ankle[2]) / 2,
        "mean_visibility": sum(visibilities) / len(visibilities),
    }
