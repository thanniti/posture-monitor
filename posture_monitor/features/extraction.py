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
