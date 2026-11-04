from dataclasses import dataclass
from math import dist


@dataclass(frozen=True)
class GesturePrediction:
    label: str
    confidence: int
    finger_count: int


DISPLAY_NAMES = {
    "THUMBS_UP": "THUMBS UP",
    "THUMBS_DOWN": "THUMBS DOWN",
    "PEACE_2": "PEACE / 2",
    "OK": "OK",
    "PINCH": "PINCH",
    "POINTING_1": "POINTING / 1",
    "CALL": "CALL",
    "CROSSED_FINGERS": "CROSSED FINGERS",
    "ROCK": "ROCK",
    "NUMBER_1": "NUMBER 1",
    "NUMBER_2": "NUMBER 2",
    "NUMBER_3": "NUMBER 3",
    "NUMBER_4": "NUMBER 4",
    "OPEN_HAND_5": "OPEN HAND / 5",
    "FIST": "FIST",
    "UNKNOWN": "UNKNOWN",
}


def _point(landmark, aspect_ratio):
    return landmark.x, landmark.y / aspect_ratio, landmark.z


def _distance(first, second):
    return dist(first, second)


def _state_margin(ratio, threshold, expected_extended):
    separation = abs(ratio - threshold)
    margin = min(1.0, separation / 0.8)
    return margin if (ratio > threshold) == expected_extended else 0.0


def _confidence(ratios, states, extra_margin=1.0):
    margins = [
        _state_margin(ratio, threshold, state)
        for ratio, state, threshold in zip(
            ratios,
            states,
            (1.25, 1.15, 1.15, 1.15, 1.15),
        )
    ]
    margins.append(max(0.0, min(1.0, extra_margin)))
    return max(50, min(99, round(70 + 29 * sum(margins) / len(margins))))


def _prediction(label, ratios, states, finger_count, extra_margin=1.0):
    return GesturePrediction(
        label=label,
        confidence=_confidence(ratios, states, extra_margin),
        finger_count=finger_count,
    )


def classify_gesture(landmarks, aspect_ratio=1.0):
    if len(landmarks) != 21:
        raise ValueError("Se necesitan exactamente 21 landmarks para clasificar una mano.")
    if aspect_ratio <= 0:
        raise ValueError("aspect_ratio debe ser mayor que cero.")

    points = [_point(landmark, aspect_ratio) for landmark in landmarks]
    wrist = points[0]
    palm_scale = max(_distance(wrist, points[9]), 1e-6)

    thumb_to_mcp = tuple(points[2][axis] - points[3][axis] for axis in range(3))
    thumb_to_tip = tuple(points[4][axis] - points[3][axis] for axis in range(3))
    thumb_dot = sum(
        thumb_to_mcp[axis] * thumb_to_tip[axis] for axis in range(3)
    )
    thumb_lengths = max(
        _distance(points[2], points[3]) * _distance(points[4], points[3]),
        1e-6,
    )
    thumb_ratio = 1.0 - thumb_dot / thumb_lengths
    finger_ratios = [
        _distance(points[tip], wrist) / max(_distance(points[pip], wrist), 1e-6)
        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))
    ]
    ratios = [thumb_ratio, *finger_ratios]
    states = [
        thumb_ratio > 1.25,
        *(ratio > 1.15 for ratio in finger_ratios),
    ]
    thumb, index, middle, ring, pinky = states
    finger_count = sum(states)
    close_threshold = 0.3
    pinch_distance = _distance(points[4], points[8]) / palm_scale
    pinch_margin = (close_threshold - pinch_distance) / close_threshold

    base_order = points[9][0] - points[5][0]
    tip_order = points[12][0] - points[8][0]
    crossed = (
        index
        and middle
        and not ring
        and not pinky
        and not thumb
        and base_order * tip_order < 0
        and _distance(points[8], points[12]) / palm_scale < 0.8
    )

    if crossed:
        return _prediction("CROSSED_FINGERS", ratios, states, finger_count)

    if pinch_distance < close_threshold:
        extended_other_fingers = sum((middle, ring, pinky))
        label = "OK" if extended_other_fingers >= 2 else "PINCH"
        return _prediction(
            label,
            ratios,
            states,
            finger_count,
            extra_margin=pinch_margin,
        )

    if thumb and pinky and not index and not middle and not ring:
        return _prediction("CALL", ratios, states, finger_count)

    if index and pinky and not middle and not ring:
        return _prediction("ROCK", ratios, states, finger_count)

    if thumb and not index and not middle and not ring and not pinky:
        vertical_delta = (points[4][1] - points[2][1]) / palm_scale
        if vertical_delta < -0.12:
            return _prediction("THUMBS_UP", ratios, states, finger_count)
        if vertical_delta > 0.12:
            return _prediction("THUMBS_DOWN", ratios, states, finger_count)

    if index and middle and not ring and not pinky and not thumb:
        return _prediction("PEACE_2", ratios, states, finger_count)

    if index and not middle and not ring and not pinky and not thumb:
        return _prediction("POINTING_1", ratios, states, finger_count)

    if finger_count == 0:
        return _prediction("FIST", ratios, states, finger_count)
    if finger_count == 1:
        return _prediction("NUMBER_1", ratios, states, finger_count)
    if finger_count == 2:
        return _prediction("NUMBER_2", ratios, states, finger_count)
    if finger_count == 3:
        return _prediction("NUMBER_3", ratios, states, finger_count)
    if finger_count == 4:
        return _prediction("NUMBER_4", ratios, states, finger_count)
    if finger_count == 5:
        return _prediction("OPEN_HAND_5", ratios, states, finger_count)
    return _prediction("UNKNOWN", ratios, states, finger_count)


def normalize_landmarks(landmarks, aspect_ratio=1.0):
    if len(landmarks) != 21:
        raise ValueError("Se necesitan exactamente 21 landmarks para normalizar una mano.")
    if aspect_ratio <= 0:
        raise ValueError("aspect_ratio debe ser mayor que cero.")

    points = [_point(landmark, aspect_ratio) for landmark in landmarks]
    wrist = points[0]
    scale = max(_distance(wrist, points[9]), 1e-6)
    return tuple(
        round((coordinate - wrist[axis]) / scale, 6)
        for point in points
        for axis, coordinate in enumerate(point)
    )
