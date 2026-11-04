import unittest
from types import SimpleNamespace

from gestures import classify_gesture, normalize_landmarks


def make_hand(extended=(), thumb=None, crossed=False, pinch=False, ok=False):
    points = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    points[0] = SimpleNamespace(x=0.5, y=0.9, z=0.0)

    finger_layout = (
        (5, 6, 7, 8, 0.4),
        (9, 10, 11, 12, 0.5),
        (13, 14, 15, 16, 0.6),
        (17, 18, 19, 20, 0.7),
    )
    for finger_index, (mcp, pip, dip, tip, x_position) in enumerate(finger_layout):
        is_extended = finger_index in extended
        points[mcp] = SimpleNamespace(x=x_position, y=0.62, z=0.0)
        points[pip] = SimpleNamespace(x=x_position, y=0.48, z=0.0)
        points[dip] = SimpleNamespace(x=x_position, y=0.36, z=0.0)
        tip_y = 0.2 if is_extended else 0.56
        points[tip] = SimpleNamespace(x=x_position, y=tip_y, z=0.0)

    points[1] = SimpleNamespace(x=0.38, y=0.78, z=0.0)
    points[2] = SimpleNamespace(x=0.34, y=0.7, z=0.0)
    points[3] = SimpleNamespace(x=0.3, y=0.62, z=0.0)
    thumb_tip = (0.43, 0.75)
    if thumb == "up":
        thumb_tip = (0.34, 0.25)
    elif thumb == "down":
        points[1] = SimpleNamespace(x=0.38, y=0.7, z=0.0)
        points[2] = SimpleNamespace(x=0.34, y=0.73, z=0.0)
        points[3] = SimpleNamespace(x=0.34, y=0.81, z=0.0)
        thumb_tip = (0.34, 0.97)
    elif thumb == "call":
        thumb_tip = (0.14, 0.3)
    elif pinch or ok:
        thumb_tip = (points[8].x, points[8].y)
    points[4] = SimpleNamespace(x=thumb_tip[0], y=thumb_tip[1], z=0.0)

    if crossed:
        points[8] = SimpleNamespace(x=0.53, y=0.2, z=0.0)
        points[12] = SimpleNamespace(x=0.47, y=0.2, z=0.0)
    return points


class GestureClassificationTests(unittest.TestCase):
    def assert_gesture(self, expected, **hand_options):
        prediction = classify_gesture(make_hand(**hand_options))
        self.assertEqual(prediction.label, expected)
        self.assertGreaterEqual(prediction.confidence, 50)
        self.assertLessEqual(prediction.confidence, 99)

    def test_thumb_directions(self):
        self.assert_gesture("THUMBS_UP", thumb="up")
        self.assert_gesture("THUMBS_DOWN", thumb="down")

    def test_peace_and_pointing(self):
        self.assert_gesture("PEACE_2", extended=(0, 1))
        self.assert_gesture("POINTING_1", extended=(0,))

    def test_ok_and_pinch(self):
        self.assert_gesture("OK", extended=(1, 2, 3), ok=True)
        self.assert_gesture("PINCH", pinch=True)

    def test_call_and_crossed_fingers(self):
        self.assert_gesture("CALL", extended=(3,), thumb="call")
        self.assert_gesture("CROSSED_FINGERS", extended=(0, 1), crossed=True)

    def test_rock_gesture(self):
        self.assert_gesture("ROCK", extended=(0, 3))

    def test_finger_counts(self):
        self.assert_gesture("NUMBER_3", extended=(0, 1, 2))
        self.assert_gesture("NUMBER_4", extended=(0, 1, 2, 3))
        self.assert_gesture("OPEN_HAND_5", extended=(0, 1, 2, 3), thumb="call")

    def test_fist_and_normalized_features(self):
        hand = make_hand()
        self.assert_gesture("FIST")
        features = normalize_landmarks(hand, aspect_ratio=4 / 3)
        self.assertEqual(len(features), 63)
        self.assertEqual(features[:3], (0.0, 0.0, 0.0))

    def test_rejects_invalid_landmark_count(self):
        with self.assertRaises(ValueError):
            classify_gesture(make_hand()[:-1])


if __name__ == "__main__":
    unittest.main()
