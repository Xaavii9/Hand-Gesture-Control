import unittest
from types import SimpleNamespace

from hand_tracking import HandTracker


def make_landmarks():
    return [
        SimpleNamespace(x=0.4 + index * 0.002, y=0.7 - index * 0.01, z=0.0)
        for index in range(21)
    ]


class HandTrackingTests(unittest.TestCase):
    def test_convert_result_keeps_handedness_and_world_points(self):
        landmarks = make_landmarks()
        world_landmarks = [
            SimpleNamespace(x=point.x, y=point.y, z=-0.02 * index)
            for index, point in enumerate(landmarks)
        ]
        result = SimpleNamespace(
            hand_landmarks=[landmarks],
            hand_world_landmarks=[world_landmarks],
            handedness=[[SimpleNamespace(category_name="Right", display_name="", score=0.93)]],
        )

        hands = HandTracker.convert_result(result, width=640, height=480)

        self.assertEqual(len(hands), 1)
        self.assertEqual(hands[0].handedness, "Right")
        self.assertEqual(hands[0].handedness_score, 0.93)
        self.assertEqual(hands[0].world_landmarks, tuple(world_landmarks))
        self.assertEqual(len(hands[0].landmarks), 21)
        self.assertAlmostEqual(hands[0].center[0], sum(landmarks[i].x for i in (0, 5, 9, 13, 17)) / 5)

    def test_missing_category_is_reported_as_unknown(self):
        landmarks = make_landmarks()
        result = SimpleNamespace(
            hand_landmarks=[landmarks],
            hand_world_landmarks=[landmarks],
            handedness=[[]],
        )
        hands = HandTracker.convert_result(result, width=640, height=480)
        self.assertEqual(hands[0].handedness, "Unknown")
        self.assertEqual(hands[0].handedness_score, 0.0)


if __name__ == "__main__":
    unittest.main()
