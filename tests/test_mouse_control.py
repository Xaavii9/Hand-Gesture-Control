import unittest
from types import SimpleNamespace

from mouse_control import (
    LEFT_BUTTON_DOWN,
    LEFT_BUTTON_UP,
    MouseController,
    is_pinch,
    map_camera_to_screen,
)


class FakeUser32:
    def __init__(self, width=1366, height=768):
        self.width = width
        self.height = height
        self.cursor_positions = []
        self.mouse_events = []

    def GetSystemMetrics(self, index):
        return self.width if index == 0 else self.height

    def SetCursorPos(self, x, y):
        self.cursor_positions.append((x, y))
        return 1

    def mouse_event(self, *arguments):
        self.mouse_events.append(arguments)


def make_hand(thumb_x, thumb_y, index_x, index_y):
    landmarks = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    landmarks[0] = SimpleNamespace(x=0.5, y=0.9, z=0.0)
    landmarks[9] = SimpleNamespace(x=0.5, y=0.5, z=0.0)
    landmarks[4] = SimpleNamespace(x=thumb_x, y=thumb_y, z=0.0)
    landmarks[8] = SimpleNamespace(x=index_x, y=index_y, z=0.0)
    return landmarks


class MouseControlTests(unittest.TestCase):
    def test_camera_coordinates_map_to_screen_edges(self):
        self.assertEqual(map_camera_to_screen(0.0, 0.0, 1366, 768), (0, 0))
        self.assertEqual(map_camera_to_screen(1.0, 1.0, 1366, 768), (1365, 767))

    def test_invalid_screen_size_is_rejected(self):
        with self.assertRaises(ValueError):
            map_camera_to_screen(0.5, 0.5, 0, 768)

    def test_sensitivity_amplifies_small_hand_motion(self):
        default = map_camera_to_screen(0.55, 0.5, 1366, 768, sensitivity=1.0)[0]
        faster = map_camera_to_screen(0.55, 0.5, 1366, 768, sensitivity=1.7)[0]
        self.assertGreater(faster - 682, default - 682)

    def test_mouse_moves_and_pinch_clicks_once_until_released(self):
        user32 = FakeUser32()
        mouse = MouseController(user32=user32)
        first = mouse.update(0.5, 0.5, pinching=False, timestamp=1.0)
        second = mouse.update(0.6, 0.6, pinching=True, timestamp=2.0)
        third = mouse.update(0.6, 0.6, pinching=True, timestamp=3.0)
        mouse.update(0.6, 0.6, pinching=False, timestamp=3.1)

        self.assertFalse(first[2])
        self.assertTrue(second[2])
        self.assertFalse(third[2])
        self.assertEqual(
            [event[0] for event in user32.mouse_events],
            [LEFT_BUTTON_DOWN, LEFT_BUTTON_UP],
        )
        self.assertEqual(len(user32.cursor_positions), 4)
        self.assertFalse(mouse._button_down)

    def test_click_cooldown_and_release_rearm(self):
        user32 = FakeUser32()
        mouse = MouseController(user32=user32, click_cooldown=0.5)
        self.assertTrue(mouse.update(0.5, 0.5, True, timestamp=1.0)[2])
        mouse.update(0.5, 0.5, False, timestamp=1.1)
        self.assertFalse(mouse.update(0.5, 0.5, True, timestamp=1.2)[2])
        mouse.update(0.5, 0.5, False, timestamp=1.3)
        self.assertTrue(mouse.update(0.5, 0.5, True, timestamp=1.6)[2])
        self.assertEqual(
            [event[0] for event in user32.mouse_events],
            [LEFT_BUTTON_DOWN, LEFT_BUTTON_UP, LEFT_BUTTON_DOWN],
        )
        mouse.reset_pinch()
        self.assertEqual(user32.mouse_events[-1][0], LEFT_BUTTON_UP)

    def test_reset_releases_button_if_hand_disappears(self):
        user32 = FakeUser32()
        mouse = MouseController(user32=user32)
        mouse.update(0.5, 0.5, pinching=True, timestamp=1.0)
        mouse.reset_pinch()
        self.assertEqual(
            [event[0] for event in user32.mouse_events],
            [LEFT_BUTTON_DOWN, LEFT_BUTTON_UP],
        )

    def test_pinching_uses_distance_relative_to_palm(self):
        self.assertTrue(is_pinch(make_hand(0.50, 0.50, 0.52, 0.50)))
        self.assertFalse(is_pinch(make_hand(0.35, 0.50, 0.60, 0.30)))


if __name__ == "__main__":
    unittest.main()
