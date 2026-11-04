import unittest
from collections import deque
import os
import tempfile
from pathlib import Path

from main import update_rock_state
from main import MODEL_PATH
from screen_library import ScreenLibrary


class RockStateTests(unittest.TestCase):
    def test_rock_requires_stable_frames_and_fires_once(self):
        history = deque(maxlen=5)
        active = False
        events = []
        for candidate in (True, True, True, True):
            active, started = update_rock_state(candidate, active, history)
            events.append(started)
        self.assertEqual(events, [False, False, True, False])

    def test_rock_can_fire_again_after_hand_releases_gesture(self):
        history = deque(maxlen=5)
        active = False
        events = []
        for candidate in (True, True, True):
            active, _ = update_rock_state(candidate, active, history)
        for candidate in (False, False, False):
            active, _ = update_rock_state(candidate, active, history)
        for candidate in (True, True, True, True):
            active, started = update_rock_state(candidate, active, history)
            events.append(started)
        self.assertEqual(events, [False, False, True, False])

    def test_assets_resolve_from_project_not_current_directory(self):
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary_directory:
            try:
                os.chdir(temporary_directory)
                project_root = Path(__file__).resolve().parents[1]
                self.assertEqual(MODEL_PATH, project_root / "hand_landmarker.task")
                self.assertTrue(MODEL_PATH.is_file())
                library = ScreenLibrary(project_root / "pantalla")
                self.assertEqual(library.root, project_root / "pantalla")
            finally:
                os.chdir(original_directory)


if __name__ == "__main__":
    unittest.main()
