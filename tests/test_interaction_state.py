import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from gestures import GesturePrediction
from interaction_state import GestureStateMachine
from screen_library import FILE_TILE_CENTER_X, FILE_TILE_START_Y, ScreenLibrary


def make_hand(
    side,
    index_x,
    index_y=0.5,
    pinching=True,
    center_x=None,
    prediction_label="FIST",
):
    center_x = index_x if center_x is None else center_x
    landmarks = [SimpleNamespace(x=center_x, y=0.5, z=0.0) for _ in range(21)]
    landmarks[0] = SimpleNamespace(x=center_x, y=0.8, z=0.0)
    landmarks[9] = SimpleNamespace(x=center_x, y=0.5, z=0.0)
    landmarks[8] = SimpleNamespace(x=index_x, y=index_y, z=0.0)
    thumb_x = index_x + (0.01 if pinching else 0.16)
    landmarks[4] = SimpleNamespace(x=thumb_x, y=index_y, z=0.0)

    for mcp, pip, dip in ((5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19)):
        landmarks[mcp] = SimpleNamespace(x=center_x, y=0.65, z=0.0)
        landmarks[pip] = SimpleNamespace(x=center_x, y=0.55, z=0.0)
        landmarks[dip] = SimpleNamespace(x=center_x, y=0.45, z=0.0)

    return SimpleNamespace(
        handedness=side,
        landmarks=landmarks,
        center=(center_x, index_y, 0.0),
        prediction=GesturePrediction(prediction_label, 90, 0),
    )


def show_test_document(machine):
    machine.panel.visible = True
    machine.panel.x = 0.5
    machine.panel.y = 0.5


def open_first_file(machine, now=1.0):
    tile_hand = make_hand("Left", FILE_TILE_CENTER_X, FILE_TILE_START_Y)
    machine.update([tile_hand], (640, 480), now)
    return machine.update(
        [make_hand("Left", FILE_TILE_CENTER_X, FILE_TILE_START_Y, pinching=False)],
        (640, 480),
        now + 0.1,
    )


class GestureStateMachineTests(unittest.TestCase):
    def test_right_hand_pinch_drags_virtual_object(self):
        machine = GestureStateMachine()
        show_test_document(machine)
        first = machine.update([make_hand("Left", 0.5)], (640, 480), 1.0)
        second = machine.update([make_hand("Left", 0.62, 0.56)], (640, 480), 1.1)
        self.assertEqual(first.mode, "DRAG")
        self.assertEqual(second.mode, "DRAG")
        self.assertAlmostEqual(machine.panel.x, 0.62)
        self.assertAlmostEqual(machine.panel.y, 0.56)

    def test_long_static_pinch_then_motion_rotates_object(self):
        machine = GestureStateMachine()
        show_test_document(machine)
        machine.update([make_hand("Left", 0.5)], (640, 480), 1.0)
        held = machine.update([make_hand("Left", 0.5)], (640, 480), 2.1)
        rotated = machine.update([make_hand("Left", 0.56, 0.53)], (640, 480), 2.2)
        self.assertEqual(held.mode, "ROTATE_3D")
        self.assertEqual(rotated.mode, "ROTATE_3D")
        self.assertNotEqual(machine.panel.rotation_y, 0.0)
        self.assertNotEqual(machine.panel.rotation_x, 0.0)

    def test_two_pinches_scale_the_virtual_object(self):
        machine = GestureStateMachine()
        show_test_document(machine)
        pair = [
            make_hand("Right", 0.35, center_x=0.35),
            make_hand("Left", 0.65, center_x=0.65),
        ]
        first = machine.update(pair, (640, 480), 1.0)
        pair = [
            make_hand("Right", 0.25, center_x=0.25),
            make_hand("Left", 0.75, center_x=0.75),
        ]
        second = machine.update(pair, (640, 480), 1.1)
        self.assertEqual(first.mode, "SCALE_2_HANDS")
        self.assertEqual(second.mode, "SCALE_2_HANDS")
        self.assertGreater(machine.panel.scale, 1.0)

    def test_two_hand_collision_resets_panel(self):
        machine = GestureStateMachine()
        machine.panel.scale = 1.8
        hands = [
            make_hand("Right", 0.45, pinching=False, center_x=0.45),
            make_hand("Left", 0.55, pinching=False, center_x=0.55),
        ]
        result = machine.update(hands, (640, 480), 1.0)
        self.assertTrue(result.reset)
        self.assertEqual(result.mode, "RESET")
        self.assertEqual(machine.panel.scale, 1.0)

    def test_double_clap_triggers_double_clap_mode_and_flag(self):
        machine = GestureStateMachine()
        touching_hands = [
            make_hand("Right", 0.48, pinching=False, center_x=0.48),
            make_hand("Left", 0.52, pinching=False, center_x=0.52),
        ]
        separated_hands = [
            make_hand("Right", 0.1, pinching=False, center_x=0.1),
            make_hand("Left", 0.9, pinching=False, center_x=0.9),
        ]
        first = machine.update(touching_hands, (640, 480), 1.0)
        self.assertTrue(first.reset)
        self.assertFalse(first.double_clap)

        machine.update(separated_hands, (640, 480), 1.2)

        second = machine.update(touching_hands, (640, 480), 1.4)
        self.assertTrue(second.reset)
        self.assertTrue(second.double_clap)
        self.assertEqual(second.mode, "DOUBLE_CLAP")

    def test_two_hands_reset_even_if_handedness_is_ambiguous(self):
        machine = GestureStateMachine()
        machine.panel.scale = 1.5
        hands = [
            make_hand("Right", 0.48, pinching=False, center_x=0.48),
            make_hand("Right", 0.52, pinching=False, center_x=0.52),
        ]
        result = machine.update(hands, (640, 480), 1.0)
        self.assertTrue(result.reset)
        self.assertEqual(machine.panel.scale, 1.0)

    def test_fast_flick_to_edge_dismisses_panel(self):
        machine = GestureStateMachine()
        show_test_document(machine)
        machine.update([make_hand("Left", 0.5)], (640, 480), 1.0)
        result = machine.update([make_hand("Left", 0.99)], (640, 480), 1.1)
        self.assertTrue(result.dismissed)
        self.assertFalse(machine.panel.visible)

    def test_camera_left_hand_label_is_the_active_hand(self):
        machine = GestureStateMachine(active_hand_label="Left")
        show_test_document(machine)
        result = machine.update([make_hand("Right", 0.5)], (640, 480), 1.0)
        self.assertEqual(result.mode, "IDLE")

    def test_short_pinch_on_a_file_opens_its_text(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "Aura.txt").write_text("Texto de Aura", encoding="utf-8")
            machine = GestureStateMachine(screen_library=ScreenLibrary(root))
            tile_x = FILE_TILE_CENTER_X
            tile_y = FILE_TILE_START_Y

            selected = machine.update(
                [make_hand("Left", tile_x, tile_y)],
                (640, 480),
                1.0,
            )
            opened = machine.update(
                [make_hand("Left", tile_x, tile_y, pinching=False)],
                (640, 480),
                1.1,
            )

            self.assertEqual(selected.mode, "SELECT_ENTRY")
            self.assertEqual(opened.mode, "OPEN_FILE")
            self.assertTrue(machine.panel.visible)
            self.assertEqual(machine.panel.title, "Aura.txt")
            self.assertEqual(machine.panel.content, "Texto de Aura")

    def test_double_pinch_outside_file_selection_does_not_open_anything(self):
        machine = GestureStateMachine()
        machine.update([make_hand("Left", 0.5)], (640, 480), 1.0)
        result = machine.update([make_hand("Left", 0.5, pinching=False)], (640, 480), 1.1)
        self.assertEqual(result.mode, "IDLE")
        self.assertFalse(machine.panel.visible)

    def test_short_pinch_enters_folder_and_shows_parent_entry(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            folder = root / "Carpeta"
            folder.mkdir()
            (folder / "inside.txt").write_text("Dentro", encoding="utf-8")
            machine = GestureStateMachine(screen_library=ScreenLibrary(root))
            first = make_hand("Left", FILE_TILE_CENTER_X, FILE_TILE_START_Y)
            released = make_hand(
                "Left",
                FILE_TILE_CENTER_X,
                FILE_TILE_START_Y,
                pinching=False,
            )
            machine.update([first], (640, 480), 1.0)
            machine.update([released], (640, 480), 1.1)
            result = machine.update([released], (640, 480), 1.1)

            self.assertEqual(result.mode, "BROWSE")
            self.assertEqual(result.current_directory, "Carpeta")
            self.assertEqual([entry.name for entry in result.entries], ["..", "inside.txt"])

    def test_file_and_folder_tiles_can_be_dragged(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "Carpeta").mkdir()
            machine = GestureStateMachine(screen_library=ScreenLibrary(root))
            machine.update(
                [make_hand("Left", FILE_TILE_CENTER_X, FILE_TILE_START_Y)],
                (640, 480),
                1.0,
            )
            moved = machine.update(
                [make_hand("Left", 0.5, 0.48)],
                (640, 480),
                1.1,
            )
            self.assertEqual(moved.mode, "MOVE_ENTRY")
            self.assertEqual(
                machine.entry_positions["Carpeta"],
                (0.5, 0.48),
            )

    def test_losing_hand_cancels_entry_drag_without_opening_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "Aura.txt").write_text("Texto", encoding="utf-8")
            machine = GestureStateMachine(screen_library=ScreenLibrary(root))
            machine.update(
                [make_hand("Left", FILE_TILE_CENTER_X, FILE_TILE_START_Y)],
                (640, 480),
                1.0,
            )
            machine.update([], (640, 480), 1.1)
            result = machine.update(
                [make_hand("Left", FILE_TILE_CENTER_X, FILE_TILE_START_Y, pinching=False)],
                (640, 480),
                1.2,
            )
            self.assertEqual(result.mode, "BROWSE")
            self.assertFalse(machine.panel.visible)
            self.assertIsNone(machine._dragging_entry)

    def test_opened_file_scales_with_simultaneous_pinches(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "Aura.txt").write_text("Texto de Aura", encoding="utf-8")
            machine = GestureStateMachine(screen_library=ScreenLibrary(root))
            open_first_file(machine)
            first_pair = [
                make_hand("Right", 0.53, center_x=0.53),
                make_hand("Left", 0.83, center_x=0.83),
            ]
            first = machine.update(first_pair, (640, 480), 1.4)
            second_pair = [
                make_hand("Right", 0.40, center_x=0.40),
                make_hand("Left", 0.96, center_x=0.96),
            ]
            second = machine.update(second_pair, (640, 480), 1.5)

            self.assertEqual(first.mode, "SCALE_2_HANDS")
            self.assertEqual(second.mode, "SCALE_2_HANDS")
            self.assertGreater(machine.panel.scale, 1.0)

    def test_double_pinch_with_physical_left_attracts_selected_document(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "Aura.txt").write_text("Texto de Aura", encoding="utf-8")
            machine = GestureStateMachine(screen_library=ScreenLibrary(root))
            opened = open_first_file(machine)
            self.assertEqual(opened.mode, "OPEN_FILE")

            machine.update(
                [make_hand("Right", 0.82, pinching=True)],
                (640, 480),
                1.2,
            )
            machine.update(
                [make_hand("Right", 0.82, pinching=False)],
                (640, 480),
                1.5,
            )
            attracted = machine.update(
                [make_hand("Right", 0.82, pinching=True)],
                (640, 480),
                1.7,
            )

            self.assertTrue(attracted.attracted)
            self.assertEqual(attracted.mode, "ATTRACT")
            self.assertAlmostEqual(machine.panel.x, 0.82)

    def test_attract_gesture_does_not_try_to_open_a_folder(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "Carpeta").mkdir()
            machine = GestureStateMachine(screen_library=ScreenLibrary(root))
            machine.selected_file = "Carpeta"
            machine.selected_entry = "Carpeta"
            machine.update([make_hand("Right", 0.6, pinching=True)], (640, 480), 1.0)
            machine.update([make_hand("Right", 0.6, pinching=False)], (640, 480), 1.2)
            result = machine.update([make_hand("Right", 0.6, pinching=True)], (640, 480), 1.4)
            self.assertEqual(result.mode, "ATTRACT")
            self.assertFalse(result.attracted)

    def test_opening_image_sets_image_texture_on_document(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            Image.new("RGB", (10, 8), (10, 20, 30)).save(root / "barca.jpg")
            machine = GestureStateMachine(screen_library=ScreenLibrary(root))
            result = open_first_file(machine)
            self.assertEqual(result.mode, "OPEN_FILE")
            self.assertEqual(machine.panel.kind, "image")
            self.assertIsInstance(machine.panel.image_rgb, np.ndarray)
            self.assertEqual(machine.panel.image_rgb.shape, (8, 10, 3))


if __name__ == "__main__":
    unittest.main()
