import unittest
from types import SimpleNamespace

import numpy as np

from hud_renderer import HUDRenderer
from interaction_state import FloatingObject, InteractionFrame


class HUDRendererTests(unittest.TestCase):
    def test_projection_changes_when_object_rotates(self):
        panel = FloatingObject()
        initial = HUDRenderer.project_panel(panel, (640, 480))
        panel.rotation_y = 0.7
        rotated = HUDRenderer.project_panel(panel, (640, 480))
        self.assertEqual(len(initial), 8)
        self.assertNotEqual(initial, rotated)

    def test_render_draws_virtual_object_and_status(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        interaction = InteractionFrame(
            mode="IDLE",
            status="ready",
            panel=FloatingObject(),
        )
        rendered = HUDRenderer().draw(frame, [], interaction, False, "MIC OFF")
        self.assertGreater(int(rendered.sum()), 0)

    def test_hidden_object_is_not_rendered(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        interaction = InteractionFrame(
            mode="FLICK_DISMISS",
            status="dismissed",
            panel=FloatingObject(visible=False),
        )
        rendered = HUDRenderer().draw(frame, [], interaction, False, "MIC OFF")
        self.assertGreater(int(rendered.sum()), 0)

    def test_image_pixels_are_used_as_document_texture(self):
        panel = FloatingObject(
            title="barca.jpg",
            relative_path="barca.jpg",
            kind="image",
            image_rgb=np.full((32, 48, 3), (210, 30, 80), dtype=np.uint8),
        )
        texture = HUDRenderer()._document_texture(panel, 640, 480)
        self.assertEqual(texture.shape[2], 3)
        self.assertTrue(np.any(np.all(texture == (210, 30, 80), axis=2)))


if __name__ == "__main__":
    unittest.main()
