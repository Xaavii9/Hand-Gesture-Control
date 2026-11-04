import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from screen_library import ScreenLibrary


class ScreenLibraryTests(unittest.TestCase):
    def test_lists_and_reads_text_files_recursively(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "Aura.txt").write_text("Texto de prueba", encoding="utf-8")
            (root / "subcarpeta").mkdir()
            (root / "subcarpeta" / "Partit").write_text("Lletra", encoding="utf-8")
            library = ScreenLibrary(root)

            self.assertEqual(library.list_files(), ("Aura.txt", "subcarpeta/Partit"))
            document = library.read_document("subcarpeta/Partit")
            self.assertTrue(document.is_text)
            self.assertEqual(document.preview, "Lletra")

    def test_rejects_paths_outside_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "pantalla"
            root.mkdir()
            outside = Path(temporary_directory) / "outside.txt"
            outside.write_text("No", encoding="utf-8")
            library = ScreenLibrary(root)
            with self.assertRaises(ValueError):
                library.read_document("../outside.txt")

    def test_binary_files_are_not_shown_as_text(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "imagen.bin").write_bytes(b"\x00\x01\x02")
            document = ScreenLibrary(root).read_document("imagen.bin")
            self.assertFalse(document.is_text)

    def test_preview_is_limited(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "large.txt").write_text("a" * 7000, encoding="utf-8")
            document = ScreenLibrary(root).read_document("large.txt")
            self.assertTrue(document.truncated)
            self.assertIn("Vista previa limitada", document.preview)

    def test_directory_navigation_lists_parent_and_children(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            folder = root / "Carpeta"
            folder.mkdir()
            (folder / "nota.txt").write_text("Dentro", encoding="utf-8")
            library = ScreenLibrary(root)

            self.assertEqual(library.list_entries()[0].name, "Carpeta")
            library.enter_directory("Carpeta")
            self.assertEqual(
                [(entry.name, entry.kind) for entry in library.list_entries()],
                [("..", "parent"), ("nota.txt", "text")],
            )
            library.enter_directory("..")
            self.assertEqual(library.current_directory, "")

    def test_image_preview_returns_rgb_pixels(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            Image.new("RGB", (8, 6), (20, 40, 60)).save(root / "foto.png")
            document = ScreenLibrary(root).read_document("foto.png")
            self.assertEqual(document.kind, "image")
            self.assertIsInstance(document.image_rgb, np.ndarray)
            self.assertEqual(document.image_rgb.shape, (6, 8, 3))
            self.assertTrue(np.all(document.image_rgb == (20, 40, 60)))

    def test_hidden_files_and_gitkeep_are_ignored(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / ".gitkeep").write_text("", encoding="utf-8")
            (root / ".hidden_file").write_text("secret", encoding="utf-8")
            (root / "visible.txt").write_text("hello", encoding="utf-8")
            library = ScreenLibrary(root)
            self.assertEqual(library.list_files(), ("visible.txt",))
            entries = library.list_entries()
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].name, "visible.txt")


if __name__ == "__main__":
    unittest.main()
