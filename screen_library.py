from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


TEXT_SUFFIXES = {
    ".c", ".cfg", ".cpp", ".css", ".csv", ".html", ".ini", ".java",
    ".js", ".json", ".jsx", ".log", ".md", ".py", ".toml", ".ts",
    ".tsx", ".txt", ".xml", ".yaml", ".yml",
}
MAX_PREVIEW_BYTES = 64 * 1024
MAX_PREVIEW_CHARACTERS = 6000
MAX_IMAGE_BYTES = 24 * 1024 * 1024
MAX_IMAGE_PIXELS = 24_000_000
FILE_TILE_CENTER_X = 0.20
FILE_TILE_START_Y = 0.30
FILE_TILE_STEP_Y = 0.16
FILE_TILE_WIDTH = 0.34
FILE_TILE_HEIGHT = 0.12
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


@dataclass(frozen=True)
class ScreenEntry:
    name: str
    relative_path: str
    kind: str


@dataclass(frozen=True)
class ScreenDocument:
    relative_path: str
    preview: str
    is_text: bool
    truncated: bool
    kind: str = "text"
    image_rgb: object | None = None


class ScreenLibrary:
    def __init__(self, root=None):
        self.root = Path(root or Path(__file__).resolve().parent / "pantalla").resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"No se encuentra la carpeta: {self.root}")
        self.current_directory = ""

    @property
    def current_path(self):
        return (self.root / self.current_directory).resolve()

    def list_entries(self):
        current = self.current_path
        if not current.is_relative_to(self.root):
            raise ValueError("La navegacion debe permanecer dentro de pantalla.")

        entries = []
        if current != self.root:
            entries.append(ScreenEntry("..", "..", "parent"))

        for path in current.iterdir():
            if path.is_symlink() or path.name.startswith("."):
                continue
            relative_path = path.resolve().relative_to(self.root).as_posix()
            if path.is_dir():
                kind = "folder"
            elif path.suffix.lower() in IMAGE_SUFFIXES:
                kind = "image"
            elif path.suffix.lower() in TEXT_SUFFIXES or not path.suffix:
                kind = "text"
            else:
                kind = "binary"
            entries.append(ScreenEntry(path.name, relative_path, kind))

        return tuple(
            sorted(entries, key=lambda entry: (entry.kind not in ("parent", "folder"), entry.name.casefold()))
        )

    def enter_directory(self, relative_path):
        if relative_path == "..":
            if self.current_directory:
                self.current_directory = Path(self.current_directory).parent.as_posix()
                if self.current_directory == ".":
                    self.current_directory = ""
            return

        candidate = (self.root / relative_path).resolve()
        if (
            not candidate.is_relative_to(self.root)
            or candidate.parent != self.current_path
            or not candidate.is_dir()
        ):
            raise ValueError("Solo se puede entrar en una subcarpeta de pantalla.")
        self.current_directory = candidate.relative_to(self.root).as_posix()

    def list_files(self):
        files = []
        for path in self.root.rglob("*"):
            if path.is_symlink() or not path.is_file() or path.name.startswith("."):
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(self.root):
                continue
            files.append(resolved.relative_to(self.root).as_posix())
        return tuple(sorted(files, key=str.casefold))

    def read_document(self, relative_path):
        candidate = (self.root / relative_path).resolve()
        if not candidate.is_relative_to(self.root) or not candidate.is_file():
            raise ValueError("El archivo debe estar dentro de pantalla.")

        if candidate.suffix.lower() in IMAGE_SUFFIXES:
            if candidate.stat().st_size > MAX_IMAGE_BYTES:
                return ScreenDocument(
                    relative_path,
                    "Imagen demasiado grande para la vista previa.",
                    False,
                    True,
                    kind="image",
                )
            with Image.open(candidate) as source:
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    return ScreenDocument(
                        relative_path,
                        "Resolucion demasiado grande para la vista previa.",
                        False,
                        True,
                        kind="image",
                    )
                image = source.convert("RGB")
                image.thumbnail((1600, 1200))
                image_rgb = np.asarray(image).copy()
            return ScreenDocument(
                relative_path,
                "Vista de imagen",
                False,
                False,
                kind="image",
                image_rgb=image_rgb,
            )

        with candidate.open("rb") as source:
            raw = source.read(MAX_PREVIEW_BYTES + 1)
        truncated = len(raw) > MAX_PREVIEW_BYTES
        raw = raw[:MAX_PREVIEW_BYTES]
        if b"\x00" in raw[:4096]:
            return ScreenDocument(
                relative_path,
                "Vista previa no disponible para binarios.",
                False,
                truncated,
                kind="binary",
            )

        suffix = candidate.suffix.lower()
        if suffix not in TEXT_SUFFIXES and suffix:
            return ScreenDocument(
                relative_path,
                "Vista previa no disponible para este formato.",
                False,
                truncated,
                kind="binary",
            )

        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            if suffix:
                return ScreenDocument(
                    relative_path,
                    "El archivo no es texto UTF-8 legible.",
                    False,
                    truncated,
                    kind="binary",
                )
            text = raw.decode("cp1252")

        printable = sum(character.isprintable() or character in "\r\n\t" for character in text)
        if text and printable / len(text) < 0.85:
            return ScreenDocument(
                relative_path,
                "Vista previa no disponible para binarios.",
                False,
                truncated,
                kind="binary",
            )

        if len(text) > MAX_PREVIEW_CHARACTERS:
            text = text[:MAX_PREVIEW_CHARACTERS]
            truncated = True
        if truncated:
            text += "\n\n[Vista previa limitada]"
        return ScreenDocument(relative_path, text or "[Archivo vacio]", True, truncated, kind="text")
