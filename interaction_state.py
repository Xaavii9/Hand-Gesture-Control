from dataclasses import dataclass
import math

from mouse_control import is_pinch
from screen_library import (
    FILE_TILE_CENTER_X,
    FILE_TILE_HEIGHT,
    FILE_TILE_START_Y,
    FILE_TILE_STEP_Y,
    FILE_TILE_WIDTH,
    ScreenLibrary,
)


@dataclass
class FloatingObject:
    x: float = 0.68
    y: float = 0.52
    width: float = 0.54
    height: float = 0.72
    scale: float = 1.0
    rotation_x: float = 0.0
    rotation_y: float = 0.0
    rotation_z: float = 0.0
    visible: bool = False
    title: str = ""
    content: str = ""
    relative_path: str = ""
    kind: str = "text"
    image_rgb: object | None = None


@dataclass(frozen=True)
class InteractionFrame:
    mode: str
    status: str
    reset: bool = False
    dismissed: bool = False
    attracted: bool = False
    double_clap: bool = False
    panel: FloatingObject | None = None
    files: tuple = ()
    selected_file: str | None = None
    entries: tuple = ()
    entry_positions: tuple = ()
    current_directory: str = ""
    active_hand_label: str = "Left"


def _distance(first, second):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def _metric_point(landmark, aspect_ratio):
    return landmark.x, landmark.y / aspect_ratio, landmark.z


def _palm_image_scale(hand, aspect_ratio):
    wrist = _metric_point(hand.landmarks[0], aspect_ratio)
    middle_base = _metric_point(hand.landmarks[9], aspect_ratio)
    return _distance(wrist, middle_base)


class GestureStateMachine:
    def __init__(
        self,
        screen_library=None,
        active_hand_label="Left",
        attractor_hand_label="Right",
        drag_threshold=0.32,
        clap_distance=1.2,
        clap_cooldown=0.2,
        rotation_hold_seconds=1.0,
        flick_speed=2.0,
    ):
        self.screen_library = screen_library
        self.entries = screen_library.list_entries() if screen_library else ()
        self.files = tuple(entry.relative_path for entry in self.entries)
        self.entry_positions = {}
        self._reset_entry_positions()
        self.active_hand_label = active_hand_label
        self.attractor_hand_label = attractor_hand_label
        self.selected_file = None
        self.selected_entry = None
        self.document = None
        self.panel = FloatingObject()
        self.drag_threshold = drag_threshold
        self.clap_distance = clap_distance
        self.clap_cooldown = clap_cooldown
        self.rotation_hold_seconds = rotation_hold_seconds
        self.flick_speed = flick_speed
        self._two_hand_pinching = False
        self._scale_start_distance = None
        self._scale_start_value = 1.0
        self._active_hand_pinching = False
        self._drag_offset = (0.0, 0.0)
        self._pinch_started_at = 0.0
        self._pinch_start_point = (0.0, 0.0)
        self._last_point = None
        self._last_point_at = None
        self._rotating = False
        self._clap_was_close = False
        self._last_reset = float("-inf")
        self._last_clap_time = float("-inf")
        self._attract_is_pinching = False
        self._attract_first_pinch = None
        self._dragging_entry = None
        self._entry_drag_offset = (0.0, 0.0)
        self._entry_drag_start = None
        self._entry_drag_moved = False

    def reset_screen(self):
        if self.screen_library is not None:
            while self.screen_library.current_directory:
                self.screen_library.enter_directory("..")
            self._refresh_entries()
        self.panel = FloatingObject()
        self.selected_file = None
        self.selected_entry = None
        self.document = None
        self.reset_gesture_state()

    def reset_gesture_state(self):
        self._reset_manipulation()
        self._clap_was_close = False
        self._attract_is_pinching = False
        self._attract_first_pinch = None
        self._dragging_entry = None
        self._entry_drag_start = None
        self._entry_drag_moved = False

    def update(self, hands, frame_size, now):
        width, height = frame_size
        if width < 1 or height < 1:
            raise ValueError("El tamano de frame debe ser mayor que cero.")
        aspect_ratio = width / height
        active_hand = self._hand_by_side(hands, self.active_hand_label.lower())
        other_label = "left" if self.active_hand_label.lower() == "right" else "right"
        other_hand = self._hand_by_side(hands, other_label)
        attractor_hand = self._hand_by_side(hands, self.attractor_hand_label.lower())

        hand_a = active_hand
        hand_b = other_hand
        if (hand_a is None or hand_b is None) and len(hands) >= 2:
            hand_a, hand_b = hands[0], hands[1]

        reset, double_clap = self._detect_clap(hand_a, hand_b, aspect_ratio, now)
        if reset:
            self.reset_screen()
            self._last_reset = now
            if double_clap:
                return self._frame(
                    "DOUBLE_CLAP",
                    "2 Palmadas: cancion AC/DC iniciada",
                    reset=True,
                    double_clap=True,
                )
            return self._frame("RESET", "Palmas juntas: interfaz restaurada", reset=True)

        both_hands_pinching = (
            active_hand is not None
            and attractor_hand is not None
            and is_pinch(active_hand.landmarks, aspect_ratio=aspect_ratio)
            and is_pinch(attractor_hand.landmarks, aspect_ratio=aspect_ratio)
        )
        if both_hands_pinching:
            self._attract_is_pinching = False
            self._attract_first_pinch = None
        elif self._detect_attract_double_pinch(attractor_hand, aspect_ratio, now):
            if not self.panel.visible and self.selected_file is not None:
                self._open_selected_file()
            if self.panel.visible:
                self.panel.x = min(0.92, max(0.08, attractor_hand.center[0]))
                self.panel.y = min(0.84, max(0.16, attractor_hand.center[1]))
                self._reset_manipulation()
                return self._frame(
                    "ATTRACT",
                    "Mano abierta - CALL - mano abierta: documento atraido",
                    attracted=True,
                )
            return self._frame(
                "ATTRACT",
                "Selecciona un documento para atraerlo a tu mano",
            )

        if active_hand is None:
            self._reset_manipulation()
            return self._frame(
                "IDLE",
                f"Muestra la mano {self.active_hand_label} de la camara",
            )

        point = (active_hand.landmarks[8].x, active_hand.landmarks[8].y)
        active_pinching = is_pinch(
            active_hand.landmarks,
            threshold=self.drag_threshold,
            aspect_ratio=aspect_ratio,
        )
        other_pinching = other_hand is not None and is_pinch(
            other_hand.landmarks,
            threshold=self.drag_threshold,
            aspect_ratio=aspect_ratio,
        )
        if active_pinching and other_pinching:
            mode = self._update_scale(active_hand, other_hand, aspect_ratio)
            self._active_hand_pinching = False
            self._dragging_entry = None
            self._entry_drag_start = None
            self._entry_drag_moved = False
            self._rotating = False
            self._last_point = None
            self._last_point_at = None
            return self._frame(
                mode,
                f"Zoom bimanual: {round(self.panel.scale * 100)}%"
                if self.panel.visible
                else "No hay documento abierto para escalar",
            )

        self._two_hand_pinching = False
        self._scale_start_distance = None

        tile_index = self._file_index_at(point)
        if self._dragging_entry is not None and not active_pinching:
            if not self._entry_drag_moved:
                result = self._activate_selected_entry()
                if result is not None:
                    self._dragging_entry = None
                    self._active_hand_pinching = False
                    return result
            self._dragging_entry = None
            self._active_hand_pinching = False
            return self._frame("MOVE_ENTRY", f"Posicion actualizada: {self.selected_entry}")

        if self._active_hand_pinching and self._dragging_entry is not None:
            center_x = min(0.94, max(0.06, point[0] + self._entry_drag_offset[0]))
            center_y = min(0.90, max(0.10, point[1] + self._entry_drag_offset[1]))
            self.entry_positions[self._dragging_entry] = (center_x, center_y)
            if self._entry_drag_start is not None:
                self._entry_drag_moved = (
                    _distance(point, self._entry_drag_start) > 0.025
                )
            return self._frame("MOVE_ENTRY", f"Moviendo: {self.selected_entry}")

        if not active_pinching:
            self._active_hand_pinching = False
            self._dragging_entry = None
            self._rotating = False
            self._last_point = None
            self._last_point_at = None
            if tile_index is None:
                return self._frame(
                    "IDLE",
                    "Pinza en un archivo para seleccionarlo; doble pinza para abrir",
                )
            return self._frame("BROWSE", "Acerca la pinza para seleccionar el archivo")

        if tile_index is not None:
            selected = self.entries[tile_index]
            self.selected_entry = selected.relative_path
            self.selected_file = selected.relative_path
            self._active_hand_pinching = True
            self._dragging_entry = selected.relative_path
            self._entry_drag_start = point
            self._entry_drag_moved = False
            self._entry_drag_offset = (
                self.entry_positions[selected.relative_path][0] - point[0],
                self.entry_positions[selected.relative_path][1] - point[1],
            )
            return self._frame("SELECT_ENTRY", f"Seleccionado: {selected.name}")

        if not self._active_hand_pinching:
            if not self.panel.visible or not self._contains(point):
                return self._frame(
                    "AIM",
                    "Selecciona un archivo o acerca la pinza al documento",
                )
            self._active_hand_pinching = True
            self._pinch_started_at = now
            self._pinch_start_point = point
            self._drag_offset = (self.panel.x - point[0], self.panel.y - point[1])
            self._last_point = point
            self._last_point_at = now
            return self._frame(
                "DRAG",
                "Documento seleccionado: arrastra o manten para rotar",
            )

        if (
            not self._rotating
            and now - self._pinch_started_at >= self.rotation_hold_seconds
            and _distance(point, self._pinch_start_point) < 0.035
        ):
            self._rotating = True
            self._last_point = point
            self._last_point_at = now

        if self._rotating:
            if self._last_point is not None:
                delta_x = point[0] - self._last_point[0]
                delta_y = point[1] - self._last_point[1]
                self.panel.rotation_y += delta_x * math.tau
                self.panel.rotation_x += delta_y * math.tau
            self._last_point = point
            self._last_point_at = now
            return self._frame("ROTATE_3D", "Rotacion 3D del documento")

        dismissed = self._is_flick(point, now, aspect_ratio)
        if dismissed:
            self.panel.visible = False
            self._reset_manipulation()
            return self._frame("FLICK_DISMISS", "Flick: documento cerrado", dismissed=True)

        self.panel.x = min(0.9, max(0.1, point[0] + self._drag_offset[0]))
        self.panel.y = min(0.82, max(0.18, point[1] + self._drag_offset[1]))
        self._last_point = point
        self._last_point_at = now
        return self._frame("DRAG", "Arrastrando documento con la pinza")

    def _frame(self, mode, status, reset=False, dismissed=False, attracted=False, double_clap=False):
        return InteractionFrame(
            mode=mode,
            status=status,
            reset=reset,
            dismissed=dismissed,
            attracted=attracted,
            double_clap=double_clap,
            panel=self.panel,
            files=self.files,
            selected_file=self.selected_file,
            entries=self.entries,
            entry_positions=tuple(
                (path, position[0], position[1])
                for path, position in self.entry_positions.items()
            ),
            current_directory=(
                self.screen_library.current_directory
                if self.screen_library is not None
                else ""
            ),
            active_hand_label=self.active_hand_label,
        )

    def _reset_entry_positions(self):
        self.entry_positions = {
            entry.relative_path: (
                FILE_TILE_CENTER_X,
                FILE_TILE_START_Y + index * FILE_TILE_STEP_Y,
            )
            for index, entry in enumerate(getattr(self, "entries", ()))
        }

    def _refresh_entries(self):
        if self.screen_library is None:
            self.entries = ()
        else:
            self.entries = self.screen_library.list_entries()
        self.files = tuple(entry.relative_path for entry in self.entries)
        self._reset_entry_positions()

    def _file_index_at(self, point):
        for index, entry in enumerate(self.entries):
            center_x, center_y = self.entry_positions[entry.relative_path]
            if (
                abs(point[0] - center_x) <= FILE_TILE_WIDTH / 2
                and abs(point[1] - center_y) <= FILE_TILE_HEIGHT / 2
            ):
                return index
        return None

    def _open_selected_file(self):
        if self.screen_library is None or self.selected_file is None:
            return False
        selected = next(
            (entry for entry in self.entries if entry.relative_path == self.selected_file),
            None,
        )
        if selected is not None and selected.kind in ("folder", "parent"):
            return False
        self.document = self.screen_library.read_document(self.selected_file)
        self.selected_entry = self.selected_file
        self.panel = FloatingObject(
            x=0.68,
            y=0.52,
            width=0.54,
            height=0.72,
            visible=True,
            title=self.document.relative_path,
            content=self.document.preview,
            relative_path=self.document.relative_path,
            kind=self.document.kind,
            image_rgb=self.document.image_rgb,
        )
        return True

    def _activate_selected_entry(self):
        if self.selected_entry is None:
            return None
        if self._navigate_selected_folder():
            return self._frame(
                "BROWSE",
                f"Carpeta: {self.screen_library.current_directory or 'pantalla'}",
            )
        if self._open_selected_file():
            return self._frame("OPEN_FILE", f"Abierto: {self.selected_file}")
        return None

    def _navigate_selected_folder(self):
        if self.screen_library is None or self.selected_entry is None:
            return False
        selected = next(
            (entry for entry in self.entries if entry.relative_path == self.selected_entry),
            None,
        )
        if selected is None or selected.kind not in ("folder", "parent"):
            return False
        self.screen_library.enter_directory(selected.relative_path)
        self.selected_entry = None
        self.selected_file = None
        self.document = None
        self.panel = FloatingObject()
        self._refresh_entries()
        self._reset_manipulation()
        return True

    def _detect_attract_double_pinch(self, hand, aspect_ratio, now):
        if hand is None:
            if self._attract_first_pinch is not None and now - self._attract_first_pinch > 1.0:
                self._attract_first_pinch = None
            self._attract_is_pinching = False
            return False

        pinching = is_pinch(hand.landmarks, aspect_ratio=aspect_ratio)
        if not pinching:
            self._attract_is_pinching = False
            return False
        if self._attract_is_pinching:
            return False

        self._attract_is_pinching = True
        if self._attract_first_pinch is not None and now - self._attract_first_pinch <= 0.75:
            self._attract_first_pinch = None
            return True
        self._attract_first_pinch = now
        return False

    @staticmethod
    def _hand_by_side(hands, side):
        return next(
            (hand for hand in hands if hand.handedness.lower() == side),
            None,
        )

    def _contains(self, point):
        half_width = self.panel.width * self.panel.scale / 2
        half_height = self.panel.height * self.panel.scale / 2
        return (
            abs(point[0] - self.panel.x) <= half_width
            and abs(point[1] - self.panel.y) <= half_height
        )

    def _update_scale(self, right_hand, left_hand, aspect_ratio):
        if not self.panel.visible:
            return "IDLE"
        right_center = _metric_point(
            right_hand.landmarks[9], aspect_ratio
        )
        left_center = _metric_point(left_hand.landmarks[9], aspect_ratio)
        current_distance = _distance(right_center, left_center)
        if current_distance <= 1e-6:
            return "SCALE_2_HANDS"
        midpoint = (
            (right_hand.landmarks[9].x + left_hand.landmarks[9].x) / 2,
            (right_hand.landmarks[9].y + left_hand.landmarks[9].y) / 2,
        )
        if not self._two_hand_pinching:
            if not self._contains(midpoint):
                return "AIM"
            self._two_hand_pinching = True
            self._scale_start_distance = current_distance
            self._scale_start_value = self.panel.scale
        if self._scale_start_distance:
            ratio = current_distance / self._scale_start_distance
            self.panel.scale = min(2.2, max(0.55, self._scale_start_value * ratio))
        return "SCALE_2_HANDS"

    def _detect_clap(self, hand_a, hand_b, aspect_ratio, now):
        if hand_a is None or hand_b is None:
            self._clap_was_close = False
            return False, False
        center_a = tuple(
            sum(_metric_point(hand_a.landmarks[index], aspect_ratio)[axis] for index in (0, 5, 9, 13, 17)) / 5
            for axis in range(2)
        )
        center_b = tuple(
            sum(_metric_point(hand_b.landmarks[index], aspect_ratio)[axis] for index in (0, 5, 9, 13, 17)) / 5
            for axis in range(2)
        )
        scale_a = _distance(
            _metric_point(hand_a.landmarks[0], aspect_ratio)[:2],
            _metric_point(hand_a.landmarks[9], aspect_ratio)[:2],
        )
        scale_b = _distance(
            _metric_point(hand_b.landmarks[0], aspect_ratio)[:2],
            _metric_point(hand_b.landmarks[9], aspect_ratio)[:2],
        )
        palm_scale = max((scale_a + scale_b) / 2, 1e-6)
        collision = _distance(center_a, center_b) / palm_scale < self.clap_distance
        triggered = collision and not self._clap_was_close and (now - self._last_reset >= self.clap_cooldown)
        self._clap_was_close = collision

        double_clap = False
        if triggered:
            elapsed = now - self._last_clap_time
            if 0.15 <= elapsed <= 1.2:
                double_clap = True
                self._last_clap_time = float("-inf")
            else:
                self._last_clap_time = now

        return triggered, double_clap

    def _is_flick(self, point, now, aspect_ratio):
        if self._last_point is None or self._last_point_at is None:
            return False
        elapsed = now - self._last_point_at
        if elapsed <= 0 or elapsed > 0.35:
            return False
        metric_point = (point[0], point[1] / aspect_ratio)
        previous = (self._last_point[0], self._last_point[1] / aspect_ratio)
        speed = _distance(metric_point, previous) / elapsed
        delta_x = point[0] - self._last_point[0]
        near_edge = (
            point[0] < 0.05
            or point[0] > 0.95
            or point[1] < 0.05
            or point[1] > 0.95
        )
        rightward_flick = delta_x >= 0.12 and point[0] >= 0.85
        return rightward_flick or (speed >= self.flick_speed and near_edge)

    def _reset_manipulation(self):
        self._two_hand_pinching = False
        self._scale_start_distance = None
        self._active_hand_pinching = False
        self._dragging_entry = None
        self._entry_drag_start = None
        self._entry_drag_moved = False
        self._last_point = None
        self._last_point_at = None
        self._rotating = False
