import ctypes
import time


LEFT_BUTTON_DOWN = 0x0002
LEFT_BUTTON_UP = 0x0004


def map_camera_to_screen(
    x,
    y,
    screen_width,
    screen_height,
    margin=0.05,
    sensitivity=1.0,
):
    if screen_width < 1 or screen_height < 1:
        raise ValueError("La resolucion de pantalla debe ser mayor que cero.")
    x = min(1.0, max(0.0, 0.5 + (x - 0.5) * sensitivity))
    y = min(1.0, max(0.0, 0.5 + (y - 0.5) * sensitivity))
    x = min(1.0 - margin, max(margin, x))
    y = min(1.0 - margin, max(margin, y))
    usable_width = screen_width - 1
    usable_height = screen_height - 1
    screen_x = round((x - margin) / (1.0 - 2 * margin) * usable_width)
    screen_y = round((y - margin) / (1.0 - 2 * margin) * usable_height)
    return screen_x, screen_y


def is_pinch(landmarks, threshold=0.32, aspect_ratio=1.0):
    wrist = landmarks[0]
    middle_base = landmarks[9]
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    if aspect_ratio <= 0:
        raise ValueError("aspect_ratio debe ser mayor que cero.")

    palm_scale = max(
        ((wrist.x - middle_base.x) ** 2
         + ((wrist.y - middle_base.y) / aspect_ratio) ** 2
         + (wrist.z - middle_base.z) ** 2) ** 0.5,
        1e-6,
    )
    fingertip_distance = (
        (thumb_tip.x - index_tip.x) ** 2
        + ((thumb_tip.y - index_tip.y) / aspect_ratio) ** 2
        + (thumb_tip.z - index_tip.z) ** 2
    ) ** 0.5
    return fingertip_distance / palm_scale < threshold


class MouseController:
    def __init__(
        self,
        user32=None,
        smoothing=0.5,
        sensitivity=1.7,
        click_cooldown=0.45,
    ):
        self._user32 = user32 or ctypes.windll.user32
        self._screen_width = self._user32.GetSystemMetrics(0)
        self._screen_height = self._user32.GetSystemMetrics(1)
        if self._screen_width < 1 or self._screen_height < 1:
            raise RuntimeError("No se pudo consultar la resolucion de pantalla.")
        self._smoothing = min(1.0, max(0.01, smoothing))
        self._sensitivity = max(0.1, sensitivity)
        self._click_cooldown = click_cooldown
        self._position = None
        self._pinching = False
        self._button_down = False
        self._last_click = float("-inf")

    def update(self, camera_x, camera_y, pinching, timestamp=None):
        target_x, target_y = map_camera_to_screen(
            camera_x,
            camera_y,
            self._screen_width,
            self._screen_height,
            sensitivity=self._sensitivity,
        )
        if self._position is None:
            self._position = (float(target_x), float(target_y))
        else:
            self._position = (
                self._position[0]
                + self._smoothing * (target_x - self._position[0]),
                self._position[1]
                + self._smoothing * (target_y - self._position[1]),
            )

        cursor_x, cursor_y = (round(value) for value in self._position)
        if not self._user32.SetCursorPos(cursor_x, cursor_y):
            raise OSError("Windows no pudo mover el cursor.")

        now = time.monotonic() if timestamp is None else timestamp
        pressed = pinching and not self._pinching
        if pressed and now - self._last_click < self._click_cooldown:
            pressed = False
        if pressed:
            self._press_left()
            self._last_click = now
        elif not pinching and self._button_down:
            self._release_left()

        self._pinching = pinching
        return cursor_x, cursor_y, pressed

    def _press_left(self):
        self._user32.mouse_event(LEFT_BUTTON_DOWN, 0, 0, 0, 0)
        self._button_down = True

    def _release_left(self):
        self._user32.mouse_event(LEFT_BUTTON_UP, 0, 0, 0, 0)
        self._button_down = False

    def reset_pinch(self):
        if self._button_down:
            self._release_left()
        self._pinching = False
