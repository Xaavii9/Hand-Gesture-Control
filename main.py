from collections import Counter, deque
from dataclasses import replace
from pathlib import Path
import time

import cv2

from audio_detector import AudioClapDetector
from browser_actions import open_clap_song, open_motivation_video
from gestures import GesturePrediction
from hand_tracking import HandTracker
from hud_renderer import HUDRenderer
from interaction_state import GestureStateMachine, InteractionFrame
from mouse_control import MouseController, is_pinch
from screen_library import ScreenLibrary


MODEL_PATH = Path(__file__).resolve().with_name("hand_landmarker.task")


def stabilize_hands(hands, histories):
    stable_hands = []
    active_sides = set()
    for hand in hands:
        side = hand.handedness
        active_sides.add(side)
        history = histories.setdefault(side, deque(maxlen=5))
        history.append(hand.prediction)
        label, count = Counter(item.label for item in history).most_common(1)[0]
        if count >= 3:
            matching = [item for item in history if item.label == label]
            hand = replace(
                hand,
                prediction=GesturePrediction(
                    label,
                    round(sum(item.confidence for item in matching) / len(matching)),
                    matching[-1].finger_count,
                ),
            )
        stable_hands.append(hand)

    for side in set(histories) - active_sides:
        del histories[side]
    return stable_hands


def open_camera():
    for index in (0, 1):
        for backend in (cv2.CAP_ANY, cv2.CAP_DSHOW):
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                return cap
            cap.release()
    return None


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH.name}")

    library = ScreenLibrary(Path(__file__).resolve().parent / "pantalla")
    machine = GestureStateMachine(
        screen_library=library,
        active_hand_label="Left",
        attractor_hand_label="Right",
    )
    renderer = HUDRenderer()
    histories = {}
    rock_history = deque(maxlen=5)
    mouse_mode = False
    mouse = None
    last_rock_action = float("-inf")
    rock_was_active = False
    timestamp_ms = 0

    audio_detector = AudioClapDetector()
    audio_detector.start()

    camera = open_camera()
    if camera is None:
        audio_detector.stop()
        raise RuntimeError(
            "Could not open any camera. Ensure your camera is connected, "
            "not being used by another application (Zoom, Chrome, etc.), and camera permissions are granted."
        )

    try:
        with HandTracker(MODEL_PATH) as tracker:
            while True:
                success, frame = camera.read()
                if not success:
                    print("Could not read a camera frame.")
                    break

                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp_ms = max(timestamp_ms + 1, int(time.monotonic() * 1000))
                hands = stabilize_hands(tracker.detect(rgb_frame, timestamp_ms), histories)
                now = time.monotonic()
                navigation_hand = next(
                    (hand for hand in hands if hand.handedness.lower() == "left"),
                    None,
                )

                rock_candidate = (
                    navigation_hand is not None
                    and navigation_hand.prediction.label == "ROCK"
                    and navigation_hand.prediction.confidence >= 80
                )
                rock_was_active, rock_started = update_rock_state(
                    rock_candidate,
                    rock_was_active,
                    rock_history,
                )
                if rock_started and now - last_rock_action >= 5.0:
                    try:
                        open_motivation_video()
                        last_rock_action = now
                        print("ROCK gesture: motivation video opened")
                    except Exception as error:
                        print(f"Could not open the motivation video: {error}")

                if audio_detector.poll_double_clap():
                    try:
                        open_clap_song()
                        print("AUDIO: 2 palmadas escuchadas por microfono -> Cancion AC/DC abierta")
                    except Exception as error:
                        print(f"Could not open the AC/DC song: {error}")

                if mouse_mode:
                    interaction = InteractionFrame(
                        mode="DESKTOP_MOUSE",
                        status="Index moves pointer; pinch-and-hold drags",
                        panel=machine.panel,
                        files=machine.files,
                        selected_file=machine.selected_file,
                        entries=machine.entries,
                        entry_positions=tuple(
                            (path, position[0], position[1])
                            for path, position in machine.entry_positions.items()
                        ),
                        current_directory=library.current_directory,
                        active_hand_label=machine.active_hand_label,
                    )
                    if navigation_hand is not None:
                        if mouse is None:
                            mouse = MouseController(smoothing=0.62, sensitivity=1.8)
                        mouse.update(
                            navigation_hand.landmarks[8].x,
                            navigation_hand.landmarks[8].y,
                            is_pinch(
                                navigation_hand.landmarks,
                                aspect_ratio=width / height,
                            ),
                        )
                    elif mouse is not None:
                        mouse.reset_pinch()
                else:
                    interaction = machine.update(hands, (width, height), now)

                renderer.draw(
                    frame,
                    hands,
                    interaction,
                    mouse_mode,
                    audio_detector.status,
                )
                cv2.imshow("Hand Gesture Control", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if key in (ord("m"), ord("M")):
                    mouse_mode = not mouse_mode
                    if mouse is not None:
                        mouse.reset_pinch()
                    if mouse_mode and mouse is None:
                        try:
                            mouse = MouseController(smoothing=0.62, sensitivity=1.8)
                        except Exception as error:
                            mouse_mode = False
                            print(f"Desktop mouse control is unavailable: {error}")
    finally:
        audio_detector.stop()
        if mouse is not None:
            mouse.reset_pinch()
        camera.release()
        cv2.destroyAllWindows()


def update_rock_state(candidate, was_active, history):
    history.append(bool(candidate))
    stable = sum(history) >= 3
    return stable, stable and not was_active


if __name__ == "__main__":
    main()
