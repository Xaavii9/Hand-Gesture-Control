from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from gestures import GesturePrediction, classify_gesture


HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS


@dataclass(frozen=True)
class HandPose:
    landmarks: tuple
    world_landmarks: tuple
    handedness: str
    handedness_score: float
    prediction: GesturePrediction

    @property
    def center(self):
        selected = (0, 5, 9, 13, 17)
        return tuple(
            sum(getattr(self.landmarks[index], axis) for index in selected)
            / len(selected)
            for axis in ("x", "y", "z")
        )

    @property
    def world_center_z(self):
        selected = (0, 5, 9, 13, 17)
        return sum(self.world_landmarks[index].z for index in selected) / len(selected)


class HandTracker:
    def __init__(self, model_path, max_hands=2):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"No se encuentra el modelo: {model_path}")

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, rgb_frame, timestamp_ms):
        height, width = rgb_frame.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        return self.convert_result(result, width, height)

    @staticmethod
    def convert_result(result, width, height):
        aspect_ratio = width / height
        hands = []
        for index, landmarks in enumerate(result.hand_landmarks):
            categories = result.handedness[index]
            category = categories[0] if categories else None
            handedness = "Unknown"
            handedness_score = 0.0
            if category is not None:
                handedness = category.category_name or category.display_name or "Unknown"
                handedness_score = category.score or 0.0

            world_landmarks = result.hand_world_landmarks[index]
            hands.append(
                HandPose(
                    landmarks=tuple(landmarks),
                    world_landmarks=tuple(world_landmarks),
                    handedness=handedness,
                    handedness_score=handedness_score,
                    prediction=classify_gesture(landmarks, aspect_ratio),
                )
            )
        return hands

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
