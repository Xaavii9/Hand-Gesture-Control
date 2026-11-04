import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hand_tracking import HAND_CONNECTIONS
from screen_library import (
    FILE_TILE_CENTER_X,
    FILE_TILE_HEIGHT,
    FILE_TILE_START_Y,
    FILE_TILE_STEP_Y,
    FILE_TILE_WIDTH,
)


class HUDRenderer:
    def __init__(self):
        self._texture_cache = {}
        self._font_path = "C:/Windows/Fonts/segoeui.ttf"

    def draw(self, frame, hands, interaction, mouse_mode, microphone_status):
        height, width = frame.shape[:2]
        self._draw_file_list(
            frame,
            interaction.entries,
            interaction.selected_file,
            interaction.entry_positions,
            interaction.current_directory,
        )
        for hand in hands:
            self._draw_hand(frame, hand, width, height)

        if interaction.panel is not None and interaction.panel.visible:
            self._draw_object(frame, interaction.panel)

        self._draw_status(
            frame,
            hands,
            interaction,
            mouse_mode,
            microphone_status,
        )
        return frame

    @staticmethod
    def project_panel(panel, frame_size):
        width, height = frame_size
        panel_width = panel.width * width * panel.scale
        panel_height = panel.height * height * panel.scale
        depth = min(panel_width, panel_height) * 0.08
        half_width = panel_width / 2
        half_height = panel_height / 2
        vertices = [
            (-half_width, -half_height, -depth),
            (half_width, -half_height, -depth),
            (half_width, half_height, -depth),
            (-half_width, half_height, -depth),
            (-half_width, -half_height, depth),
            (half_width, -half_height, depth),
            (half_width, half_height, depth),
            (-half_width, half_height, depth),
        ]

        cos_x, sin_x = math.cos(panel.rotation_x), math.sin(panel.rotation_x)
        cos_y, sin_y = math.cos(panel.rotation_y), math.sin(panel.rotation_y)
        cos_z, sin_z = math.cos(panel.rotation_z), math.sin(panel.rotation_z)
        focal_length = height * 2.2
        projected = []
        center_x = panel.x * width
        center_y = panel.y * height
        for x, y, z in vertices:
            rotated_y = y * cos_x - z * sin_x
            rotated_z = y * sin_x + z * cos_x
            rotated_x = x * cos_y + rotated_z * sin_y
            rotated_z = -x * sin_y + rotated_z * cos_y
            final_x = rotated_x * cos_z - rotated_y * sin_z
            final_y = rotated_x * sin_z + rotated_y * cos_z
            perspective = focal_length / max(focal_length + rotated_z, focal_length * 0.25)
            projected.append(
                (
                    round(center_x + final_x * perspective),
                    round(center_y + final_y * perspective),
                )
            )
        return projected

    @staticmethod
    def _draw_hand(frame, hand, width, height):
        points = [
            (
                max(0, min(width - 1, round(landmark.x * width))),
                max(0, min(height - 1, round(landmark.y * height))),
            )
            for landmark in hand.landmarks
        ]
        color = (80, 220, 180) if hand.handedness.lower() == "right" else (220, 150, 70)
        for connection in HAND_CONNECTIONS:
            cv2.line(
                frame,
                points[connection.start],
                points[connection.end],
                color,
                2,
                cv2.LINE_AA,
            )
        cv2.circle(frame, points[8], 8, (40, 245, 245), -1, cv2.LINE_AA)
        cv2.circle(frame, points[8], 12, color, 2, cv2.LINE_AA)

    def _draw_object(self, frame, panel):
        points = self.project_panel(panel, (frame.shape[1], frame.shape[0]))
        front = np.array([points[index] for index in (0, 1, 2, 3)], dtype=np.int32)
        texture = self._document_texture(panel, frame.shape[1], frame.shape[0])
        texture_height, texture_width = texture.shape[:2]
        source = np.array(
            [
                [0, 0],
                [texture_width - 1, 0],
                [texture_width - 1, texture_height - 1],
                [0, texture_height - 1],
            ],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(source, front.astype(np.float32))
        frame_size = (frame.shape[1], frame.shape[0])
        warped = cv2.warpPerspective(texture, transform, frame_size)
        source_mask = np.full((texture_height, texture_width), 255, dtype=np.uint8)
        mask = cv2.warpPerspective(source_mask, transform, frame_size) > 0
        blended = cv2.addWeighted(warped, 0.94, frame, 0.06, 0)
        frame[mask] = blended[mask]

        edges = (
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        )
        for start, end in edges:
            cv2.line(frame, points[start], points[end], (70, 235, 205), 2, cv2.LINE_AA)

    def _document_texture(self, panel, frame_width, frame_height):
        texture_width = max(160, round(panel.width * panel.scale * frame_width))
        texture_height = max(160, round(panel.height * panel.scale * frame_height))
        cache_key = (
            panel.relative_path,
            panel.title,
            panel.content,
            texture_width,
            texture_height,
        )
        if cache_key in self._texture_cache:
            return self._texture_cache[cache_key]

        image = Image.new("RGB", (texture_width, texture_height), (18, 30, 34))
        draw = ImageDraw.Draw(image)
        title_font = ImageFont.truetype(self._font_path, max(15, round(texture_height * 0.043)))
        body_font = ImageFont.truetype(self._font_path, max(12, round(texture_height * 0.031)))
        margin = max(14, round(texture_width * 0.045))
        draw.text((margin, margin), panel.title or "Documento", font=title_font, fill=(105, 241, 210))
        draw.line(
            (margin, margin + round(texture_height * 0.075), texture_width - margin, margin + round(texture_height * 0.075)),
            fill=(54, 127, 119),
            width=2,
        )

        if panel.image_rgb is not None:
            image_top = margin + round(texture_height * 0.10)
            available_width = texture_width - margin * 2
            available_height = texture_height - image_top - margin
            photo = Image.fromarray(panel.image_rgb).convert("RGB")
            photo.thumbnail((available_width, available_height), Image.Resampling.LANCZOS)
            image_x = (texture_width - photo.width) // 2
            image_y = image_top + (available_height - photo.height) // 2
            image.paste(photo, (image_x, image_y))
            draw.rectangle(
                (image_x, image_y, image_x + photo.width - 1, image_y + photo.height - 1),
                outline=(105, 241, 210),
                width=2,
            )
            image_array = np.asarray(image)
            self._cache_texture(cache_key, image_array)
            return image_array

        content = panel.content or "Sin vista previa disponible."
        available_width = texture_width - margin * 2
        body_top = margin + round(texture_height * 0.10)
        line_height = max(14, round(body_font.size * 1.42))
        column_gap = max(12, round(texture_width * 0.025))
        column_width = max(50, (available_width - column_gap) // 2)
        column_limit = max(1, (texture_height - body_top - margin) // line_height)
        columns = [[], []]
        column_index = 0

        for paragraph in content.splitlines() or [""]:
            words = paragraph.split()
            if not words:
                columns[column_index].append("")
                continue
            line = ""
            for word in words:
                candidate = f"{line} {word}".strip()
                if line and draw.textlength(candidate, font=body_font) > column_width:
                    columns[column_index].append(line)
                    line = word
                else:
                    line = candidate
            if line:
                columns[column_index].append(line)

            if len(columns[column_index]) >= column_limit:
                column_index = 1

        for index, lines in enumerate(columns):
            x = margin + index * (column_width + column_gap)
            for line_index, line in enumerate(lines[:column_limit]):
                y = body_top + line_index * line_height
                draw.text((x, y), line, font=body_font, fill=(226, 236, 232))

        image_array = np.asarray(image)
        self._cache_texture(cache_key, image_array)
        return image_array

    def _cache_texture(self, cache_key, image_array):
        if len(self._texture_cache) > 8:
            self._texture_cache.clear()
        self._texture_cache[cache_key] = image_array

    @staticmethod
    def _draw_file_list(frame, entries, selected_file, entry_positions, current_directory):
        height, width = frame.shape[:2]
        positions = {
            relative_path: (center_x, center_y)
            for relative_path, center_x, center_y in entry_positions
        }
        tile_width = round(width * FILE_TILE_WIDTH)
        tile_height = round(height * FILE_TILE_HEIGHT)
        current = current_directory.replace("\\", "/") or "pantalla"
        cv2.putText(
            frame,
            f"PANTALLA / {current}"[:42],
            (round(width * (FILE_TILE_CENTER_X - FILE_TILE_WIDTH / 2)), round(height * 0.17)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (100, 240, 210),
            1,
            cv2.LINE_AA,
        )
        for index, entry in enumerate(entries):
            center_x, center_y = positions.get(
                entry.relative_path,
                (FILE_TILE_CENTER_X, FILE_TILE_START_Y + index * FILE_TILE_STEP_Y),
            )
            left = round(width * (center_x - FILE_TILE_WIDTH / 2))
            center_y = round(height * center_y)
            top = center_y - tile_height // 2
            bottom = center_y + tile_height // 2
            right = left + tile_width
            selected = entry.relative_path == selected_file
            fill = (42, 68, 68) if selected else (20, 37, 42)
            border = (90, 245, 205) if selected else (58, 130, 123)
            overlay = frame.copy()
            cv2.rectangle(overlay, (left, top), (right, bottom), fill, -1)
            cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
            cv2.rectangle(frame, (left, top), (right, bottom), border, 2, cv2.LINE_AA)
            icon = {
                "folder": "[DIR]",
                "image": "[IMG]",
                "parent": "[..]",
                "text": "[TXT]",
            }.get(entry.kind, "[FILE]")
            label = f"{icon} {entry.name}"
            cv2.putText(
                frame,
                ("> " if selected else "") + label[:25],
                (left + 12, center_y + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (235, 248, 244),
                1,
                cv2.LINE_AA,
            )

    @staticmethod
    def _draw_status(frame, hands, interaction, mouse_mode, microphone_status):
        height, width = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 86), (8, 17, 23), -1)
        cv2.rectangle(overlay, (0, height - 94), (width, height), (8, 17, 23), -1)
        cv2.addWeighted(overlay, 0.66, frame, 0.34, 0, frame)

        cv2.putText(
            frame,
            "GESTURE INTERFACE / LOCAL VISION",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (100, 240, 210),
            2,
            cv2.LINE_AA,
        )
        if hands:
            primary = next(
                (
                    hand
                    for hand in hands
                    if hand.handedness.lower() == interaction.active_hand_label.lower()
                ),
                hands[0],
            )
            gesture_text = (
                f"CAMERA {interaction.active_hand_label.upper()} / "
                f"GESTURE: {primary.prediction.label}   "
                f"SCORE: {primary.prediction.confidence}%"
            )
        else:
            gesture_text = "GESTURE: NO HAND"
        cv2.putText(
            frame,
            gesture_text,
            (16, 61),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (235, 245, 242),
            1,
            cv2.LINE_AA,
        )

        footer = (
            f"MODE: {interaction.mode} | {interaction.status}",
            f"DESKTOP MOUSE: {'ON' if mouse_mode else 'OFF'} [M] | {microphone_status}",
            "CAMERA LEFT: pinch-release file to open; hold/move to drag; hold 1s then move to rotate",
            "BOTH PINCH: zoom | CAMERA RIGHT (your left): double pinch attracts selected document",
            "SPACE: save labeled sample | U D P O I S C F R 1-5: label | M: mouse | Q: quit",
        )
        for index, text in enumerate(footer):
            cv2.putText(
                frame,
                text[:110],
                (14, height - 96 + index * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (215, 230, 230),
                1,
                cv2.LINE_AA,
            )
