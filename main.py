#!/usr/bin/env python3
"""
AIR CANVAS
----------
A real-time hand-gesture drawing application using OpenCV + MediaPipe + NumPy.

Gestures:
    * Index finger only raised   -> DRAW mode (draw smooth lines)
    * Index + Middle raised      -> SELECT mode (hover over palette to pick a color)
    * Open palm (four fingers)   -> CLEAR the canvas

Keyboard:
    q / ESC   -> Quit
    c         -> Clear canvas
    + / -     -> Increase / decrease brush thickness
"""

import sys
import time

import cv2
import mediapipe as mp
import numpy as np


class AirCanvas:
    def __init__(self, camera_id: int = 0, width: int = 1280, height: int = 720):
        self.camera_id = camera_id
        self.width = width
        self.height = height

        # ------------------------------------------------------------------ #
        # MediaPipe Hands
        # ------------------------------------------------------------------ #
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils

        # ------------------------------------------------------------------ #
        # Webcam
        # ------------------------------------------------------------------ #
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open webcam (camera id {camera_id}). "
                "Check that the camera is connected, not in use by another app, "
                "and that camera permissions are granted."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # ------------------------------------------------------------------ #
        # Drawing state
        # ------------------------------------------------------------------ #
        self.canvas = None
        self.prev_point = None          # (x, y) index tip of previous frame
        self.draw_color = (255, 0, 0)   # BGR -> Blue
        self.brush_thickness = 5

        # ------------------------------------------------------------------ #
        # Smooth-workflow settings
        # ------------------------------------------------------------------ #
        # Run MediaPipe hand detection every N-th frame; reuse the last result
        # in between so drawing stays fluid even when detection is slow.
        self.detect_every_n = 2
        self.frame_counter = 0
        self.cached_landmarks = None
        self.cached_handedness = None

        # ------------------------------------------------------------------ #
        # On-screen color palette
        # ------------------------------------------------------------------ #
        # (label, BGR color)
        self.palette = [
            ("BLUE", (255, 0, 0)),
            ("GREEN", (0, 255, 0)),
            ("RED", (0, 0, 255)),
            ("YELLOW", (0, 255, 255)),
            ("PURPLE", (255, 0, 255)),
            ("WHITE", (255, 255, 255)),
            ("ERASER", (0, 0, 0)),
        ]
        self.palette_x_start = 50
        self.palette_y = 25
        self.palette_w = 130
        self.palette_h = 55
        self.palette_gap = 15

        # ------------------------------------------------------------------ #
        # FPS / HUD state
        # ------------------------------------------------------------------ #
        self.prev_time = 0
        self.fps = 0.0
        self.status = "No hand detected"
        self.status_color = (0, 0, 255)  # red

        # ------------------------------------------------------------------ #
        # Gesture cooldowns (seconds)
        # ------------------------------------------------------------------ #
        self.last_clear_time = 0.0
        # ------------------------------------------------------------------ #
    def fingers_up(self, hand_landmarks, handedness) -> list:
        """
        Return [thumb, index, middle, ring, pinky] booleans.
        A finger is 'up' when its tip is above its PIP joint (smaller y).
        Thumb direction depends on handedness.
        """
        lm = hand_landmarks.landmark
        fingers = []

        # Thumb: tip vs IP joint, mirrored for right/left hands
        thumb_tip_x = lm[4].x
        thumb_ip_x = lm[3].x
        label = handedness.classification[0].label
        if label == "Right":
            fingers.append(thumb_tip_x < thumb_ip_x)
        else:
            fingers.append(thumb_tip_x > thumb_ip_x)

        # Four fingers: tip above PIP => raised
        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
            fingers.append(lm[tip].y < lm[pip].y)

        return fingers

    # ------------------------------------------------------------------ #
    def draw_palette(self, frame):
        """Draw color-selection buttons along the top of the frame."""
        x = self.palette_x_start
        y = self.palette_y
        for label, color in self.palette:
            selected = color == self.draw_color
            thickness = cv2.FILLED if selected else 3
            cv2.rectangle(
                frame,
                (x, y),
                (x + self.palette_w, y + self.palette_h),
                color,
                thickness,
            )
            # White outline on every button
            cv2.rectangle(
                frame,
                (x, y),
                (x + self.palette_w, y + self.palette_h),
                (255, 255, 255),
                2,
            )
            txt_color = (0, 0, 0) if selected else (255, 255, 255)
            cv2.putText(
                frame,
                label,
                (x + 12, y + 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                txt_color,
                2,
            )
            x += self.palette_w + self.palette_gap

    def check_palette(self, point):
        """If the point is over a palette button, select that color."""
        x, y = point
        if not (self.palette_y <= y <= self.palette_y + self.palette_h):
            return
        px = self.palette_x_start
        for label, color in self.palette:
            if px <= x <= px + self.palette_w:
                self.draw_color = color
                self.status = f"Color: {label}"
                self.status_color = color
                break
            px += self.palette_w + self.palette_gap
    # ------------------------------------------------------------------ #
    def run(self):
        print("[INFO] Air Canvas started. Press 'q' or ESC to quit.")
        try:
            # Warm up the camera: macOS AVFoundation sometimes drops the first
            # few frames. Retry up to 10 times before giving up.
            warmup_retries = 10
            for _ in range(warmup_retries):
                ret, _ = self.cap.read()
                if ret:
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError(
                    "Webcam opened but could not read any frames. "
                    "Another app may be using the camera."
                )

            while self.cap.isOpened():
                try:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("[WARNING] Dropped frame, retrying...")
                        time.sleep(0.01)
                        continue

                    # Mirror for a natural selfie view
                    frame = cv2.flip(frame, 1)
                    h, w, _ = frame.shape

                    # Lazy canvas init (matches actual frame size)
                    if self.canvas is None or self.canvas.shape[:2] != (h, w):
                        self.canvas = np.zeros((h, w, 3), dtype=np.uint8)

                    # MediaPipe expects RGB
                    # Run full detection only every N frames; reuse cached
                    # results in between for a smooth, fluid drawing workflow.
                    self.frame_counter += 1
                    if self.frame_counter % self.detect_every_n == 0:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        rgb.flags.writeable = False
                        results = self.hands.process(rgb)
                        rgb.flags.writeable = True
                        if results.multi_hand_landmarks and results.multi_handedness:
                            self.cached_landmarks = results.multi_hand_landmarks
                            self.cached_handedness = results.multi_handedness
                        else:
                            self.cached_landmarks = None
                            self.cached_handedness = None

                    index_tip = None
                    mode = "IDLE"

                    if self.cached_landmarks and self.cached_handedness:
                        # Track only the first detected hand
                        hand_landmarks = self.cached_landmarks[0]
                        handedness = self.cached_handedness[0]

                        fingers = self.fingers_up(hand_landmarks, handedness)
                        index_tip = (
                            int(hand_landmarks.landmark[8].x * w),
                            int(hand_landmarks.landmark[8].y * h),
                        )

                        # Draw skeleton landmarks
                        self.mp_draw.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                        )
                        # ---------------- Gesture classification ----------------
                        four_fingers = all(fingers[1:])  # index, middle, ring, pinky

                        if four_fingers:
                            now = time.time()
                            if now - self.last_clear_time > self.clear_cooldown:
                                self.canvas[:] = 0
                                self.last_clear_time = now
                                self.prev_point = None
                                self.status = "CANVAS CLEARED"
                                self.status_color = (0, 255, 0)
                            mode = "CLEAR"
                        elif fingers[1] and fingers[2]:
                            mode = "SELECT"
                            self.prev_point = None
                            self.status = "SELECT MODE"
                            self.status_color = (0, 255, 255)
                        elif fingers[1] and not fingers[2]:
                            mode = "DRAW"
                            self.status = "DRAW MODE"
                            self.status_color = (0, 255, 0)
                        else:
                            mode = "IDLE"
                            self.prev_point = None
                            self.status = "HAND DETECTED"
                            self.status_color = (0, 255, 0)

                        # ---------------- Act on mode ----------------
                        if mode == "DRAW" and index_tip:
                            if self.prev_point is not None:
                                cv2.line(
                                    self.canvas,
                                    self.prev_point,
                                    index_tip,
                                    self.draw_color,
                                    self.brush_thickness,
                                    cv2.LINE_AA,
                                )
                            self.prev_point = index_tip
                        elif mode == "SELECT" and index_tip:
                            self.check_palette(index_tip)
                        else:
                            self.prev_point = None
                    else:
                        self.prev_point = None
                        self.status = "No hand detected"
                        self.status_color = (0, 0, 255)
                except Exception as err:
                    print(f"[ERROR] Frame processing error: {err}")
                    continue

                # ---------------- FPS ----------------
                curr_time = time.time()
                self.fps = (
                    1.0 / (curr_time - self.prev_time) if self.prev_time else 0.0
                )
                self.prev_time = curr_time

                # ---------------- Blend canvas onto webcam feed ----------------
                canvas_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
                _, inv_mask = cv2.threshold(
                    canvas_gray, 20, 255, cv2.THRESH_BINARY_INV
                )
                inv_mask = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR)
                frame = cv2.bitwise_and(frame, inv_mask)
                frame = cv2.add(frame, self.canvas)

                # ---------------- UI ----------------
                self.draw_palette(frame)

                # Cursor at the index fingertip
                if index_tip:
                    cv2.circle(frame, index_tip, 11, self.draw_color, cv2.FILLED)
                    cv2.circle(frame, index_tip, 16, self.draw_color, 2)

                # HUD text
                hud = [
                    (f"FPS: {self.fps:5.1f}", (20, 50), (0, 255, 0)),
                    (f"Mode: {mode}", (20, 90), self.status_color),
                    (f"Status: {self.status}", (20, 130), self.status_color),
                    (f"Brush: {self.brush_thickness}", (20, 170), (255, 255, 255)),
                ]
                for text, pos, color in hud:
                    cv2.putText(
                        frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                    )

                # Hint line
                hints = (
                    "Index: Draw | Index+Middle: Select | Open Palm: Clear | "
                    "c: Clear | +/-: Brush | q: Quit"
                )
                cv2.putText(
                    frame,
                    hints,
                    (20, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 200, 200),
                    1,
                )

                cv2.imshow("Air Canvas", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q or ESC
                    break
                elif key == ord("c"):
                    self.canvas[:] = 0
                    self.prev_point = None
                    self.status = "CANVAS CLEARED"
                elif key in (ord("+"), ord("=")):
                    self.brush_thickness = min(100, self.brush_thickness + 1)
                elif key == ord("-"):
                    self.brush_thickness = max(1, self.brush_thickness - 1)
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("[INFO] Application closed.")


def main():
    try:
        app = AirCanvas()
        app.run()
    except RuntimeError as err:
        print(f"[FATAL] {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()