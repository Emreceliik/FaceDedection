"""
Kamera onunde sektirilen sanal top oyunu.
- Yer cekimi + hava surtunmesi ile fizik.
- Kafa carpismasi: yuz kutularini kullanir.
- El carpismasi: ardisik karelerin farkindan cikarilan hareket maskesi
  (yuz bolgesi haric) "el" olarak kabul edilir; hareket merkezinden
  topa dogru bir vurus uygulanir.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple

import cv2
import numpy as np


Box = Tuple[int, int, int, int]


class BallGame:
    def __init__(self, width: int = 720, height: int = 540):
        self.w = width
        self.h = height
        self.ball_r = 30
        self.gravity = 0.55
        self.air_drag = 0.997
        self.max_speed = 24.0
        self.score = 0
        self.best = 0
        self._prev_gray: np.ndarray | None = None
        self._cooldown = 0
        self._flash = 0
        self._reset_ball()

    def _reset_ball(self) -> None:
        self.x = float(self.w) * 0.5
        self.y = float(self.h) * 0.15
        self.vx = random.uniform(-3.0, 3.0)
        self.vy = 0.0
        self._cooldown = 0

    def reset(self) -> None:
        self.score = 0
        self._prev_gray = None
        self._reset_ball()

    def _resize_if_needed(self, w: int, h: int) -> None:
        if w == self.w and h == self.h:
            return
        self.w, self.h = w, h
        self.x = min(max(self.x, self.ball_r), self.w - self.ball_r)
        self.y = min(max(self.y, self.ball_r), self.h - self.ball_r)
        self._prev_gray = None

    # ------------------------------------------------------------------
    def step(self, frame_bgr: np.ndarray, face_boxes: List[Box]) -> None:
        h, w = frame_bgr.shape[:2]
        self._resize_if_needed(w, h)

        motion = self._motion_mask(frame_bgr, face_boxes)
        self._apply_physics()
        self._wall_collisions()
        if not self._fell_off():
            self._collide_with_head(face_boxes)
            self._collide_with_motion(motion, face_boxes)
        if self._cooldown > 0:
            self._cooldown -= 1
        if self._flash > 0:
            self._flash -= 1
        self._draw(frame_bgr, motion)

    # ------------------------------------------------------------------
    def _motion_mask(
        self, frame_bgr: np.ndarray, face_boxes: List[Box]
    ) -> np.ndarray | None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)
        mask: np.ndarray | None = None
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            diff = cv2.absdiff(gray, self._prev_gray)
            _, mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
            # Yuz bolgelerini hareket maskesinden cikar (kendi ifaden hareket sansin)
            for (x1, y1, x2, y2) in face_boxes:
                pad = 12
                fx1 = max(0, x1 - pad)
                fy1 = max(0, y1 - pad)
                fx2 = min(self.w, x2 + pad)
                fy2 = min(self.h, y2 + pad)
                mask[fy1:fy2, fx1:fx2] = 0
        self._prev_gray = gray
        return mask

    def _apply_physics(self) -> None:
        self.vy += self.gravity
        self.vx *= self.air_drag
        self.vy *= self.air_drag
        speed = math.hypot(self.vx, self.vy)
        if speed > self.max_speed:
            self.vx *= self.max_speed / speed
            self.vy *= self.max_speed / speed
        self.x += self.vx
        self.y += self.vy

    def _wall_collisions(self) -> None:
        if self.x - self.ball_r < 0:
            self.x = float(self.ball_r)
            self.vx = abs(self.vx) * 0.85
        elif self.x + self.ball_r > self.w:
            self.x = float(self.w - self.ball_r)
            self.vx = -abs(self.vx) * 0.85
        if self.y - self.ball_r < 0:
            self.y = float(self.ball_r)
            self.vy = abs(self.vy) * 0.6

    def _fell_off(self) -> bool:
        if self.y - self.ball_r > self.h + 4:
            if self.score > self.best:
                self.best = self.score
            self.score = 0
            self._reset_ball()
            self._flash = 6
            return True
        return False

    # ------------------------------------------------------------------
    def _collide_with_head(self, face_boxes: List[Box]) -> None:
        if self._cooldown > 0:
            return
        for (x1, y1, x2, y2) in face_boxes:
            cx = (x1 + x2) * 0.5
            half_w = max(1.0, (x2 - x1) * 0.5)

            if self.x + self.ball_r < x1 or self.x - self.ball_r > x2:
                continue

            # Ust kenara carpma: top yukaridan dusuyorken
            top_y = y1
            if (
                self.vy > 0
                and abs((self.y + self.ball_r) - top_y) < 18
                and y1 - 30 < self.y < y2
            ):
                self.y = float(top_y - self.ball_r)
                offset = (self.x - cx) / half_w
                self.vx = float(max(-14.0, min(14.0, self.vx + offset * 7.0)))
                self.vy = -max(8.0, abs(self.vy) * 1.05 + 2.0)
                self._register_hit()
                return

            # Yuz icine girdiyse yukari it
            if x1 < self.x < x2 and y1 < self.y < y2:
                self.y = float(y1 - self.ball_r)
                self.vy = -10.0
                self._register_hit()
                return

    def _collide_with_motion(
        self, motion: np.ndarray | None, face_boxes: List[Box]
    ) -> None:
        if motion is None or self._cooldown > 0:
            return
        check_r = self.ball_r + 6
        x1 = max(0, int(self.x - check_r))
        y1 = max(0, int(self.y - check_r))
        x2 = min(self.w, int(self.x + check_r))
        y2 = min(self.h, int(self.y + check_r))
        if x2 <= x1 or y2 <= y1:
            return

        roi = motion[y1:y2, x1:x2]
        motion_pixels = int(cv2.countNonZero(roi))
        if motion_pixels < 220:
            return

        ys, xs = np.nonzero(roi)
        if xs.size == 0:
            return
        mx = float(xs.mean()) + x1
        my = float(ys.mean()) + y1

        dx = self.x - mx
        dy = self.y - my
        norm = math.hypot(dx, dy)
        if norm < 1e-3:
            dx, dy = 0.0, -1.0
        else:
            dx /= norm
            dy /= norm

        power = min(20.0, 9.0 + motion_pixels / 450.0)
        self.vx = dx * power
        self.vy = dy * power - 2.5  # her zaman biraz yukari yonelt
        self._register_hit()

    def _register_hit(self) -> None:
        self.score += 1
        self._cooldown = 6
        self._flash = 4

    # ------------------------------------------------------------------
    def _draw(self, frame_bgr: np.ndarray, motion: np.ndarray | None) -> None:
        if motion is not None:
            # Hareketli alanlari hafifce isaretle (gorsel ipucu)
            tint = np.zeros_like(frame_bgr)
            tint[:, :, 1] = motion  # yesil kanal
            cv2.addWeighted(frame_bgr, 1.0, tint, 0.18, 0, dst=frame_bgr)

        cx, cy = int(self.x), int(self.y)
        r = self.ball_r
        cv2.circle(frame_bgr, (cx, cy), r, (40, 90, 230), -1)
        cv2.circle(frame_bgr, (cx, cy), r, (255, 255, 255), 2)
        cv2.circle(frame_bgr, (cx, cy), max(4, r // 4), (255, 255, 255), -1)

        if self._flash > 0:
            cv2.circle(frame_bgr, (cx, cy), r + 8, (0, 255, 255), 3)

        cv2.rectangle(frame_bgr, (10, 10), (260, 78), (0, 0, 0), -1)
        cv2.rectangle(frame_bgr, (10, 10), (260, 78), (255, 255, 255), 1)
        cv2.putText(
            frame_bgr,
            f"Skor: {self.score}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            f"En iyi: {self.best}",
            (20, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
