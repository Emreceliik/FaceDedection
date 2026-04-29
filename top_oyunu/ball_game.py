"""
Kamera onunde sektirilen sanal top oyunu.
- Yer cekimi + hava surtunmesi ile fizik.
- Kafa carpismasi: yuz kutusunu kullanir.
- El carpismasi: ardisik karelerin farkindan cikarilan hareket maskesi
  (yuz bolgesi haric) "el" olarak kabul edilir.

Eglenceli eklentiler:
- Topun ardinda renkli iz (trail).
- Topa vuruldugunda parcacik patlamasi.
- Combo carpani: ust uste vuruslar puani katlar.
- Seviye sistemi: belirli skorda LEVEL UP, top hizlanir, rengi degisir.
- Ses efektleri (opsiyonel, SoundFx ile).
"""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from sounds import SoundFx


Box = Tuple[int, int, int, int]


# Seviyeye gore top renkleri (BGR). Listenin sonuna ulasilirsa son renk kalir.
LEVEL_COLORS = [
    (40, 90, 230),    # turuncu-kirmizi
    (40, 200, 230),   # sari-turuncu
    (40, 220, 120),   # yesil
    (220, 120, 40),   # mavi
    (220, 60, 200),   # mor
    (60, 60, 240),    # parlak kirmizi
]


class _Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "color", "size")

    def __init__(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        life: int,
        color: Tuple[int, int, int],
        size: int,
    ):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life = life
        self.max_life = life
        self.color = color
        self.size = size

    def step(self) -> bool:
        self.vy += 0.25
        self.vx *= 0.96
        self.vy *= 0.96
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        return self.life > 0


class BallGame:
    def __init__(
        self,
        width: int = 720,
        height: int = 540,
        sound: Optional[SoundFx] = None,
    ):
        self.w = width
        self.h = height
        self.ball_r = 30
        self.gravity = 0.55
        self.air_drag = 0.997
        self.max_speed = 24.0

        self.score = 0
        self.best = 0
        self.level = 1
        self.combo = 0
        self.best_combo = 0

        self._prev_gray: Optional[np.ndarray] = None
        self._cooldown = 0
        self._flash = 0
        self._level_flash = 0
        self._shake = 0

        self.trail: Deque[Tuple[float, float]] = deque(maxlen=14)
        self.particles: List[_Particle] = []

        self.sound = sound

        self._reset_ball()

    # ------------------------------------------------------------------
    def _reset_ball(self) -> None:
        self.x = float(self.w) * 0.5
        self.y = float(self.h) * 0.15
        self.vx = random.uniform(-3.0, 3.0)
        self.vy = 0.0
        self._cooldown = 0
        self.trail.clear()

    def reset(self) -> None:
        self.score = 0
        self.level = 1
        self.combo = 0
        self.particles.clear()
        self._level_flash = 0
        self._shake = 0
        self._prev_gray = None
        self._reset_ball()
        if self.sound:
            self.sound.start()

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
        if self._level_flash > 0:
            self._level_flash -= 1
        if self._shake > 0:
            self._shake -= 1

        # parcaciklari ilerlet
        self.particles = [p for p in self.particles if p.step()]

        self.trail.append((self.x, self.y))
        self._draw(frame_bgr, motion, face_boxes)

    # ------------------------------------------------------------------
    def _motion_mask(
        self, frame_bgr: np.ndarray, face_boxes: List[Box]
    ) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)
        mask: Optional[np.ndarray] = None
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            diff = cv2.absdiff(gray, self._prev_gray)
            _, mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
            for (x1, y1, x2, y2) in face_boxes:
                pad = 12
                fx1 = max(0, x1 - pad)
                fy1 = max(0, y1 - pad)
                fx2 = min(self.w, x2 + pad)
                fy2 = min(self.h, y2 + pad)
                mask[fy1:fy2, fx1:fx2] = 0
        self._prev_gray = gray
        return mask

    def _level_factor(self) -> float:
        # 1. seviye => 1.0; her seviyede %8 daha hizli, ust limit ~1.6
        return min(1.6, 1.0 + (self.level - 1) * 0.08)

    def _apply_physics(self) -> None:
        f = self._level_factor()
        self.vy += self.gravity * f
        self.vx *= self.air_drag
        self.vy *= self.air_drag
        speed = math.hypot(self.vx, self.vy)
        cap = self.max_speed * f
        if speed > cap:
            self.vx *= cap / speed
            self.vy *= cap / speed
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
            if self.combo > self.best_combo:
                self.best_combo = self.combo
            self._spawn_burst(
                self.w * 0.5,
                self.h - 10,
                count=18,
                color=(60, 60, 230),
                spread=8.0,
                upward=True,
            )
            self.score = 0
            self.level = 1
            self.combo = 0
            self._reset_ball()
            self._flash = 6
            self._shake = 8
            if self.sound:
                self.sound.fall()
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
                self._register_hit(kind="head")
                return

            if x1 < self.x < x2 and y1 < self.y < y2:
                self.y = float(y1 - self.ball_r)
                self.vy = -10.0
                self._register_hit(kind="head")
                return

    def _collide_with_motion(
        self, motion: Optional[np.ndarray], face_boxes: List[Box]
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
        self.vy = dy * power - 2.5
        self._register_hit(kind="hand")

    # ------------------------------------------------------------------
    def _multiplier(self) -> int:
        if self.combo >= 10:
            return 4
        if self.combo >= 6:
            return 3
        if self.combo >= 3:
            return 2
        return 1

    def _register_hit(self, kind: str) -> None:
        self.combo += 1
        mult = self._multiplier()
        self.score += mult
        self._cooldown = 6
        self._flash = 4

        # seviye atlama her 10 puanda
        new_level = self.score // 10 + 1
        leveled_up = new_level > self.level
        self.level = new_level
        if leveled_up:
            self._level_flash = 35
            self._shake = 6
            self._spawn_burst(
                self.x,
                self.y,
                count=24,
                color=self._level_color(),
                spread=9.0,
                upward=False,
            )
            if self.sound:
                self.sound.level_up()

        # carpisma parcaciklari
        self._spawn_burst(
            self.x,
            self.y,
            count=10 + min(self.combo, 8),
            color=self._level_color(),
            spread=6.0,
            upward=False,
        )

        if self.sound and not leveled_up:
            self.sound.hit(self.combo)

    def _level_color(self) -> Tuple[int, int, int]:
        idx = min(self.level - 1, len(LEVEL_COLORS) - 1)
        return LEVEL_COLORS[max(0, idx)]

    def _spawn_burst(
        self,
        cx: float,
        cy: float,
        count: int,
        color: Tuple[int, int, int],
        spread: float = 6.0,
        upward: bool = False,
    ) -> None:
        for _ in range(count):
            ang = random.uniform(0.0, math.tau)
            sp = random.uniform(2.0, spread)
            vx = math.cos(ang) * sp
            vy = math.sin(ang) * sp
            if upward:
                vy = -abs(vy) - 2.0
            life = random.randint(18, 32)
            size = random.randint(2, 5)
            jitter = (
                max(0, min(255, color[0] + random.randint(-25, 25))),
                max(0, min(255, color[1] + random.randint(-25, 25))),
                max(0, min(255, color[2] + random.randint(-25, 25))),
            )
            self.particles.append(_Particle(cx, cy, vx, vy, life, jitter, size))

    # ------------------------------------------------------------------
    def _draw(
        self,
        frame_bgr: np.ndarray,
        motion: Optional[np.ndarray],
        face_boxes: List[Box],
    ) -> None:
        # hareket tinti
        if motion is not None:
            tint = np.zeros_like(frame_bgr)
            tint[:, :, 1] = motion
            cv2.addWeighted(frame_bgr, 1.0, tint, 0.18, 0, dst=frame_bgr)

        # ekran sarsintisi (level up / dusme an'inda)
        ox = oy = 0
        if self._shake > 0:
            ox = random.randint(-3, 3)
            oy = random.randint(-3, 3)

        # iz (trail)
        ball_color = self._level_color()
        if len(self.trail) >= 2:
            n = len(self.trail)
            for i in range(1, n):
                p1 = self.trail[i - 1]
                p2 = self.trail[i]
                t = i / n
                thick = max(1, int(self.ball_r * 0.4 * t))
                col = (
                    int(ball_color[0] * t + 30 * (1 - t)),
                    int(ball_color[1] * t + 30 * (1 - t)),
                    int(ball_color[2] * t + 30 * (1 - t)),
                )
                cv2.line(
                    frame_bgr,
                    (int(p1[0]) + ox, int(p1[1]) + oy),
                    (int(p2[0]) + ox, int(p2[1]) + oy),
                    col,
                    thick,
                    cv2.LINE_AA,
                )

        # parcaciklar
        for p in self.particles:
            t = p.life / p.max_life
            col = (
                int(p.color[0] * t),
                int(p.color[1] * t),
                int(p.color[2] * t),
            )
            cv2.circle(
                frame_bgr,
                (int(p.x) + ox, int(p.y) + oy),
                max(1, int(p.size * (0.3 + 0.7 * t))),
                col,
                -1,
            )

        # yuz kutulari (silik, oyun karismasin diye ince)
        for (x1, y1, x2, y2) in face_boxes:
            cv2.rectangle(frame_bgr, (x1 + ox, y1 + oy), (x2 + ox, y2 + oy), (255, 200, 0), 1)

        # top
        cx, cy = int(self.x) + ox, int(self.y) + oy
        r = self.ball_r
        cv2.circle(frame_bgr, (cx, cy), r, ball_color, -1)
        cv2.circle(frame_bgr, (cx, cy), r, (255, 255, 255), 2)
        cv2.circle(frame_bgr, (cx - r // 3, cy - r // 3), max(4, r // 4), (255, 255, 255), -1)

        if self._flash > 0:
            cv2.circle(frame_bgr, (cx, cy), r + 8, (0, 255, 255), 3)

        # skor paneli
        self._draw_panel(frame_bgr)

        # LEVEL UP yazisi
        if self._level_flash > 0:
            self._draw_level_up(frame_bgr)

    def _draw_panel(self, frame_bgr: np.ndarray) -> None:
        cv2.rectangle(frame_bgr, (10, 10), (300, 110), (0, 0, 0), -1)
        cv2.rectangle(frame_bgr, (10, 10), (300, 110), (255, 255, 255), 1)
        cv2.putText(
            frame_bgr,
            f"Skor: {self.score}",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            f"Seviye: {self.level}",
            (20, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 230, 255),
            2,
            cv2.LINE_AA,
        )
        mult = self._multiplier()
        combo_color = (255, 255, 255) if mult == 1 else (40, 220, 255)
        cv2.putText(
            frame_bgr,
            f"Combo: {self.combo}  x{mult}",
            (20, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            combo_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            f"En iyi: {self.best}  (combo {self.best_combo})",
            (20, 106),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    def _draw_level_up(self, frame_bgr: np.ndarray) -> None:
        text = f"LEVEL {self.level}!"
        scale = 1.6 + (35 - self._level_flash) * 0.02
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 3)
        x = (self.w - tw) // 2
        y = (self.h + th) // 2
        # golge
        cv2.putText(
            frame_bgr,
            text,
            (x + 3, y + 3),
            cv2.FONT_HERSHEY_DUPLEX,
            scale,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            text,
            (x, y),
            cv2.FONT_HERSHEY_DUPLEX,
            scale,
            self._level_color(),
            3,
            cv2.LINE_AA,
        )
