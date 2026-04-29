"""
Balon Patlatma:
- Ekranin altinda rastgele renkli balonlar olusur, yukari dogru yuzer.
- Elin balonun uzerinden hizla gectiginde balon patlar (hareket+temas).
- Patlama: parcaciklar + skor.
- Cizgiyi (ust seridi) gecen balonlar can kaybettirir, 3 can.
- Combo: ust uste patlatmalar puan carpani verir.
"""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from hand_detector import HandResult


BALLOON_COLORS = [
    (60, 60, 240),
    (60, 180, 240),
    (60, 220, 130),
    (220, 120, 60),
    (220, 60, 200),
    (40, 200, 240),
]


class _Balloon:
    __slots__ = ("x", "y", "r", "vy", "wobble", "phase", "color", "alive")

    def __init__(self, x: float, y: float, r: int, vy: float, color):
        self.x = x
        self.y = y
        self.r = r
        self.vy = vy
        self.wobble = random.uniform(0.6, 1.6)
        self.phase = random.uniform(0.0, math.tau)
        self.color = color
        self.alive = True

    def step(self) -> None:
        self.y += self.vy
        self.phase += 0.08
        self.x += math.sin(self.phase) * self.wobble


class _Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "color", "size")

    def __init__(self, x, y, vx, vy, life, color, size):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life = life
        self.max_life = life
        self.color = color
        self.size = size

    def step(self) -> bool:
        self.vy += 0.18
        self.vx *= 0.97
        self.vy *= 0.97
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        return self.life > 0


class BalloonGame:
    def __init__(self, width: int = 720, height: int = 540):
        self.w = width
        self.h = height
        self.balloons: List[_Balloon] = []
        self.particles: List[_Particle] = []
        self._spawn_timer = 0
        self.score = 0
        self.best = 0
        self.combo = 0
        self.best_combo = 0
        self.lives = 3
        self.level = 1

        self._tip_trail: Deque[Tuple[int, int]] = deque(maxlen=8)
        self._game_over = False
        self._flash_msg = ""
        self._flash_frames = 0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.balloons.clear()
        self.particles.clear()
        self.score = 0
        self.combo = 0
        self.lives = 3
        self.level = 1
        self._spawn_timer = 0
        self._tip_trail.clear()
        self._game_over = False
        self._flash_msg = "Hadi baslayalim!"
        self._flash_frames = 28

    def _resize(self, w: int, h: int) -> None:
        if w == self.w and h == self.h:
            return
        self.w, self.h = w, h

    # ------------------------------------------------------------------
    def step(self, frame_bgr: np.ndarray, hand: Optional[HandResult]) -> None:
        h, w = frame_bgr.shape[:2]
        self._resize(w, h)

        if self._game_over:
            self._draw(frame_bgr, hand)
            return

        self._spawn_logic()
        for b in self.balloons:
            b.step()
        # ekrandan cikanlari ele al (ustten kacanlar can kaybettirir)
        new_balloons: List[_Balloon] = []
        for b in self.balloons:
            if not b.alive:
                continue
            if b.y + b.r < 60:
                self._lose_life()
                continue
            new_balloons.append(b)
        self.balloons = new_balloons

        if hand is not None:
            self._tip_trail.append(hand.fingertip)
            self._check_pops(hand)
        else:
            self._tip_trail.clear()
            self.combo = 0

        self.particles = [p for p in self.particles if p.step()]

        self._draw(frame_bgr, hand)

    # ------------------------------------------------------------------
    def _spawn_logic(self) -> None:
        self._spawn_timer -= 1
        if self._spawn_timer > 0:
            return
        # seviye yukseldikce daha sik spawn ve daha hizli balon
        base = max(14, 40 - self.level * 3)
        self._spawn_timer = random.randint(base, base + 18)
        r = random.randint(28, 44)
        x = random.randint(r + 20, max(r + 21, self.w - r - 20))
        y = self.h + r
        speed = -random.uniform(1.6, 2.2) - 0.18 * self.level
        color = random.choice(BALLOON_COLORS)
        self.balloons.append(_Balloon(x, y, r, speed, color))

    def _lose_life(self) -> None:
        self.lives -= 1
        self.combo = 0
        self._flash_msg = f"Kacti! Can: {max(0, self.lives)}"
        self._flash_frames = 22
        if self.lives <= 0:
            self._game_over = True
            if self.score > self.best:
                self.best = self.score
            self._flash_msg = "OYUN BITTI! Yeni oyun icin Space"
            self._flash_frames = 9999

    def _check_pops(self, hand: HandResult) -> None:
        if len(self._tip_trail) < 2:
            return
        # son hareket vektoru: bu sayede sadece "vurus" anlarinda patliyor
        p_prev = self._tip_trail[-2]
        p_now = self._tip_trail[-1]
        dx = p_now[0] - p_prev[0]
        dy = p_now[1] - p_prev[1]
        speed = math.hypot(dx, dy)
        if speed < 6.0:
            # cok yavas hareket: yine de el direkt balonun ustundeyse pat
            need_speed = False
        else:
            need_speed = True

        for b in self.balloons:
            if not b.alive:
                continue
            d = math.hypot(b.x - p_now[0], b.y - p_now[1])
            if d <= b.r + 6 and (need_speed or d < b.r * 0.6):
                self._pop(b)

    def _pop(self, b: _Balloon) -> None:
        b.alive = False
        self.combo += 1
        mult = 1
        if self.combo >= 8:
            mult = 4
        elif self.combo >= 5:
            mult = 3
        elif self.combo >= 3:
            mult = 2
        self.score += mult
        if self.combo > self.best_combo:
            self.best_combo = self.combo
        # seviye atla
        new_level = self.score // 12 + 1
        if new_level > self.level:
            self.level = new_level
            self._flash_msg = f"LEVEL {self.level}!"
            self._flash_frames = 30

        # parcacik patlamasi
        for _ in range(18):
            ang = random.uniform(0.0, math.tau)
            sp = random.uniform(2.0, 6.0)
            self.particles.append(
                _Particle(
                    b.x,
                    b.y,
                    math.cos(ang) * sp,
                    math.sin(ang) * sp,
                    random.randint(18, 30),
                    b.color,
                    random.randint(2, 5),
                )
            )

    # ------------------------------------------------------------------
    def _draw(self, frame_bgr: np.ndarray, hand: Optional[HandResult]) -> None:
        # ust serit (kacma cizgisi)
        cv2.line(frame_bgr, (0, 60), (self.w, 60), (40, 40, 220), 2)
        cv2.putText(
            frame_bgr,
            "Buradan kacarsa can gider",
            (10, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (40, 40, 220),
            1,
            cv2.LINE_AA,
        )

        # balonlar
        for b in self.balloons:
            cv2.circle(
                frame_bgr,
                (int(b.x), int(b.y)),
                b.r,
                b.color,
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                frame_bgr,
                (int(b.x), int(b.y)),
                b.r,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            # parlak nokta
            cv2.circle(
                frame_bgr,
                (int(b.x - b.r // 3), int(b.y - b.r // 3)),
                max(3, b.r // 6),
                (255, 255, 255),
                -1,
                cv2.LINE_AA,
            )
            # ip
            cv2.line(
                frame_bgr,
                (int(b.x), int(b.y + b.r)),
                (int(b.x + 6), int(b.y + b.r + 18)),
                (200, 200, 200),
                1,
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
                (int(p.x), int(p.y)),
                max(1, int(p.size * (0.3 + 0.7 * t))),
                col,
                -1,
            )

        # parmak ucu izi
        if len(self._tip_trail) >= 2:
            n = len(self._tip_trail)
            for i in range(1, n):
                p1 = self._tip_trail[i - 1]
                p2 = self._tip_trail[i]
                t = i / n
                cv2.line(
                    frame_bgr,
                    p1,
                    p2,
                    (40, 220, 255),
                    max(1, int(5 * t)),
                    cv2.LINE_AA,
                )

        if hand is not None:
            cv2.circle(frame_bgr, hand.fingertip, 10, (40, 220, 255), 2)

        self._draw_panel(frame_bgr)

        if self._flash_frames > 0:
            self._flash_frames -= 1
            (tw, th), _ = cv2.getTextSize(
                self._flash_msg, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2
            )
            x = (self.w - tw) // 2
            y = (self.h + th) // 2
            cv2.putText(
                frame_bgr,
                self._flash_msg,
                (x + 2, y + 2),
                cv2.FONT_HERSHEY_DUPLEX,
                1.0,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame_bgr,
                self._flash_msg,
                (x, y),
                cv2.FONT_HERSHEY_DUPLEX,
                1.0,
                (40, 220, 255),
                2,
                cv2.LINE_AA,
            )

    def _draw_panel(self, frame_bgr: np.ndarray) -> None:
        cv2.rectangle(frame_bgr, (self.w - 230, 70), (self.w - 10, 180), (0, 0, 0), -1)
        cv2.rectangle(frame_bgr, (self.w - 230, 70), (self.w - 10, 180), (255, 255, 255), 1)
        cv2.putText(
            frame_bgr,
            f"Skor: {self.score}",
            (self.w - 220, 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            f"Seviye: {self.level}",
            (self.w - 220, 122),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 230, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            f"Combo: {self.combo}",
            (self.w - 220, 144),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 220, 255),
            2,
            cv2.LINE_AA,
        )
        # canlar
        for i in range(3):
            color = (60, 60, 230) if i < self.lives else (60, 60, 60)
            cv2.circle(
                frame_bgr, (self.w - 220 + i * 28, 168), 8, color, -1, cv2.LINE_AA
            )
