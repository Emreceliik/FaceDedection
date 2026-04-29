"""
Parmak Sayma modu:
- Elin algilanan parmak sayisini buyuk fontla ekrana yansitir.
- Algilamanin titrememesi icin son N karenin medyaninda yumusatma yapar.
- Konturu, konveks gobegi ve parmak ucunu gorsellestirir.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import cv2
import numpy as np

from hand_detector import HandResult


class CounterMode:
    def __init__(self, smoothing: int = 7):
        self._history: Deque[int] = deque(maxlen=smoothing)
        self._stable_count = 0

    def reset(self) -> None:
        self._history.clear()
        self._stable_count = 0

    def step(self, frame_bgr: np.ndarray, hand: Optional[HandResult]) -> None:
        h, w = frame_bgr.shape[:2]

        if hand is None:
            self._history.append(-1)
        else:
            self._history.append(hand.finger_count)
            self._draw_geometry(frame_bgr, hand)

        # medyan ile yumusat
        valid = [v for v in self._history if v >= 0]
        if valid:
            self._stable_count = int(np.median(valid))
        else:
            self._stable_count = -1

        self._draw_overlay(frame_bgr, w, h)

    # ------------------------------------------------------------------
    def _draw_geometry(self, frame_bgr: np.ndarray, hand: HandResult) -> None:
        cv2.drawContours(frame_bgr, [hand.contour], -1, (60, 220, 120), 2)
        hull = cv2.convexHull(hand.contour)
        cv2.drawContours(frame_bgr, [hull], -1, (240, 220, 60), 1)
        cv2.circle(frame_bgr, hand.center, 6, (40, 40, 240), -1)
        cv2.circle(frame_bgr, hand.fingertip, 9, (40, 220, 255), -1)
        cv2.circle(frame_bgr, hand.fingertip, 14, (40, 220, 255), 2)

    def _draw_overlay(self, frame_bgr: np.ndarray, w: int, h: int) -> None:
        if self._stable_count < 0:
            text = "El yok"
            color = (180, 180, 180)
        else:
            text = str(self._stable_count)
            color = (40, 220, 255)

        # buyuk sayi - sag alt
        scale = 4.0
        thickness = 6
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness
        )
        x = w - tw - 30
        y = h - 30
        # arka plan kutusu
        cv2.rectangle(
            frame_bgr,
            (x - 14, y - th - 14),
            (x + tw + 14, y + 14),
            (15, 15, 15),
            -1,
        )
        cv2.rectangle(
            frame_bgr,
            (x - 14, y - th - 14),
            (x + tw + 14, y + 14),
            color,
            2,
        )
        cv2.putText(
            frame_bgr,
            text,
            (x, y),
            cv2.FONT_HERSHEY_DUPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        # ust solda mod basligi
        cv2.putText(
            frame_bgr,
            "Parmak Sayma",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # parmak gorseli
        if self._stable_count >= 0:
            self._draw_finger_dots(frame_bgr, w)

    def _draw_finger_dots(self, frame_bgr: np.ndarray, w: int) -> None:
        n = max(0, min(5, self._stable_count))
        gap = 36
        start_x = w - 5 * gap - 30
        y = 32
        for i in range(5):
            color = (40, 220, 120) if i < n else (60, 60, 60)
            cv2.circle(frame_bgr, (start_x + i * gap, y), 12, color, -1)
            cv2.circle(frame_bgr, (start_x + i * gap, y), 12, (255, 255, 255), 1)
