"""
Hava Cizimi:
- 1 parmak: parmak ucunla cizer.
- 2 parmak: tasi (cizmeden imleci tasir).
- >=4 parmak: TUM tuvali temizler.
- 0 parmak (yumruk): silgi modu, tuvalde gezdiginiz alanlari siler.

Renk paleti ekranin ust serisinde gosterilir, parmak ucunu uzerinde
yarim saniye tutarsaniz seciliyor.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from hand_detector import HandResult


PALETTE: List[Tuple[int, int, int]] = [
    (50, 50, 240),    # kirmizi
    (40, 140, 240),   # turuncu
    (40, 220, 240),   # sari
    (60, 220, 80),    # yesil
    (240, 180, 50),   # mavi
    (220, 80, 200),   # mor
    (240, 240, 240),  # beyaz
]


class PaintGame:
    def __init__(self, width: int = 720, height: int = 540):
        self.w = width
        self.h = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.color_idx = 0
        self.brush = 8
        self._last_point: Optional[Tuple[int, int]] = None
        self._last_mode: str = "idle"
        self._hover_idx: Optional[int] = None
        self._hover_frames = 0
        self._flash_msg = ""
        self._flash_frames = 0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.canvas[:] = 0
        self._last_point = None
        self._flash_msg = "Tuval temizlendi"
        self._flash_frames = 30

    def _resize(self, w: int, h: int) -> None:
        if w == self.w and h == self.h:
            return
        self.w, self.h = w, h
        self.canvas = cv2.resize(self.canvas, (w, h))

    # ------------------------------------------------------------------
    def step(self, frame_bgr: np.ndarray, hand: Optional[HandResult]) -> None:
        h, w = frame_bgr.shape[:2]
        self._resize(w, h)

        if hand is None:
            self._last_point = None
            self._last_mode = "idle"
        else:
            self._handle_hand(hand)

        self._compose(frame_bgr)

    # ------------------------------------------------------------------
    def _handle_hand(self, hand: HandResult) -> None:
        fc = hand.finger_count
        tip = hand.fingertip
        center = hand.center

        # ust serideki renk paletinin uzerinde mi?
        if self._check_palette_hover(tip):
            self._last_point = None
            self._last_mode = "palette"
            return

        if fc >= 4:
            # tum tuvali temizle
            if self._last_mode != "clear":
                self.reset()
            self._last_mode = "clear"
            self._last_point = None
            return

        if fc == 0:
            # silgi: el merkezinin etrafinda silgi cember
            r = max(20, int(np.sqrt(hand.area) * 0.18))
            cv2.circle(self.canvas, center, r, (0, 0, 0), -1)
            self._last_mode = "erase"
            self._last_point = None
            return

        if fc == 1:
            # cizim
            color = PALETTE[self.color_idx]
            if self._last_mode == "draw" and self._last_point is not None:
                cv2.line(
                    self.canvas, self._last_point, tip, color, self.brush, cv2.LINE_AA
                )
            cv2.circle(self.canvas, tip, max(1, self.brush // 2), color, -1)
            self._last_point = tip
            self._last_mode = "draw"
            return

        # 2-3 parmak: hareket modu (cizmez)
        self._last_mode = "hover"
        self._last_point = None

    def _check_palette_hover(self, tip: Tuple[int, int]) -> bool:
        n = len(PALETTE)
        cell_w = self.w // (n + 1)
        cell_h = 50
        if tip[1] > cell_h:
            self._hover_idx = None
            self._hover_frames = 0
            return False
        idx = tip[0] // cell_w
        if idx >= n:
            self._hover_idx = None
            self._hover_frames = 0
            return False
        if self._hover_idx == idx:
            self._hover_frames += 1
        else:
            self._hover_idx = idx
            self._hover_frames = 1
        if self._hover_frames >= 12 and self.color_idx != idx:
            self.color_idx = int(idx)
            self._flash_msg = "Renk degisti"
            self._flash_frames = 20
        return True

    # ------------------------------------------------------------------
    def _compose(self, frame_bgr: np.ndarray) -> None:
        # tuvali kameraya bindir (cizilen yerler tam opak)
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        inv = cv2.bitwise_not(mask)
        bg = cv2.bitwise_and(frame_bgr, frame_bgr, mask=inv)
        fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
        cv2.add(bg, fg, dst=frame_bgr)

        self._draw_palette(frame_bgr)
        self._draw_hud(frame_bgr)

    def _draw_palette(self, frame_bgr: np.ndarray) -> None:
        n = len(PALETTE)
        cell_w = self.w // (n + 1)
        cell_h = 50
        cv2.rectangle(frame_bgr, (0, 0), (self.w, cell_h), (20, 20, 20), -1)
        for i, color in enumerate(PALETTE):
            x1 = i * cell_w + 6
            x2 = (i + 1) * cell_w - 6
            cv2.rectangle(frame_bgr, (x1, 6), (x2, cell_h - 6), color, -1)
            if i == self.color_idx:
                cv2.rectangle(
                    frame_bgr, (x1 - 3, 3), (x2 + 3, cell_h - 3), (255, 255, 255), 2
                )
            if self._hover_idx == i and self.color_idx != i:
                # progress yayi
                prog = min(1.0, self._hover_frames / 12.0)
                bar_x2 = x1 + int((x2 - x1) * prog)
                cv2.rectangle(
                    frame_bgr, (x1, cell_h - 8), (bar_x2, cell_h - 4), (255, 255, 255), -1
                )
        # silgi gostergesi en sagda
        ex1 = n * cell_w + 6
        ex2 = (n + 1) * cell_w - 6
        cv2.rectangle(frame_bgr, (ex1, 6), (ex2, cell_h - 6), (40, 40, 40), -1)
        cv2.putText(
            frame_bgr,
            "TEMIZLE",
            (ex1 + 4, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def _draw_hud(self, frame_bgr: np.ndarray) -> None:
        text = f"Mod: {self._last_mode.upper()}"
        cv2.putText(
            frame_bgr,
            text,
            (10, self.h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if self._flash_frames > 0:
            self._flash_frames -= 1
            cv2.putText(
                frame_bgr,
                self._flash_msg,
                (self.w // 2 - 100, self.h // 2),
                cv2.FONT_HERSHEY_DUPLEX,
                1.0,
                (40, 220, 255),
                2,
                cv2.LINE_AA,
            )
